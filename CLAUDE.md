# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is **NeuroMeld** — a reproducible multimodal EEG/fMRI fusion pipeline for psychiatric prediction, built for Brainhack School 2026. It is built on the [`invoke`](https://www.pyinvoke.org/) task runner. The `airoh` pip package provides reusable invoke tasks; this repo customizes them via `tasks.py` and `invoke.yaml`.

**Goal:** predict a configurable phenotypic target (e.g. diagnosis, age) using EEG features only, fMRI connectivity features only, and both combined — to compare modality contributions.

**Input formats:** the pipeline accepts flat TSVs or raw tool outputs for each modality. The format is auto-detected from the path set in `invoke.yaml`:
- EEG (`eeg_path`): any `.tsv` file → flat feature table; any directory containing `*_eeg.fif` → MNE-BIDS (band-power features extracted automatically via `load_eeg_mne()`). Configure `eeg_mne_task` to select the BIDS task label (default: `rest`).
- fMRI (`fmri_path`): any `.tsv` file → flat connectivity table; any directory containing `*_desc-correlation_matrix.tsv` → Halfpipe derivatives. Configure `fmri_halfpipe_strategy` for the denoising tag.

**Chunk concept:** subjects (`participant_id`) are the unit of processing.

**Smoke data:** `invoke generate-smoke-data` populates `source_data/smoke/` with 30 synthetic subjects in all four input formats. `invoke run-smoke` uses this data to test the pipeline end-to-end. The MNE-BIDS smoke data contains real `.fif` files generated with `mne.io.RawArray`; the EEG TSV is derived from them by running the same feature extractor. The synthetic data includes ~2% NaN in fMRI features, one subject with missing age (exercises confound-drop in `run_intersect`), and two independent latent signals so both `diagnosis` and `age` targets are weakly predictable even after confound correction.

## Persona

Respond as Uncle Airoh: patient, warm, and wise. Assume the user may be new to coding. Explain errors gently, encourage before correcting, and frame tradeoffs as learning opportunities. When things get heated, offer a calming cup of jasmine tea.

## Setup

```bash
# uv (recommended):
uv sync

# pip:
pip install -r requirements.txt

# conda:
conda env create -n airoh_env -f environment.yml && conda activate airoh_env
```

## Common Commands

With `uv`:
```bash
uv run invoke fetch           # Download source data
uv run invoke run             # Full pipeline (project-specific pre= chain)
uv run invoke run-notebooks   # Execute notebooks, save figures to output_data/
uv run invoke clean           # Remove output_data/ contents
uv run invoke --list          # Show all available tasks
```

Without `uv` (activate your environment first):
```bash
invoke fetch              # Download source data (configured in invoke.yaml under files:)
invoke run                # Full pipeline (project-specific pre= chain)
invoke run-notebooks      # Execute notebooks, save figures to output_data/
invoke clean              # Remove output_data/ contents
invoke --list             # Show all available tasks
```

## Pipeline steps

The full pipeline runs as: `fetch → run-load-eeg → run-load-fmri → run-predict → run-notebooks`

| Task | Input | Output | Key module |
|---|---|---|---|
| `run-intersect` | EEG source + fMRI source + phenotype | `output_data/subjects.txt` | `tasks.py` |
| `run-load-eeg` | TSV or MNE folder (filtered to `subjects.txt`) | `output_data/eeg_features.tsv` | `analysis/load_eeg.py` |
| `run-load-fmri` | TSV or Halfpipe folder (filtered to `subjects.txt`) | `output_data/fmri_features.tsv` | `analysis/load_fmri.py` |
| `run-predict` | both feature TSVs + phenotype | `output_data/results/metrics.tsv`, `fold_scores.tsv` | `analysis/predict.py` |
| `run-notebooks` | `output_data/results/` | `output_data/scores_by_condition_{target}.png`, `feature_importance_{target}.png` | `notebooks/results_overview.ipynb` |

**Cleaning tasks:** `clean-intersect` removes `subjects.txt`; `clean-outputs` removes flat TSVs and PNGs; `clean-predict` removes `output_data/results/`; `clean-smoke` removes `source_data/smoke/`. The top-level `clean` calls all four.

**Subject intersection:** `run-intersect` reads only subject IDs (not features) from each source — directory listing for MNE/Halfpipe, `participant_id` column for TSVs — and writes the common set to `output_data/subjects.txt`. It also reads the phenotype file and drops subjects that have missing values in any confound column (`age`, `gender`, `study_site`, excluding the target). This ensures subjects with incomplete confounds never reach `correct_confounds`, where a single NaN row would silently corrupt the entire residual matrix. Both `run-load-eeg` and `run-load-fmri` have `pre=[run_intersect]` so the intersection always runs first.

## Prediction pipeline (`analysis/predict.py`)

`run_prediction()` is the main entry point. It executes three conditions — EEG-only, fMRI-only, multimodal — with identical methodology:

1. **Subject alignment** — keep only subjects present in all three inputs (EEG, fMRI, phenotype).
2. **Task detection** — binary/low-cardinality integer target → classification; continuous → regression.
3. **GLM confound correction** — OLS regression of `age`, `gender`, `study_site` out of features; the target column is never used as a confound. Subjects with missing confound values are excluded upstream in `run-intersect`.
4. **NaN handling** — before entering the CV loop, `load_eeg` and `load_fmri` both drop all-NaN feature columns and median-impute remaining sparse NaN at the group level.
5. **Nested cross-validation** — outer k-fold for generalisation, inner k-fold for hyperparameter tuning. Inside each outer fold:
   - `SimpleImputer` (median) → `StandardScaler` → PCA (fitted on the training split only) → model. In the multimodal condition the scaler+PCA is fitted **independently per modality** (EEG, fMRI) via a `ColumnTransformer` and the per-modality components are concatenated before the model, so the larger fMRI feature block cannot dominate a shared PCA and crush the EEG block.
   - `GridSearchCV` on inner splits optimises **AUC** for classification, **neg-MAE** for regression.
   - Best model evaluated on the held-out outer fold.
6. **Permutation test (vs chance)** — `n_permutations` shuffles of `y` build a null distribution; p-value = fraction of null scores ≥ observed (primary metric: AUC for classification, MAE for regression — for MAE the count is ≤ observed, lower is better). One shared set of permutations is drawn once and applied to all three conditions at the same permutation index.
7. **Inter-modality permutation test** — significance of the score *difference* between conditions (EEG-only vs fMRI-only, EEG-only vs multimodal, fMRI-only vs multimodal). For each pair the observed Δ = score(a) − score(b); the null Δ reuses the **same permuted labels** for both conditions (the shared permutations from step 6, aligned by index), so the test is paired. Two-sided p-value = (#{|Δ_null| ≥ |Δ_obs|} + 1) / (n_permutations + 1). This replaces an earlier paired t-test on the few, non-independent CV folds, which was both underpowered and miscalibrated (overlapping training sets violate the t-test independence assumption).

**Primary metrics:**
- Classification: `roc_auc` (AUC-ROC). `balanced_accuracy` is also reported.
- Regression: `mae` (used for permutation tests and figures). `pearson_r` and `r2` also reported. For MAE the permutation p-value counts null scores ≤ observed (lower is better).

**Supported models** (`model_type` in `invoke.yaml`): `logistic`, `ridge`, `elasticnet`, `svm`, `random_forest`. For regression targets, `logistic` maps to `Ridge`.

**Outputs:**
- `output_data/results/{target}/metrics.tsv` — one row per condition: mean/std for all metrics, `p_vs_chance`, and the inter-modality difference p-values (`p_eeg_only_vs_fmri_only`, `p_eeg_only_vs_multimodal`, `p_fmri_only_vs_multimodal`).
- `output_data/results/{target}/fold_scores.tsv` — raw per-fold scores in long format.
- `output_data/results/{target}/feature_importances.tsv` — per-feature importance (mean/std across outer folds), columns: `condition`, `feature`, `modality`, `importance_mean`, `importance_std`. Not produced for SVM RBF.

**Feature importance:** `_extract_feature_importance(best_pipe)` projects model-native importance back to original feature space. For linear models (`coef_`): `mean(|coef|) @ |pca.components_| / scaler.scale_`. For RandomForest (`feature_importances_`): `fi @ |pca.components_|` (MDI in PCA space, approximate). For SVM RBF: returns None (skipped with warning). In the multimodal condition the pipeline has one PCA per modality, so each modality block of the model coefficients is projected back through its **own** PCA/scaler and the results concatenated in original feature order (EEG first, then fMRI); features are tagged `eeg` or `fmri`.

**Known methodological limitations (acceptable for Brainhack, revisit before publication):**
- *Confound correction before CV*: `correct_confounds` fits OLS betas on all subjects (including test folds) before the CV loop. The strictly correct approach is to fit the confound model on each training fold and apply it to the held-out fold. This refactor would require passing the confound matrix into `_run_nested_cv`. The bias introduced by the current approach is small when confounds are weakly correlated with features, but could inflate performance estimates in edge cases (e.g. strong site effects with small N per site).
- *Group-level NaN imputation before CV*: `impute_eeg_features` and `impute_fmri_features` compute column medians across all subjects (test included) and impute sparse NaN before the CV loop. The `SimpleImputer` inside the CV fold is therefore a no-op. Strictly, the median should be computed on the training fold only. In practice the bias is negligible (~2% NaN, large N) — moving imputation inside CV would not lose any subjects (the imputer still fills NaN, just with a training-only median).
- *Null hypothesis of the inter-modality test*: the difference test permutes the labels, so its null is "neither condition carries any signal". It therefore answers "is the observed performance difference larger than what no-signal data would produce?" rather than the stricter "do both modalities carry signal but to a different degree?". This is the standard, conservative label-permutation approach and is far better calibrated than a t-test on overlapping CV folds, but it is not a test of equal-but-nonzero performance.

## Notebooks

`notebooks/results_overview.ipynb` reads `output_data/results/` and produces figures per target:
- `scores_by_condition_{target}.png` — bar chart (mean ± std) with per-fold overlay, the exact metric value + significance vs chance annotated above each bar, and significance brackets (inter-modality permutation test).
- `feature_importance_{target}.png` — top-20 features per condition (horizontal bar chart), coloured by modality.
- A significance summary cell prints p-values (vs chance + inter-modality permutation tests) with star notation.

Notebooks receive `OUTPUT_DATA_DIR` and `SOURCE_DATA_DIR` as environment variables (injected by `airoh.utils.run_notebooks`). All heavy computation must remain in `analysis/` — notebooks are visualization only.

## Architecture

**Always read `tasks.py` first** before proposing or implementing any pipeline change — it is the authoritative source of what tasks exist, how they are wired, and what parameters they accept.

**Execution flow:** `invoke run` triggers the project's analysis pipeline via `pre=` dependencies declared in `tasks.py`. The three permanent tasks — `fetch`, `run`, `clean` — are always present; intermediate steps are project-specific.

- `invoke.yaml` — all path, data, and model config (see Configuration section in README.md)
- `tasks.py` — project-specific invoke tasks; imports reusable tasks from `airoh.utils`
- `analysis/` — pure Python analysis logic, called by tasks in `tasks.py`
- `notebooks/` — Jupyter notebooks executed by `run_notebooks` via `airoh.utils.run_notebooks`; notebooks receive `OUTPUT_DATA_DIR` and `SOURCE_DATA_DIR` as environment variables
- `source_data/CONTENT.md` and `output_data/CONTENT.md` — authoritative docs for what each data folder contains; update these when data assets change, do not duplicate their content elsewhere

**Analysis vs. notebooks:** Heavy computation belongs in `analysis/` Python code, invoked by `run-{name}` tasks, which write results to `output_data/`. Notebooks are for visualization only — they read from `output_data/` and produce figures. This keeps notebooks fast and focused.

**Idempotent tasks:** Each `run-{name}` task must check whether its outputs already exist and skip execution if they do. This means `invoke run` can be called repeatedly during development of a later step — earlier steps are skipped automatically. To force a full rerun, call `invoke clean` first, then `invoke run`.

**Task naming conventions:**
- Analysis tasks are named `run-{name}` (e.g. `run-preprocessing`, `run-model`).
- Cleaning tasks mirror them: `clean-{name}` removes only the outputs of the corresponding step.
- The top-level `clean` task calls all `clean-{name}` tasks in sequence.
- The top-level `run` task wires all steps together via `pre=` chains in `tasks.py`.

**Task parameters:** `run-{name}` tasks should expose chunk or subset parameters (e.g. a subject ID, a chunk index) so that individual pieces can be rerun in isolation. They should also support a `smoke` flag for a fast minimal run useful for testing the pipeline end-to-end without running the full analysis.

**Adding a new analysis step:** add a function to `analysis/`, add a `run-{name}` task and a matching `clean-{name}` task in `tasks.py`, wire both into the top-level `run` and `clean` tasks via `pre=` chains, and create or extend a notebook in `notebooks/` for visualization.

**Evolving CLAUDE.md:** Keep this file current as the project grows. It should always reflect the actual scope of the project — what it does, what data it uses, and what analysis steps it contains. When adding or removing a task, rename a folder, or change the pipeline structure, update CLAUDE.md in the same commit. Stale guidance here misleads future AI sessions and collaborators alike.

**Keeping README.md current:** README.md is the user-facing documentation for this project. Any structural or workflow change — new tasks, renamed folders, updated commands, new dependencies — must be reflected there in the same commit. The task list in README.md should match `invoke --list` exactly; if a task is added or removed, update README.md accordingly. For data folder contents, point to `source_data/CONTENT.md` and `output_data/CONTENT.md` rather than duplicating their content inline.
