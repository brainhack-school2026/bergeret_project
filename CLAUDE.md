# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is **brainhack-2026-multimodal** — a reproducible multimodal EEG/fMRI fusion pipeline for psychiatric prediction, built for Brainhack School 2026. It is built on the [`invoke`](https://www.pyinvoke.org/) task runner. The `airoh` pip package provides reusable invoke tasks; this repo customizes them via `tasks.py` and `invoke.yaml`.

**Goal:** predict a configurable phenotypic target (e.g. diagnosis, age) using EEG features only, fMRI connectivity features only, and both combined — to compare modality contributions.

**Input formats:** the pipeline accepts flat TSVs or raw tool outputs for each modality:
- EEG: `eeg_features.tsv` OR `mne_output/` (MNE feature export, one CSV per subject)
- fMRI: `fmri_features.tsv` OR `halfpipe_output/` (Halfpipe connectivity matrices, one TSV per subject/run/strategy)

**Chunk concept:** subjects (`participant_id`) are the unit of processing.

**Smoke data:** `invoke generate-smoke-data` populates `source_data/smoke/` with 30 synthetic subjects in all four input formats. `invoke run-smoke` uses this data to test the pipeline end-to-end. The synthetic data includes ~2% NaN (sparse values + deliberately all-NaN columns), one subject with missing age (exercises confound-drop in `run_intersect`), and two independent latent signals so both `diagnosis` and `gender` targets are weakly predictable even after confound correction.

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
| `run-notebooks` | `output_data/results/` | `output_data/scores_by_condition.png`, `fold_distribution.png` | `notebooks/results_overview.ipynb` |

**Cleaning tasks:** `clean-intersect` removes `subjects.txt`; `clean-outputs` removes flat TSVs and PNGs; `clean-predict` removes `output_data/results/`; `clean-smoke` removes `source_data/smoke/`. The top-level `clean` calls all four.

**Subject intersection:** `run-intersect` reads only subject IDs (not features) from each source — directory listing for MNE/Halfpipe, `participant_id` column for TSVs — and writes the common set to `output_data/subjects.txt`. It also reads the phenotype file and drops subjects that have missing values in any confound column (`age`, `gender`, `study_site`, excluding the target). This ensures subjects with incomplete confounds never reach `correct_confounds`, where a single NaN row would silently corrupt the entire residual matrix. Both `run-load-eeg` and `run-load-fmri` have `pre=[run_intersect]` so the intersection always runs first.

## Prediction pipeline (`analysis/predict.py`)

`run_prediction()` is the main entry point. It executes three conditions — EEG-only, fMRI-only, multimodal — with identical methodology:

1. **Subject alignment** — keep only subjects present in all three inputs (EEG, fMRI, phenotype).
2. **Task detection** — binary/low-cardinality integer target → classification; continuous → regression.
3. **GLM confound correction** — OLS regression of `age`, `gender`, `study_site` out of features; the target column is never used as a confound. Subjects with missing confound values are excluded upstream in `run-intersect`.
4. **NaN handling** — before entering the CV loop, `load_eeg` and `load_fmri` both drop all-NaN feature columns and median-impute remaining sparse NaN at the group level.
5. **Nested cross-validation** — outer k-fold for generalisation, inner k-fold for hyperparameter tuning. Inside each outer fold:
   - `SimpleImputer` (median) → PCA (fitted on training split only) → `StandardScaler` → model.
   - `GridSearchCV` on inner splits optimises **AUC** for classification, **neg-MAE** for regression.
   - Best model evaluated on the held-out outer fold.
6. **Permutation test** — `n_permutations` shuffles of `y` build a null distribution; p-value = fraction of null scores ≥ observed (primary metric: AUC for classification, Pearson r for regression).
7. **Paired t-tests** — fold-level scores compared between EEG-only vs fMRI-only, EEG-only vs multimodal, fMRI-only vs multimodal.

**Primary metrics:**
- Classification: `roc_auc` (AUC-ROC). `balanced_accuracy` is also reported.
- Regression: `pearson_r` (used for permutation tests). `mae` and `r2` also reported.

**Supported models** (`model_type` in `invoke.yaml`): `logistic`, `ridge`, `elasticnet`, `svm`, `random_forest`. For regression targets, `logistic` maps to `Ridge`.

**Outputs:**
- `output_data/results/metrics.tsv` — one row per condition: mean/std for all metrics, `p_vs_chance`, paired p-values.
- `output_data/results/fold_scores.tsv` — raw per-fold scores in long format.

## Notebooks

`notebooks/results_overview.ipynb` reads `output_data/results/` and produces two figures:
- `scores_by_condition.png` — bar chart (mean ± std) with per-fold overlay and p-vs-chance annotations.
- `fold_distribution.png` — violin plot of per-fold score distribution per condition.
- A significance summary cell prints p-values (vs chance + paired t-tests) with star notation.

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
