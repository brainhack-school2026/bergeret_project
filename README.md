# brainhack-2026-multimodal

Reproducible multimodal EEG/fMRI fusion pipeline for psychiatric prediction.
Built for Brainhack School 2026 using the [`airoh`](https://pypi.org/project/airoh/) task runner.

The pipeline trains separate prediction models on EEG features, fMRI connectivity features,
and both combined — so you can compare whether one modality, the other, or their fusion
best predicts a chosen phenotypic target (e.g. diagnosis, age, a clinical score).
Primary metric is **AUC-ROC** for classification and **Pearson r** for regression.

**Designed to be used with [Claude Code](https://claude.ai/code).** Run `/init-airoh-project`
in a fresh clone to set up or extend the pipeline.

---

## Quick Start

```bash
uv sync
uv run invoke run-smoke   # end-to-end test with synthetic data
uv run invoke run         # full pipeline (requires real data in source_data/)
```

---

## Setup

```bash
uv sync
```

---

## Data inputs

Place your data files in `source_data/` and configure paths in `invoke.yaml`.
See [`source_data/CONTENT.md`](source_data/CONTENT.md) for the expected formats.

The pipeline accepts two input formats for each modality:

| Modality | Format A | Format B |
|---|---|---|
| EEG | `eeg_features.tsv` (flat table) | `mne_output/` (MNE feature export folder) |
| fMRI | `fmri_features.tsv` (flat table) | `halfpipe_output/` (Halfpipe connectivity matrices) |

Set `eeg_input_type` and `fmri_input_type` in `invoke.yaml` to `"tsv"` or `"mne"` / `"halfpipe"`.

---

## Task Overview

| Task | Description |
|---|---|
| `fetch` | Print instructions for placing real source data |
| `generate-smoke-data` | Generate lightweight synthetic data for testing |
| `run-intersect` | Compute subject intersection across EEG, fMRI, and phenotype; drop subjects with missing confound values → `output_data/subjects.txt` |
| `run-load-eeg` | Load and harmonise EEG features → `output_data/eeg_features.tsv` |
| `run-load-fmri` | Load and harmonise fMRI connectivity → `output_data/fmri_features.tsv` |
| `run-predict` | Train and evaluate EEG-only, fMRI-only, and multimodal models → `output_data/results/` |
| `run-notebooks` | Execute notebooks and save figures to `output_data/` |
| `run` | Full pipeline (all steps in order) |
| `run-smoke` | Smoke test: synthetic data + minimal end-to-end pass |
| `clean` | Remove all generated outputs and synthetic data |
| `clean-intersect` | Remove `output_data/subjects.txt` |
| `clean-outputs` | Remove flat TSV and PNG outputs from `output_data/` |
| `clean-predict` | Remove prediction results from `output_data/results/` |
| `clean-smoke` | Remove synthetic smoke data from `source_data/smoke/` |

Use `invoke --list` or `invoke --help <task>` for details.

---

## Configuration

All settings live in `invoke.yaml`. Key options for the prediction step:

| Key | Default | Description |
|---|---|---|
| `target_column` | `diagnosis` | Phenotype column to predict (binary/integer → classification with AUC, continuous → regression with Pearson r + MAE) |
| `model_type` | `ridge` | Model: `logistic`, `ridge`, `elasticnet`, `svm`, `random_forest` |
| `cv_outer_folds` | `5` | Number of outer cross-validation folds (evaluation) |
| `cv_inner_folds` | `5` | Number of inner folds (hyperparameter tuning, optimises AUC / neg-MAE) |
| `pca_variance` | `0.95` | Fraction of variance retained by PCA per modality |
| `n_permutations` | `100` | Number of permutations for the null distribution (p-value vs chance) — use ≥500 for publication |
| `eeg_input_type` | `tsv` | EEG input format: `tsv` or `mne` |
| `fmri_input_type` | `tsv` | fMRI input format: `tsv` or `halfpipe` |
| `fmri_halfpipe_strategy` | `36P` | Halfpipe denoising strategy tag (e.g. `36P`, `aCompCor`) |

---

## Output

See [`output_data/CONTENT.md`](output_data/CONTENT.md) for a description of all generated files.

### Notebooks

| Notebook | Description | Figures produced |
|---|---|---|
| `notebooks/results_overview.ipynb` | Visualises prediction results from `output_data/results/` | `scores_by_condition.png` (bar + fold overlay, p-value annotations), `fold_distribution.png` (violin per condition) |

---

## Philosophy

- **Analysis in code, visualization in notebooks.** Heavy computation lives in `analysis/`; notebooks only read results and produce figures.
- **Idempotent steps.** Each `run-{name}` task skips if outputs already exist. Call `invoke clean` to force a full rerun.
- **Smoke tests.** `invoke run-smoke` generates synthetic data and runs the full pipeline in seconds.
- **Two input formats per modality.** Both flat TSVs and raw tool outputs (MNE, Halfpipe) are supported.
