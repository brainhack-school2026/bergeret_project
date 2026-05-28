# brainhack-2026-multimodal

<a href="https://github.com/pbergeret12">
  <img src="https://avatars.githubusercontent.com/u/81371004?v=4&s=100" width="100px;" alt=""/>
  <br /><sub><b>Pierre Bergeret</b></sub>
</a>

### About Me

PhD Student in Psychiatry at the Université de Montréal specialized in neuroimaging analysis.

### Project slides

https://docs.google.com/presentation/d/1ZMddc8o5beXn3dZDECPOYwVRuh-DVODSOPS99YFSp7M/edit?slide=id.p#slide=id.p

---

Reproducible multimodal EEG/fMRI fusion pipeline for psychiatric prediction.
Built for Brainhack School 2026 using the [`airoh`](https://pypi.org/project/airoh/) task runner.

The pipeline trains separate prediction models on EEG features, fMRI connectivity features,
and both combined — so you can compare whether one modality, the other, or their fusion
best predicts a chosen phenotypic target (e.g. diagnosis, age, a clinical score).
Primary metric is **AUC-ROC** for classification and **MAE** for regression.

**Designed to be used with [Claude Code](https://claude.ai/code).** Run `/init-airoh-project`
in a fresh clone to set up or extend the pipeline.

---

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Smoke test — generates synthetic data and runs the full pipeline end-to-end
uv run invoke run-smoke

# 3. Switch to real data
#    - Edit invoke.yaml: set phenotype_file, eeg_tsv / eeg_mne_dir,
#      fmri_tsv / fmri_halfpipe_dir, eeg_input_type, fmri_input_type
#    - Place your data files in source_data/ (see source_data/CONTENT.md)
uv run invoke clean       # remove smoke outputs so the real run is not skipped
uv run invoke run         # full pipeline with your data
```

---

## Setup

```bash
# uv (recommended — handles virtualenv automatically):
uv sync

# pip (e.g. on HPC without uv):
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows
pip install -r requirements.txt

# conda:
conda env create -f environment.yml
conda activate airoh_env
```

> **HPC note:** if your cluster uses environment modules, load Python first:
> ```bash
> module load python/3.11   # adapt to your cluster's module name
> python -m venv venv
> source venv/bin/activate
> pip install -r requirements.txt
> ```

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
| `run-predict` | Train and evaluate EEG-only, fMRI-only, and multimodal models → `output_data/results/{target}/` |
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
| `notebooks/results_overview.ipynb` | Visualises prediction results from `output_data/results/` | `scores_by_condition_{target}.png` (bar + fold overlay, p-value annotations), `fold_distribution_{target}.png` (violin per condition) — one pair of figures per prediction target |

---

## Philosophy

- **Analysis in code, visualization in notebooks.** Heavy computation lives in `analysis/`; notebooks only read results and produce figures.
- **Idempotent steps.** Each `run-{name}` task skips if outputs already exist. Call `invoke clean` to force a full rerun.
- **Smoke tests.** `invoke run-smoke` generates synthetic data and runs the full pipeline in seconds.
- **Two input formats per modality.** Both flat TSVs and raw tool outputs (MNE, Halfpipe) are supported.
