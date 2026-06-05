# NeuroMeld

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
#    - Edit invoke.yaml: set phenotype_file, eeg_path, fmri_path
#      (point each to a .tsv file or a directory — format is auto-detected)
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

## Container (Singularity / Apptainer)

For maximum reproducibility on HPC clusters, a Singularity image bakes the
code and all dependencies. Data is provided at runtime via bind mounts.
Input format (TSV vs MNE-BIDS / Halfpipe) is **auto-detected** from the path.

### Get the image

**Option A — Pull from GitHub Container Registry (recommended, no build needed):**
```bash
# Apptainer (Compute Canada / most HPC):
apptainer pull neuromeld.sif docker://ghcr.io/pbergeret12/neuromeld:latest

# Singularity:
singularity pull neuromeld.sif docker://ghcr.io/pbergeret12/neuromeld:latest
```

**Option B — Build from source locally, transfer to HPC:**
```bash
# 1. Build for linux/amd64 (required for HPC — Mac users must specify platform)
docker buildx build --platform linux/amd64 -t neuromeld:amd64 --load .

# 2. Save as tar
docker save neuromeld:amd64 -o neuromeld.tar

# 3. Transfer and convert on HPC
scp neuromeld.tar user@hpc.cluster.ca:~/
apptainer build neuromeld.sif docker-archive://neuromeld.tar
```

**Option C — Build directly on HPC (fakeroot required):**
```bash
apptainer build --fakeroot neuromeld.sif singularity.def
```

### Smoke test (no data needed)

Verify the image works end-to-end with synthetic data:
```bash
apptainer run neuromeld.sif --smoke

# Keep the outputs:
apptainer run neuromeld.sif --smoke --output-dir ./smoke_outputs
```

### Run

```bash
singularity run \
  -B /path/to/source_data:/data/source_data \
  -B /path/to/output_data:/data/output_data \
  brainhack_multimodal.sif \
  --eeg-path  /data/source_data/eeg_features.tsv \
  --fmri-path /data/source_data/halfpipe_output/ \
  --target-column diagnosis \
  --model-type ridge \
  --n-permutations 100
```

`--eeg-path` and `--fmri-path` accept any file (flat TSV) or directory (MNE-BIDS / Halfpipe) — format is auto-detected. They default to `/data/source_data/eeg_features.tsv` and `/data/source_data/fmri_features.tsv`.

All options:
```
--eeg-path PATH              EEG data: .tsv file or MNE-BIDS directory
                             [/data/source_data/eeg_features.tsv]
--fmri-path PATH             fMRI data: .tsv file or Halfpipe directory
                             [/data/source_data/fmri_features.tsv]
--target-column STR          Column to predict                    [diagnosis]
--model-type STR             logistic|ridge|elasticnet|svm|rf     [ridge]
--n-permutations INT         Permutations for null distribution    [100]
--fmri-halfpipe-strategy STR Halfpipe denoising strategy tag      [Baseline]
--cv-outer-folds INT         Outer CV folds                        [5]
--cv-inner-folds INT         Inner CV folds                        [5]
--pca-variance FLOAT         PCA explained variance threshold      [0.95]
--smoke                      Self-contained smoke test (no mounts)
```

### SLURM example

```bash
#!/bin/bash
#SBATCH --job-name=neuromeld
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

singularity run \
  -B $SLURM_SUBMIT_DIR/source_data:/data/source_data \
  -B $SLURM_SUBMIT_DIR/output_data:/data/output_data \
  brainhack_multimodal.sif \
  --eeg-path  /data/source_data/eeg_features.tsv \
  --fmri-path /data/source_data/halfpipe_output/ \
  --target-column diagnosis \
  --n-permutations 100
```

---

## Data inputs

Configure `eeg_path` and `fmri_path` in `invoke.yaml` to point to your data.
See [`source_data/CONTENT.md`](source_data/CONTENT.md) for the expected formats.

The format is **auto-detected from the path** — no extra flag needed:

| `eeg_path` / `fmri_path` value | Detected as |
|---|---|
| Path to a file (any name) | flat TSV — `participant_id` + one column per feature |
| Path to a directory containing `sub-*/eeg/*_eeg.fif` | MNE-BIDS — band-power features extracted automatically |
| Path to a directory containing `sub-*/**/task-rest/*_desc-correlation_matrix.tsv` | Halfpipe derivatives |

```yaml
# invoke.yaml examples
eeg_path: /data/my_eeg_table.tsv          # flat TSV, any filename
eeg_path: /data/bids_dataset/             # MNE-BIDS directory
fmri_path: /data/connectivity_matrix.tsv  # flat TSV, any filename
fmri_path: /data/halfpipe_output/         # Halfpipe directory
```

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
| `eeg_path` | `source_data/eeg_features.tsv` | Path to EEG data: a `.tsv` file or a MNE-BIDS directory |
| `fmri_path` | `source_data/fmri_features.tsv` | Path to fMRI data: a `.tsv` file or a Halfpipe directory |
| `fmri_halfpipe_strategy` | `Baseline` | Halfpipe denoising strategy tag (e.g. `Baseline`, `36P`, `aCompCor`) — only used when `fmri_path` is a directory |
| `eeg_mne_task` | `rest` | BIDS task label for `.fif` files — only used when `eeg_path` is a directory |

---

## Output

See [`output_data/CONTENT.md`](output_data/CONTENT.md) for a description of all generated files.

### Notebooks

| Notebook | Description | Figures produced |
|---|---|---|
| `notebooks/results_overview.ipynb` | Visualises prediction results from `output_data/results/` | `scores_by_condition_{target}.png` (bar + fold overlay, significance brackets), `fold_distribution_{target}.png` (violin per condition), `feature_importance_{target}.png` (top-20 features per condition, coloured by modality), `modality_importance_{target}.png` (EEG vs fMRI aggregate for multimodal) — one set of figures per prediction target |

---

## Philosophy

- **Analysis in code, visualization in notebooks.** Heavy computation lives in `analysis/`; notebooks only read results and produce figures.
- **Idempotent steps.** Each `run-{name}` task skips if outputs already exist. Call `invoke clean` to force a full rerun.
- **Smoke tests.** `invoke run-smoke` generates synthetic data and runs the full pipeline in seconds.
- **Two input formats per modality.** Both flat TSVs and raw tool outputs (MNE, Halfpipe) are supported.
