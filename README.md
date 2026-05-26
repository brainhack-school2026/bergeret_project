# brainhack-2026-multimodal

Reproducible multimodal EEG/fMRI fusion pipeline for psychiatric prediction.
Built for Brainhack School 2026 using the [`airoh`](https://pypi.org/project/airoh/) task runner.

The pipeline trains separate prediction models on EEG features, fMRI connectivity features,
and both combined — so you can compare whether one modality, the other, or their fusion
best predicts a chosen phenotypic target (e.g. diagnosis, age, a clinical score).

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
| `run-load-eeg` | Load and harmonise EEG features → `output_data/eeg_features.tsv` |
| `run-load-fmri` | Load and harmonise fMRI connectivity → `output_data/fmri_features.tsv` |
| `run-predict` | Train and evaluate EEG-only, fMRI-only, and multimodal models |
| `run-notebooks` | Execute notebooks and save figures to `output_data/` |
| `run` | Full pipeline (all steps in order) |
| `run-smoke` | Smoke test: synthetic data + minimal end-to-end pass |
| `clean` | Remove all generated outputs and synthetic data |
| `clean-outputs` | Remove analysis outputs only |
| `clean-smoke` | Remove synthetic smoke data only |

Use `invoke --list` or `invoke --help <task>` for details.

---

## Output

See [`output_data/CONTENT.md`](output_data/CONTENT.md) for a description of all generated files.

---

## Philosophy

- **Analysis in code, visualization in notebooks.** Heavy computation lives in `analysis/`; notebooks only read results and produce figures.
- **Idempotent steps.** Each `run-{name}` task skips if outputs already exist. Call `invoke clean` to force a full rerun.
- **Smoke tests.** `invoke run-smoke` generates synthetic data and runs the full pipeline in seconds.
- **Two input formats per modality.** Both flat TSVs and raw tool outputs (MNE, Halfpipe) are supported.
