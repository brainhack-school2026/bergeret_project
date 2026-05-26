"""
Generate lightweight synthetic neuroimaging data for smoke testing.

Produces all four input formats the pipeline accepts:
  - phenotype.tsv           : participant phenotypes
  - eeg_features.tsv        : flat EEG feature table
  - mne_output/             : per-subject CSV files mimicking MNE feature export
  - fmri_features.tsv       : flat fMRI connectivity table (upper triangle)
  - halfpipe_output/        : per-subject ROI x ROI matrix TSVs mimicking Halfpipe output
"""

import numpy as np
import pandas as pd
from pathlib import Path

N_SUBJECTS = 15
N_ROIS = 20
N_CONNECTIVITY = N_ROIS * (N_ROIS - 1) // 2  # 190

EEG_FEATURE_NAMES = [
    "alpha_power",
    "theta_power",
    "beta_power",
    "delta_power",
    "gamma_power",
    "theta_alpha_ratio",
    "frontal_asymmetry",
    "spectral_entropy",
    "hjorth_mobility",
    "hjorth_complexity",
]

ROI_NAMES = [f"roi_{i + 1:02d}" for i in range(N_ROIS)]


def _subject_ids(n):
    return [f"sub-{i + 1:03d}" for i in range(n)]


def _connectivity_col_names():
    return [
        f"conn_{ROI_NAMES[i]}_{ROI_NAMES[j]}"
        for i in range(N_ROIS)
        for j in range(i + 1, N_ROIS)
    ]


def _symmetric_correlation_matrix(rng):
    """Random symmetric matrix with values in [-1, 1] and diagonal = 1."""
    raw = rng.uniform(-1, 1, (N_ROIS, N_ROIS))
    sym = (raw + raw.T) / 2
    np.fill_diagonal(sym, 1.0)
    return sym


def generate_phenotype(out_dir, n_subjects=N_SUBJECTS, seed=42):
    rng = np.random.default_rng(seed)
    ids = _subject_ids(n_subjects)
    df = pd.DataFrame({
        "participant_id": ids,
        "age": rng.integers(20, 66, n_subjects),
        "sex": rng.choice(["M", "F"], n_subjects),
        "site": rng.choice(["site1", "site2"], n_subjects),
        "diagnosis": rng.choice([0, 1], n_subjects, p=[0.6, 0.4]),
        "iq": rng.integers(85, 131, n_subjects).astype(float),
        "anxiety_score": rng.integers(0, 41, n_subjects).astype(float),
    })
    out_path = Path(out_dir) / "phenotype.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)


def generate_eeg_tsv(out_dir, n_subjects=N_SUBJECTS, seed=43):
    rng = np.random.default_rng(seed)
    ids = _subject_ids(n_subjects)
    data = rng.uniform(0, 1, (n_subjects, len(EEG_FEATURE_NAMES)))
    df = pd.DataFrame(data, columns=EEG_FEATURE_NAMES)
    df.insert(0, "participant_id", ids)
    out_path = Path(out_dir) / "eeg_features.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)


def generate_mne_output(out_dir, n_subjects=N_SUBJECTS, seed=44):
    """One CSV per subject in mne_output/{sub_id}/{sub_id}_eeg_features.csv."""
    rng = np.random.default_rng(seed)
    ids = _subject_ids(n_subjects)
    mne_dir = Path(out_dir) / "mne_output"
    for sub_id in ids:
        sub_dir = mne_dir / sub_id
        sub_dir.mkdir(parents=True, exist_ok=True)
        features = rng.uniform(0, 1, len(EEG_FEATURE_NAMES))
        pd.DataFrame([features], columns=EEG_FEATURE_NAMES).to_csv(
            sub_dir / f"{sub_id}_eeg_features.csv", index=False
        )


def generate_fmri_tsv(out_dir, n_subjects=N_SUBJECTS, seed=45):
    rng = np.random.default_rng(seed)
    ids = _subject_ids(n_subjects)
    col_names = _connectivity_col_names()
    rows = [
        _symmetric_correlation_matrix(rng)[np.triu_indices(N_ROIS, k=1)]
        for _ in range(n_subjects)
    ]
    df = pd.DataFrame(rows, columns=col_names)
    df.insert(0, "participant_id", ids)
    out_path = Path(out_dir) / "fmri_features.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)


def generate_halfpipe_output(out_dir, n_subjects=N_SUBJECTS, strategies=None, seed=46):
    """
    One ROI x ROI matrix TSV per subject/run/strategy.
    Subjects 1-5 have 2 runs; the rest have 1 run (to test run-merging logic).
    """
    if strategies is None:
        strategies = ["36P"]
    rng = np.random.default_rng(seed)
    ids = _subject_ids(n_subjects)
    halfpipe_dir = Path(out_dir) / "halfpipe_output"
    for i, sub_id in enumerate(ids):
        func_dir = halfpipe_dir / sub_id / "func"
        func_dir.mkdir(parents=True, exist_ok=True)
        n_runs = 2 if i < 5 else 1
        for strategy in strategies:
            for run in range(1, n_runs + 1):
                mat = _symmetric_correlation_matrix(rng)
                df_mat = pd.DataFrame(mat, index=ROI_NAMES, columns=ROI_NAMES)
                fname = f"{sub_id}_task-rest_run-{run}_desc-{strategy}_matrix.tsv"
                df_mat.to_csv(func_dir / fname, sep="\t")


def generate_all(out_dir, n_subjects=N_SUBJECTS, strategies=None, seed=42):
    """Generate all synthetic input formats into out_dir."""
    out_dir = Path(out_dir)
    generate_phenotype(out_dir, n_subjects=n_subjects, seed=seed)
    generate_eeg_tsv(out_dir, n_subjects=n_subjects, seed=seed + 1)
    generate_mne_output(out_dir, n_subjects=n_subjects, seed=seed + 2)
    generate_fmri_tsv(out_dir, n_subjects=n_subjects, seed=seed + 3)
    generate_halfpipe_output(out_dir, n_subjects=n_subjects, strategies=strategies, seed=seed + 4)
    print(f"[smoke] {n_subjects} synthetic subjects written to {out_dir}/")
