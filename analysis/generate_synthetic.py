"""
Generate lightweight synthetic neuroimaging data for smoke testing.

Produces all four input formats the pipeline accepts:
  - phenotype.tsv           : participant phenotypes (participant_id, gender, age, study_site, diagnosis)
  - eeg_features.tsv        : flat EEG feature table (FreeSurfer-style region z-scores)
  - mne_output/             : per-subject CSV files mimicking MNE feature export
  - fmri_features.tsv       : flat fMRI connectivity table (corr_i_j upper triangle)
  - halfpipe_output/        : per-subject ROI x ROI matrix TSVs (no header, as Halfpipe produces)

Diagnosis is weakly predictable from both EEG and fMRI features via a shared latent
signal, so the multimodal fusion pipeline has something real to learn from.
"""

import numpy as np
import pandas as pd
from pathlib import Path

N_SUBJECTS = 15
N_ROIS = 20
N_CONNECTIVITY = N_ROIS * (N_ROIS - 1) // 2  # 190

# Subset of FreeSurfer Desikan-Killiany region names (realistic column naming)
_EEG_REGIONS = [
    "bankssts",
    "caudalanteriorcingulate",
    "caudalmiddlefrontal",
    "cuneus",
    "entorhinal",
]
EEG_FEATURE_NAMES = [
    f"{region}_{hemi}_offset_zscore"
    for region in _EEG_REGIONS
    for hemi in ("lh", "rh")
]  # 10 features: 5 regions × 2 hemispheres


def _subject_ids(n):
    return [f"sub-{i + 1:03d}" for i in range(n)]


def _connectivity_col_names():
    """Upper triangle of N_ROIS x N_ROIS matrix, named corr_i_j (integer indices)."""
    return [f"corr_{i}_{j}" for i in range(1, N_ROIS + 1) for j in range(i + 1, N_ROIS + 1)]


def _latent_signal(n, seed):
    """Shared latent factor driving diagnosis probability and both modalities."""
    return np.random.default_rng(seed).standard_normal(n)


def _make_connectivity_matrix(rng, signal_i):
    """
    Random symmetric correlation matrix with a weak signal embedded.
    signal_i shifts mean connectivity slightly to make diagnosis predictable.
    """
    raw = rng.uniform(-0.3, 0.3, (N_ROIS, N_ROIS)) + 0.1 * signal_i
    sym = (raw + raw.T) / 2
    np.fill_diagonal(sym, 1.0)
    return np.clip(sym, -1.0, 1.0)


def generate_phenotype(out_dir, n_subjects=N_SUBJECTS, seed=42):
    rng = np.random.default_rng(seed)
    ids = _subject_ids(n_subjects)
    signal = _latent_signal(n_subjects, seed)

    # Diagnosis probability increases with signal value (weak but real effect)
    diag_prob = 1 / (1 + np.exp(-0.8 * signal))
    diagnosis = rng.binomial(1, diag_prob)

    df = pd.DataFrame({
        "participant_id": ids,
        "gender": rng.choice(["Female", "Male"], n_subjects),
        "age": rng.integers(8, 22, n_subjects),
        "study_site": rng.choice(["HBNsiteSI", "HBNsiteRU"], n_subjects),
        "diagnosis": diagnosis,
    })
    out_path = Path(out_dir) / "phenotype.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)


def generate_eeg_tsv(out_dir, n_subjects=N_SUBJECTS, seed=42):
    rng = np.random.default_rng(seed + 1)
    ids = _subject_ids(n_subjects)
    signal = _latent_signal(n_subjects, seed)
    signal_weights = rng.uniform(0.2, 0.5, len(EEG_FEATURE_NAMES))

    data = rng.standard_normal((n_subjects, len(EEG_FEATURE_NAMES))) + np.outer(signal, signal_weights)
    df = pd.DataFrame(data, columns=EEG_FEATURE_NAMES)
    df.insert(0, "participant_id", ids)
    out_path = Path(out_dir) / "eeg_features.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)


def generate_mne_output(out_dir, n_subjects=N_SUBJECTS, seed=42):
    """One CSV per subject: mne_output/{sub_id}/{sub_id}_eeg_features.csv"""
    rng = np.random.default_rng(seed + 2)
    ids = _subject_ids(n_subjects)
    signal = _latent_signal(n_subjects, seed)
    signal_weights = rng.uniform(0.2, 0.5, len(EEG_FEATURE_NAMES))
    mne_dir = Path(out_dir) / "mne_output"
    for i, sub_id in enumerate(ids):
        sub_dir = mne_dir / sub_id
        sub_dir.mkdir(parents=True, exist_ok=True)
        features = rng.standard_normal(len(EEG_FEATURE_NAMES)) + signal[i] * signal_weights
        pd.DataFrame([features], columns=EEG_FEATURE_NAMES).to_csv(
            sub_dir / f"{sub_id}_eeg_features.csv", index=False
        )


def generate_fmri_tsv(out_dir, n_subjects=N_SUBJECTS, seed=42):
    rng = np.random.default_rng(seed + 3)
    ids = _subject_ids(n_subjects)
    signal = _latent_signal(n_subjects, seed)
    col_names = _connectivity_col_names()
    rows = [
        _make_connectivity_matrix(rng, signal[i])[np.triu_indices(N_ROIS, k=1)]
        for i in range(n_subjects)
    ]
    df = pd.DataFrame(rows, columns=col_names)
    df.insert(0, "participant_id", ids)
    out_path = Path(out_dir) / "fmri_features.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)


def generate_halfpipe_output(out_dir, n_subjects=N_SUBJECTS, strategies=None, seed=42):
    """
    One ROI x ROI matrix TSV per subject/run/strategy.
    Subjects 1-5 have 2 runs to test run-merging logic.
    Matrices have NO header (raw tab-separated numbers), matching real Halfpipe output.
    """
    if strategies is None:
        strategies = ["36P"]
    rng = np.random.default_rng(seed + 4)
    ids = _subject_ids(n_subjects)
    signal = _latent_signal(n_subjects, seed)
    halfpipe_dir = Path(out_dir) / "halfpipe_output"
    for i, sub_id in enumerate(ids):
        func_dir = halfpipe_dir / sub_id / "func"
        func_dir.mkdir(parents=True, exist_ok=True)
        n_runs = 2 if i < 5 else 1
        for strategy in strategies:
            for run in range(1, n_runs + 1):
                mat = _make_connectivity_matrix(rng, signal[i])
                fname = f"{sub_id}_task-rest_run-{run}_desc-{strategy}_matrix.tsv"
                # No header, no index — raw numbers only, as Halfpipe produces
                np.savetxt(func_dir / fname, mat, delimiter="\t")


def generate_all(out_dir, n_subjects=N_SUBJECTS, strategies=None, seed=42):
    """Generate all synthetic input formats into out_dir."""
    out_dir = Path(out_dir)
    generate_phenotype(out_dir, n_subjects=n_subjects, seed=seed)
    generate_eeg_tsv(out_dir, n_subjects=n_subjects, seed=seed)
    generate_mne_output(out_dir, n_subjects=n_subjects, seed=seed)
    generate_fmri_tsv(out_dir, n_subjects=n_subjects, seed=seed)
    generate_halfpipe_output(out_dir, n_subjects=n_subjects, strategies=strategies, seed=seed)
    print(f"[smoke] {n_subjects} synthetic subjects written to {out_dir}/")
