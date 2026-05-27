"""
Generate lightweight synthetic neuroimaging data for smoke testing.

Two independent latent signals are embedded in EEG and fMRI features:
  - signal_diag  → drives diagnosis (and partially features)
  - signal_gender → drives gender  (and partially features)

Because the signals are independent, confound correction does not remove
prediction-relevant variance: regressing out gender when predicting
diagnosis removes only gender variance, leaving the diagnosis signal intact,
and vice versa.  Both targets should achieve AUC > 0.5 in the smoke test.

Produces all four input formats the pipeline accepts:
  - phenotype.tsv     : balanced 50/50 diagnosis and gender; one subject has
                        missing age (exercises confound-drop in run_intersect)
  - eeg_features.tsv  : 50 EEG band-power features, ~2% NaN total
                        (sparse + two entirely-NaN columns)
  - mne_output/       : per-subject CSV mimicking MNE feature export
                        (no NaN — exercises the clean path)
  - fmri_features.tsv : 300 connectivity features, ~2% NaN total
                        (sparse + three entirely-NaN columns)
  - halfpipe_output/  : BIDS-compliant Halfpipe structure
                        (ses-1/func/task-rest/, feature- tag, desc-correlation)
                        subjects 1–5: 2 runs (run-merging path)
                        subjects 1–3: NaN ROI pairs (group-imputation path)
"""

import numpy as np
import pandas as pd
from pathlib import Path

N_SUBJECTS = 30
N_ROIS = 25
N_CONNECTIVITY = N_ROIS * (N_ROIS - 1) // 2  # 300

# EEG: 5 frequency bands × 10 standard 10-20 channels = 50 features
_EEG_CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
_EEG_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
EEG_FEATURE_NAMES = [
    f"{ch}_{band}_power" for band in _EEG_BANDS for ch in _EEG_CHANNELS
]  # 50 features

_N_ALL_NAN_EEG = 2    # entirely-NaN columns in EEG TSV
_N_ALL_NAN_FMRI = 3   # entirely-NaN columns in fMRI TSV
_SPARSE_NAN_RATE = 0.010  # sparse NaN on top of all-NaN columns → ~2% total

_NAN_ROI_PAIRS = [(2, 5), (10, 14)]   # ROI pairs set to NaN in Halfpipe matrices
_NAN_HALFPIPE_N = 3                    # first N subjects get NaN ROI pairs


def _subject_ids(n):
    return [f"sub-{i + 1:03d}" for i in range(n)]


def _connectivity_col_names():
    return [f"corr_{i}_{j}" for i in range(1, N_ROIS + 1) for j in range(i + 1, N_ROIS + 1)]


def _signal_diag_eeg(n, seed):
    """Diagnostic signal captured by EEG — independent of the fMRI component."""
    return np.random.default_rng(seed).standard_normal(n)


def _signal_diag_fmri(n, seed):
    """Diagnostic signal captured by fMRI — independent of the EEG component."""
    return np.random.default_rng(seed + 50).standard_normal(n)


def _signal_gender(n, seed):
    """Gender signal embedded in both modalities — independent of diagnosis signals."""
    return np.random.default_rng(seed + 100).standard_normal(n)


def _balanced_binary(signal):
    """Return 0/1 array balanced 50/50 by ranking signal values."""
    n = len(signal)
    rank = np.argsort(np.argsort(signal))
    return (rank >= n // 2).astype(int)


def _add_sparse_nan(rng, X, rate):
    n_nan = max(1, int(X.size * rate))
    flat = X.flatten().copy()
    idx = rng.choice(len(flat), size=n_nan, replace=False)
    flat[idx] = np.nan
    return flat.reshape(X.shape)


def _make_connectivity_matrix(rng, signal_i, nan_roi_pairs=None):
    raw = rng.uniform(-0.3, 0.3, (N_ROIS, N_ROIS)) + 0.12 * signal_i
    sym = (raw + raw.T) / 2
    np.fill_diagonal(sym, 1.0)
    mat = np.clip(sym, -1.0, 1.0)
    if nan_roi_pairs:
        for i, j in nan_roi_pairs:
            mat[i, j] = np.nan
            mat[j, i] = np.nan
    return mat


def generate_phenotype(out_dir, n_subjects=N_SUBJECTS, seed=42):
    rng = np.random.default_rng(seed)
    ids = _subject_ids(n_subjects)

    # Diagnosis: sum of two complementary components (one per modality)
    # so each modality alone has only partial information
    sig_d = _signal_diag_eeg(n_subjects, seed) + _signal_diag_fmri(n_subjects, seed)
    sig_g = _signal_gender(n_subjects, seed)
    diagnosis = _balanced_binary(sig_d)
    gender = ["Female" if g else "Male" for g in _balanced_binary(sig_g)]

    age = rng.integers(8, 22, n_subjects).astype(float)
    age[-1] = np.nan  # last subject missing age → dropped in run_intersect

    df = pd.DataFrame({
        "participant_id": ids,
        "gender": gender,
        "age": age,
        "study_site": rng.choice(["HBNsiteSI", "HBNsiteRU"], n_subjects),
        "diagnosis": diagnosis,
    })
    out_path = Path(out_dir) / "phenotype.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)


def generate_eeg_tsv(out_dir, n_subjects=N_SUBJECTS, seed=42):
    rng = np.random.default_rng(seed + 1)
    ids = _subject_ids(n_subjects)
    # EEG captures its own diagnostic component + gender, but NOT the fMRI component
    sig_d_eeg = _signal_diag_eeg(n_subjects, seed)
    sig_g = _signal_gender(n_subjects, seed)

    w_d = rng.uniform(0.09, 0.13, len(EEG_FEATURE_NAMES))
    w_g = rng.uniform(0.07, 0.11, len(EEG_FEATURE_NAMES))

    data = (rng.standard_normal((n_subjects, len(EEG_FEATURE_NAMES)))
            + np.outer(sig_d_eeg, w_d)
            + np.outer(sig_g, w_g))

    data = _add_sparse_nan(rng, data, _SPARSE_NAN_RATE)
    all_nan_cols = rng.choice(len(EEG_FEATURE_NAMES), size=_N_ALL_NAN_EEG, replace=False)
    data[:, all_nan_cols] = np.nan

    df = pd.DataFrame(data, columns=EEG_FEATURE_NAMES)
    df.insert(0, "participant_id", ids)
    out_path = Path(out_dir) / "eeg_features.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)


def generate_mne_output(out_dir, n_subjects=N_SUBJECTS, seed=42):
    """
    One CSV per subject: mne_output/{sub_id}/{sub_id}_eeg_features.csv
    One row, feature columns only (no participant_id), no NaN — clean path.
    Same feature names as the TSV (same pipeline, different export format).
    """
    rng = np.random.default_rng(seed + 2)
    ids = _subject_ids(n_subjects)
    sig_d_eeg = _signal_diag_eeg(n_subjects, seed)
    sig_g = _signal_gender(n_subjects, seed)
    w_d = rng.uniform(0.09, 0.13, len(EEG_FEATURE_NAMES))
    w_g = rng.uniform(0.07, 0.11, len(EEG_FEATURE_NAMES))
    mne_dir = Path(out_dir) / "mne_output"
    for i, sub_id in enumerate(ids):
        sub_dir = mne_dir / sub_id
        sub_dir.mkdir(parents=True, exist_ok=True)
        features = (rng.standard_normal(len(EEG_FEATURE_NAMES))
                    + sig_d_eeg[i] * w_d + sig_g[i] * w_g)
        pd.DataFrame([features], columns=EEG_FEATURE_NAMES).to_csv(
            sub_dir / f"{sub_id}_eeg_features.csv", index=False
        )


def generate_fmri_tsv(out_dir, n_subjects=N_SUBJECTS, seed=42):
    rng = np.random.default_rng(seed + 3)
    ids = _subject_ids(n_subjects)
    # fMRI captures its own diagnostic component + gender, but NOT the EEG component
    sig_d_fmri = _signal_diag_fmri(n_subjects, seed)
    sig_g = _signal_gender(n_subjects, seed)
    col_names = _connectivity_col_names()

    rows = np.array([
        _make_connectivity_matrix(rng, 0.14 * sig_d_fmri[i] + 0.08 * sig_g[i])[np.triu_indices(N_ROIS, k=1)]
        for i in range(n_subjects)
    ], dtype=float)

    rows = _add_sparse_nan(rng, rows, _SPARSE_NAN_RATE)
    all_nan_cols = rng.choice(len(col_names), size=_N_ALL_NAN_FMRI, replace=False)
    rows[:, all_nan_cols] = np.nan

    df = pd.DataFrame(rows, columns=col_names)
    df.insert(0, "participant_id", ids)
    out_path = Path(out_dir) / "fmri_features.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)


def generate_halfpipe_output(out_dir, n_subjects=N_SUBJECTS, strategies=None, seed=42):
    """
    BIDS-compliant Halfpipe output matching the real HBN structure:
        halfpipe_output/sub-{id}/ses-1/func/task-rest/
            sub-{id}_ses-1_task-rest_run-{n}_feature-{strategy}_atlas-SynthAtlas_desc-correlation_matrix.tsv

    - No header, raw tab-separated numbers
    - Subjects 1–5: 2 runs (NaN-aware run-merging)
    - Subjects 1–3: NaN for two ROI pairs (group imputation)
    """
    if strategies is None:
        strategies = ["36P"]
    rng = np.random.default_rng(seed + 4)
    ids = _subject_ids(n_subjects)
    sig_d_fmri = _signal_diag_fmri(n_subjects, seed)
    sig_g = _signal_gender(n_subjects, seed)
    halfpipe_dir = Path(out_dir) / "halfpipe_output"

    for i, sub_id in enumerate(ids):
        task_dir = halfpipe_dir / sub_id / "ses-1" / "func" / "task-rest"
        task_dir.mkdir(parents=True, exist_ok=True)
        n_runs = 2 if i < 5 else 1
        nan_pairs = _NAN_ROI_PAIRS if i < _NAN_HALFPIPE_N else None
        for strategy in strategies:
            for run in range(1, n_runs + 1):
                mat = _make_connectivity_matrix(
                    rng, 0.14 * sig_d_fmri[i] + 0.08 * sig_g[i], nan_roi_pairs=nan_pairs
                )
                fname = (
                    f"{sub_id}_ses-1_task-rest_run-{run}"
                    f"_feature-{strategy}_atlas-SynthAtlas_desc-correlation_matrix.tsv"
                )
                np.savetxt(task_dir / fname, mat, delimiter="\t")


def generate_all(out_dir, n_subjects=N_SUBJECTS, strategies=None, seed=42):
    """Generate all synthetic input formats into out_dir."""
    out_dir = Path(out_dir)
    generate_phenotype(out_dir, n_subjects=n_subjects, seed=seed)
    generate_eeg_tsv(out_dir, n_subjects=n_subjects, seed=seed)
    generate_mne_output(out_dir, n_subjects=n_subjects, seed=seed)
    generate_fmri_tsv(out_dir, n_subjects=n_subjects, seed=seed)
    generate_halfpipe_output(out_dir, n_subjects=n_subjects, strategies=strategies, seed=seed)
    print(f"[smoke] {n_subjects} synthetic subjects written to {out_dir}/")
