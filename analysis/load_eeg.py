"""
Load EEG features from either a flat TSV or a MNE-BIDS directory of .fif files.

TSV format:
    participant_id  feature_1  feature_2  ...

MNE-BIDS format:
    {bids_dir}/sub-{id}/ses-{ses}/eeg/sub-{id}_ses-{ses}_task-{task}_eeg.fif
    Band-power features (delta/theta/alpha/beta/gamma × channel) are extracted
    automatically using Welch's method.
"""

from pathlib import Path

import numpy as np
import pandas as pd

_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def load_eeg_tsv(tsv_path: Path, subjects: list[str] | None = None) -> pd.DataFrame:
    """Load pre-computed EEG feature table."""
    tsv_path = Path(tsv_path)
    df = pd.read_csv(tsv_path, sep="\t", dtype={"participant_id": str})
    if subjects:
        df = df[df["participant_id"].isin(subjects)]
    return df.reset_index(drop=True)


def _extract_band_power(raw) -> dict:
    """Extract mean band power per channel from a MNE Raw object."""
    spectrum = raw.compute_psd(method="welch", fmin=1.0, fmax=45.0, verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)
    features = {}
    for band_name, (fmin, fmax) in _BANDS.items():
        mask = (freqs >= fmin) & (freqs < fmax)
        band_power = psds[:, mask].mean(axis=1)
        for ch_name, power in zip(raw.ch_names, band_power):
            features[f"{ch_name}_{band_name}_power"] = float(power)
    return features


def load_eeg_mne(bids_dir: Path, subjects: list[str] | None = None, task: str = "rest") -> pd.DataFrame:
    """
    Extract band-power features from MNE-BIDS .fif files.

    Expected structure:
        {bids_dir}/sub-{id}/[ses-{ses}/]eeg/sub-{id}[_ses-{ses}]_task-{task}_eeg.fif

    Features: delta/theta/alpha/beta/gamma band power per channel (Welch's method).
    """
    import mne
    mne.set_log_level("WARNING")

    bids_dir = Path(bids_dir)
    sub_dirs = sorted(d for d in bids_dir.iterdir() if d.is_dir() and d.name.startswith("sub-"))
    if subjects:
        sub_dirs = [d for d in sub_dirs if d.name in subjects]

    records = []
    for sub_dir in sub_dirs:
        sub_id = sub_dir.name
        fif_files = sorted(sub_dir.rglob(f"*_task-{task}_eeg.fif"))
        if not fif_files:
            print(f"[load-eeg] Skipping {sub_id}: no *_task-{task}_eeg.fif found")
            continue
        try:
            raw = mne.io.read_raw_fif(fif_files[0], preload=True, verbose=False)
        except Exception as exc:
            print(f"[load-eeg] Skipping {sub_id}: could not read {fif_files[0]}: {exc}")
            continue
        row = _extract_band_power(raw)
        row["participant_id"] = sub_id
        records.append(row)

    if not records:
        raise ValueError(
            f"No *_task-{task}_eeg.fif files found in {bids_dir}. "
            "Check that your MNE-BIDS directory contains sub-*/[ses-*/]eeg/*.fif files."
        )

    df = pd.DataFrame(records)
    cols = ["participant_id"] + [c for c in df.columns if c != "participant_id"]
    return df[cols].reset_index(drop=True)


def impute_eeg_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    NaN handling at the group level (applied after loading regardless of input type):
      1. Drop features that are NaN for every subject.
      2. Impute remaining NaN with the cross-subject median for that feature.
      3. Print a single summary line with the fraction of values affected.
    """
    feat_cols = [c for c in df.columns if c != "participant_id"]
    X = df[feat_cols].values.astype(float)

    total_values = X.size
    n_nan_initial = int(np.isnan(X).sum())

    # Step 1: drop all-NaN columns
    all_nan_mask = np.all(np.isnan(X), axis=0)
    n_dropped_cols = int(all_nan_mask.sum())
    n_dropped_values = n_dropped_cols * X.shape[0]
    X = X[:, ~all_nan_mask]
    kept_cols = [c for c, drop in zip(feat_cols, all_nan_mask) if not drop]

    # Step 2: impute remaining NaN with column median
    nan_remaining = np.isnan(X)
    n_imputed = int(nan_remaining.sum())
    col_medians = np.nanmedian(X, axis=0)
    X[nan_remaining] = np.take(col_medians, np.where(nan_remaining)[1])

    pct_affected = 100 * (n_dropped_values + n_imputed) / total_values
    print(
        f"[load-eeg] NaN summary: {n_imputed} values imputed (median), "
        f"{n_dropped_cols} all-NaN columns dropped "
        f"({pct_affected:.1f}% of {total_values} total values affected)"
    )

    result = pd.DataFrame(X, columns=kept_cols)
    result.insert(0, "participant_id", df["participant_id"].values)
    return result


def load_eeg(path: Path, input_type: str, subjects: list[str] | None = None, task: str = "rest") -> pd.DataFrame:
    """
    Load EEG features from either a TSV file or a MNE-BIDS directory of .fif files.

    Args:
        path: Path to the TSV file or to the MNE-BIDS root directory.
        input_type: 'tsv' or 'mne'.
        subjects: Optional list of participant_ids to keep.
        task: BIDS task label to load from .fif files (default: 'rest'). Only used for MNE input.
    """
    path = Path(path)

    if input_type == "tsv":
        df = load_eeg_tsv(path, subjects=subjects)
    elif input_type == "mne":
        df = load_eeg_mne(path, subjects=subjects, task=task)
    else:
        raise ValueError(f"Unknown eeg_input_type: '{input_type}'. Expected 'tsv' or 'mne'.")

    return impute_eeg_features(df)
