"""
Load EEG features from either a flat TSV or a MNE per-subject output folder.

Auto-detection: if the configured path is a directory, MNE mode is used;
if it is a file, TSV mode is used.

TSV format:
    participant_id  feature_1  feature_2  ...

MNE format:
    {mne_dir}/{sub_id}/{sub_id}_eeg_features.csv
    (one row per subject, feature columns only — no participant_id in the file)
"""

from pathlib import Path

import pandas as pd


def detect_input_type(path: Path) -> str:
    """Return 'tsv' if path is a file, 'mne' if it is a directory."""
    if path.is_file():
        return "tsv"
    if path.is_dir():
        return "mne"
    raise FileNotFoundError(f"EEG input not found: {path}")


def load_eeg_tsv(tsv_path: Path, subjects: list[str] | None = None) -> pd.DataFrame:
    """Load pre-computed EEG feature table."""
    tsv_path = Path(tsv_path)
    df = pd.read_csv(tsv_path, sep="\t", dtype={"participant_id": str})
    if subjects:
        df = df[df["participant_id"].isin(subjects)]
    return df.reset_index(drop=True)


def load_eeg_mne(mne_dir: Path, subjects: list[str] | None = None) -> pd.DataFrame:
    """
    Reconstruct a flat EEG feature table from per-subject MNE CSV exports.

    Expected structure:
        {mne_dir}/{sub_id}/{sub_id}_eeg_features.csv
    Each file has one row and feature columns only (no participant_id column).
    """
    mne_dir = Path(mne_dir)
    records = []

    sub_dirs = sorted(d for d in mne_dir.iterdir() if d.is_dir())
    if subjects:
        sub_dirs = [d for d in sub_dirs if d.name in subjects]

    for sub_dir in sub_dirs:
        sub_id = sub_dir.name
        csv_files = list(sub_dir.glob(f"{sub_id}_eeg_features.csv"))
        if not csv_files:
            print(f"[load-eeg] Skipping {sub_id}: no *_eeg_features.csv found")
            continue
        row = pd.read_csv(csv_files[0]).iloc[0].to_dict()
        row["participant_id"] = sub_id
        records.append(row)

    if not records:
        raise ValueError(f"No EEG feature files found in {mne_dir}")

    df = pd.DataFrame(records)
    cols = ["participant_id"] + [c for c in df.columns if c != "participant_id"]
    return df[cols].reset_index(drop=True)


def load_eeg(path: Path, input_type: str = "auto", subjects: list[str] | None = None) -> pd.DataFrame:
    """
    Load EEG features from either a TSV file or a MNE output folder.

    Args:
        path: Path to the TSV file or to the MNE output directory.
        input_type: 'tsv', 'mne', or 'auto' (auto-detect from path type).
        subjects: Optional list of participant_ids to keep.
    """
    path = Path(path)
    if input_type == "auto":
        input_type = detect_input_type(path)

    if input_type == "tsv":
        return load_eeg_tsv(path, subjects=subjects)
    elif input_type == "mne":
        return load_eeg_mne(path, subjects=subjects)
    else:
        raise ValueError(f"Unknown eeg_input_type: '{input_type}'. Expected 'tsv' or 'mne'.")
