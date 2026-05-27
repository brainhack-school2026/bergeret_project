"""
Load fMRI connectivity features from either a flat TSV or a Halfpipe output folder.

Auto-detection: if the configured path is a directory, Halfpipe mode is used;
if it is a file, TSV mode is used.

TSV format:
    participant_id  corr_1_2  corr_1_3  ...

Halfpipe format (BIDS derivatives):
    {halfpipe_dir}/sub-{id}/ses-{ses}/func/task-rest/
        sub-{id}_ses-{ses}_task-rest_run-{n}_feature-{strategy}_atlas-{atlas}_desc-correlation_matrix.tsv

    - `desc-correlation` is always the matrix type (fixed by Halfpipe)
    - `feature-` is the denoising strategy tag (e.g. 'Baseline', '36P', 'aCompCor')
      -> this is what `fmri_halfpipe_strategy` selects
    - No header (raw tab-separated numbers), consistent with wonkyconn has_header=False
    - Run merging: NaN-aware — if one run is NaN use the other; if both NaN keep NaN
    - Output columns are named corr_i_j with 1-based integer ROI indices

NaN handling (applied to both TSV and Halfpipe output):
    1. Impute remaining NaN per feature with the median across subjects.
    2. Drop features that are NaN for every subject (no information).
    3. Print a single summary of the fraction of values imputed or dropped.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def detect_input_type(path: Path) -> str:
    """Return 'tsv' if path is a file, 'halfpipe' if it is a directory."""
    if path.is_file():
        return "tsv"
    if path.is_dir():
        return "halfpipe"
    raise FileNotFoundError(f"fMRI input not found: {path}")


def load_fmri_tsv(tsv_path: Path, subjects: list[str] | None = None) -> pd.DataFrame:
    """Load pre-computed fMRI connectivity table."""
    tsv_path = Path(tsv_path)
    df = pd.read_csv(tsv_path, sep="\t", dtype={"participant_id": str})
    if subjects:
        df = df[df["participant_id"].isin(subjects)]
    return df.reset_index(drop=True)


def load_fmri_halfpipe(
    halfpipe_dir: Path,
    strategy: str,
    subjects: list[str] | None = None,
) -> pd.DataFrame:
    """
    Reconstruct a flat fMRI connectivity table from Halfpipe per-subject matrix files.

    Run merging is NaN-aware per element:
      - both runs valid  -> mean of the two
      - one run NaN      -> use the valid run directly (no averaging)
      - both runs NaN    -> NaN (will be imputed later at the group level)

    Args:
        halfpipe_dir: Root of the Halfpipe derivatives (contains one subfolder per subject).
        strategy: Value of the feature- BIDS tag (e.g. 'Baseline', '36P', 'aCompCor').
        subjects: Optional list of participant_ids to process.
    """
    halfpipe_dir = Path(halfpipe_dir)
    records = []

    sub_dirs = sorted(d for d in halfpipe_dir.iterdir() if d.is_dir())
    if subjects:
        sub_dirs = [d for d in sub_dirs if d.name in subjects]

    for sub_dir in sub_dirs:
        sub_id = sub_dir.name
        run_files = sorted(
            sub_dir.glob(f"**/task-rest/*_feature-{strategy}_*_desc-correlation_matrix.tsv")
        )
        if not run_files:
            print(f"[load-fmri] Skipping {sub_id}: no files for feature '{strategy}'")
            continue

        matrices = [np.loadtxt(f, delimiter="\t") for f in run_files]

        # NaN-aware run merging: nanmean returns NaN only when all inputs are NaN
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            avg_mat = np.nanmean(matrices, axis=0)

        n_rois = avg_mat.shape[0]
        idx_i, idx_j = np.triu_indices(n_rois, k=1)
        upper = avg_mat[idx_i, idx_j]
        col_names = [f"corr_{i + 1}_{j + 1}" for i, j in zip(idx_i, idx_j)]

        records.append({"participant_id": sub_id, **dict(zip(col_names, upper))})

    if not records:
        raise ValueError(
            f"No subjects found in {halfpipe_dir} for strategy '{strategy}'. "
            "Check fmri_halfpipe_dir and fmri_halfpipe_strategy in invoke.yaml."
        )

    return pd.DataFrame(records).reset_index(drop=True)


def impute_fmri_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    NaN handling at the group level (applied after vectorisation regardless of input type):
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
        f"[load-fmri] NaN summary: {n_imputed} values imputed (median), "
        f"{n_dropped_cols} all-NaN columns dropped "
        f"({pct_affected:.1f}% of {total_values} total values affected)"
    )

    result = pd.DataFrame(X, columns=kept_cols)
    result.insert(0, "participant_id", df["participant_id"].values)
    return result


def load_fmri(
    path: Path,
    input_type: str = "auto",
    strategy: str | None = None,
    subjects: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load fMRI connectivity features from either a TSV file or a Halfpipe output folder.

    Args:
        path: Path to the TSV file or to the Halfpipe output directory.
        input_type: 'tsv', 'halfpipe', or 'auto' (auto-detect from path type).
        strategy: Halfpipe feature tag value (required when input_type='halfpipe').
        subjects: Optional list of participant_ids to keep.
    """
    path = Path(path)
    if input_type == "auto":
        input_type = detect_input_type(path)

    if input_type == "tsv":
        df = load_fmri_tsv(path, subjects=subjects)
    elif input_type == "halfpipe":
        if not strategy:
            raise ValueError("strategy must be provided for Halfpipe input.")
        df = load_fmri_halfpipe(path, strategy=strategy, subjects=subjects)
    else:
        raise ValueError(f"Unknown fmri_input_type: '{input_type}'. Expected 'tsv' or 'halfpipe'.")

    return impute_fmri_features(df)
