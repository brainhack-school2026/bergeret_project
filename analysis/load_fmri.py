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
      → this is what `fmri_halfpipe_strategy` selects
    - `atlas-` is ignored: all atlases matching the strategy are averaged
    - No header (raw tab-separated numbers), consistent with wonkyconn has_header=False
    - Multiple runs and/or sessions per subject are averaged before extracting
      the upper triangle
    - Output columns are named corr_i_j with 1-based integer ROI indices
"""

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

    Each subject's connectivity matrix is the average across runs (if multiple runs exist).
    The upper triangle is extracted and flattened into columns named corr_i_j (1-based).

    Args:
        halfpipe_dir: Root of the Halfpipe output (contains one subfolder per subject).
        strategy: The desc- tag value to filter on (e.g. '36P', 'aCompCor').
        subjects: Optional list of participant_ids to process.
    """
    halfpipe_dir = Path(halfpipe_dir)
    records = []

    sub_dirs = sorted(d for d in halfpipe_dir.iterdir() if d.is_dir())
    if subjects:
        sub_dirs = [d for d in sub_dirs if d.name in subjects]

    for sub_dir in sub_dirs:
        sub_id = sub_dir.name
        # Halfpipe BIDS layout: ses-*/func/task-rest/*_feature-{strategy}_*_desc-correlation_matrix.tsv
        # Use ** to handle any session depth; desc-correlation is always fixed for Halfpipe matrices
        run_files = sorted(sub_dir.glob(f"**/task-rest/*_feature-{strategy}_*_desc-correlation_matrix.tsv"))
        if not run_files:
            print(f"[load-fmri] Skipping {sub_id}: no files for feature '{strategy}'")
            continue

        # Load each run matrix (no header — raw numbers only)
        matrices = [np.loadtxt(f, delimiter="\t") for f in run_files]
        if len(run_files) > 1:
            avg_mat = np.mean(matrices, axis=0)
        else:
            avg_mat = matrices[0]

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
        strategy: Halfpipe denoising strategy name (required when input_type='halfpipe').
        subjects: Optional list of participant_ids to keep.
    """
    path = Path(path)
    if input_type == "auto":
        input_type = detect_input_type(path)

    if input_type == "tsv":
        return load_fmri_tsv(path, subjects=subjects)
    elif input_type == "halfpipe":
        if not strategy:
            raise ValueError("strategy must be provided for Halfpipe input.")
        return load_fmri_halfpipe(path, strategy=strategy, subjects=subjects)
    else:
        raise ValueError(f"Unknown fmri_input_type: '{input_type}'. Expected 'tsv' or 'halfpipe'.")
