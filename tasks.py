import shutil
from pathlib import Path
from invoke import task


@task
def fetch(c):
    """Print instructions for configuring real source data."""
    print("Set your data paths in invoke.yaml:")
    print("  phenotype_file: path/to/phenotype.tsv")
    print("  eeg_path:       path/to/eeg_features.tsv      # flat TSV")
    print("                  path/to/mne_bids_dir/          # or MNE-BIDS directory with sub-*/eeg/*.fif")
    print("  fmri_path:      path/to/fmri_features.tsv     # flat TSV")
    print("                  path/to/halfpipe_output/       # or Halfpipe derivatives directory")


@task
def generate_smoke_data(c, n_subjects=30, strategies="36P"):
    """Generate lightweight synthetic data for smoke testing into source_data/smoke/."""
    from analysis.generate_synthetic import generate_all
    out_dir = Path(c.config.get("source_data_dir")) / "smoke"
    generate_all(out_dir, n_subjects=int(n_subjects), strategies=strategies.split(","))


@task
def run_intersect(c, smoke=False):
    """Compute subject intersection across EEG, fMRI, and phenotype → output_data/subjects.txt"""
    import pandas as pd
    from airoh.utils import ensure_dir_exist

    out_path = Path(c.config.get("output_data_dir")) / "subjects.txt"
    if out_path.exists():
        print(f"[run-intersect] Skipping — {out_path} already exists")
        return

    ensure_dir_exist(c, "output_data_dir")
    smoke_dir = Path(c.config.get("source_data_dir")) / "smoke"
    def ids_from_tsv(path):
        return set(pd.read_csv(path, sep="\t", usecols=["participant_id"], dtype=str)["participant_id"])

    def ids_from_dir(path):
        return {d.name for d in Path(path).iterdir() if d.is_dir()}

    # EEG subjects
    _, eeg_path = _resolve_path(c, "eeg", smoke_dir if smoke else None)
    eeg_ids = ids_from_dir(eeg_path) if eeg_path.is_dir() else ids_from_tsv(eeg_path)

    # fMRI subjects
    _, fmri_path = _resolve_path(c, "fmri", smoke_dir if smoke else None)
    fmri_ids = ids_from_dir(fmri_path) if fmri_path.is_dir() else ids_from_tsv(fmri_path)

    # Phenotype subjects
    phenotype_path = smoke_dir / "phenotype.tsv" if smoke else Path(c.config.get("phenotype_file"))
    phenotype_df = pd.read_csv(phenotype_path, sep="\t", dtype={"participant_id": str})
    phenotype_ids = set(phenotype_df["participant_id"])

    common = sorted(eeg_ids & fmri_ids & phenotype_ids)

    # Drop subjects with missing values in confound columns (age, gender, study_site)
    target_col = c.config.get("target_column", "diagnosis")
    confound_cols = [col for col in ("age", "gender", "study_site") if col != target_col]
    available_confounds = [col for col in confound_cols if col in phenotype_df.columns]
    if available_confounds:
        pheno_common = phenotype_df[phenotype_df["participant_id"].isin(common)]
        missing_mask = pheno_common[available_confounds].isna().any(axis=1)
        n_dropped = int(missing_mask.sum())
        if n_dropped:
            dropped_ids = set(pheno_common.loc[missing_mask, "participant_id"])
            common = [s for s in common if s not in dropped_ids]
            print(
                f"[run-intersect] Dropped {n_dropped} subjects with missing confound values "
                f"({', '.join(available_confounds)}) → {len(common)} subjects remaining"
            )

    out_path.write_text("\n".join(common))
    print(
        f"[run-intersect] EEG: {len(eeg_ids)} | fMRI: {len(fmri_ids)} | phenotype: {len(phenotype_ids)}"
        f" → {len(common)} subjects with complete data → {out_path}"
    )


def _find_sub_root(file_path: Path) -> Path:
    """Walk up from file_path to find the directory that directly contains sub-* dirs."""
    p = file_path.parent
    while p != p.parent:
        if p.name.startswith("sub-"):
            return p.parent
        p = p.parent
    raise ValueError(f"Could not find a sub-* directory in the path: {file_path}")


def _detect_input_type(path: Path, modality: str) -> tuple[str, Path]:
    """Detect input type from the user-provided path.

    - File → tsv mode (any filename, any extension — user points directly to their table).
    - Directory → inspect contents:
        EEG:  *_eeg.fif files present → mne mode
        fMRI: *_desc-correlation_matrix.tsv files present → halfpipe mode
    """
    path = Path(path)

    if path.is_file():
        print(f"[auto-detect] {modality.upper()} → tsv ({path})", flush=True)
        return ("tsv", path)

    if path.is_dir():
        if modality == "eeg":
            hits = list(path.rglob("*_eeg.fif"))
            if hits:
                root = _find_sub_root(hits[0])
                print(f"[auto-detect] EEG → mne ({root})", flush=True)
                return ("mne", root)
            raise FileNotFoundError(
                f"No MNE-BIDS .fif files found under {path}.\n"
                "Expected sub-*/[ses-*/]eeg/*_eeg.fif files.\n"
                "If your EEG data is a flat TSV, set eeg_path to the file path in invoke.yaml."
            )
        else:
            hits = list(path.rglob("*_desc-correlation_matrix.tsv"))
            if hits:
                root = _find_sub_root(hits[0])
                print(f"[auto-detect] fMRI → halfpipe ({root})", flush=True)
                return ("halfpipe", root)
            raise FileNotFoundError(
                f"No Halfpipe correlation matrix files found under {path}.\n"
                "Expected sub-*/**/task-rest/*_desc-correlation_matrix.tsv files.\n"
                "If your fMRI data is a flat TSV, set fmri_path to the file path in invoke.yaml."
            )

    raise FileNotFoundError(
        f"Path not found: {path}\n"
        f"Set {'eeg_path' if modality == 'eeg' else 'fmri_path'} in invoke.yaml."
    )


def _resolve_path(c, modality: str, smoke_dir: Path | None = None) -> tuple[str, Path]:
    """Return (input_type, path) for a given modality.

    In smoke mode: use the flat TSV generated by generate-smoke-data (fixed names).
    Otherwise: read eeg_path / fmri_path from invoke.yaml and detect from the path itself.
    """
    if smoke_dir is not None:
        tsv = smoke_dir / ("eeg_features.tsv" if modality == "eeg" else "fmri_features.tsv")
        return ("tsv", tsv)

    path_key = "eeg_path" if modality == "eeg" else "fmri_path"
    path = Path(c.config.get(path_key))
    return _detect_input_type(path, modality)


def _load_subjects(output_data_dir: Path) -> list[str] | None:
    """Read subjects.txt if it exists, else return None (load all)."""
    path = output_data_dir / "subjects.txt"
    if path.exists():
        return [s for s in path.read_text().splitlines() if s]
    return None


@task(pre=[run_intersect])
def run_load_eeg(c, subjects=None, smoke=False):
    """Load EEG features (TSV or MNE folder) → output_data/eeg_features.tsv"""
    from analysis.load_eeg import load_eeg
    from airoh.utils import ensure_dir_exist

    out_path = Path(c.config.get("output_data_dir")) / "eeg_features.tsv"
    if out_path.exists():
        print(f"[run-load-eeg] Skipping — {out_path} already exists")
        return

    ensure_dir_exist(c, "output_data_dir")
    smoke_dir = Path(c.config.get("source_data_dir")) / "smoke"
    eeg_type, path = _resolve_path(c, "eeg", smoke_dir if smoke else None)

    output_data_dir = Path(c.config.get("output_data_dir"))
    subjects_list = subjects.split(",") if subjects else _load_subjects(output_data_dir)
    eeg_task = c.config.get("eeg_mne_task", "rest")
    df = load_eeg(path, input_type=eeg_type, subjects=subjects_list, task=eeg_task)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"[run-load-eeg] {len(df)} subjects, {len(df.columns) - 1} features → {out_path}")


@task(pre=[run_intersect])
def run_load_fmri(c, subjects=None, smoke=False):
    """Load fMRI connectivity (TSV or Halfpipe folder) → output_data/fmri_features.tsv"""
    from analysis.load_fmri import load_fmri
    from airoh.utils import ensure_dir_exist

    out_path = Path(c.config.get("output_data_dir")) / "fmri_features.tsv"
    if out_path.exists():
        print(f"[run-load-fmri] Skipping — {out_path} already exists")
        return

    ensure_dir_exist(c, "output_data_dir")
    smoke_dir = Path(c.config.get("source_data_dir")) / "smoke"
    fmri_type, path = _resolve_path(c, "fmri", smoke_dir if smoke else None)
    strategy = c.config.get("fmri_halfpipe_strategy", "36P")

    output_data_dir = Path(c.config.get("output_data_dir"))
    subjects_list = subjects.split(",") if subjects else _load_subjects(output_data_dir)
    df = load_fmri(path, input_type=fmri_type, strategy=strategy, subjects=subjects_list)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"[run-load-fmri] {len(df)} subjects, {len(df.columns) - 1} features → {out_path}")


@task
def run_predict(c, target=None, smoke=False):
    """Run EEG-only, fMRI-only, and multimodal prediction → output_data/results/"""
    from analysis.predict import run_prediction
    from airoh.utils import ensure_dir_exist

    target_col = target or c.config.get("target_column", "diagnosis")
    results_dir = Path(c.config.get("output_data_dir")) / "results" / target_col
    metrics_path = results_dir / "metrics.tsv"
    if metrics_path.exists():
        print(f"[run-predict] Skipping — {metrics_path} already exists")
        return

    ensure_dir_exist(c, "output_data_dir")
    results_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(c.config.get("output_data_dir"))
    eeg_path = output_dir / "eeg_features.tsv"
    fmri_path = output_dir / "fmri_features.tsv"

    if smoke:
        smoke_dir = Path(c.config.get("source_data_dir")) / "smoke"
        phenotype_path = smoke_dir / "phenotype.tsv"
    else:
        phenotype_path = Path(c.config.get("phenotype_file"))

    import pandas as pd
    eeg_df = pd.read_csv(eeg_path, sep="\t", dtype={"participant_id": str})
    fmri_df = pd.read_csv(fmri_path, sep="\t", dtype={"participant_id": str})
    phenotype_df = pd.read_csv(phenotype_path, sep="\t", dtype={"participant_id": str})

    model_type = c.config.get("model_type", "ridge")
    n_outer = int(c.config.get("cv_outer_folds", 5))
    n_inner = int(c.config.get("cv_inner_folds", 5))
    pca_variance = float(c.config.get("pca_variance", 0.95))
    n_permutations = int(c.config.get("n_permutations", 100))

    metrics_df, fold_df, importance_df = run_prediction(
        eeg_df=eeg_df,
        fmri_df=fmri_df,
        phenotype_df=phenotype_df,
        target_col=target_col,
        model_type=model_type,
        n_outer=n_outer,
        n_inner=n_inner,
        pca_variance=pca_variance,
        n_permutations=n_permutations,
    )

    metrics_df.to_csv(metrics_path, sep="\t", index=False)
    fold_df.to_csv(results_dir / "fold_scores.tsv", sep="\t", index=False)
    if not importance_df.empty:
        imp_path = results_dir / "feature_importances.tsv"
        importance_df.to_csv(imp_path, sep="\t", index=False)
        print(f"[run-predict] Feature importances → {imp_path}")
    print(f"[run-predict] Done → {metrics_path}")
    print(metrics_df.to_string(index=False))


@task
def run_notebooks(c):
    """Execute notebooks and save figures to output_data/."""
    from airoh.utils import run_notebooks as airoh_run_notebooks, ensure_dir_exist
    notebooks_dir = Path(c.config.get("notebooks_dir"))
    output_dir = Path(c.config.get("output_data_dir")).resolve()
    ensure_dir_exist(c, "output_data_dir")
    airoh_run_notebooks(c, notebooks_dir, output_dir, keys=["source_data_dir", "output_data_dir"])


@task(pre=[fetch, run_intersect, run_load_eeg, run_load_fmri, run_predict, run_notebooks])
def run(c):
    """Full pipeline: fetch → intersect → load EEG → load fMRI → predict → notebooks."""
    print("Pipeline complete.")


@task
def run_smoke(c):
    """Smoke test: generate synthetic data and run minimal end-to-end pipeline."""
    generate_smoke_data(c)
    run_intersect(c, smoke=True)
    run_load_eeg(c, smoke=True)
    run_load_fmri(c, smoke=True)
    run_predict(c, smoke=True)                    # classification: diagnosis
    run_predict(c, target="age", smoke=True)      # regression: age
    run_notebooks(c)


@task
def clean_predict(c):
    """Remove prediction results from output_data/results/."""
    results_dir = Path(c.config.get("output_data_dir")) / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
        print(f"Removed {results_dir}")


@task
def clean_intersect(c):
    """Remove subject intersection file from output_data/."""
    path = Path(c.config.get("output_data_dir")) / "subjects.txt"
    if path.exists():
        path.unlink()
        print(f"Removed {path}")


@task
def clean_outputs(c):
    """Remove flat TSV and PNG outputs from output_data/."""
    from airoh.utils import clean_folder
    clean_folder(c, "output_data_dir", "*.tsv")
    clean_folder(c, "output_data_dir", "*.png")


@task
def clean_smoke(c):
    """Remove generated synthetic data from source_data/smoke/."""
    smoke_dir = Path(c.config.get("source_data_dir")) / "smoke"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
        print(f"Removed {smoke_dir}")
    else:
        print("Nothing to clean (source_data/smoke/ does not exist)")


@task(pre=[clean_intersect, clean_outputs, clean_predict, clean_smoke])
def clean(c):
    """Remove all generated outputs and synthetic smoke data."""
    pass
