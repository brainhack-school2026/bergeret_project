import shutil
from pathlib import Path
from invoke import task


@task
def fetch(c):
    """Print instructions for placing real source data."""
    print("Place your data files in source_data/ (or update paths in invoke.yaml):")
    print("  phenotype.tsv       — participants x phenotypic variables")
    print("  eeg_features.tsv    — OR set eeg_input_type=mne and eeg_mne_dir in invoke.yaml")
    print("  fmri_features.tsv   — OR set fmri_input_type=halfpipe and fmri_halfpipe_dir in invoke.yaml")


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


def _detect_input_type(search_dir: Path, modality: str) -> tuple[str, Path]:
    """Auto-detect EEG or fMRI input format by inspecting search_dir.

    EEG: looks for 'eeg_features.tsv' (→ tsv) then 'mne_output/' (→ mne).
    fMRI: looks for 'fmri_features.tsv' (→ tsv) then 'halfpipe_output/' (→ halfpipe).
    TSV takes priority when both are present.
    """
    if modality == "eeg":
        if (search_dir / "eeg_features.tsv").exists():
            detected = ("tsv", search_dir / "eeg_features.tsv")
        elif (search_dir / "mne_output").is_dir():
            detected = ("mne", search_dir / "mne_output")
        else:
            raise FileNotFoundError(
                f"No EEG input found in {search_dir} — "
                "expected 'eeg_features.tsv' or 'mne_output/'"
            )
    else:
        if (search_dir / "fmri_features.tsv").exists():
            detected = ("tsv", search_dir / "fmri_features.tsv")
        elif (search_dir / "halfpipe_output").is_dir():
            detected = ("halfpipe", search_dir / "halfpipe_output")
        else:
            raise FileNotFoundError(
                f"No fMRI input found in {search_dir} — "
                "expected 'fmri_features.tsv' or 'halfpipe_output/'"
            )
    print(f"[auto-detect] {modality.upper()} → {detected[0]} ({detected[1]})", flush=True)
    return detected


def _resolve_path(c, modality: str, smoke_dir: Path | None = None) -> tuple[str, Path]:
    """Return (input_type, path) for a given modality.

    In smoke mode (smoke_dir provided): always auto-detect from smoke_dir.
    Otherwise: auto-detect from source_data_dir when input_type is 'auto',
    or use the explicit config paths.
    """
    if smoke_dir is not None:
        return _detect_input_type(smoke_dir, modality)

    if modality == "eeg":
        input_type = c.config.get("eeg_input_type", "auto")
        if input_type == "auto":
            return _detect_input_type(Path(c.config.get("source_data_dir")), "eeg")
        path = Path(c.config.get("eeg_mne_dir") if input_type == "mne" else c.config.get("eeg_tsv"))
        return input_type, path

    input_type = c.config.get("fmri_input_type", "auto")
    if input_type == "auto":
        return _detect_input_type(Path(c.config.get("source_data_dir")), "fmri")
    path = Path(c.config.get("fmri_halfpipe_dir") if input_type == "halfpipe" else c.config.get("fmri_tsv"))
    return input_type, path


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
    df = load_eeg(path, input_type=eeg_type, subjects=subjects_list)
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
