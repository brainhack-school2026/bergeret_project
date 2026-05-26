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
def generate_smoke_data(c, n_subjects=15, strategies="36P"):
    """Generate lightweight synthetic data for smoke testing into source_data/smoke/."""
    from analysis.generate_synthetic import generate_all
    out_dir = Path(c.config.get("source_data_dir")) / "smoke"
    generate_all(out_dir, n_subjects=int(n_subjects), strategies=strategies.split(","))


@task
def run_load_eeg(c, smoke=False):
    """Load EEG features from TSV or MNE output folder → output_data/eeg_features.tsv"""
    # TODO: implement — check invoke.yaml for eeg_input_type ("tsv" or "mne")
    print("TODO: run-load-eeg not yet implemented")


@task
def run_load_fmri(c, smoke=False):
    """Load fMRI connectivity from TSV or Halfpipe output → output_data/fmri_features.tsv"""
    # TODO: implement — check invoke.yaml for fmri_input_type ("tsv" or "halfpipe")
    print("TODO: run-load-fmri not yet implemented")


@task
def run_predict(c, target=None, smoke=False):
    """Run EEG-only, fMRI-only, and multimodal prediction → output_data/results/"""
    # TODO: implement — target column is read from invoke.yaml or passed as argument
    print("TODO: run-predict not yet implemented")


@task
def run_notebooks(c):
    """Execute notebooks and save figures to output_data/."""
    from airoh.utils import run_notebooks as airoh_run_notebooks, ensure_dir_exist
    notebooks_dir = Path(c.config.get("notebooks_dir"))
    output_dir = Path(c.config.get("output_data_dir")).resolve()
    ensure_dir_exist(c, "output_data_dir")
    airoh_run_notebooks(c, notebooks_dir, output_dir, keys=["source_data_dir", "output_data_dir"])


@task(pre=[fetch, run_load_eeg, run_load_fmri, run_predict, run_notebooks])
def run(c):
    """Full pipeline: fetch → load EEG → load fMRI → predict → notebooks."""
    print("Pipeline complete.")


@task
def run_smoke(c):
    """Smoke test: generate synthetic data and run minimal end-to-end pipeline."""
    generate_smoke_data(c)
    run_load_eeg(c, smoke=True)
    run_load_fmri(c, smoke=True)
    run_predict(c, smoke=True)
    run_notebooks(c)


@task
def clean_outputs(c):
    """Remove analysis outputs from output_data/."""
    from airoh.utils import clean_folder
    clean_folder(c, "output_data_dir", "*.tsv")
    clean_folder(c, "output_data_dir", "*.png")
    clean_folder(c, "output_data_dir", "results")


@task
def clean_smoke(c):
    """Remove generated synthetic data from source_data/smoke/."""
    smoke_dir = Path(c.config.get("source_data_dir")) / "smoke"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
        print(f"Removed {smoke_dir}")
    else:
        print("Nothing to clean (source_data/smoke/ does not exist)")


@task(pre=[clean_outputs, clean_smoke])
def clean(c):
    """Remove all generated outputs and synthetic smoke data."""
    pass
