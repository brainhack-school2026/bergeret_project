#!/usr/bin/env bash
# Container entrypoint for Docker / Singularity.
# Parses CLI arguments, writes invoke.yaml, then runs the pipeline.
#
# Input data is provided via bind mounts:
#   -B /your/source_data:/data/source_data
#   -B /your/output_data:/data/output_data
#
# Input format (tsv vs mne / halfpipe) is auto-detected from source_data.
set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
TARGET_COLUMN="diagnosis"
MODEL_TYPE="ridge"
CV_OUTER_FOLDS="5"
CV_INNER_FOLDS="5"
PCA_VARIANCE="0.95"
N_PERMUTATIONS="100"
FMRI_HALFPIPE_STRATEGY="Baseline"
SMOKE="0"
OUTPUT_DIR=""

# ── help ──────────────────────────────────────────────────────────────────────
usage() {
cat << 'EOF'
Multimodal EEG/fMRI fusion pipeline — container entrypoint.

Usage (Singularity):
  singularity run \
    -B /path/to/source_data:/data/source_data \
    -B /path/to/output_data:/data/output_data \
    neuromeld.sif [OPTIONS]

Input format (tsv vs MNE / Halfpipe) is detected automatically from source_data.

Options:
  --target-column STR          Phenotype column to predict        [diagnosis]
  --model-type STR             logistic|ridge|elasticnet|          [ridge]
                               svm|random_forest
  --n-permutations INT         Permutations for p-value vs chance  [500]
  --fmri-halfpipe-strategy STR Halfpipe denoising strategy tag     [Baseline]
  --cv-outer-folds INT         Outer CV folds                      [5]
  --cv-inner-folds INT         Inner CV folds (hyperparameter)     [5]
  --pca-variance FLOAT         PCA explained variance threshold    [0.95]
  --smoke                      Self-contained smoke test (no mounts needed)
  --output-dir PATH            Save smoke outputs to PATH (default: ephemeral)
  --help                       Show this message
EOF
exit 0
}

# ── parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --target-column)           TARGET_COLUMN="$2";           shift 2 ;;
        --model-type)              MODEL_TYPE="$2";              shift 2 ;;
        --cv-outer-folds)          CV_OUTER_FOLDS="$2";          shift 2 ;;
        --cv-inner-folds)          CV_INNER_FOLDS="$2";          shift 2 ;;
        --pca-variance)            PCA_VARIANCE="$2";            shift 2 ;;
        --n-permutations)          N_PERMUTATIONS="$2";          shift 2 ;;
        --fmri-halfpipe-strategy)  FMRI_HALFPIPE_STRATEGY="$2";  shift 2 ;;
        --smoke)                   SMOKE="1";                    shift   ;;
        --output-dir)              OUTPUT_DIR="$2";              shift 2 ;;
        --help|-h)                 usage ;;
        *) echo "[container] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Run invoke from a tmpdir so invoke.yaml is always writable (Apptainer
# mounts the container read-only by default).
WORKDIR=$(mktemp -d)
ln -s /app/tasks.py    "$WORKDIR/tasks.py"
ln -s /app/analysis    "$WORKDIR/analysis"
cp -r /app/notebooks   "$WORKDIR/notebooks"
cd "$WORKDIR"

# ── smoke test — no bind mounts needed ───────────────────────────────────────
if [[ "${SMOKE}" == "1" ]]; then
    SMOKE_SOURCE=$(mktemp -d)
    if [[ -n "${OUTPUT_DIR}" ]]; then
        SMOKE_OUTPUT="${OUTPUT_DIR}"
        mkdir -p "${SMOKE_OUTPUT}"
        echo "[container] Smoke test mode — outputs → ${SMOKE_OUTPUT}"
    else
        SMOKE_OUTPUT=$(mktemp -d)
        echo "[container] Smoke test mode — ephemeral (use --output-dir PATH to keep outputs)"
    fi
    cat > "$WORKDIR/invoke.yaml" << EOF
code_dir: analysis
notebooks_dir: notebooks
source_data_dir: ${SMOKE_SOURCE}
output_data_dir: ${SMOKE_OUTPUT}
phenotype_file: ${SMOKE_SOURCE}/smoke/phenotype.tsv
eeg_input_type: auto
fmri_input_type: auto
fmri_halfpipe_strategy: ${FMRI_HALFPIPE_STRATEGY}
target_column: ${TARGET_COLUMN}
model_type: ${MODEL_TYPE}
cv_outer_folds: ${CV_OUTER_FOLDS}
cv_inner_folds: ${CV_INNER_FOLDS}
pca_variance: ${PCA_VARIANCE}
n_permutations: ${N_PERMUTATIONS}
EOF
    exec invoke run-smoke
fi

# ── production run ────────────────────────────────────────────────────────────
if [[ ! -d "/data/source_data" ]] || [[ -z "$(ls -A /data/source_data 2>/dev/null)" ]]; then
    echo "[container] ERROR: /data/source_data is empty or not mounted." >&2
    echo "            Add: -B /your/source_data:/data/source_data" >&2
    exit 1
fi

cat > "$WORKDIR/invoke.yaml" << EOF
code_dir: analysis
notebooks_dir: notebooks
source_data_dir: /data/source_data
output_data_dir: /data/output_data
phenotype_file: /data/source_data/phenotype.tsv
eeg_input_type: auto
fmri_input_type: auto
fmri_halfpipe_strategy: ${FMRI_HALFPIPE_STRATEGY}
target_column: ${TARGET_COLUMN}
model_type: ${MODEL_TYPE}
cv_outer_folds: ${CV_OUTER_FOLDS}
cv_inner_folds: ${CV_INNER_FOLDS}
pca_variance: ${PCA_VARIANCE}
n_permutations: ${N_PERMUTATIONS}
EOF

echo "[container] Configuration:"
cat "$WORKDIR/invoke.yaml"
echo ""

exec invoke run-intersect run-load-eeg run-load-fmri run-predict run-notebooks
