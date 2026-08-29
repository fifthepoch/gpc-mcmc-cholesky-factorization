#!/bin/bash
#SBATCH --job-name=exp4_tabpfn
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/tabpfn_ting/%x_%j.out
#SBATCH --error=slurm_logs/tabpfn_ting/%x_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

# Run Experiment 4 (TabPFN) on existing embeddings.
# By default, if datasets/<dataset>-hg exists, use that embedding root and
# therefore prefer projected_512.npy when available.
#
# Optional overrides:
#   DATASET              pcam | camelyon17 | embed    (default: pcam)
#   MAX_TRAIN_SAMPLES    int                          (default: 3500)
#   EMBEDDING_DIR        path to embedding root       (default: HG root when present)
#   TABPFN_TOKEN         Prior Labs API token         (optional)
#   TABPFN_TOKEN_FILE    file containing token        (optional, safer)
#   CONDA_ENV            /path/to/env                 (auto-detected)
#   PROJECT_ROOT         /path/to/project             (auto-detected)
#   ENV_FILE             /path/to/.env                (optional)
#
# Preferred safer auth:
#   printf '%s\n' '<your-prior-labs-token>' > /scratch/ab1234/tabpfn_token.txt
#   chmod 600 /scratch/ab1234/tabpfn_token.txt
#   sbatch --account=torch_pr_xxx_yyy \
#          --export=ALL,NETID=ab1234,DATASET=pcam,CONDA_ENV=/scratch/ab1234/conda-envs/gpc,TABPFN_TOKEN_FILE=/scratch/ab1234/tabpfn_token.txt \
#          scripts/exp4_tabpfn_baseline.sbatch
#
# Inline option, less safe because it may end up in shell history:
#   sbatch --account=torch_pr_xxx_yyy \
#          --export=ALL,NETID=ab1234,DATASET=pcam,CONDA_ENV=/scratch/ab1234/conda-envs/gpc,TABPFN_TOKEN='<your-token>' \
#          scripts/exp4_tabpfn_baseline.sbatch

SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "${SBATCH_DIR}/.." && pwd)}}"
ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/.env}"
NETID="${NETID:-}"

if [ -z "${NETID}" ]; then
  case "${PROJECT_ROOT}" in
    /scratch/*/*)
      NETID="${PROJECT_ROOT#/scratch/}"
      NETID="${NETID%%/*}"
      ;;
    *)
      NETID="${USER:-}"
      ;;
  esac
fi

if [ -z "${NETID}" ]; then
  echo "ERROR: NETID could not be determined." >&2
  echo "Submit with --export=ALL,NETID=<your-netid> if auto-detection fails." >&2
  exit 1
fi

DATASET="${DATASET:-pcam}"
DATA_ROOT="${DATA_ROOT:-datasets}"
EMBEDDING_DIR="${EMBEDDING_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-data}"
DEVICE="${DEVICE:-cuda}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-3500}"
TABPFN_TOKEN_FILE="${TABPFN_TOKEN_FILE:-}"
PREDICT_CHUNK_SIZE="${PREDICT_CHUNK_SIZE:-1000}"
SEED="${SEED:-42}"

SCRATCH_BASE="/scratch/${NETID}"
if [ ! -d "${SCRATCH_BASE}" ] && [ -d "/gpfs/scratch/${NETID}" ]; then
  SCRATCH_BASE="/gpfs/scratch/${NETID}"
fi
CONDA_ENV="${CONDA_ENV:-${SCRATCH_BASE}/conda-envs/gpc}"

module purge || true

CONDA_SH=""
if module --ignore_cache load anaconda3/2025.06 >/dev/null 2>&1 && \
   [ -f /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh ]; then
  CONDA_SH="/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh"
elif [ -f "${SCRATCH_BASE}/miniconda3/etc/profile.d/conda.sh" ]; then
  CONDA_SH="${SCRATCH_BASE}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: Could not find conda.sh." >&2
  echo "Tried cluster module anaconda3/2025.06 and local miniconda installs under:" >&2
  echo "  - ${SCRATCH_BASE}/miniconda3" >&2
  echo "  - ${HOME}/miniconda3" >&2
  exit 1
fi

source "${CONDA_SH}"

export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
set +u
conda activate "${CONDA_ENV}"
set -u

if [ -z "${CONDA_PREFIX:-}" ] || [ ! -d "${CONDA_PREFIX}" ]; then
  echo "ERROR: Failed to activate conda environment: ${CONDA_ENV}" >&2
  exit 1
fi

export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset PYTHONHOME
unset PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
if [ -f "${ENV_FILE}" ]; then
  echo "Loading env from ${ENV_FILE}"
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

export TABPFN_DISABLE_TELEMETRY="${TABPFN_DISABLE_TELEMETRY:-1}"
# Headless: do not try to open a browser for PriorLabs login on compute nodes.
export TABPFN_NO_BROWSER="${TABPFN_NO_BROWSER:-1}"
export TABPFN_TOKEN="${TABPFN_TOKEN:-}"
# Optional: path to a one-line token file (keeps secrets out of --export=...), e.g.
#   echo "$TABPFN_TOKEN" > /scratch/${NETID}/.tabpfn_token
#   chmod 600 /scratch/${NETID}/.tabpfn_token
#   export TABPFN_TOKEN_FILE=/scratch/${NETID}/.tabpfn_token
export TABPFN_TOKEN_FILE="${TABPFN_TOKEN_FILE:-}"

PYTHON="${CONDA_PREFIX}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  echo "ERROR: Python not found at ${PYTHON}" >&2
  exit 1
fi

"${PYTHON}" -c "import dotenv; print('python-dotenv imported OK')" >/dev/null 2>&1 || {
  echo "Installing missing python-dotenv into ${CONDA_ENV} ..."
  "${PYTHON}" -m pip install --no-cache-dir python-dotenv
}

hg_dataset_root_name() {
  case "${DATASET}" in
    pcam) echo "pcam-hg" ;;
    camelyon17) echo "camelyon17-hg" ;;
    *) echo "${DATASET}" ;;
  esac
}

default_hg_embedding_dir="${DATA_ROOT}/$(hg_dataset_root_name)"
if [ -z "${EMBEDDING_DIR}" ]; then
  if [ -d "${default_hg_embedding_dir}/train/embeddings" ]; then
    EMBEDDING_DIR="${default_hg_embedding_dir}"
  else
    EMBEDDING_DIR="data/embeddings"
  fi
fi

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs
mkdir -p "${OUTPUT_DIR}"

if [ ! -f "experiments/exp4_tabpfn_baseline.py" ]; then
  echo "ERROR: experiments/exp4_tabpfn_baseline.py not found under ${PROJECT_ROOT}" >&2
  exit 1
fi

"${PYTHON}" -c "import tabpfn; print('TabPFN imported OK')" || {
  echo "ERROR: TabPFN import check failed. Install it in ${CONDA_ENV}." >&2
  exit 1
}

if [ -z "${TABPFN_TOKEN}" ] && [ -z "${TABPFN_TOKEN_FILE}" ]; then
  echo "WARN: Neither TABPFN_TOKEN nor TABPFN_TOKEN_FILE is set." >&2
  echo "WARN: If TabPFN model weights are not already cached, this job will fail." >&2
  echo "WARN: Get a token from https://ux.priorlabs.ai (License tab), then either:" >&2
  echo "WARN:   --export=ALL,NETID=${NETID},TABPFN_TOKEN=<token>,DATASET=${DATASET}" >&2
  echo "WARN: or put the token in a file and set TABPFN_TOKEN_FILE (see script header)." >&2
fi

have_flat_embeddings() {
  for suffix in train_embeddings train_labels val_embeddings val_labels test_embeddings test_labels; do
    if [ ! -f "${EMBEDDING_DIR}/${DATASET}_${suffix}.npy" ]; then
      return 1
    fi
  done
  return 0
}

have_hg_embeddings() {
  local hg_root_name
  local dataset_root
  local split_dir
  hg_root_name="$(hg_dataset_root_name)"
  dataset_root="${EMBEDDING_DIR}"

  if [ ! -d "${dataset_root}/train/embeddings" ] && [ -d "${EMBEDDING_DIR}/${hg_root_name}/train/embeddings" ]; then
    dataset_root="${EMBEDDING_DIR}/${hg_root_name}"
  fi

  for split_dir in train valid test; do
    if [ ! -d "${dataset_root}/${split_dir}/embeddings" ]; then
      return 1
    fi
    if [ -f "${dataset_root}/${split_dir}/embeddings/projected_512.npy" ] || \
       [ -f "${dataset_root}/${split_dir}/embeddings/embeddings.npy" ]; then
      :
    else
      return 1
    fi
  done

  return 0
}

if ! have_flat_embeddings && ! have_hg_embeddings; then
  echo "ERROR: Missing embeddings for dataset=${DATASET} in ${EMBEDDING_DIR}" >&2
  echo "Expected either flat files like ${EMBEDDING_DIR}/${DATASET}_train_embeddings.npy" >&2
  echo "or a Hugging Face export layout like ${EMBEDDING_DIR}/$(hg_dataset_root_name)/train/embeddings/projected_512.npy" >&2
  exit 1
fi

echo "=============================================================================="
echo "Experiment 4: TabPFN Tabular Model Baseline"
echo "=============================================================================="
echo "Job ID            : ${SLURM_JOB_ID}"
echo "Node              : $(hostname)"
echo "GPU               : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time        : $(date)"
echo "Project root      : ${PROJECT_ROOT}"
echo "Conda env         : ${CONDA_ENV}"
echo "Dataset           : ${DATASET}"
echo "Data root         : ${DATA_ROOT}"
echo "Embedding dir     : ${EMBEDDING_DIR}"
echo "Output dir        : ${OUTPUT_DIR}"
echo "Device            : ${DEVICE}"
echo "Max train samples : ${MAX_TRAIN_SAMPLES}"
echo "Predict chunk     : ${PREDICT_CHUNK_SIZE}"
echo "Seed              : ${SEED}"
echo "Token set         : $( [ -n "${TABPFN_TOKEN}" ] && echo yes || echo no )"
echo "=============================================================================="

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

"${PYTHON}" experiments/exp4_tabpfn_baseline.py \
    --dataset "${DATASET}" \
    --embedding-dir "${EMBEDDING_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-train-samples "${MAX_TRAIN_SAMPLES}" \
    --predict-chunk-size "${PREDICT_CHUNK_SIZE}" \
    --device "${DEVICE}" \
    --seed "${SEED}"

echo "=============================================================================="
echo "Done: $(date)"
echo "Expected outputs in ${PROJECT_ROOT}/${OUTPUT_DIR}:"
echo "  - exp4_${DATASET}_results.json"
echo "  - exp4_${DATASET}_calibration.png"
echo "  - exp4_${DATASET}_roc.png"
echo "=============================================================================="
