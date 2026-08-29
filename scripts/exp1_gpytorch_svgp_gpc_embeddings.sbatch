#!/bin/bash
#SBATCH --job-name=exp1_svgp
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp1_gpytorch_svgp/%x_%j.out
#SBATCH --error=slurm_logs/exp1_gpytorch_svgp/%x_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

# Submit with, for example:
#   sbatch --account=csci_ga_sh 2565-2026sp --partition=<partition> \
#          --export=ALL,NETID=sd6701 scripts/exp1_gpytorch_svgp_gpc_embeddings.sbatch
#
# Optional overrides:
#   NETID, PROJECT_ROOT, CONDA_ENV
#   DATASET_ROOT       datasets/pcam-hg
#   EMBEDDING_FILE     projected_512.npy | embeddings.npy
#   LABEL_FILE         y_embeddings.npy | labels.csv
#   OUTPUT_DIR         data
#   OUTPUT_NAME        exp1_gpytorch_svgp_pcam_results.npz
#   NUM_INDUCING       512
#   BATCH_SIZE         1024
#   NUM_EPOCHS         50
#   LEARNING_RATE      0.01
#   MAX_TRAIN          empty means full train split
#   MAX_VALID          empty means full valid split
#   MAX_TEST           empty means full test split
#   SEED               42
#   N_BINS             15
#   THRESHOLD          0.5
#   NUM_WORKERS        0
#   NO_CUDA            1 to force CPU
#   SAVE_MODEL         1 to also save the model .pt

infer_netid() {
  case "$1" in
    /scratch/*/*)
      printf '%s\n' "$1" | awk -F/ '{print $3}'
      ;;
    *)
      return 1
      ;;
  esac
}

SBATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "${SBATCH_DIR}/.." && pwd)}}"

NETID="${NETID:-}"
if [ -z "${NETID}" ]; then
  NETID="$(infer_netid "${PROJECT_ROOT}" 2>/dev/null || true)"
fi
if [ -z "${NETID}" ] && [ -n "${USER:-}" ]; then
  NETID="${USER}"
fi
if [ -z "${NETID}" ]; then
  echo "ERROR: Could not infer NETID. Submit with --export=ALL,NETID=<your-netid>." >&2
  exit 1
fi

SCRATCH_BASE="/scratch/${NETID}"
if [ ! -d "${SCRATCH_BASE}" ] && [ -d "/gpfs/scratch/${NETID}" ]; then
  SCRATCH_BASE="/gpfs/scratch/${NETID}"
fi
CONDA_ENV="${CONDA_ENV:-${SCRATCH_BASE}/conda-envs/gpc}"
DATASET_ROOT="${DATASET_ROOT:-datasets/pcam-hg}"
EMBEDDING_FILE="${EMBEDDING_FILE:-projected_512.npy}"
LABEL_FILE="${LABEL_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-data}"
OUTPUT_NAME="${OUTPUT_NAME:-exp1_gpytorch_svgp_pcam_results.npz}"
NUM_INDUCING="${NUM_INDUCING:-512}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"
MAX_TRAIN="${MAX_TRAIN:-}"
MAX_VALID="${MAX_VALID:-}"
MAX_TEST="${MAX_TEST:-}"
SEED="${SEED:-42}"
N_BINS="${N_BINS:-15}"
THRESHOLD="${THRESHOLD:-0.5}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NO_CUDA="${NO_CUDA:-0}"
SAVE_MODEL="${SAVE_MODEL:-0}"

if [ -f "${PROJECT_ROOT}/scripts/set_scratch_env.sh" ]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/scripts/set_scratch_env.sh"
fi

load_conda() {
  module purge >/dev/null 2>&1 || true

  local candidates=(
    "/scratch/${NETID}/miniconda3/etc/profile.d/conda.sh"
    "${HOME}/miniconda3/etc/profile.d/conda.sh"
    "/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [ -f "${candidate}" ]; then
      # shellcheck disable=SC1090
      source "${candidate}"
      return 0
    fi
  done

  echo "ERROR: Could not find conda.sh for NETID=${NETID}." >&2
  return 1
}

load_conda

export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
set +u
conda activate "${CONDA_ENV}"
set -u

if [ -z "${CONDA_PREFIX:-}" ] || [ ! -d "${CONDA_PREFIX}" ]; then
  echo "ERROR: Failed to activate conda environment: ${CONDA_ENV}" >&2
  exit 1
fi

unset PYTHONHOME
unset PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID:-$$}"

mkdir -p "${PROJECT_ROOT}/slurm_logs/exp1_gpytorch_svgp"
mkdir -p "${PROJECT_ROOT}/${OUTPUT_DIR}"
mkdir -p "${MPLCONFIGDIR}"

PYTHON="${CONDA_PREFIX}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  echo "ERROR: Python not found at ${PYTHON}" >&2
  exit 1
fi

"${PYTHON}" -c "import numpy, torch, gpytorch; print('Env import check: OK')" || {
  echo "ERROR: Import check failed in ${CONDA_ENV}." >&2
  exit 1
}

cd "${PROJECT_ROOT}"

if [ ! -f "experiments/exp1_gpytorch_svgp_gpc_embeddings.py" ]; then
  echo "ERROR: experiments/exp1_gpytorch_svgp_gpc_embeddings.py not found under ${PROJECT_ROOT}" >&2
  exit 1
fi

echo "=============================================================================="
echo "Experiment 1: GPyTorch SVGP binary GP classification on PCam-HG embeddings"
echo "=============================================================================="
echo "Job ID         : ${SLURM_JOB_ID:-N/A}"
echo "Node           : $(hostname)"
echo "Start time     : $(date)"
echo "Project root   : ${PROJECT_ROOT}"
echo "NETID          : ${NETID}"
echo "Conda env      : ${CONDA_ENV}"
echo "Dataset root   : ${DATASET_ROOT}"
echo "Embedding file : ${EMBEDDING_FILE}"
echo "Label file     : ${LABEL_FILE:-auto}"
echo "Output dir     : ${OUTPUT_DIR}"
echo "Output name    : ${OUTPUT_NAME}"
echo "Num inducing   : ${NUM_INDUCING}"
echo "Batch size     : ${BATCH_SIZE}"
echo "Num epochs     : ${NUM_EPOCHS}"
echo "Learning rate  : ${LEARNING_RATE}"
echo "Max train      : ${MAX_TRAIN:-full}"
echo "Max valid      : ${MAX_VALID:-full}"
echo "Max test       : ${MAX_TEST:-full}"
echo "Seed           : ${SEED}"
echo "N bins         : ${N_BINS}"
echo "Threshold      : ${THRESHOLD}"
echo "Num workers    : ${NUM_WORKERS}"
echo "No CUDA        : ${NO_CUDA}"
echo "Save model     : ${SAVE_MODEL}"
echo "=============================================================================="

CMD=(
  "${PYTHON}" "experiments/exp1_gpytorch_svgp_gpc_embeddings.py"
  "--dataset-root" "${DATASET_ROOT}"
  "--embedding-file" "${EMBEDDING_FILE}"
  "--output-dir" "${OUTPUT_DIR}"
  "--output-name" "${OUTPUT_NAME}"
  "--num-inducing" "${NUM_INDUCING}"
  "--batch-size" "${BATCH_SIZE}"
  "--num-epochs" "${NUM_EPOCHS}"
  "--learning-rate" "${LEARNING_RATE}"
  "--seed" "${SEED}"
  "--n-bins" "${N_BINS}"
  "--threshold" "${THRESHOLD}"
  "--num-workers" "${NUM_WORKERS}"
)

if [ -n "${LABEL_FILE}" ]; then
  CMD+=("--label-file" "${LABEL_FILE}")
fi
if [ -n "${MAX_TRAIN}" ]; then
  CMD+=("--max-train" "${MAX_TRAIN}")
fi
if [ -n "${MAX_VALID}" ]; then
  CMD+=("--max-valid" "${MAX_VALID}")
fi
if [ -n "${MAX_TEST}" ]; then
  CMD+=("--max-test" "${MAX_TEST}")
fi
if [ "${NO_CUDA}" = "1" ]; then
  CMD+=("--no-cuda")
fi
if [ "${SAVE_MODEL}" = "1" ]; then
  CMD+=("--save-model")
fi

"${CMD[@]}"

echo "=============================================================================="
echo "Done: $(date)"
echo "Artifacts written under ${PROJECT_ROOT}/${OUTPUT_DIR}"
echo "=============================================================================="
