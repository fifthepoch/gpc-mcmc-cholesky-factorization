#!/bin/bash
#SBATCH --job-name=svgp_full8
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --output=slurm_logs/exp1_gpytorch_svgp_full8/%x_%A_%a.out
#SBATCH --error=slurm_logs/exp1_gpytorch_svgp_full8/%x_%A_%a.err

set -euo pipefail
export PYTHONNOUSERSITE=1

# One script for the 8 full-dataset SVGP jobs:
#   dataset in {pcam, camelyon17}
#   standardization in {standardized, nostd}
#   inducing points in {200, 512}
#
# Submit:
#   sbatch --export=ALL,NETID=<your-netid> scripts/exp1_gpytorch_svgp_full_8way_sweep.sbatch
#
# Optional overrides:
#   PROJECT_ROOT, CONDA_ENV, OUTPUT_ROOT
#   NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, LENGTHSCALE
#   NO_CUDA, SAVE_MODEL

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
CONDA_ENV="${CONDA_ENV:-${SCRATCH_BASE}/conda-envs/gpc}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/svgp_full_8way_standardization_inducing_sweep}"

EMBEDDING_FILE="${EMBEDDING_FILE:-projected_512.npy}"
LABEL_FILE="${LABEL_FILE:-}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LEARNING_RATE="${LEARNING_RATE:-0.01}"
LENGTHSCALE="${LENGTHSCALE:-}"
N_BINS="${N_BINS:-15}"
THRESHOLD="${THRESHOLD:-0.5}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NO_CUDA="${NO_CUDA:-0}"
SAVE_MODEL="${SAVE_MODEL:-0}"
SEED="${SEED:-42}"

DATASET_NAMES=(pcam pcam pcam pcam camelyon17 camelyon17 camelyon17 camelyon17)
DATASET_ROOTS=(datasets/pcam-hg datasets/pcam-hg datasets/pcam-hg datasets/pcam-hg datasets/camelyon17-hg datasets/camelyon17-hg datasets/camelyon17-hg datasets/camelyon17-hg)
STANDARDIZE_VALUES=(1 1 0 0 1 1 0 0)
STANDARDIZE_TAGS=(standardized standardized nostd nostd standardized standardized nostd nostd)
INDUCING_VALUES=(200 512 200 512 200 512 200 512)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${TASK_ID}" -lt 0 ] || [ "${TASK_ID}" -gt 7 ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID must be in 0..7, got ${TASK_ID}" >&2
  exit 1
fi

DATASET_NAME="${DATASET_NAMES[${TASK_ID}]}"
DATASET_ROOT="${DATASET_ROOTS[${TASK_ID}]}"
STANDARDIZE="${STANDARDIZE_VALUES[${TASK_ID}]}"
STANDARDIZE_TAG="${STANDARDIZE_TAGS[${TASK_ID}]}"
NUM_INDUCING="${INDUCING_VALUES[${TASK_ID}]}"
OUTPUT_NAME="exp1_gpytorch_svgp_${DATASET_NAME}_full_${STANDARDIZE_TAG}_m${NUM_INDUCING}_results.npz"

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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID:-$$}-${TASK_ID}"

PYTHON="${CONDA_PREFIX}/bin/python"

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs/exp1_gpytorch_svgp_full8 "${OUTPUT_ROOT}" "${MPLCONFIGDIR}"

"${PYTHON}" -c "import numpy, torch, gpytorch; print('Env import check: OK')" || {
  echo "ERROR: Import check failed in ${CONDA_ENV}." >&2
  exit 1
}

if [ ! -f "experiments/exp1_gpytorch_svgp_gpc_embeddings.py" ]; then
  echo "ERROR: experiments/exp1_gpytorch_svgp_gpc_embeddings.py not found under ${PROJECT_ROOT}" >&2
  exit 1
fi
if [ ! -d "${DATASET_ROOT}/train/embeddings" ]; then
  echo "ERROR: Dataset root appears missing or incomplete: ${PROJECT_ROOT}/${DATASET_ROOT}" >&2
  exit 1
fi

MANIFEST="${OUTPUT_ROOT}/sweep_manifest.tsv"
if [ "${TASK_ID}" = "0" ]; then
  printf "array_task\tdataset\tdataset_root\tstandardize\tnum_inducing\toutput\n" > "${MANIFEST}"
fi

echo "=============================================================================="
echo "SVGP full 8-way sweep task"
echo "=============================================================================="
echo "Array task      : ${TASK_ID}"
echo "Dataset         : ${DATASET_NAME}"
echo "Dataset root    : ${DATASET_ROOT}"
echo "Standardize     : ${STANDARDIZE}"
echo "Num inducing    : ${NUM_INDUCING}"
echo "Output root     : ${OUTPUT_ROOT}"
echo "Output name     : ${OUTPUT_NAME}"
echo "Epochs          : ${NUM_EPOCHS}"
echo "Batch size      : ${BATCH_SIZE}"
echo "Learning rate   : ${LEARNING_RATE}"
echo "Seed            : ${SEED}"
echo "=============================================================================="

CMD=(
  "${PYTHON}" "experiments/exp1_gpytorch_svgp_gpc_embeddings.py"
  "--dataset" "${DATASET_NAME}"
  "--dataset-root" "${DATASET_ROOT}"
  "--embedding-file" "${EMBEDDING_FILE}"
  "--output-dir" "${OUTPUT_ROOT}"
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
if [ -n "${LENGTHSCALE}" ]; then
  CMD+=("--lengthscale" "${LENGTHSCALE}")
fi
if [ "${STANDARDIZE}" != "1" ]; then
  CMD+=("--no-standardize")
fi
if [ "${NO_CUDA}" = "1" ]; then
  CMD+=("--no-cuda")
fi
if [ "${SAVE_MODEL}" = "1" ]; then
  CMD+=("--save-model")
fi

"${CMD[@]}"

printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
  "${TASK_ID}" "${DATASET_NAME}" "${DATASET_ROOT}" "${STANDARDIZE}" \
  "${NUM_INDUCING}" "${OUTPUT_ROOT}/${OUTPUT_NAME}" >> "${MANIFEST}"

echo "=============================================================================="
echo "Done: $(date)"
echo "Saved: ${PROJECT_ROOT}/${OUTPUT_ROOT}/${OUTPUT_NAME}"
echo "=============================================================================="
