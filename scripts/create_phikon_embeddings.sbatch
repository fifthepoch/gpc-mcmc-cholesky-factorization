#!/bin/bash
#SBATCH --job-name=phikon_embed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/phikon/%x_%j.out
#SBATCH --error=slurm_logs/phikon/%x_%j.err
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1

set -euo pipefail
export PYTHONNOUSERSITE=1

# =============================================================================
# Submit with:
#   sbatch --export=ALL,NETID=sd6701 scripts/create_phikon_embeddings.sbatch
#
# Optional overrides:
#   DATASET            pcam | camelyon17 | all          (default: camelyon17)
#   PCAM_DIRNAME       folder name for PCam under datasets/    (default: pcam-hg)
#   CAMELYON17_DIRNAME folder name for CAMELYON17 under datasets/ (default: camelyon17-hg-2)
#   SPLITS             "train valid test" or any subset (default: train valid test)
#   BATCH_SIZE         encode batch size                (default: 64)
#   PROJECT_DIM        target projected dim (or empty)  (default: 512)
#   MODEL_NAME         hf model id                      (default: owkin/phikon)
#   DTYPE              float32 | float16                (default: float32)
#   FEATURE_POOLING    cls | mean                       (default: cls)
#   OVERWRITE          1 to recompute existing outputs  (default: 0)
# =============================================================================

NETID="${NETID:-<netid>}"
if [ "${NETID}" = "<netid>" ]; then
  echo "ERROR: NETID is required. Submit with --export=ALL,NETID=<your-netid>." >&2
  exit 1
fi

SCRATCH_BASE="/scratch/${NETID}"
if [ ! -d "${SCRATCH_BASE}" ] && [ -d "/gpfs/scratch/${NETID}" ]; then
  SCRATCH_BASE="/gpfs/scratch/${NETID}"
fi
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/gpc-mcmc-cholesky-factorization}"
CONDA_ENV="${CONDA_ENV:-${SCRATCH_BASE}/conda-envs/gpc}"
CONDA_SH="${CONDA_SH:-${SCRATCH_BASE}/miniconda3/etc/profile.d/conda.sh}"

DATASET="${DATASET:-camelyon17}"
PCAM_DIRNAME="${PCAM_DIRNAME:-pcam-hg}"
CAMELYON17_DIRNAME="${CAMELYON17_DIRNAME:-camelyon17-hg-2}"
SPLITS="${SPLITS:-train valid test}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PROJECT_DIM="${PROJECT_DIM:-512}"
MODEL_NAME="${MODEL_NAME:-owkin/phikon}"
DTYPE="${DTYPE:-float32}"
FEATURE_POOLING="${FEATURE_POOLING:-cls}"
OVERWRITE="${OVERWRITE:-0}"

SCRIPT_DIR="${PROJECT_ROOT}/scripts"
if [ -f "${SCRIPT_DIR}/set_scratch_env.sh" ]; then
  source "${SCRIPT_DIR}/set_scratch_env.sh"
fi

if [ ! -f "${CONDA_SH}" ]; then
  echo "ERROR: conda.sh not found at ${CONDA_SH}" >&2
  exit 1
fi
if [ ! -d "${CONDA_ENV}" ]; then
  echo "ERROR: Conda env not found at ${CONDA_ENV}" >&2
  exit 1
fi

export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
set +u
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u

if [ -z "${CONDA_PREFIX:-}" ] || [ ! -d "${CONDA_PREFIX}" ]; then
  echo "ERROR: Failed to activate conda environment: ${CONDA_ENV}" >&2
  exit 1
fi
echo "Activated conda env: ${CONDA_PREFIX}"

export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset PYTHONHOME
unset PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

PYTHON="${CONDA_PREFIX}/bin/python"
echo "Using: ${PYTHON} ($(${PYTHON} --version 2>&1))"

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs/phikon

# -----------------------------------------------------------------------------
# Build a virtual data-root with symlinks so create_phikon_embeddings.py picks
# up custom directory names (e.g. camelyon17-hg-2) without code changes.
# Writes through the symlink land in the *real* directory.
# -----------------------------------------------------------------------------
VIRTUAL_DATA_ROOT="${TMPDIR:-/tmp}/phikon_data_root_${SLURM_JOB_ID:-$$}"
rm -rf "${VIRTUAL_DATA_ROOT}"
mkdir -p "${VIRTUAL_DATA_ROOT}"

REAL_PCAM_DIR="${PROJECT_ROOT}/datasets/${PCAM_DIRNAME}"
REAL_CAMELYON17_DIR="${PROJECT_ROOT}/datasets/${CAMELYON17_DIRNAME}"

if [ -d "${REAL_PCAM_DIR}" ]; then
  ln -sfn "${REAL_PCAM_DIR}" "${VIRTUAL_DATA_ROOT}/pcam-hg"
elif [ "${DATASET}" = "pcam" ] || [ "${DATASET}" = "all" ]; then
  echo "ERROR: PCam directory not found: ${REAL_PCAM_DIR}" >&2
  exit 1
fi

if [ -d "${REAL_CAMELYON17_DIR}" ]; then
  ln -sfn "${REAL_CAMELYON17_DIR}" "${VIRTUAL_DATA_ROOT}/camelyon17-hg"
elif [ "${DATASET}" = "camelyon17" ] || [ "${DATASET}" = "all" ]; then
  echo "ERROR: CAMELYON17 directory not found: ${REAL_CAMELYON17_DIR}" >&2
  exit 1
fi

trap 'rm -rf "${VIRTUAL_DATA_ROOT}"' EXIT

echo "=============================================================================="
echo "Phikon embedding extraction"
echo "=============================================================================="
echo "Job ID            : ${SLURM_JOB_ID:-N/A}"
echo "Node              : $(hostname)"
echo "GPU               : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Dataset           : ${DATASET}"
echo "PCam dir          : ${REAL_PCAM_DIR}"
echo "CAMELYON17 dir    : ${REAL_CAMELYON17_DIR}"
echo "Virtual data root : ${VIRTUAL_DATA_ROOT}"
echo "Splits            : ${SPLITS}"
echo "Batch size        : ${BATCH_SIZE}"
echo "Project dim       : ${PROJECT_DIM}"
echo "Model             : ${MODEL_NAME}"
echo "Dtype             : ${DTYPE}"
echo "Pooling           : ${FEATURE_POOLING}"
echo "Overwrite         : ${OVERWRITE}"
echo "Project root      : ${PROJECT_ROOT}"
echo "Conda env         : ${CONDA_ENV}"
echo "=============================================================================="

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

EXTRA_ARGS=()
if [ -n "${PROJECT_DIM}" ]; then
  EXTRA_ARGS+=(--project-dim "${PROJECT_DIM}")
fi
if [ "${OVERWRITE}" = "1" ]; then
  EXTRA_ARGS+=(--overwrite)
fi
# shellcheck disable=SC2206
SPLITS_ARR=( ${SPLITS} )

"${PYTHON}" scripts/create_phikon_embeddings.py \
  --dataset "${DATASET}" \
  --data-root "${VIRTUAL_DATA_ROOT}" \
  --device cuda \
  --batch-size "${BATCH_SIZE}" \
  --model-name "${MODEL_NAME}" \
  --dtype "${DTYPE}" \
  --feature-pooling "${FEATURE_POOLING}" \
  --splits "${SPLITS_ARR[@]}" \
  "${EXTRA_ARGS[@]}"

echo "=============================================================================="
echo "Done: $(date)"
echo "Embeddings written under each <split>/embeddings/ in:"
[ -d "${REAL_PCAM_DIR}" ]      && echo "  ${REAL_PCAM_DIR}"
[ -d "${REAL_CAMELYON17_DIR}" ] && echo "  ${REAL_CAMELYON17_DIR}"
echo "=============================================================================="
