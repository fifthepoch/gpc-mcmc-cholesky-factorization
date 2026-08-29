#!/bin/bash
#SBATCH --job-name=exp3_unet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/exp3_unet/%x_%j.out
#SBATCH --error=slurm_logs/exp3_unet/%x_%j.err
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1

set -euo pipefail
export PYTHONNOUSERSITE=1

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
# Space-separated list, e.g. "pcam", "camelyon17", or "pcam camelyon17".
DATASETS="${DATASETS:-pcam camelyon17}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-ML-Final_project}"
WANDB_ENTITY="${WANDB_ENTITY:-a-salt}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-exp3-unet}"

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
cd "${PROJECT_ROOT}"
mkdir -p slurm_logs
mkdir -p data/exp3_unet

echo "=============================================================================="
echo "Experiment 3: U-Net pathology classifier"
echo "=============================================================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "Datasets     : ${DATASETS}"
echo "Epochs       : ${EPOCHS}"
echo "Batch size   : ${BATCH_SIZE}"
echo "Use W&B      : ${USE_WANDB}"
echo "Project root : ${PROJECT_ROOT}"
echo "Conda env    : ${CONDA_ENV}"
echo "=============================================================================="

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

WANDB_ARGS=()
if [ "${USE_WANDB}" = "1" ]; then
  WANDB_ARGS+=(--wandb --wandb-project "${WANDB_PROJECT}" --wandb-mode "${WANDB_MODE}")
  if [ -n "${WANDB_ENTITY}" ]; then
    WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
  fi
  if [ -n "${WANDB_GROUP}" ]; then
    WANDB_ARGS+=(--wandb-group "${WANDB_GROUP}")
  fi
fi

# shellcheck disable=SC2206
DATASETS_ARR=( ${DATASETS} )
for DS in "${DATASETS_ARR[@]}"; do
  echo ""
  echo ">>>>>>>>>>>>>>>> Training U-Net on dataset: ${DS} <<<<<<<<<<<<<<<<"
  "${PYTHON}" experiments/exp3_unet_classifier.py \
    --dataset "${DS}" \
    --data-root "${PROJECT_ROOT}/datasets" \
    --output-root "${PROJECT_ROOT}/data/exp3_unet" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    "${WANDB_ARGS[@]}"
done

echo "=============================================================================="
echo "Done: $(date)"
echo "Outputs saved under ${PROJECT_ROOT}/data/exp3_unet/<dataset>/"
echo "=============================================================================="
