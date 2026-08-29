#!/bin/bash
#SBATCH --job-name=exp5_vit_mask
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/exp5_vit_small/%x_%j.out
#SBATCH --error=slurm_logs/exp5_vit_small/%x_%j.err
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
# Submit examples:
#   sbatch --export=ALL,NETID=sd6701,DATASET=pcam scripts/exp5_vit_random_masking.sbatch
#   sbatch --export=ALL,NETID=sd6701,DATASET=camelyon17 scripts/exp5_vit_random_masking.sbatch
#   sbatch --export=ALL,NETID=sd6701,DATASET=combined scripts/exp5_vit_random_masking.sbatch
#
# DATASET controls both training data and the default output subdirectory.
DATASET="${DATASET:-combined}"   # one of: pcam, camelyon17, combined (joint training)
EPOCHS="${EPOCHS:-30}"
EVAL_TEST_EVERY="${EVAL_TEST_EVERY:-10}"   # snapshot test metrics every N epochs (0 disables)
BATCH_SIZE="${BATCH_SIZE:-32}"
MASK_RATIO="${MASK_RATIO:-0.0}"   # 0 disables masking; set >0 to enable
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-${DATASET}}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-ML-Final_project}"
WANDB_ENTITY="${WANDB_ENTITY:-a-salt}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-exp5-vit-mask}"

CONDA_SH="${CONDA_SH:-${SCRATCH_BASE}/miniconda3/etc/profile.d/conda.sh}"

SCRIPT_DIR="${PROJECT_ROOT}/scripts"
if [ -f "${SCRIPT_DIR}/set_scratch_env.sh" ]; then
  source "${SCRIPT_DIR}/set_scratch_env.sh"
fi

if [ ! -f "${CONDA_SH}" ]; then
  echo "ERROR: conda.sh not found at ${CONDA_SH}" >&2
  echo "Pass the correct path with --export=ALL,CONDA_SH=/scratch/${NETID}/miniconda3/etc/profile.d/conda.sh" >&2
  exit 1
fi

if [ ! -d "${CONDA_ENV}" ]; then
  echo "ERROR: Conda env not found at ${CONDA_ENV}" >&2
  exit 1
fi

# Avoid conda hook issues under strict nounset shells.
export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
set +u
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u

if [ -z "${CONDA_PREFIX:-}" ] || [ ! -d "${CONDA_PREFIX}" ]; then
  echo "ERROR: Failed to activate conda environment: ${CONDA_ENV}" >&2
  echo "Pass the correct env with --export=ALL,CONDA_ENV=/scratch/${NETID}/conda-envs/<env>" >&2
  exit 1
fi
echo "Activated conda env: ${CONDA_PREFIX}"

# Ensure the activated env's Python wins over any module-level paths.
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset PYTHONHOME
unset PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

PYTHON="${CONDA_PREFIX}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  echo "ERROR: Python not found at ${PYTHON}" >&2
  exit 1
fi
echo "Using: ${PYTHON} ($(${PYTHON} --version 2>&1))"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs/exp5_vit_small
mkdir -p data/exp5_vit

echo "=============================================================================="
echo "Experiment 5: ViT-Small + random patch masking"
echo "=============================================================================="
echo "Job ID         : ${SLURM_JOB_ID}"
echo "Node           : $(hostname)"
echo "Dataset        : ${DATASET}"
echo "Epochs         : ${EPOCHS}"
echo "Eval test every: ${EVAL_TEST_EVERY} epochs (0 = only at end)"
echo "Batch size     : ${BATCH_SIZE}"
echo "Mask ratio     : ${MASK_RATIO}"
echo "Learning rate  : ${LEARNING_RATE}"
echo "Weight decay   : ${WEIGHT_DECAY}"
echo "Output subdir  : ${OUTPUT_SUBDIR}"
echo "Use W&B        : ${USE_WANDB}"
echo "Project root   : ${PROJECT_ROOT}"
echo "Conda env      : ${CONDA_ENV}"
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

"${PYTHON}" experiments/exp5/exp5_vit_small_random_masking.py \
  --dataset "${DATASET}" \
  --data-root "${PROJECT_ROOT}/datasets" \
  --output-root "${PROJECT_ROOT}/data/exp5_vit/${OUTPUT_SUBDIR}" \
  --epochs "${EPOCHS}" \
  --eval-test-every "${EVAL_TEST_EVERY}" \
  --batch-size "${BATCH_SIZE}" \
  --mask-ratio "${MASK_RATIO}" \
  --learning-rate "${LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  "${WANDB_ARGS[@]}"

echo "=============================================================================="
echo "Done: $(date)"
echo "Outputs saved under ${PROJECT_ROOT}/data/exp5_vit"
echo "=============================================================================="
