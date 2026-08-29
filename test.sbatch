#!/bin/bash
#SBATCH --job-name=exp3_test
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

# Run only experiments/exp3_nn_baseline.py.
# If embeddings are missing, it can optionally run extract_embeddings.py first.
# It also supports the precomputed Hugging Face export layout under
# datasets/pcam-hg or datasets/camelyon17-hg.
#
# Example:
#   sbatch --export=ALL,DATASET=pcam test.sbatch

SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SBATCH_DIR}}}"
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
ENCODER="${ENCODER:-densenet121}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"
AUTO_EXTRACT="${AUTO_EXTRACT:-}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
DROPOUT="${DROPOUT:-0.3}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SEED="${SEED:-42}"

SCRATCH_BASE="/scratch/${NETID}"
if [ ! -d "${SCRATCH_BASE}" ] && [ -d "/gpfs/scratch/${NETID}" ]; then
  SCRATCH_BASE="/gpfs/scratch/${NETID}"
fi
CONDA_ENV="${CONDA_ENV:-${SCRATCH_BASE}/conda-envs/gpc}"

# Conda activation: prefer the cluster module when available, otherwise fall
# back to a user-installed miniconda under /scratch.
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

PYTHON="${CONDA_PREFIX}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  echo "ERROR: Python not found at ${PYTHON}" >&2
  exit 1
fi

default_hg_embedding_dir="${DATA_ROOT}"
case "${DATASET}" in
  pcam) default_hg_embedding_dir="${DATA_ROOT}/pcam-hg" ;;
  camelyon17) default_hg_embedding_dir="${DATA_ROOT}/camelyon17-hg" ;;
  *) default_hg_embedding_dir="${DATA_ROOT}/${DATASET}" ;;
esac

if [ -z "${EMBEDDING_DIR}" ]; then
  if [ -d "${default_hg_embedding_dir}/train/embeddings" ]; then
    EMBEDDING_DIR="${default_hg_embedding_dir}"
  else
    EMBEDDING_DIR="data/embeddings"
  fi
fi

if [ -z "${AUTO_EXTRACT}" ]; then
  if [ "${EMBEDDING_DIR}" = "${default_hg_embedding_dir}" ] && \
     [ -d "${default_hg_embedding_dir}/train/embeddings" ]; then
    AUTO_EXTRACT=0
  else
    AUTO_EXTRACT=1
  fi
fi

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${EMBEDDING_DIR}"

if [ ! -f "experiments/exp3_nn_baseline.py" ]; then
  echo "ERROR: experiments/exp3_nn_baseline.py not found under ${PROJECT_ROOT}" >&2
  exit 1
fi

if [ ! -f "experiments/extract_embeddings.py" ]; then
  echo "ERROR: experiments/extract_embeddings.py not found under ${PROJECT_ROOT}" >&2
  exit 1
fi

hg_dataset_root_name() {
  case "${DATASET}" in
    pcam) echo "pcam-hg" ;;
    camelyon17) echo "camelyon17-hg" ;;
    *) echo "${DATASET}" ;;
  esac
}

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
    if [ ! -f "${dataset_root}/${split_dir}/embeddings/y_embeddings.npy" ]; then
      return 1
    fi
    if [ -f "${dataset_root}/${split_dir}/embeddings/embeddings.npy" ] || \
       [ -f "${dataset_root}/${split_dir}/embeddings/projected_512.npy" ]; then
      :
    else
      return 1
    fi
  done

  return 0
}

need_embeddings=0
if ! have_flat_embeddings && ! have_hg_embeddings; then
  need_embeddings=1
fi

if [ "${need_embeddings}" = "1" ] && [ "${AUTO_EXTRACT}" = "1" ]; then
  echo "=============================================================================="
  echo "[Step 1] Missing embeddings detected. Extracting ${DATASET} embeddings first."
  echo "Data root    : ${DATA_ROOT}"
  echo "Embed dir    : ${EMBEDDING_DIR}"
  echo "Encoder      : ${ENCODER}"
  echo "Embed batch  : ${EMBED_BATCH_SIZE}"
  echo "=============================================================================="

  "${PYTHON}" experiments/extract_embeddings.py \
      --dataset "${DATASET}" \
      --data-root "${DATA_ROOT}" \
      --output-dir "${EMBEDDING_DIR}" \
      --encoder "${ENCODER}" \
      --batch-size "${EMBED_BATCH_SIZE}" \
      --device "${DEVICE}"
fi

if ! have_flat_embeddings && ! have_hg_embeddings; then
  echo "ERROR: Missing embeddings for dataset=${DATASET} in ${EMBEDDING_DIR}" >&2
  echo "Expected either flat files like ${EMBEDDING_DIR}/${DATASET}_train_embeddings.npy" >&2
  echo "or a Hugging Face export layout like ${EMBEDDING_DIR}/$(hg_dataset_root_name)/train/embeddings/embeddings.npy" >&2
  echo "with y_embeddings.npy." >&2
  echo "Either rerun with AUTO_EXTRACT=1, or point EMBEDDING_DIR to an existing embedding directory." >&2
  exit 1
fi

echo "=============================================================================="
echo "Experiment 3 Test Job"
echo "=============================================================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Project root : ${PROJECT_ROOT}"
echo "Conda env    : ${CONDA_ENV}"
echo "Dataset      : ${DATASET}"
echo "Data root    : ${DATA_ROOT}"
echo "Embed dir    : ${EMBEDDING_DIR}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "Device       : ${DEVICE}"
echo "Encoder      : ${ENCODER}"
echo "Auto extract : ${AUTO_EXTRACT}"
echo "Epochs       : ${EPOCHS}"
echo "Batch size   : ${BATCH_SIZE}"
echo "=============================================================================="

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo ""
echo "[Step 2] Training exp3_nn_baseline.py on ${DATASET}"
"${PYTHON}" experiments/exp3_nn_baseline.py \
    --dataset "${DATASET}" \
    --embedding-dir "${EMBEDDING_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --dropout "${DROPOUT}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --seed "${SEED}"

echo "=============================================================================="
echo "Done: $(date)"
echo "Expected outputs in ${PROJECT_ROOT}/${OUTPUT_DIR}:"
echo "  - exp3_${DATASET}_results.json"
echo "  - exp3_${DATASET}_calibration.png"
echo "  - exp3_${DATASET}_roc.png"
echo "=============================================================================="
