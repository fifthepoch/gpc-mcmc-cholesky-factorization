#!/bin/bash
#SBATCH --job-name=exp3_nn_baseline
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp_3_nn_ting/%x_%j.out
#SBATCH --error=slurm_logs/exp_3_nn_ting/%x_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

# =============================================================================
# User-configurable paths
# =============================================================================
# Submit with:
#   sbatch --export=ALL,NETID=ab1234 scripts/exp3_nn_baseline.sbatch
#
# Optional overrides:
#   DATASET       pcam | camelyon17 | embed          (default: pcam)
#   ENCODER       densenet121 | dinov2_vitl14        (default: densenet121)
#   EMBEDDING_DIR path to embedding root             (default: dataset-specific HG root when present)
#   MODEL_ARCH    residual_mlp | mlp | linear        (default: residual_mlp)
#   HIDDEN_DIM    classifier hidden width            (default: 512)
#   NUM_LAYERS    residual blocks for residual_mlp   (default: 3)
#   DROPOUT       classifier dropout                 (default: 0.3)
#   LR            AdamW learning rate                (default: 1e-3)
#   WEIGHT_DECAY  AdamW weight decay                 (default: 1e-4)
#   EPOCHS        max classifier epochs              (default: 50)
#   PATIENCE      early-stopping patience            (default: 5)
#   BATCH_SIZE    embedding batch size               (default: 512)
#   CONDA_ENV     /path/to/env                       (auto-detected)
#   PROJECT_ROOT  /path/to/project                   (auto-detected)
#   SKIP_EMBED    1                                  (skip embedding extraction; auto-set for HG layout)

NETID="${NETID:-<netid>}"
if [ "${NETID}" = "<netid>" ]; then
  echo "ERROR: NETID is required. Submit with --export=ALL,NETID=<your-netid>." >&2
  exit 1
fi

DATASET="${DATASET:-pcam}"
ENCODER="${ENCODER:-densenet121}"
MODEL_ARCH="${MODEL_ARCH:-residual_mlp}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
NUM_LAYERS="${NUM_LAYERS:-3}"
DROPOUT="${DROPOUT:-0.3}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-50}"
BATCH_SIZE="${BATCH_SIZE:-512}"

SCRATCH_BASE="/scratch/${NETID}"
if [ ! -d "${SCRATCH_BASE}" ] && [ -d "/gpfs/scratch/${NETID}" ]; then
  SCRATCH_BASE="/gpfs/scratch/${NETID}"
fi
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/gpc-mcmc-cholesky-factorization}"
CONDA_ENV="${CONDA_ENV:-${SCRATCH_BASE}/conda-envs/gpc}"
CONDA_SH="${CONDA_SH:-${SCRATCH_BASE}/miniconda3/etc/profile.d/conda.sh}"

if [ -z "${EMBEDDING_DIR:-}" ]; then
  case "${DATASET}" in
    pcam)
      if [ -d "${PROJECT_ROOT}/datasets/pcam-hg/train/embeddings" ]; then
        EMBEDDING_DIR="datasets/pcam-hg"
      else
        EMBEDDING_DIR="data/embeddings"
      fi
      ;;
    camelyon17)
      if [ -d "${PROJECT_ROOT}/datasets/camelyon17-hg-2/train/embeddings" ]; then
        EMBEDDING_DIR="datasets/camelyon17-hg-2"
      elif [ -d "${PROJECT_ROOT}/datasets/camelyon17-hg/train/embeddings" ]; then
        EMBEDDING_DIR="datasets/camelyon17-hg"
      else
        EMBEDDING_DIR="data/embeddings"
      fi
      ;;
    embed)
      if [ -d "${PROJECT_ROOT}/datasets/embed/train/embeddings" ]; then
        EMBEDDING_DIR="datasets/embed"
      else
        EMBEDDING_DIR="data/embeddings"
      fi
      ;;
    *)
      EMBEDDING_DIR="data/embeddings"
      ;;
  esac
fi

if [ -z "${SKIP_EMBED:-}" ]; then
  case "${EMBEDDING_DIR}" in
    *pcam-hg|*camelyon17-hg|*camelyon17-hg-2|*embed)
      SKIP_EMBED=1
      ;;
    *)
      SKIP_EMBED=0
      ;;
  esac
fi

# =============================================================================
# Load scratch environment (caches, tmp dirs, etc.)
# =============================================================================
SCRIPT_DIR="${PROJECT_ROOT}/scripts"
if [ -f "${SCRIPT_DIR}/set_scratch_env.sh" ]; then
  source "${SCRIPT_DIR}/set_scratch_env.sh"
else
  echo "WARN: set_scratch_env.sh not found, continuing without it." >&2
fi

# =============================================================================
# Conda activation
# =============================================================================
if [ ! -f "${CONDA_SH}" ]; then
  echo "ERROR: conda.sh not found at ${CONDA_SH}" >&2
  echo "Pass the correct path with --export=ALL,CONDA_SH=/scratch/${NETID}/miniconda3/etc/profile.d/conda.sh" >&2
  exit 1
fi

# Avoid occasional conda hook issues under strict nounset shells on this cluster.
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

# Ensure the activated env's Python wins over module-level Anaconda paths.
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

"${PYTHON}" -c "import torch; print(f'PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}')" || {
  echo "ERROR: PyTorch import check failed." >&2
  exit 1
}

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# =============================================================================
# Run Experiment 3
# =============================================================================
cd "${PROJECT_ROOT}"
mkdir -p slurm_logs
mkdir -p "${EMBEDDING_DIR}"

echo "=============================================================================="
echo "Experiment 3: Deterministic Neural Network Baseline"
echo "=============================================================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time   : $(date)"
echo "Project root : ${PROJECT_ROOT}"
echo "Conda env    : ${CONDA_ENV}"
echo "Conda sh     : ${CONDA_SH}"
echo "Dataset      : ${DATASET}"
echo "Encoder      : ${ENCODER}"
echo "Model arch   : ${MODEL_ARCH}"
echo "Hidden dim   : ${HIDDEN_DIM}"
echo "Layers       : ${NUM_LAYERS}"
echo "Dropout      : ${DROPOUT}"
echo "Embed dir    : ${EMBEDDING_DIR}"
echo "Embed path   : ${PROJECT_ROOT}/${EMBEDDING_DIR}"
echo "Skip embed   : ${SKIP_EMBED}"
echo "=============================================================================="

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# --- Step 1: Extract embeddings (GPU) ----------------------------------------
if [ "${SKIP_EMBED}" != "1" ]; then
  echo ""
  echo "[Step 1] Extracting embeddings: ${DATASET} with ${ENCODER}"
  "${PYTHON}" experiments/extract_embeddings.py \
      --dataset "${DATASET}" \
      --data-root datasets \
      --output-dir "${EMBEDDING_DIR}" \
      --encoder "${ENCODER}" \
      --batch-size 256 \
      --device cuda
else
  echo ""
  echo "[Step 1] Skipped (SKIP_EMBED=1). Using existing embeddings in ${EMBEDDING_DIR}"
  echo "         exp3_nn_baseline.py will prefer projected_512.npy when it exists."
fi

# --- Step 2: Train NN head + evaluate -----------------------------------------
echo ""
echo "[Step 2] Training ${MODEL_ARCH} head and evaluating on ${DATASET}"
"${PYTHON}" experiments/exp3_nn_baseline.py \
    --dataset "${DATASET}" \
    --embedding-dir "${EMBEDDING_DIR}" \
    --output-dir data \
    --model-arch "${MODEL_ARCH}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --num-layers "${NUM_LAYERS}" \
    --dropout "${DROPOUT}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --batch-size "${BATCH_SIZE}" \
    --device cuda

echo "=============================================================================="
echo "Done: $(date)"
echo "Expected outputs in ${PROJECT_ROOT}/data:"
echo "  - exp3_${DATASET}_results.json"
echo "  - exp3_${DATASET}_calibration.png"
echo "  - exp3_${DATASET}_roc.png"
echo "=============================================================================="
