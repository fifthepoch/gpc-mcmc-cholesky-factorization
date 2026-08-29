#!/bin/bash
#SBATCH --job-name=exp2_embed_full_k200_lf25
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/exp2_embed_full_k200_lf25/%x_%j.out
#SBATCH --error=slurm_logs/exp2_embed_full_k200_lf25/%x_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

# Experiment 2 (RPCholesky + HMC GPC) on the EMBED embeddings.
#
# Assumes scripts/create_embed_splits.py has already written
#   datasets/embed/{train,valid,test}/embeddings/{projected_512.npy,y_embeddings.npy}
#
# Submit with:
#   sbatch --export=ALL,NETID=sd6701 scripts/exp2_embed_full_k200_lf25.sbatch
#
# Optional overrides:
#   PROJECT_ROOT, CONDA_ENV, DATASET_ROOT, EMBEDDING_FILE,
#   K, LF, N_WARMUP, N_SAMPLES, BANDWIDTH, SUBSAMPLE_TEST, SEED

NETID="${NETID:-${USER:-}}"
if [ -z "${NETID}" ]; then
  echo "ERROR: NETID is required. Submit with --export=ALL,NETID=<your-netid>." >&2
  exit 1
fi

SBATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "${SBATCH_DIR}/.." && pwd)}}"

SCRATCH_BASE="/scratch/${NETID}"
if [ ! -d "${SCRATCH_BASE}" ] && [ -d "/gpfs/scratch/${NETID}" ]; then
  SCRATCH_BASE="/gpfs/scratch/${NETID}"
fi
CONDA_ENV="${CONDA_ENV:-${SCRATCH_BASE}/conda-envs/gpc}"

DATASET_ROOT="${DATASET_ROOT:-datasets/embed}"
EMBEDDING_FILE="${EMBEDDING_FILE:-projected_512.npy}"
K="${K:-200}"
LF="${LF:-25}"
N_WARMUP="${N_WARMUP:-1000}"
N_SAMPLES="${N_SAMPLES:-1000}"
BANDWIDTH="${BANDWIDTH:-approx_median}"
SUBSAMPLE_TEST="${SUBSAMPLE_TEST:-}"
SEED="${SEED:-42}"

if [ -f "${PROJECT_ROOT}/scripts/set_scratch_env.sh" ]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/scripts/set_scratch_env.sh"
fi

load_conda() {
  module purge >/dev/null 2>&1 || true
  local candidates=(
    "${SCRATCH_BASE}/miniconda3/etc/profile.d/conda.sh"
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
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID:-$$}"
mkdir -p "${MPLCONFIGDIR}"

PYTHON="${CONDA_PREFIX}/bin/python"

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs/exp2_embed_full_k200_lf25

TRAIN_EMB="${DATASET_ROOT}/train/embeddings/${EMBEDDING_FILE}"
TRAIN_LABELS="${DATASET_ROOT}/train/embeddings/y_embeddings.npy"
TEST_EMB="${DATASET_ROOT}/test/embeddings/${EMBEDDING_FILE}"
TEST_LABELS="${DATASET_ROOT}/test/embeddings/y_embeddings.npy"

for required in "${TRAIN_EMB}" "${TRAIN_LABELS}" "${TEST_EMB}" "${TEST_LABELS}"; do
  if [ ! -f "${required}" ]; then
    echo "ERROR: missing ${required}" >&2
    echo "Run scripts/create_embed_splits.py first." >&2
    exit 1
  fi
done

FACTOR_DIR="${DATASET_ROOT}/train/embeddings/exp2_embed_full_k=${K}"
OUT_DIR="data/exp2_embed_sweep/k${K}_lf${LF}"
mkdir -p "${OUT_DIR}"

echo "=============================================================================="
echo "Experiment 2: RPCholesky + HMC GPC on EMBED"
echo "=============================================================================="
echo "Job ID       : ${SLURM_JOB_ID:-N/A}"
echo "Node         : $(hostname)"
echo "Start time   : $(date)"
echo "Project root : ${PROJECT_ROOT}"
echo "Dataset root : ${DATASET_ROOT}"
echo "Train emb    : ${TRAIN_EMB}"
echo "Test emb     : ${TEST_EMB}"
echo "Factor dir   : ${FACTOR_DIR}"
echo "Output dir   : ${OUT_DIR}"
echo "k / leapfrog : ${K} / ${LF}"
echo "warmup/sample: ${N_WARMUP} / ${N_SAMPLES}"
echo "bandwidth    : ${BANDWIDTH}"
echo "=============================================================================="

echo ""
echo "[Step 1] Accelerated RPCholesky factor at k=${K}"
"${PYTHON}" experiments/exp_rpcholesky_embeddings.py \
    --embeddings "${TRAIN_EMB}" \
    --labels "${TRAIN_LABELS}" \
    --k-values "${K}" \
    --kernel gaussian \
    --bandwidth "${BANDWIDTH}" \
    --block-size 10 \
    --stoptol 1e-13 \
    --output-dir "${FACTOR_DIR}" \
    --seed "${SEED}"

echo ""
echo "[Step 2] HMC posterior sampling + pivot-inducing-point predictive"
CMD=(
  "${PYTHON}" experiments/exp2.py
  --factor-dir "${FACTOR_DIR}"
  --test-embeddings "${TEST_EMB}"
  --test-labels "${TEST_LABELS}"
  --k "${K}"
  --n-samples "${N_SAMPLES}"
  --n-warmup "${N_WARMUP}"
  --n-leapfrog "${LF}"
  --output-dir "${OUT_DIR}"
  --seed "${SEED}"
)
if [ -n "${SUBSAMPLE_TEST}" ]; then
  CMD+=(--subsample-test "${SUBSAMPLE_TEST}")
fi
"${CMD[@]}"

echo "=============================================================================="
echo "Done: $(date)"
echo "Artifacts in ${PROJECT_ROOT}/${OUT_DIR}"
echo "=============================================================================="
