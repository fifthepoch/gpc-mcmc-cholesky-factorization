#!/bin/bash
#SBATCH --job-name=exp2_rpchol_hmc
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/hmc_rose/%x_%j.out
#SBATCH --error=slurm_logs/hmc_rose/%x_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

# Submit with:
#   sbatch --export=ALL,NETID=ab1234 scripts/exp2_rpchol_hmc_pcam.sbatch
#
# Optional overrides:
#   CONDA_ENV, PROJECT_ROOT
#   EMBEDDINGS, LABELS, TEST_EMBEDDINGS, TEST_LABELS
#   K_VALUES, FACTOR_K, BANDWIDTH, FACTOR_DIR, EXP2_OUTPUT_DIR
#   N_SAMPLES, N_WARMUP

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

EMBEDDINGS="${EMBEDDINGS:-datasets/pcam-hg/train/embeddings/projected_512.npy}"
LABELS="${LABELS:-datasets/pcam-hg/train/embeddings/y_embeddings.npy}"
TEST_EMBEDDINGS="${TEST_EMBEDDINGS:-datasets/pcam-hg/test/embeddings/projected_512.npy}"
TEST_LABELS="${TEST_LABELS:-datasets/pcam-hg/test/embeddings/y_embeddings.npy}"
K_VALUES="${K_VALUES:-200}"
FACTOR_K="${FACTOR_K:-200}"
BANDWIDTH="${BANDWIDTH:-approx_median}"
FACTOR_DIR="${FACTOR_DIR:-data/rpchol_full}"
EXP2_OUTPUT_DIR="${EXP2_OUTPUT_DIR:-data}"
N_SAMPLES="${N_SAMPLES:-2000}"
N_WARMUP="${N_WARMUP:-500}"

SCRIPT_DIR="${PROJECT_ROOT}/scripts"
if [ -f "${SCRIPT_DIR}/set_scratch_env.sh" ]; then
  source "${SCRIPT_DIR}/set_scratch_env.sh"
else
  echo "WARN: set_scratch_env.sh not found, continuing without it." >&2
fi

source "/scratch/${NETID}/miniconda3/etc/profile.d/conda.sh"

export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
set +u
conda activate "/scratch/${NETID}/conda-envs/gpc"
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

"${PYTHON}" -c "import numpy, scipy, emcee; print('import check: OK')" || {
  echo "ERROR: Import check failed. Install required dependencies in ${CONDA_ENV}." >&2
  exit 1
}

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs
mkdir -p slurm_logs/hmc_rose
mkdir -p "${FACTOR_DIR}"
mkdir -p "${EXP2_OUTPUT_DIR}"

if [ ! -f "experiments/exp_rpcholesky_embeddings.py" ]; then
  echo "ERROR: experiments/exp_rpcholesky_embeddings.py not found under ${PROJECT_ROOT}" >&2
  exit 1
fi
if [ ! -f "experiments/exp2.py" ]; then
  echo "ERROR: experiments/exp2.py not found under ${PROJECT_ROOT}" >&2
  exit 1
fi

echo "=============================================================================="
echo "Experiment 2 pipeline: RPCholesky factors + HMC GP prediction"
echo "=============================================================================="
echo "Job ID          : ${SLURM_JOB_ID:-N/A}"
echo "Node            : $(hostname)"
echo "Start time      : $(date)"
echo "Project root    : ${PROJECT_ROOT}"
echo "Conda env       : ${CONDA_ENV}"
echo "Embeddings      : ${EMBEDDINGS}"
echo "Labels          : ${LABELS}"
echo "Test embeddings : ${TEST_EMBEDDINGS}"
echo "Test labels     : ${TEST_LABELS}"
echo "K values        : ${K_VALUES}"
echo "Factor k        : ${FACTOR_K}"
echo "Bandwidth       : ${BANDWIDTH}"
echo "Factor dir      : ${FACTOR_DIR}"
echo "Exp2 output dir : ${EXP2_OUTPUT_DIR}"
echo "HMC samples     : ${N_SAMPLES}"
echo "HMC warmup      : ${N_WARMUP}"
echo "=============================================================================="

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "[1/2] Building RPCholesky factors..."
"${PYTHON}" experiments/exp_rpcholesky_embeddings.py \
  --embeddings "${EMBEDDINGS}" \
  --labels "${LABELS}" \
  --k-values ${K_VALUES} \
  --output-dir "${FACTOR_DIR}" \
  --bandwidth "${BANDWIDTH}"

echo "[2/2] Running low-rank HMC predictive experiment..."
"${PYTHON}" experiments/exp2.py \
  --factor-dir "${FACTOR_DIR}" \
  --test-embeddings "${TEST_EMBEDDINGS}" \
  --test-labels "${TEST_LABELS}" \
  --k "${FACTOR_K}" \
  --n-samples "${N_SAMPLES}" \
  --n-warmup "${N_WARMUP}" \
  --output-dir "${EXP2_OUTPUT_DIR}"

echo "=============================================================================="
echo "Done: $(date)"
echo "Artifacts:"
echo "  - ${PROJECT_ROOT}/${FACTOR_DIR}/factor_k*.npy"
echo "  - ${PROJECT_ROOT}/${FACTOR_DIR}/summary.json"
echo "  - ${PROJECT_ROOT}/${EXP2_OUTPUT_DIR}/exp2_* (metrics/samples depending on args)"
echo "=============================================================================="
