#!/bin/bash
#SBATCH --job-name=exp2_only_hmc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/exp2_rose/%x_%j.out
#SBATCH --error=slurm_logs/exp2_rose/%x_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

# Submit with, for example:
#   sbatch --account=csci_ga_2565-2026sp --partition=<partition> \
#          --export=ALL,NETID=sd6701 scripts/exp2_only_hmc_pcam.sbatch
#
# Optional overrides:
#   NETID, PROJECT_ROOT, CONDA_ENV
#   FACTOR_DIR              datasets/pcam-hg/train/embeddings/exp2_rpchol_full
#   EMBEDDINGS              optional; default is FACTOR_DIR/embeddings.npy
#   TEST_EMBEDDINGS         datasets/pcam-hg/test/embeddings/projected_512.npy
#   TEST_LABELS             datasets/pcam-hg/test/embeddings/y_embeddings.npy
#   K                       200
#   BANDWIDTH               optional; default uses FACTOR_DIR/summary.json
#   N_SAMPLES               2000
#   N_WARMUP                500
#   N_LEAPFROG              12
#   PREDICTION_BATCH_SIZE   512
#   HMC_STEP_CONSTANT       0.05
#   HMC_TARGET_ACCEPT       0.8
#   SUBSAMPLE_TEST          optional
#   EXP2_OUTPUT_DIR         auto -> data/<factor_dir_basename>
#   SEED                    42

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

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${PROJECT_ROOT}" "$1" ;;
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
FACTOR_DIR="${FACTOR_DIR:-datasets/pcam-hg/train/embeddings/exp2_rpchol_full}"
EMBEDDINGS="${EMBEDDINGS:-}"
TEST_EMBEDDINGS="${TEST_EMBEDDINGS:-datasets/pcam-hg/test/embeddings/projected_512.npy}"
TEST_LABELS="${TEST_LABELS:-datasets/pcam-hg/test/embeddings/y_embeddings.npy}"
K="${K:-200}"
BANDWIDTH="${BANDWIDTH:-}"
N_SAMPLES="${N_SAMPLES:-2000}"
N_WARMUP="${N_WARMUP:-500}"
N_LEAPFROG="${N_LEAPFROG:-12}"
PREDICTION_BATCH_SIZE="${PREDICTION_BATCH_SIZE:-512}"
HMC_STEP_CONSTANT="${HMC_STEP_CONSTANT:-0.05}"
HMC_TARGET_ACCEPT="${HMC_TARGET_ACCEPT:-0.8}"
SUBSAMPLE_TEST="${SUBSAMPLE_TEST:-}"
SEED="${SEED:-42}"

FACTOR_DIR_ABS="$(resolve_path "${FACTOR_DIR}")"
TEST_EMBEDDINGS_ABS="$(resolve_path "${TEST_EMBEDDINGS}")"
TEST_LABELS_ABS="$(resolve_path "${TEST_LABELS}")"

if [ -z "${EXP2_OUTPUT_DIR:-}" ]; then
  FACTOR_TAG="$(basename "${FACTOR_DIR}")"
  EXP2_OUTPUT_DIR="data/${FACTOR_TAG}"
fi
EXP2_OUTPUT_DIR_ABS="$(resolve_path "${EXP2_OUTPUT_DIR}")"

if [ -n "${EMBEDDINGS}" ]; then
  EMBEDDINGS_ABS="$(resolve_path "${EMBEDDINGS}")"
else
  EMBEDDINGS_ABS=""
fi

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

mkdir -p "${PROJECT_ROOT}/slurm_logs"
mkdir -p "${PROJECT_ROOT}/slurm_logs/exp2_rose"
mkdir -p "${EXP2_OUTPUT_DIR_ABS}"

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

if [ ! -f "experiments/exp2_rose.py" ]; then
  echo "ERROR: experiments/exp2_rose.py not found under ${PROJECT_ROOT}" >&2
  exit 1
fi

if [ ! -d "${FACTOR_DIR_ABS}" ]; then
  echo "ERROR: Factor directory not found: ${FACTOR_DIR_ABS}" >&2
  exit 1
fi
if [ ! -f "${FACTOR_DIR_ABS}/factor_k${K}.npy" ]; then
  echo "ERROR: Missing factor file: ${FACTOR_DIR_ABS}/factor_k${K}.npy" >&2
  exit 1
fi
if [ ! -f "${FACTOR_DIR_ABS}/pivots_k${K}.npy" ]; then
  echo "ERROR: Missing pivots file: ${FACTOR_DIR_ABS}/pivots_k${K}.npy" >&2
  exit 1
fi
if [ ! -f "${FACTOR_DIR_ABS}/kernel_submatrix_k${K}.npy" ]; then
  echo "ERROR: Missing kernel submatrix file: ${FACTOR_DIR_ABS}/kernel_submatrix_k${K}.npy" >&2
  exit 1
fi
if [ ! -f "${FACTOR_DIR_ABS}/labels.npy" ]; then
  echo "ERROR: Missing labels file: ${FACTOR_DIR_ABS}/labels.npy" >&2
  exit 1
fi
if [ -z "${EMBEDDINGS_ABS}" ] && [ ! -f "${FACTOR_DIR_ABS}/embeddings.npy" ]; then
  echo "ERROR: No EMBEDDINGS override provided and ${FACTOR_DIR_ABS}/embeddings.npy is missing." >&2
  exit 1
fi
if [ ! -f "${TEST_EMBEDDINGS_ABS}" ]; then
  echo "ERROR: Missing test embeddings file: ${TEST_EMBEDDINGS_ABS}" >&2
  exit 1
fi
if [ ! -f "${TEST_LABELS_ABS}" ]; then
  echo "ERROR: Missing test labels file: ${TEST_LABELS_ABS}" >&2
  exit 1
fi

echo "=============================================================================="
echo "Experiment 2: low-rank HMC GP prediction from prebuilt RPCholesky factors"
echo "=============================================================================="
echo "Job ID                 : ${SLURM_JOB_ID:-N/A}"
echo "Node                   : $(hostname)"
echo "Start time             : $(date)"
echo "Project root           : ${PROJECT_ROOT}"
echo "NETID                  : ${NETID}"
echo "Conda env              : ${CONDA_ENV}"
echo "Factor dir             : ${FACTOR_DIR}"
echo "Test embeddings        : ${TEST_EMBEDDINGS}"
echo "Test labels            : ${TEST_LABELS}"
if [ -n "${EMBEDDINGS}" ]; then
  echo "Train embeddings       : ${EMBEDDINGS}"
else
  echo "Train embeddings       : ${FACTOR_DIR}/embeddings.npy"
fi
echo "k                      : ${K}"
if [ -n "${BANDWIDTH}" ]; then
  echo "Bandwidth override     : ${BANDWIDTH}"
else
  echo "Bandwidth override     : <from ${FACTOR_DIR}/summary.json>"
fi
echo "HMC samples            : ${N_SAMPLES}"
echo "HMC warmup             : ${N_WARMUP}"
echo "HMC leapfrog steps     : ${N_LEAPFROG}"
echo "Prediction batch size  : ${PREDICTION_BATCH_SIZE}"
echo "HMC step constant      : ${HMC_STEP_CONSTANT}"
echo "HMC target accept      : ${HMC_TARGET_ACCEPT}"
if [ -n "${SUBSAMPLE_TEST}" ]; then
  echo "Test subsample         : ${SUBSAMPLE_TEST}"
else
  echo "Test subsample         : <full test split>"
fi
echo "Output dir             : ${EXP2_OUTPUT_DIR}"
echo "=============================================================================="

CMD=(
  "${PYTHON}" "experiments/exp2_rose.py"
  "--factor-dir" "${FACTOR_DIR}"
  "--test-embeddings" "${TEST_EMBEDDINGS}"
  "--test-labels" "${TEST_LABELS}"
  "--k" "${K}"
  "--n-samples" "${N_SAMPLES}"
  "--n-warmup" "${N_WARMUP}"
  "--n-leapfrog" "${N_LEAPFROG}"
  "--prediction-batch-size" "${PREDICTION_BATCH_SIZE}"
  "--hmc-step-constant" "${HMC_STEP_CONSTANT}"
  "--hmc-target-accept" "${HMC_TARGET_ACCEPT}"
  "--output-dir" "${EXP2_OUTPUT_DIR}"
  "--seed" "${SEED}"
)

if [ -n "${EMBEDDINGS}" ]; then
  CMD+=("--embeddings" "${EMBEDDINGS}")
fi
if [ -n "${BANDWIDTH}" ]; then
  CMD+=("--bandwidth" "${BANDWIDTH}")
fi
if [ -n "${SUBSAMPLE_TEST}" ]; then
  CMD+=("--subsample-test" "${SUBSAMPLE_TEST}")
fi

"${CMD[@]}"

echo "=============================================================================="
echo "Done: $(date)"
echo "Artifacts:"
echo "  - ${EXP2_OUTPUT_DIR_ABS}/exp2_k${K}_results.npz"
echo "=============================================================================="
