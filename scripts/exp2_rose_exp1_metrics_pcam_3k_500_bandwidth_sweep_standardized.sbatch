#!/bin/bash
#SBATCH --job-name=exp2_bw_sweep_std
#SBATCH --account=csci_ga_2565-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/hmc_rose/%x_%j.out
#SBATCH --error=slurm_logs/hmc_rose/%x_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

# Submit with:
#   sbatch --export=ALL,NETID=<your-netid> scripts/exp2_rose_exp1_metrics_pcam_3k_500_bandwidth_sweep_standardized.sbatch
#
# Optional overrides:
#   DATASET_NAME, DATASET_ROOT
#   BANDWIDTH_VALUES="5 10 15 20 30 40 50"
#   K_VALUE, N_SAMPLES, N_WARMUP, N_LEAPFROG
#   TRAIN_SUBSAMPLE, TEST_SUBSAMPLE, SEED
#   STANDARDIZE_FACTOR, STANDARDIZE_FROM_FACTOR
#   SWEEP_OUTPUT_ROOT, SWEEP_FACTOR_ROOT

DATASET_NAME="${DATASET_NAME:-pcam}"
BANDWIDTH_VALUES="${BANDWIDTH_VALUES:-5 10 15 20 30 40 50}"
K_VALUE="${K_VALUE:-200}"
N_SAMPLES="${N_SAMPLES:-2000}"
N_WARMUP="${N_WARMUP:-200}"
N_LEAPFROG="${N_LEAPFROG:-12}"
TRAIN_SUBSAMPLE="${TRAIN_SUBSAMPLE:-3000}"
TEST_SUBSAMPLE="${TEST_SUBSAMPLE:-500}"
SEED="${SEED:-42}"
STANDARDIZE_FACTOR="${STANDARDIZE_FACTOR:-1}"
STANDARDIZE_FROM_FACTOR="${STANDARDIZE_FROM_FACTOR:-1}"

SBATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "${SBATCH_DIR}/.." && pwd)}}"

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
DATASET_ROOT="${DATASET_ROOT:-datasets/pcam-hg}"
TRAIN_EMBEDDINGS="${TRAIN_EMBEDDINGS:-${DATASET_ROOT}/train/embeddings/projected_512.npy}"
TRAIN_LABELS="${TRAIN_LABELS:-${DATASET_ROOT}/train/embeddings/y_embeddings.npy}"
TEST_EMBEDDINGS="${TEST_EMBEDDINGS:-${DATASET_ROOT}/test/embeddings/projected_512.npy}"
TEST_LABELS="${TEST_LABELS:-${DATASET_ROOT}/test/embeddings/y_embeddings.npy}"
BLOCK_SIZE="${BLOCK_SIZE:-10}"
STOPTOL="${STOPTOL:-1e-13}"
PREDICTION_BATCH_SIZE="${PREDICTION_BATCH_SIZE:-512}"
standardize_tag="nostd"
if [ "${STANDARDIZE_FACTOR}" = "1" ]; then
  standardize_tag="standardized"
fi
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-data/exp2_bw_sweep_${DATASET_NAME}_${TRAIN_SUBSAMPLE}_${TEST_SUBSAMPLE}_${standardize_tag}}"
SWEEP_FACTOR_ROOT="${SWEEP_FACTOR_ROOT:-${DATASET_ROOT}/train/embeddings/exp2_bw_sweep_${TRAIN_SUBSAMPLE}_${standardize_tag}}"

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

PYTHON="${CONDA_PREFIX}/bin/python"

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs/hmc_rose "${MPLCONFIGDIR}"

"${PYTHON}" -c "import numpy, scipy, emcee; print('Env import check: OK')" || {
  echo "ERROR: Import check failed in ${CONDA_ENV}." >&2
  exit 1
}

for path in \
  "${TRAIN_EMBEDDINGS}" \
  "${TRAIN_LABELS}" \
  "${TEST_EMBEDDINGS}" \
  "${TEST_LABELS}" \
  "experiments/exp_rpcholesky_embeddings.py" \
  "experiments/exp2_rose_exp1_metrics.py"; do
  if [ ! -f "${path}" ]; then
    echo "ERROR: Missing required file: ${PROJECT_ROOT}/${path}" >&2
    exit 1
  fi
done

echo "=============================================================================="
echo "Running bandwidth sweep for exp2_rose_exp1_metrics"
echo "Dataset         : ${DATASET_NAME}"
echo "Dataset root    : ${DATASET_ROOT}"
echo "Bandwidths      : ${BANDWIDTH_VALUES}"
echo "Train/test      : ${TRAIN_SUBSAMPLE}/${TEST_SUBSAMPLE}"
echo "k               : ${K_VALUE}"
echo "samples/warmup  : ${N_SAMPLES}/${N_WARMUP}"
echo "leapfrog        : ${N_LEAPFROG}"
echo "seed            : ${SEED}"
echo "Standardize fac : ${STANDARDIZE_FACTOR}"
echo "Standardize test: ${STANDARDIZE_FROM_FACTOR}"
echo "Output root     : ${SWEEP_OUTPUT_ROOT}"
echo "Factor root     : ${SWEEP_FACTOR_ROOT}"
echo "=============================================================================="

for bw in ${BANDWIDTH_VALUES}; do
  bw_tag="$(printf '%s' "${bw}" | tr '.' 'p')"
  factor_dir="${SWEEP_FACTOR_ROOT}/bw${bw_tag}"
  output_dir="${SWEEP_OUTPUT_ROOT}/bw${bw_tag}"

  echo
  echo "Running bandwidth=${bw}"
  echo "  factor_dir=${factor_dir}"
  echo "  output_dir=${output_dir}"

  mkdir -p "${factor_dir}" "${output_dir}"

  FACTOR_CMD=(
    "${PYTHON}" "experiments/exp_rpcholesky_embeddings.py"
    --embeddings "${TRAIN_EMBEDDINGS}" \
    --labels "${TRAIN_LABELS}" \
    --subsample "${TRAIN_SUBSAMPLE}" \
    --stratified-subsample \
    --k-values "${K_VALUE}" \
    --output-dir "${factor_dir}" \
    --bandwidth "${bw}" \
    --block-size "${BLOCK_SIZE}" \
    --stoptol "${STOPTOL}" \
    --seed "${SEED}"
  )

  if [ "${STANDARDIZE_FACTOR}" = "1" ]; then
    FACTOR_CMD+=("--standardize")
  fi

  "${FACTOR_CMD[@]}"

  METRICS_CMD=(
    "${PYTHON}" "experiments/exp2_rose_exp1_metrics.py"
    --factor-dir "${factor_dir}" \
    --test-embeddings "${TEST_EMBEDDINGS}" \
    --test-labels "${TEST_LABELS}" \
    --k "${K_VALUE}" \
    --subsample-test "${TEST_SUBSAMPLE}" \
    --stratified-subsample-test \
    --n-samples "${N_SAMPLES}" \
    --n-warmup "${N_WARMUP}" \
    --n-leapfrog "${N_LEAPFROG}" \
    --prediction-batch-size "${PREDICTION_BATCH_SIZE}" \
    --seed "${SEED}" \
    --output-dir "${output_dir}"
  )

  if [ "${STANDARDIZE_FROM_FACTOR}" = "1" ]; then
    METRICS_CMD+=("--standardize-from-factor")
  fi

  "${METRICS_CMD[@]}"
done

echo
echo "Bandwidth sweep complete."
