#!/bin/bash
# Experiment 2 (RPCholesky + low-rank HMC), 5 seeds x 2 datasets.
#
# exp2 is pure NumPy/SciPy -- NO GPU is requested or used. Each array task runs
# one seed on both datasets (~25 min per dataset on the recorded configs).
#
# Submit from the project root on BigPurple:
#   sbatch scripts/paper_rerun_exp2.sbatch
#
# Overridable via --export, e.g.:
#   sbatch --export=ALL,CONDA_ENV=/gpfs/scratch/sd6701/conda-envs/gpc \
#          scripts/paper_rerun_exp2.sbatch
#
# NOTE: verify the partition name against `sinfo -s` before submitting --
# the value below is a BigPurple guess, not something this script can confirm.
#SBATCH --job-name=paper_rerun_exp2
#SBATCH --partition=cpu_medium
#SBATCH --array=1-5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/paper_rerun/exp2_%A_%a.out
#SBATCH --error=slurm_logs/paper_rerun/exp2_%A_%a.err

set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/scratch/sd6701/gpc-mcmc-cholesky-factorization}"
CONDA_ENV="${CONDA_ENV:-/gpfs/scratch/sd6701/conda-envs/gpc}"
CONDA_SH="${CONDA_SH:-/gpfs/scratch/sd6701/miniconda3/etc/profile.d/conda.sh}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/experiments/ML-Final-Project/Datasets}"
# Every batch writes into its own data/paper_rerun/<RUN_ID>/ so a new run can
# never mix with or overwrite an earlier one. Pass RUN_ID explicitly so the
# exp1 and exp2 arrays share one directory -- the fallback below is per-array
# and would split them into two.
if [ -z "${RUN_ID:-}" ]; then
  RUN_ID="batch-${SLURM_ARRAY_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
  echo "WARNING: RUN_ID not supplied; defaulting to ${RUN_ID}." >&2
  echo "         exp1 and exp2 will land in DIFFERENT directories." >&2
fi
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/data/paper_rerun/${RUN_ID}/runs}"
SEED="${SLURM_ARRAY_TASK_ID:-1}"

# Per-test-point predictions are always saved now (needed for reliability
# diagrams). Set MAX_TRAIN to cap the training split, e.g. MAX_TRAIN=3000 to
# reproduce the old subsampled pcam baseline.
EXTRA_ARGS=""
if [ -n "${MAX_TRAIN:-}" ]; then
  EXTRA_ARGS="--max-train ${MAX_TRAIN}"
fi

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs/paper_rerun "${RUN_DIR}"

# shellcheck disable=SC1090
source "${CONDA_SH}"
set +u
conda activate "${CONDA_ENV}"
set -u

PYTHON="${CONDA_PREFIX}/bin/python"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=============================================================="
echo "exp2 paper rerun | seed=${SEED} | job=${SLURM_JOB_ID:-local}"
echo "node=$(hostname) | start=$(date)"
echo "run id : ${RUN_ID}"
echo "project=${PROJECT_ROOT}"
echo "run id =${RUN_ID}"
echo "data=${DATA_ROOT}"
echo "=============================================================="

"${PYTHON}" -c "import numpy, scipy, emcee; print('imports OK')"

for DATASET in pcam camelyon17; do
  echo ""
  echo "--- exp2 ${DATASET} seed ${SEED} ---"
  "${PYTHON}" experiments/paper_rerun/run_exp2.py \
    --dataset "${DATASET}" \
    --data-root "${DATA_ROOT}" \
    --seed "${SEED}" \
    --run-dir "${RUN_DIR}" ${EXTRA_ARGS}
done

echo ""
echo "Done: $(date)"
