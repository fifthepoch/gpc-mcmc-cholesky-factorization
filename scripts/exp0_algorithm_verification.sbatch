#!/bin/bash
#SBATCH --job-name=exp0_rpcholesky
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

# =============================================================================
# User-configurable paths
# =============================================================================
# Submit with:
#   sbatch --export=ALL,NETID=ab1234 scripts/exp0_algorithm_verification.sbatch
#
# Optionally override CONDA_ENV or PROJECT_ROOT:
#   sbatch --export=ALL,NETID=ab1234,CONDA_ENV=/scratch/ab1234/my-env \
#          scripts/exp0_algorithm_verification.sbatch
#
# If you haven't created a conda env yet, run on the login node first:
#   NETID=ab1234 bash scripts/setup_env.sh

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
# Conda activation (cluster-specific, CRITICAL sequence)
#
# Steps that MUST happen in this exact order:
#   1. module purge           - clear any stale module state
#   2. module load anaconda3  - load the cluster's base anaconda
#   3. source conda.sh        - make `conda activate` available in bash
#   4. conda activate <env>   - switch into user env
#   5. unset PYTHONHOME/PATH  - the module load leaks these, breaking imports
# =============================================================================
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

if [ ! -d "${CONDA_ENV}" ]; then
  echo "ERROR: Conda env not found at ${CONDA_ENV}" >&2
  echo "Create it first on the login node:" >&2
  echo "  NETID=${NETID} bash scripts/setup_env.sh" >&2
  exit 1
fi

conda activate "${CONDA_ENV}"
echo "Activated conda env: ${CONDA_ENV}"

# CRITICAL: module load anaconda3 sets PYTHONHOME and PYTHONPATH to the
# system anaconda, which poisons the user env's sys.path and causes
# "No module named ..." errors. We MUST unset them after conda activate.
unset PYTHONHOME
unset PYTHONPATH

PYTHON="${CONDA_ENV}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  echo "ERROR: Python not found at ${PYTHON}" >&2
  exit 1
fi
echo "Using: ${PYTHON} ($(${PYTHON} --version 2>&1))"

"${PYTHON}" -c "import numpy, scipy, sklearn, matplotlib; print('Env import check: OK')" || {
  echo "ERROR: import check failed -- conda activation is likely broken." >&2
  exit 1
}

export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

# =============================================================================
# Run Experiment 0
# =============================================================================
cd "${PROJECT_ROOT}"
mkdir -p slurm_logs
mkdir -p data

echo "=============================================================================="
echo "Experiment 0: RPCholesky algorithm verification"
echo "=============================================================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "Start time   : $(date)"
echo "Project root : ${PROJECT_ROOT}"
echo "Conda env    : ${CONDA_ENV}"
echo "=============================================================================="

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

"${PYTHON}" experiments/exp0_algorithm_verification.py

echo "=============================================================================="
echo "Done: $(date)"
echo "Expected outputs in ${PROJECT_ROOT}/data:"
echo "  - exp0_results.mat"
echo "  - exp0_error_vs_k.png"
echo "  - exp0_time_vs_k.png"
echo "  - exp0_time_vs_N.png"
echo "=============================================================================="
