#!/bin/bash
#SBATCH --job-name=cam17_hf_export
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

# Required
NETID="${NETID:-<netid>}"
if [ "${NETID}" = "<netid>" ]; then
  echo "ERROR: NETID is required. Submit with --export=ALL,NETID=<your-netid>" >&2
  exit 1
fi

# Optional overrides
SCRATCH_BASE="/scratch/${NETID}"
if [ ! -d "${SCRATCH_BASE}" ] && [ -d "/gpfs/scratch/${NETID}" ]; then
  SCRATCH_BASE="/gpfs/scratch/${NETID}"
fi
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/gpc-mcmc-cholesky-factorization}"
CONDA_ENV="${CONDA_ENV:-${SCRATCH_BASE}/conda-envs/longcat}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/datasets/camelyon17-hg}"

# Scratch env setup (if present)
SCRIPT_DIR="${PROJECT_ROOT}/scripts"
if [ -f "${SCRIPT_DIR}/set_scratch_env.sh" ]; then
  source "${SCRIPT_DIR}/set_scratch_env.sh"
fi

# Conda activation (cluster-safe pattern)
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
set +u
conda activate "${CONDA_ENV}"
set -u

if [ -z "${CONDA_PREFIX:-}" ] || [ ! -d "${CONDA_PREFIX}" ]; then
  echo "ERROR: Failed to activate conda env: ${CONDA_ENV}" >&2
  exit 1
fi

export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset PYTHONHOME
unset PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${CONDA_PREFIX}/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "ERROR: Python not found at ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs

echo "Project root: ${PROJECT_ROOT}"
echo "Conda env:    ${CONDA_PREFIX}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Python:       ${PYTHON_BIN}"

# Run existing export script
bash scripts/create_camelyon17_hg.sh "${OUTPUT_DIR}"
