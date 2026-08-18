#!/bin/bash
#SBATCH --job-name=embed_embeddings
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

# Extract Phikon 768-d embeddings for the EMBED dataset and project them to
# 512-d with PCA. Safe to resubmit: the script resumes from its checkpoint.
#
# Example:
#   sbatch embed_embeddings.sbatch
#   sbatch --export=ALL,DATA_ROOT=/gpfs/scratch/wh2757/EMBED,PROJECT_DIM=512 embed_embeddings.sbatch

SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SBATCH_DIR}}}"
NETID="${NETID:-}"

if [ -z "${NETID}" ]; then
  case "${PROJECT_ROOT}" in
    /scratch/*/*)
      NETID="${PROJECT_ROOT#/scratch/}"
      NETID="${NETID%%/*}"
      ;;
    /gpfs/scratch/*/*)
      NETID="${PROJECT_ROOT#/gpfs/scratch/}"
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

# The EMBED images live as one HDF5 file per view under hdf5/ffdm_screening
# and hdf5/ffdm_diagnostic, so point the scan at the hdf5 subtree only.
DATA_ROOT="${DATA_ROOT:-/gpfs/scratch/wh2757/EMBED/hdf5}"

# The dataset directory belongs to another user, so default the outputs to a
# location the submitting user can write to.
if [ -z "${OUTPUT_DIR:-}" ]; then
  if [ -w "${DATA_ROOT}" ]; then
    OUTPUT_DIR="${DATA_ROOT}/embeddings"
  elif [ -d "/gpfs/scratch/${USER}" ]; then
    OUTPUT_DIR="/gpfs/scratch/${USER}/EMBED_embeddings"
  else
    OUTPUT_DIR="${PROJECT_ROOT}/data/embed_embeddings"
  fi
fi
PROJECT_DIM="${PROJECT_DIM:-512}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MODEL_NAME="${MODEL_NAME:-owkin/phikon}"
DTYPE="${DTYPE:-float32}"
DEVICE="${DEVICE:-cuda}"
LIMIT="${LIMIT:-}"
HDF5_KEY="${HDF5_KEY:-}"

SCRATCH_BASE="/scratch/${NETID}"
CONDA_ENV="${CONDA_ENV:-${SCRATCH_BASE}/conda-envs/gpc}"

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

PYTHON="${CONDA_PREFIX}/bin/python"

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs

echo "=============================================================================="
echo "EMBED Embedding Extraction Job"
echo "=============================================================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Data root    : ${DATA_ROOT}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "Project dim  : ${PROJECT_DIM}"
echo "Batch size   : ${BATCH_SIZE}"
echo "Model        : ${MODEL_NAME}"
echo "=============================================================================="

EXTRA_ARGS=()
if [ -n "${LIMIT}" ]; then
  EXTRA_ARGS+=(--limit "${LIMIT}")
fi
if [ -n "${HDF5_KEY}" ]; then
  EXTRA_ARGS+=(--hdf5-key "${HDF5_KEY}")
fi

"${PYTHON}" scripts/create_embed_embeddings.py \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-name "${MODEL_NAME}" \
    --project-dim "${PROJECT_DIM}" \
    --batch-size "${BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo "=============================================================================="
echo "Done: $(date)"
echo "Outputs in ${OUTPUT_DIR}:"
echo "  - manifest.csv"
echo "  - embeddings.npy          (N x 768)"
echo "  - projected_${PROJECT_DIM}.npy       (N x ${PROJECT_DIM})"
echo "=============================================================================="
