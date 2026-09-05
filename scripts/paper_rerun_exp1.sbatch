#!/bin/bash
# Experiment 1 (GPyTorch SVGP), 5 seeds x 2 datasets.
#
# This one DOES use the GPU. Each array task runs one seed on both datasets
# (pcam is a 3000-point subsample and is quick; camelyon17 trains on all
# 302,436 rows -- ~6 min on a GPU, far longer on CPU).
#
# Submit from the project root on BigPurple:
#   sbatch scripts/paper_rerun_exp1.sbatch
#
# NOTE: verify the partition name against `sinfo -s` before submitting --
# the value below is a BigPurple guess, not something this script can confirm.
#SBATCH --job-name=paper_rerun_exp1
#SBATCH --partition=gpu4_medium
#SBATCH --array=1-5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/paper_rerun/exp1_%A_%a.out
#SBATCH --error=slurm_logs/paper_rerun/exp1_%A_%a.err

set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/scratch/sd6701/gpc-mcmc-cholesky-factorization}"
CONDA_ENV="${CONDA_ENV:-/gpfs/scratch/sd6701/conda-envs/gpc}"
CONDA_SH="${CONDA_SH:-/gpfs/scratch/sd6701/miniconda3/etc/profile.d/conda.sh}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/experiments/ML-Final-Project/Datasets}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/data/paper_rerun/runs}"
SEED="${SLURM_ARRAY_TASK_ID:-1}"

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs/paper_rerun "${RUN_DIR}"

# shellcheck disable=SC1090
source "${CONDA_SH}"
set +u
conda activate "${CONDA_ENV}"
set -u

PYTHON="${CONDA_PREFIX}/bin/python"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "=============================================================="
echo "exp1 paper rerun | seed=${SEED} | job=${SLURM_JOB_ID:-local}"
echo "node=$(hostname) | start=$(date)"
echo "=============================================================="

"${PYTHON}" -c "import torch, gpytorch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

for DATASET in pcam camelyon17; do
  echo ""
  echo "--- exp1 ${DATASET} seed ${SEED} ---"
  "${PYTHON}" experiments/paper_rerun/run_exp1.py \
    --dataset "${DATASET}" \
    --data-root "${DATA_ROOT}" \
    --seed "${SEED}" \
    --run-dir "${RUN_DIR}"
done

echo ""
echo "Done: $(date)"
