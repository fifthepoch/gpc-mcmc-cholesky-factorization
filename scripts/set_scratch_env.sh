#!/bin/bash
# Environment setup script to redirect all installations and caches to /scratch.
# This prevents hitting /home directory quotas on the cluster.
#
# Usage:
#   source scripts/set_scratch_env.sh              (uses NETID env var)
#   NETID=ab1234 source scripts/set_scratch_env.sh (inline)
#
# This file is sourced by setup_env.sh and the sbatch templates automatically.

set -eo pipefail

# ============================================================================
# NETID configuration
# ============================================================================
NETID="${NETID:-${USER:-<netid>}}"
if [ "${NETID}" = "<netid>" ] || [ -z "${NETID}" ]; then
  echo "ERROR: NETID is not set." >&2
  echo "  Option A: export NETID=ab1234 before sourcing this script" >&2
  echo "  Option B: NETID=ab1234 source scripts/set_scratch_env.sh" >&2
  return 1 2>/dev/null || exit 1
fi

# Course cluster uses /scratch/<netid>; BigPurple uses /gpfs/scratch/<id>.
if [ -z "${SCRATCH_BASE:-}" ]; then
  if [ -d "/scratch/${NETID}" ] || { [ -d /scratch ] && [ ! -d "/gpfs/scratch/${NETID}" ]; }; then
    SCRATCH_BASE="/scratch/${NETID}"
  elif [ -d "/gpfs/scratch/${NETID}" ]; then
    SCRATCH_BASE="/gpfs/scratch/${NETID}"
  else
    SCRATCH_BASE="/scratch/${NETID}"
  fi
fi
export SCRATCH_BASE

# ============================================================================
# Create directory structure under /scratch
# ============================================================================
mkdir -p "${SCRATCH_BASE}"/{conda-envs,conda-pkgs,py-cache/{pip,python},hf-cache/{datasets,hub,transformers},tmp}

# ============================================================================
# Conda directories
# ============================================================================
export CONDA_ENVS_DIRS="${SCRATCH_BASE}/conda-envs"
export CONDA_PKGS_DIRS="${SCRATCH_BASE}/conda-pkgs"

# ============================================================================
# General cache directories
# ============================================================================
export XDG_CACHE_HOME="${SCRATCH_BASE}/py-cache"

# Pip cache
export PIP_CACHE_DIR="${SCRATCH_BASE}/py-cache/pip"
export PIP_NO_CACHE_DIR=1

# Python bytecode cache
export PYTHONPYCACHEPREFIX="${SCRATCH_BASE}/py-cache/python"

# Hugging Face caches (used by dataset download scripts)
export HF_HOME="${SCRATCH_BASE}/hf-cache"
export HF_DATASETS_CACHE="${SCRATCH_BASE}/hf-cache/datasets"
export HF_HUB_CACHE="${SCRATCH_BASE}/hf-cache/hub"
export TRANSFORMERS_CACHE="${SCRATCH_BASE}/hf-cache/transformers"

# Temporary directories
export TMPDIR="${SCRATCH_BASE}/tmp"
export TMP="${SCRATCH_BASE}/tmp"
export TEMP="${SCRATCH_BASE}/tmp"

# ============================================================================
# Summary
# ============================================================================
echo "=================================="
echo "Scratch Environment Variables Set"
echo "=================================="
echo "NETID:           ${NETID}"
echo "SCRATCH_BASE:    ${SCRATCH_BASE}"
echo ""
echo "CONDA_ENVS_DIRS: ${CONDA_ENVS_DIRS}"
echo "CONDA_PKGS_DIRS: ${CONDA_PKGS_DIRS}"
echo "PIP_CACHE_DIR:   ${PIP_CACHE_DIR}"
echo "HF_HOME:         ${HF_HOME}"
echo "TMPDIR:          ${TMPDIR}"
echo "=================================="
echo ""
echo "All installations and caches will be redirected to /scratch/${NETID}"
