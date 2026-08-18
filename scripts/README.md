# Scripts — Cluster Job Submission Guide

This directory contains environment setup scripts, dataset downloaders, and
SLURM sbatch templates for submitting experiments on the cluster.

## Prerequisites

All commands below are run **on the cluster login node** unless otherwise noted.
Replace `ab1234` with your own NetID throughout.

### 1. Set up scratch environment variables

```bash
NETID=ab1234 source scripts/set_scratch_env.sh
```

This creates the directory layout under `/scratch/<NETID>` and exports
`CONDA_ENVS_DIRS`, `PIP_CACHE_DIR`, `HF_HOME`, `TMPDIR`, etc., so that
nothing is written to `/home` (which has a small quota on this cluster).

### 2. Create the conda environment

```bash
NETID=ab1234 bash scripts/setup_env.sh
```

This creates a conda env at `/scratch/<NETID>/conda-envs/gpc` with Python 3.12
and installs all project dependencies from `requirements.txt` (numpy, scipy,
scikit-learn, torch, torchvision, h5py, tabpfn-client, etc.).

To also install WILDS dataset download dependencies:

```bash
NETID=ab1234 bash scripts/setup_env.sh --with-datasets
```

### 3. Download datasets (optional, for Experiments 3–4)

```bash
python scripts/download_datasets.py --datasets pcam --root datasets
python scripts/download_datasets.py --datasets camelyon17 --root datasets
```

---

## Submitting Experiments

### Experiment 1: GPyTorch SVGP Binary GP Classification

Runs the sparse variational GP classifier on frozen embeddings. By default the
job prefers the HG-style embedding layout, so on PCam it will use:

- `datasets/pcam-hg/train/embeddings/projected_512.npy`
- `datasets/pcam-hg/valid/embeddings/projected_512.npy`
- `datasets/pcam-hg/test/embeddings/projected_512.npy`

```bash
sbatch --export=ALL,NETID=ab1234,DATASET=pcam scripts/exp1_gpytorch_svgp_gpc.sbatch
```

| Variable               | Default         | Description                                      |
| ---------------------- | --------------- | ------------------------------------------------ |
| `NETID`                | auto-detected   | Your cluster NetID                               |
| `DATASET`              | `pcam`          | `pcam`, `camelyon17`, or `embed`                 |
| `EMBEDDING_DIR`        | auto-detected   | Defaults to `datasets/<dataset>-hg` if present   |
| `CONDA_ENV`            | auto-detected   | Override conda env path                          |
| `PROJECT_ROOT`         | auto-detected   | Override project directory                       |
| `NUM_INDUCING`         | `256`           | Number of inducing points                        |
| `BATCH_SIZE`           | `2048`          | Training batch size                              |
| `PREDICT_BATCH_SIZE`   | `4096`          | Validation/test prediction batch size            |
| `EPOCHS`               | `15`            | Maximum number of epochs                         |
| `PATIENCE`             | `4`             | Early stopping patience on validation AUROC      |
| `LEARNING_RATE`        | `0.01`          | Adam learning rate                               |
| `MAX_TRAIN_SAMPLES`    | `0`             | Subsample train set if nonzero                   |
| `MAX_VAL_SAMPLES`      | `0`             | Subsample validation set if nonzero              |
| `MAX_TEST_SAMPLES`     | `0`             | Subsample test set if nonzero                    |
| `DISABLE_STANDARDIZE`  | `0`             | Set to `1` to disable train-stat standardization |

**Examples:**

```bash
# Full PCam run on the existing 512-d projected embeddings
sbatch --export=ALL,NETID=ab1234,DATASET=pcam scripts/exp1_gpytorch_svgp_gpc.sbatch

# Faster smoke test with a capped train set and fewer inducing points
sbatch --export=ALL,NETID=ab1234,DATASET=pcam,MAX_TRAIN_SAMPLES=50000,MAX_VAL_SAMPLES=10000,MAX_TEST_SAMPLES=10000,NUM_INDUCING=128 \
       scripts/exp1_gpytorch_svgp_gpc.sbatch
```

**Outputs:** `data/exp1_gpytorch_svgp_<dataset>_results.json`,
`data/exp1_gpytorch_svgp_<dataset>_posterior.npz`,
`data/exp1_gpytorch_svgp_<dataset>_calibration.png`,
`data/exp1_gpytorch_svgp_<dataset>_roc.png`

---

### Experiment 0: RPCholesky Algorithm Verification

CPU-only job. No GPU required.

```bash
sbatch --export=ALL,NETID=ab1234 scripts/exp0_algorithm_verification.sbatch
```

**Outputs:** `data/exp0_results.mat`, `data/exp0_*.png`

---

### Experiment 3: Deterministic Neural Network Baseline

Two-step GPU job: (1) extract frozen DenseNet-121 or DINOv2 embeddings, then
(2) train a neural network classifier head on the embeddings. The default head
is now `residual_mlp`, a stronger residual MLP with LayerNorm/GELU/dropout. Set
`MODEL_ARCH=mlp` to reproduce the old 2-layer MLP baseline, or
`MODEL_ARCH=linear` for the linear-probe protocol commonly used with frozen
pathology foundation-model embeddings.

```bash
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=pcam \
       scripts/exp3_nn_baseline.sbatch
```

| Variable       | Default        | Description                                      |
| -------------- | -------------- | ------------------------------------------------ |
| `NETID`        | *(required)*   | Your cluster NetID                               |
| `DATASET`      | `pcam`         | `pcam`, `camelyon17`, or `embed`                 |
| `ENCODER`      | `densenet121`  | `densenet121` or `dinov2_vitl14`                 |
| `SKIP_EMBED`   | `0`            | Set to `1` to skip embedding extraction          |
| `EMBEDDING_DIR`| `data/embeddings` | Embedding root (project format or partner HG layout) |
| `MODEL_ARCH`   | `residual_mlp` | `residual_mlp`, `mlp`, or `linear`               |
| `HIDDEN_DIM`   | `512`          | Classifier hidden width                          |
| `NUM_LAYERS`   | `3`            | Residual blocks for `residual_mlp`               |
| `DROPOUT`      | `0.3`          | Classifier dropout                               |
| `LR`           | `1e-3`         | AdamW learning rate                              |
| `WEIGHT_DECAY` | `1e-4`         | AdamW weight decay                               |
| `EPOCHS`       | `50`           | Max classifier epochs                            |
| `PATIENCE`     | `5`            | Early-stopping patience on validation AUROC      |
| `BATCH_SIZE`   | `512`          | Embedding batch size                             |
| `CONDA_ENV`    | auto-detected  | Override conda env path                          |
| `PROJECT_ROOT` | auto-detected  | Override project directory                       |

**Examples:**

```bash
# Run on PCam with default encoder
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=pcam \
       scripts/exp3_nn_baseline.sbatch

# Run on CAMELYON17 with DINOv2 encoder
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=camelyon17,ENCODER=dinov2_vitl14 \
       scripts/exp3_nn_baseline.sbatch

# Reuse existing embeddings (skip extraction step)
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=pcam,SKIP_EMBED=1 \
       scripts/exp3_nn_baseline.sbatch

# Reproduce the old 2-layer MLP baseline
sbatch --account=torch_pr_xxx_yyy \
       --export=ALL,NETID=ab1234,DATASET=pcam,SKIP_EMBED=1,MODEL_ARCH=mlp,HIDDEN_DIM=256 \
       scripts/exp3_nn_baseline.sbatch

# Run a linear probe on frozen embeddings
sbatch --account=torch_pr_xxx_yyy \
       --export=ALL,NETID=ab1234,DATASET=pcam,SKIP_EMBED=1,MODEL_ARCH=linear \
       scripts/exp3_nn_baseline.sbatch

# Use partner embeddings (HG layout) for Exp3; skip extraction
sbatch --account=torch_pr_xxx_yyy \
       --export=ALL,NETID=ab1234,DATASET=pcam,SKIP_EMBED=1,EMBEDDING_DIR=/scratch/sd6701/gpc-mcmc-cholesky-factorization/datasets \
       scripts/exp3_nn_baseline.sbatch
```

**Outputs:** `data/exp3_<dataset>_results.json`, `data/exp3_<dataset>_calibration.png`,
`data/exp3_<dataset>_roc.png`

---

### Experiment 4: TabPFN Tabular Model Baseline

Runs TabPFN through the Prior Labs `tabpfn-client` API on the same frozen
embeddings produced by Experiment 3. **You must run Experiment 3 first** (or at
least the embedding extraction step) so that `data/embeddings/` is populated.

```bash
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=pcam \
       scripts/exp4_tabpfn_baseline.sbatch
```

| Variable             | Default        | Description                                       |
| -------------------- | -------------- | ------------------------------------------------- |
| `NETID`              | *(required)*   | Your cluster NetID                                |
| `DATASET`            | `pcam`         | `pcam`, `camelyon17`, or `embed`                  |
| `MAX_TRAIN_SAMPLES`  | `3500`         | Subsample training set to this size for TabPFN    |
| `EMBEDDING_DIR`      | `data/embeddings` | Embedding root (project format or partner HG layout) |
| `TABPFN_TOKEN`       | unset          | Prior Labs API token for `tabpfn-client` auth     |
| `TABPFN_TOKEN_FILE`  | unset          | Path to file containing token; safer than inline token |
| `CONDA_ENV`          | auto-detected  | Override conda env path                           |
| `PROJECT_ROOT`       | auto-detected  | Override project directory                        |

**Examples:**

```bash
# Run on PCam (default 50K train subsample)
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=pcam \
       scripts/exp4_tabpfn_baseline.sbatch

# Run on CAMELYON17 with larger train subsample
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=camelyon17,MAX_TRAIN_SAMPLES=100000 \
       scripts/exp4_tabpfn_baseline.sbatch

# Use partner embeddings (HG layout) for Exp4
sbatch --account=torch_pr_xxx_yyy \
       --export=ALL,NETID=ab1234,DATASET=camelyon17,EMBEDDING_DIR=/scratch/sd6701/gpc-mcmc-cholesky-factorization/datasets \
       scripts/exp4_tabpfn_baseline.sbatch

# Headless TabPFN auth using a token file
printf '%s\n' '<your-prior-labs-token>' > /scratch/ab1234/tabpfn_token.txt
chmod 600 /scratch/ab1234/tabpfn_token.txt
sbatch --account=torch_pr_xxx_yyy \
       --export=ALL,NETID=ab1234,DATASET=pcam,TABPFN_TOKEN_FILE=/scratch/ab1234/tabpfn_token.txt \
       scripts/exp4_tabpfn_baseline.sbatch

# Inline token option (works, but less safe because it may end up in shell history)
sbatch --account=torch_pr_xxx_yyy \
       --export=ALL,NETID=ab1234,DATASET=pcam,TABPFN_TOKEN='<your-token>' \
       scripts/exp4_tabpfn_baseline.sbatch
```

**Outputs:** `data/exp4_<dataset>_results.json`, `data/exp4_<dataset>_calibration.png`,
`data/exp4_<dataset>_roc.png`

---

## Recommended submission order

Run all three datasets through the full pipeline:

```bash
# Step 1: Extract embeddings + train NN for each dataset
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=pcam       scripts/exp3_nn_baseline.sbatch
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=camelyon17  scripts/exp3_nn_baseline.sbatch
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=embed       scripts/exp3_nn_baseline.sbatch

# Step 2: After Exp3 jobs complete, run TabPFN on the same embeddings
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=pcam        scripts/exp4_tabpfn_baseline.sbatch
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=camelyon17  scripts/exp4_tabpfn_baseline.sbatch
sbatch --account=torch_pr_xxx_yyy --export=ALL,NETID=ab1234,DATASET=embed       scripts/exp4_tabpfn_baseline.sbatch
```

Use `squeue -u $USER` to monitor job status and check `slurm_logs/` for output.

---

## Embedding formats supported by Exp3/Exp4

Both `exp3_nn_baseline.py` and `exp4_tabpfn_baseline.py` can load either format:

1) **Project-native format** (generated by `extract_embeddings.py`)

```text
<EMBEDDING_DIR>/pcam_train_embeddings.npy
<EMBEDDING_DIR>/pcam_train_labels.npy
<EMBEDDING_DIR>/pcam_val_embeddings.npy
<EMBEDDING_DIR>/pcam_val_labels.npy
<EMBEDDING_DIR>/pcam_test_embeddings.npy
<EMBEDDING_DIR>/pcam_test_labels.npy
```

2) **Partner HG layout**

```text
<EMBEDDING_DIR>/pcam-hg/train/embeddings/projected_512.npy
<EMBEDDING_DIR>/pcam-hg/train/embeddings/y_embeddings.npy
<EMBEDDING_DIR>/pcam-hg/valid/embeddings/projected_512.npy
<EMBEDDING_DIR>/pcam-hg/test/embeddings/projected_512.npy
```

Notes:
- For HG layout, labels are loaded from `y_embeddings.npy` if present, otherwise
  from `<split>/labels.csv`.
- `val` in code maps to `valid` in the HG directory naming.
- For Exp3 with partner embeddings, set `SKIP_EMBED=1` so the extraction stage is skipped.

---

## EMBED embeddings (768-d Phikon + 512-d PCA)

`scripts/create_embed_embeddings.py` converts the EMBED mammography images into
frozen-encoder embeddings. It recursively discovers image files under
`--data-root` — `.hdf5/.h5` (one image per file, dataset key auto-detected or
set via `--hdf5-key`), `.png/.jpg/.jpeg`, and `.dcm` (windowed via `pydicom`) —
freezes their order into `manifest.csv`, extracts 768-d Phikon (`owkin/phikon`)
features, then fits an IncrementalPCA projection to produce 512-d embeddings.

On BigPurple the EMBED copy stores each FFDM view as an HDF5 file named
`{patient}_{date}_{side}_{view}_1.hdf5` under `hdf5/ffdm_screening` and
`hdf5/ffdm_diagnostic`, so point the scan at the `hdf5/` subtree:

```bash
# Direct run (GPU node):
python scripts/create_embed_embeddings.py \
    --data-root /gpfs/scratch/wh2757/EMBED/hdf5 \
    --output-dir /gpfs/scratch/$USER/EMBED_embeddings \
    --project-dim 512

# Or submit as a SLURM job (from the project root):
sbatch embed_embeddings.sbatch
# Override paths / dataset key as needed:
sbatch --export=ALL,DATA_ROOT=/gpfs/scratch/wh2757/EMBED/hdf5,HDF5_KEY=image embed_embeddings.sbatch
```

Note: the `#SBATCH` account/partition lines in `embed_embeddings.sbatch` are
for the course cluster; on BigPurple change them to a GPU partition you have
access to (e.g. `--partition=gpu4_medium`) and drop the `--account` line.

Outputs (in `--output-dir`):

- `manifest.csv` — row i maps to image path i (alignment source of truth)
- `embeddings.npy` — `(N, 768)` Phikon features
- `projected_512.npy` — `(N, 512)` PCA-projected features
- `pca_512.npz` — fitted PCA components/mean (reusable)

The job checkpoints every 500 rows (`progress.json`), so if it hits the
walltime limit, resubmit the same command and it resumes where it stopped.
Use `--limit 100` for a quick smoke test first.

---

## Monitoring and troubleshooting

```bash
# Check job queue
squeue -u $USER

# View job output in real time
tail -f slurm_logs/exp3_nn_baseline_<JOBID>.out

# Check GPU utilization on a running node
srun --jobid=<JOBID> nvidia-smi
```

**Common issues:**

| Symptom                                  | Cause                                      | Fix                                                  |
| ---------------------------------------- | ------------------------------------------ | ---------------------------------------------------- |
| `No module named ...`                    | `PYTHONHOME`/`PYTHONPATH` not unset        | Already handled in sbatch templates                  |
| `Conda env not found`                    | Env not created yet                        | Run `NETID=... bash scripts/setup_env.sh` first      |
| `CUDA out of memory`                     | Batch size too large for GPU               | Reduce `--batch-size` in the Python script           |
| `Permission denied` on `/home`           | Cache writing to home dir                  | Verify `set_scratch_env.sh` is sourced               |
| `TabPFN import failed`                   | `tabpfn-client` not installed              | Run `pip install tabpfn-client` in the conda env     |

---

## File inventory

| File                                  | Purpose                                              |
| ------------------------------------- | ---------------------------------------------------- |
| `set_scratch_env.sh`                  | Redirects all caches to `/scratch/<NETID>`           |
| `setup_env.sh`                        | Creates conda env and installs dependencies          |
| `download_datasets.py`               | Downloads PCam, CAMELYON17-WILDS, EMBED              |
| `create_embed_embeddings.py`         | EMBED → Phikon 768-d + PCA 512-d embeddings          |
| `../embed_embeddings.sbatch`         | SLURM template for EMBED embedding extraction (1 GPU)|
| `exp0_algorithm_verification.sbatch`  | SLURM template for RPCholesky benchmark (CPU)        |
| `exp3_nn_baseline.sbatch`             | SLURM template for NN baseline (1 GPU)               |
| `exp4_tabpfn_baseline.sbatch`         | SLURM template for TabPFN baseline (1 GPU)           |
