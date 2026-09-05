# Paper-format rerun harness

Reruns **exp1 (SVGP)** and **exp2 (RPCholesky + low-rank HMC)** on **pcam** and
**camelyon17** across several seeds, then averages them into two CSVs that use
the exact 146-column schema of `experiment_results_pcam.csv` /
`experiment_results_camelyon17.csv`, so results diff directly against the
recorded rows.

No existing experiment script was modified. `run_exp2.py` imports `run_hmc` and
`sample_predictive_probabilities_pivots` from `experiments/exp2.py` and reuses
them unchanged.

## Before you run: the code is not on BigPurple yet

As of writing, `/gpfs/scratch/sd6701/gpc-mcmc-cholesky-factorization` contains
only `experiments/ML-Final-Project`. These scripts additionally need:

    src/my_cholesky/         # arpcholesky, KernelMatrix, kernels
    predictive_metrics2.py   # metric helpers
    experiments/exp2.py      # HMC + predictive sampler, imported as-is
    experiments/paper_rerun/ # this directory
    scripts/paper_rerun_*    # sbatch + aggregator

Sync those across (`git clone`/`git pull` on the cluster, or `rsync`) or the
jobs fail at import.

## Running

    # 5 seeds x 2 datasets, one array task per seed
    sbatch scripts/paper_rerun_exp2.sbatch    # CPU only, ~50 min/task
    sbatch scripts/paper_rerun_exp1.sbatch    # needs a GPU, ~10 min/task

    # once both arrays finish
    python scripts/paper_rerun_aggregate.py --expect-runs 5

Change the seed count by editing `--array=1-5` in both sbatch files.

Partition names (`cpu_medium`, `gpu4_medium`) are guesses -- check `sinfo -s`
and adjust. Everything else is overridable:

    sbatch --export=ALL,PROJECT_ROOT=...,CONDA_ENV=...,DATA_ROOT=... <script>

Single run, no SLURM:

    python experiments/paper_rerun/run_exp2.py \
      --dataset pcam --data-root <.../Datasets> --seed 1

## Output

    data/paper_rerun/
      runs/<dataset>/<experiment>/seed###.json      one row per run
      experiment_results_<dataset>_rerun.csv        <- the deliverable (means)
      experiment_results_<dataset>_rerun_std.csv    sample std of each field
      experiment_results_<dataset>_allruns.csv      every seed, for traceability

Each `_rerun.csv` holds two rows, exp1 and exp2, in the original column order.

## Defaults reproduce the recorded rows

exp2: `k=200`, `b=10`, `n_samples=1000`, `n_warmup=1000`, `n_leapfrog=25`,
`target_accept=0.8`, initial step `0.05 / k**0.25 = 0.013296` (matches the
recorded `step_size`), bandwidth via `approx_median`.

exp1: `num_inducing=512`, `batch_size=1024`, `50` epochs, `lr=0.01`,
`lengthscale_init=64.0`, `outputscale_init=1.0`. Split sizes follow the
recorded rows -- pcam used a stratified 3000/500 subsample, camelyon17 used
everything. Override with `--max-train` / `--max-valid`.

## Two deviations from the recorded rows

1. **`feature_dim` is 512, not 768.** The recorded camelyon17 rows used the
   raw 768-d embeddings; only `projected_512.npy` (a PCA projection of them) is
   present on disk. camelyon17 numbers will therefore not match exactly.
2. **The RPCholesky factor is rebuilt every seed.** Pivot selection is
   randomized, so reusing one fixed factor would understate the method's
   run-to-run spread. This is why `relative_trace_error` and
   `kernel_bandwidth` vary across seeds.

## Reading the results

Check `_rerun_std.csv` alongside the means. Also watch `tau_logp` on exp2: the
recorded camelyon17 run had `tau_logp=144.82` against `n_samples=1000`, i.e. an
effective sample size near 7, with the step size collapsing from 0.013296 to
0.000141 during adaptation. If that recurs, the posterior behind those metrics
rests on very few independent draws regardless of how good the accuracy looks.
