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

    # one id for the whole batch, shared by both arrays
    RUN_ID="run-$(date -u +%Y%m%dT%H%M%SZ)"

    sbatch --export=ALL,RUN_ID=$RUN_ID scripts/paper_rerun_exp2.sbatch  # CPU
    sbatch --export=ALL,RUN_ID=$RUN_ID scripts/paper_rerun_exp1.sbatch  # GPU

    # once both arrays finish
    python scripts/paper_rerun_aggregate.py --run-id $RUN_ID --expect-runs 5
    python scripts/paper_rerun_reliability.py --run-id $RUN_ID --seed 1

Change the seed count by editing `--array=1-5` in both sbatch files.

Partition names (`cpu_medium`, `gpu4_medium`) are guesses -- check `sinfo -s`
and adjust. Everything else is overridable:

    sbatch --export=ALL,PROJECT_ROOT=...,CONDA_ENV=...,DATA_ROOT=... <script>

Single run, no SLURM:

    python experiments/paper_rerun/run_exp2.py \
      --dataset pcam --data-root <.../Datasets> --seed 1

## Output

Every submission writes into its own batch directory, so a new run can never
overwrite an earlier one:

    data/paper_rerun/<RUN_ID>/
      runs/<dataset>/<experiment>/seed###.json      one row per run
      runs/<dataset>/exp{1,2}_seed<N>_preds.npz     per-test-point predictions
      experiment_results_<dataset>_rerun.csv        <- the deliverable
      reliability_<dataset>.png                     calibration curves
      reliability_summary.csv                       per-bin numbers

Pass RUN_ID on the sbatch command line so the exp1 and exp2 arrays share one
directory. Without it each array falls back to its own SLURM array job id and
they land separately.

The post-processing scripts read the most recently modified batch by default;
use `--run-id <name>` to pick another, or `--run-dir <path>` for an explicit
location. A pre-existing flat `data/paper_rerun/runs` is still handled.

Each `_rerun.csv` holds 14 rows per dataset: the 5 individual seeds, then a
MEAN row, then a STD row -- for exp1, then again for exp2.

The `_preds.npz` files carry `predictive_prob` and `y_test`; exp2 adds
`prob_std`, `latent_mean` and `latent_std`. Turn them into reliability
diagrams with:

    python scripts/paper_rerun_reliability.py --seed 1

## Defaults reproduce the recorded rows

exp2: `k=200`, `b=10`, `n_samples=1000`, `n_warmup=1000`, `n_leapfrog=25`,
`target_accept=0.8`, initial step `0.05 / k**0.25 = 0.013296` (matches the
recorded `step_size`), bandwidth via `approx_median`.

exp1: `num_inducing=512`, `batch_size=1024`, `50` epochs, `lr=0.01`,
`lengthscale_init=64.0`, `outputscale_init=1.0`. Both datasets train on their
FULL splits so SVGP sees the same data as the exp2 run it is compared against.
The recorded pcam row used a stratified 3000/500 subsample; reproduce it with
`--max-train 3000 --max-valid 500` (or `MAX_TRAIN=3000` for the sbatch).

## Timing fields

Both experiments populate `fit_or_train_time_sec` / `train_time_sec` so the two
are directly comparable: for SVGP that is the epoch loop, for HMC it is
warmup + sampling. `total_pipeline_time_sec` covers everything including data
loading and evaluation.

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
