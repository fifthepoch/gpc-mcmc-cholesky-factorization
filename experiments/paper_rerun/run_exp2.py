"""
Paper-format rerun of Experiment 2: RPCholesky factor + low-rank HMC GP
classification, evaluated on a held-out test split.

One invocation == one seed. Emits a single JSON row in the 146-column paper
schema; scripts/paper_rerun_aggregate.py averages the seeds afterwards.

The RPCholesky factor is rebuilt on every seed on purpose: pivot selection is
randomized, so holding one factor fixed across runs would understate the
method's true run-to-run spread.

Pure NumPy/SciPy -- no GPU is used or needed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
for path in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import schema  # noqa: E402
from my_cholesky.arpcholesky import arpcholesky  # noqa: E402
from my_cholesky.matrix import KernelMatrix  # noqa: E402
from predictive_metrics2 import (  # noqa: E402
    evaluate_binary_probabilistic_predictions,
    summarize_predictive_distribution,
)

# Reused verbatim from the original experiment -- not reimplemented.
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
from exp2 import (  # noqa: E402
    compute_tau_emcee,
    run_hmc,
    sample_predictive_probabilities_pivots,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["pcam", "camelyon17"])
    p.add_argument(
        "--data-root",
        required=True,
        help="Directory holding <dataset>/{train,test}/projected_512.npy",
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run-dir", default="data/paper_rerun/runs")

    # Defaults below reproduce the recorded rows exactly.
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--arpcholesky-b", type=int, default=10)
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--n-warmup", type=int, default=1000)
    p.add_argument("--n-leapfrog", type=int, default=25)
    p.add_argument("--hmc-step-constant", type=float, default=0.05)
    p.add_argument("--target-accept", type=float, default=0.8)
    p.add_argument("--prediction-batch-size", type=int, default=512)
    p.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Subsample the training split (default: use all rows).",
    )
    return p.parse_args()


def load_split(data_root: str, dataset: str, split: str):
    base = Path(data_root) / dataset / split
    X = np.load(base / "projected_512.npy").astype(np.float32, copy=False)
    y = np.load(base / "y_embeddings.npy").astype(np.float32, copy=False).squeeze()
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"{split}: {X.shape[0]} embeddings vs {y.shape[0]} labels -- misaligned"
        )
    return X, y


def main() -> None:
    args = parse_args()
    wall_t0 = time.perf_counter()

    # KernelMatrix's "approx_median" bandwidth draws from the legacy global RNG,
    # so seed that too or the bandwidth is not reproducible for a given seed.
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    t0 = time.perf_counter()
    X_train, y_train = load_split(args.data_root, args.dataset, "train")
    X_test, y_test = load_split(args.data_root, args.dataset, "test")
    if args.max_train is not None and args.max_train < X_train.shape[0]:
        idx = np.sort(rng.choice(X_train.shape[0], size=args.max_train, replace=False))
        X_train, y_train = X_train[idx], y_train[idx]
    data_loading_time = time.perf_counter() - t0

    n_train, feature_dim = X_train.shape
    n_test = X_test.shape[0]
    print(f"[exp2/{args.dataset}/seed{args.seed}] train={X_train.shape} test={X_test.shape}")

    # --- kernel + RPCholesky factor -------------------------------------
    t0 = time.perf_counter()
    A = KernelMatrix(X_train, kernel="gaussian", bandwidth="approx_median")
    kernel_time = time.perf_counter() - t0
    bandwidth = float(A.bandwidth)

    t0 = time.perf_counter()
    lra = arpcholesky(A, k=args.k, b=args.arpcholesky_b, seed=args.seed)
    cholesky_time = time.perf_counter() - t0
    factor_time = kernel_time + cholesky_time

    F = np.asarray(lra.get_left_factor()).astype(np.float64, copy=False)  # (N, k)
    pivot_indices = np.asarray(lra.get_indices(), dtype=np.int64)
    actual_rank = int(F.shape[1])

    trace_K = float(A.trace())
    trace_FFt = float(np.sum(F * F))
    rel_trace_error = (trace_K - trace_FFt) / trace_K if trace_K > 0 else float("nan")
    kernel_queries = int(A.num_queries())

    K_pivots = np.asarray(A[pivot_indices, pivot_indices], dtype=np.float64)
    K_pivots = 0.5 * (K_pivots + K_pivots.T)  # symmetrize, as the original does
    F_pivots = F[pivot_indices, :]
    pivot_consistency = float(np.max(np.abs(F_pivots @ F_pivots.T - K_pivots)))
    print(f"  factor k={actual_rank} in {cholesky_time:.2f}s  "
          f"rel_trace_err={rel_trace_error:.4e}  pivot_max={pivot_consistency:.2e}")

    # --- HMC ------------------------------------------------------------
    initial_step = args.hmc_step_constant / (args.k ** 0.25)
    hmc = run_hmc(
        factor=F,
        y=y_train,
        n_samples=args.n_samples,
        n_warmup=args.n_warmup,
        seed=args.seed,
        initial_step_size=initial_step,
        n_leapfrog=args.n_leapfrog,
        target_accept=args.target_accept,
        adapt_step_size=True,
    )
    nu_samples = hmc["nu_samples"]
    tau_logp = float(compute_tau_emcee(hmc["logp_trace"]))
    tau_nu_mean = float(compute_tau_emcee(np.mean(nu_samples, axis=1)))
    print(f"  HMC accept={hmc['accept_rate']:.3f}  step={hmc['step_size']:.6f}  "
          f"tau_logp={tau_logp:.2f}")

    # --- predictive distribution ----------------------------------------
    t0 = time.perf_counter()
    pred = sample_predictive_probabilities_pivots(
        F=F,
        X_train=X_train,
        X_test=X_test,
        pivot_indices=pivot_indices,
        K_pp=K_pivots,
        nu_samples=nu_samples,
        bandwidth=bandwidth,
        batch_size=args.prediction_batch_size,
        seed=args.seed + 999,
    )
    inference_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    summary = summarize_predictive_distribution(
        pred["p_samples"], pred["latent_samples"]
    )
    metrics = evaluate_binary_probabilistic_predictions(
        y_true=y_test,
        p_pred=summary["prob_mean"],
        p_samples=pred["p_samples"],
        latent_samples=pred["latent_samples"],
    )
    evaluation_time = time.perf_counter() - t0
    total_wall = time.perf_counter() - wall_t0

    ess_logp = args.n_samples / tau_logp if tau_logp > 0 else float("nan")
    total_mcmc = hmc["sampling_time"]

    # --- assemble the 146-column row ------------------------------------
    row = schema.blank_row()
    row.update(schema.slurm_context())
    row.update({
        "record_id": f"rerun-{args.dataset}-exp2-k{args.k}-lf{args.n_leapfrog}-seed{args.seed}",
        "experiment": "exp2",
        "dataset": args.dataset,
        "method_name": "RPChol+HMC",
        "sampler": "HMC",
        "script_path": "experiments/paper_rerun/run_exp2.py",
        "run_timestamp_utc": schema.utc_now(),
        "code_ref": schema.git_commit(),
        "notes": (
            f"Paper-format rerun, seed={args.seed}. Factor rebuilt this run "
            f"(pivots are randomized). Pivot |F_P F_P^T - K_PP| max="
            f"{pivot_consistency:.2e}. Embeddings: projected_512 (PCA of the "
            f"768-d source), so feature_dim={feature_dim} not 768."
        ),
        "dataset_sources": f"['{args.dataset}']",
        "embedding_source": f"{args.dataset}-projected512",
        "embedding_variant": f"paper_rerun_k={args.k}",
        "feature_dim": feature_dim,
        "n_train": n_train,
        "n_test": n_test,
        "max_train_samples": "full" if args.max_train is None else args.max_train,
        "max_test_samples": "full",
        "seed": args.seed,
        "standardize": 0,
        "kernel": "rbf",
        "kernel_bandwidth": bandwidth,
        "k": args.k,
        "arpcholesky_b": args.arpcholesky_b,
        "actual_rank": actual_rank,
        "relative_trace_error": rel_trace_error,
        "kernel_entries_queried": kernel_queries,
        "factor_MB": F.nbytes / 1e6,
        "trace_K": trace_K,
        "trace_FF_T": trace_FFt,
        "pivot_consistency_max": pivot_consistency,
        "n_samples": args.n_samples,
        "n_warmup": args.n_warmup,
        "n_leapfrog": args.n_leapfrog,
        "step_size": initial_step,
        "initial_step": initial_step,
        "final_step_size": hmc["step_size"],
        "target_accept": args.target_accept,
        "adapt": "True",
        "accept_rate": hmc["accept_rate"],
        "tau_logp": tau_logp,
        "tau_nu_mean": tau_nu_mean,
        "ess_logp": ess_logp,
        "ess_per_sec": ess_logp / total_mcmc if total_mcmc > 0 else float("nan"),
        "prob_min": float(np.min(summary["prob_mean"])),
        "prob_mean": float(np.mean(summary["prob_mean"])),
        "prob_max": float(np.max(summary["prob_mean"])),
        "latent_mean_min": float(np.min(summary["latent_mean"])),
        "latent_mean_mean": float(np.mean(summary["latent_mean"])),
        "latent_mean_max": float(np.max(summary["latent_mean"])),
        "latent_var_min": float(np.min(summary["latent_variance"])),
        "latent_var_mean": float(np.mean(summary["latent_variance"])),
        "latent_var_max": float(np.max(summary["latent_variance"])),
        # timing
        "data_loading_time_sec": data_loading_time,
        "kernel_time_sec": kernel_time,
        "cholesky_time_sec": cholesky_time,
        "factor_time_sec": factor_time,
        "warmup_time_sec": hmc["warmup_time"],
        "sampling_time_sec": hmc["sampling_time"],
        "total_mcmc_time_sec": total_mcmc,
        "predictive_sampling_time_sec": inference_time,
        "inference_time_sec": inference_time,
        "prediction_time_sec": inference_time,
        "evaluation_time_sec": evaluation_time,
        "total_pipeline_time_sec": total_wall,
        "total_runtime_sec": total_wall,
        "timing_scope": (
            "factor, warmup, sampling, predictive_sampling "
            "(post-warmup MCMC time = sampling_time_sec)"
        ),
    })

    # Metric helper uses its own key spellings; map onto the CSV schema.
    row.update({
        "log_likelihood_mean": metrics["elpd_mean"],
        "negative_log_likelihood_mean": metrics["negative_log_likelihood_mean"],
        "negative_log_loss": metrics["negative_log_likelihood_mean"],
        "mean_predictive_log_likelihood": metrics["elpd_mean"],
        "elpd": metrics["elpd"],
        "elpd_mean": metrics["elpd_mean"],
        "pell": metrics["pell"],
        "pell_mean": metrics["pell_mean"],
        "posterior_expected_log_loss": metrics["posterior_expected_log_loss"],
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "accuracy": metrics["accuracy"],
        "brier": metrics["brier"],
        "ece": metrics["ece"],
        "number_errors": metrics["number_errors"],
        "tp": metrics["TP"],
        "fp": metrics["FP"],
        "tn": metrics["TN"],
        "fn": metrics["FN"],
        "sensitivity_tpr": metrics["sensitivity_TPR"],
        "specificity_tnr": metrics["specificity_TNR"],
        "true_positive_rate": metrics["sensitivity_TPR"],
        "true_negative_rate": metrics["specificity_TNR"],
        "false_positive_rate": metrics["FPR"],
        "false_negative_rate": metrics["FNR"],
        "target_positive_rate": float(np.mean(y_test)),
    })

    path = schema.write_run(row, args.run_dir, args.dataset, "exp2", args.seed)
    print(f"  accuracy={metrics['accuracy']:.6f}  auroc={metrics['auroc']:.6f}  "
          f"total={total_wall:.2f}s")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
