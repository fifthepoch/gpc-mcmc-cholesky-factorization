"""
Experiment 1: Speed comparison of Dense Cholesky vs RPCholesky in
Random Walk Metropolis (RWM) for fake GP binary classification.

We compare:
    Method A (Dense):  f = L_dense @ nu,  L_dense = chol(K)
    Method B (RPChol): f = F @ nu,        F from arpcholesky(A, k, b=10)

across k in [20, 50, 100], reporting factor time, MCMC per-step time,
total MCMC time, acceptance rate, and Frobenius approximation error.
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
from scipy.special import expit


# Allow direct script execution without package install.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from my_cholesky.arpcholesky import arpcholesky
from my_cholesky.matrix import KernelMatrix


def make_fake_blobs(seed: int = 42):
    """Generate N=2000 two-blob binary data in R^2."""
    rng = np.random.default_rng(seed)
    n_per_class = 1000
    cov = 0.5 * np.eye(2)
    x0 = rng.multivariate_normal(mean=[-1.0, 0.0], cov=cov, size=n_per_class)
    x1 = rng.multivariate_normal(mean=[1.0, 0.0], cov=cov, size=n_per_class)
    X = np.vstack([x0, x1])
    y = np.concatenate(
        [np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)]
    )
    return X, y


def log_posterior(nu: np.ndarray, factor: np.ndarray, y: np.ndarray) -> float:
    """Log posterior for Bernoulli likelihood and standard normal prior."""
    f = factor @ nu
    p = expit(f)
    log_lik = np.sum(y * np.log(p + 1e-10) + (1 - y) * np.log(1 - p + 1e-10))
    log_prior = -0.5 * np.dot(nu, nu)
    return float(log_lik + log_prior)


def run_rwm(
    factor: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    n_warmup: int,
    seed: int,
    target_accept: float = 0.30,
    adapt_interval: int = 50,
):
    """Run Random Walk Metropolis and return timing/statistics."""
    rng = np.random.default_rng(seed)
    dim = factor.shape[1]
    total_steps = n_warmup + n_samples
    step_size = 2.38 / np.sqrt(dim)

    nu = np.zeros(dim, dtype=float)
    logp = log_posterior(nu, factor, y)

    step_times = np.zeros(total_steps, dtype=float)
    accepts = np.zeros(total_steps, dtype=bool)

    for i in range(total_steps):
        t0 = time.perf_counter()
        nu_prop = nu + step_size * rng.standard_normal(dim)
        logp_prop = log_posterior(nu_prop, factor, y)

        if np.log(rng.random()) < (logp_prop - logp):
            nu = nu_prop
            logp = logp_prop
            accepts[i] = True

        # Adaptive tuning during warmup only.
        if i < n_warmup and (i + 1) % adapt_interval == 0:
            window_start = i + 1 - adapt_interval
            accept_rate_window = float(np.mean(accepts[window_start : i + 1]))
            lower = target_accept - 0.10
            upper = target_accept + 0.10
            if accept_rate_window > upper:
                step_size *= 1.1
            elif accept_rate_window < lower:
                step_size *= 0.9

        step_times[i] = time.perf_counter() - t0

    post = slice(n_warmup, total_steps)
    warmup = slice(0, n_warmup)
    warmup_time = float(np.sum(step_times[warmup]))
    per_step_time = float(np.mean(step_times[post]))
    sampling_time = float(np.sum(step_times[post]))
    accept_rate = float(np.mean(accepts[post]))

    return {
        "per_step_time": per_step_time,
        "warmup_time": warmup_time,
        "sampling_time": sampling_time,
        "total_mcmc_time": sampling_time,
        "total_sampler_time": warmup_time + sampling_time,
        "accept_rate": accept_rate,
        "final_step_size": float(step_size),
    }


def summarize_trials(values: list[float]) -> dict[str, float | int | list[float]]:
    """Return mean/std summary while preserving raw trial values."""
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "n_trials": int(arr.size),
        "trial_values": [float(x) for x in arr],
    }


def summarize_trial_metric(
    trials: list[dict[str, float]],
    metric: str,
) -> dict[str, float | int | list[float]]:
    return summarize_trials([trial[metric] for trial in trials])


def metric_text(metric: dict[str, float | int | list[float]], precision: int) -> str:
    return f"{metric['mean']:.{precision}f} ± {metric['std']:.{precision}f}"


def write_summary_csv(results: list[dict], path: str) -> None:
    """Write a compact summary that is easy to inspect in HPC logs."""
    metrics = [
        "accepted_rank",
        "factor_time",
        "per_step_time",
        "total_time",
        "total_model_compute_time",
        "accept_rate",
        "approx_error",
        "final_step_size",
    ]
    fieldnames = ["method", "k"]
    for metric in metrics:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_n_trials"])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {"method": result["method"], "k": result["k"]}
            for metric in metrics:
                summary = result[metric]
                row[f"{metric}_mean"] = summary["mean"]
                row[f"{metric}_std"] = summary["std"]
                row[f"{metric}_n_trials"] = summary["n_trials"]
            writer.writerow(row)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    # MCMC setup
    n_samples = 2000
    n_warmup = 500
    N_REPEATS = 10
    rp_k_values = [20, 50, 100]
    print(f"Running {N_REPEATS} trials per configuration for noise-robust timing.")

    # Fake data and kernel
    X, y = make_fake_blobs(seed=42)
    A = KernelMatrix(X, kernel="gaussian", bandwidth=1.0)
    K_dense = A[:, :]
    n = K_dense.shape[0]

    # Dense factorization baseline
    dense_trials = []
    for trial_idx in range(N_REPEATS):
        seed = 42 + trial_idx
        t0 = time.perf_counter()
        L_dense = np.linalg.cholesky(K_dense + 1e-6 * np.eye(n))
        dense_factor_time = time.perf_counter() - t0

        trial_stats = run_rwm(
            factor=L_dense,
            y=y,
            n_samples=n_samples,
            n_warmup=n_warmup,
            seed=seed,
        )
        dense_trials.append(
            {
                "factor_time": float(dense_factor_time),
                "per_step_time": trial_stats["per_step_time"],
                "warmup_time": trial_stats["warmup_time"],
                "sampling_time": trial_stats["sampling_time"],
                "total_time": trial_stats["total_mcmc_time"],
                "total_model_compute_time": (
                    float(dense_factor_time) + trial_stats["total_sampler_time"]
                ),
                "accept_rate": trial_stats["accept_rate"],
                "approx_error": 0.0,
                "accepted_rank": float(n),
                "final_step_size": trial_stats["final_step_size"],
            }
        )

    if not dense_trials:
        raise RuntimeError("Dense baseline did not run any trials.")

    results = [
        {
            "method": "Dense",
            "k": int(n),
            "accepted_rank": summarize_trial_metric(dense_trials, "accepted_rank"),
            "factor_time": summarize_trial_metric(dense_trials, "factor_time"),
            "per_step_time": summarize_trial_metric(dense_trials, "per_step_time"),
            "warmup_time": summarize_trial_metric(dense_trials, "warmup_time"),
            "sampling_time": summarize_trial_metric(dense_trials, "sampling_time"),
            "total_time": summarize_trial_metric(dense_trials, "total_time"),
            "total_model_compute_time": summarize_trial_metric(
                dense_trials, "total_model_compute_time"
            ),
            "accept_rate": summarize_trial_metric(dense_trials, "accept_rate"),
            "approx_error": summarize_trial_metric(dense_trials, "approx_error"),
            "final_step_size": summarize_trial_metric(dense_trials, "final_step_size"),
        }
    ]

    fro_norm_K = np.linalg.norm(K_dense, "fro")

    for k in rp_k_values:
        rp_trials = []

        for trial_idx in range(N_REPEATS):
            seed = 42 + trial_idx + k

            # RPCholesky: O(Nk^2) vs dense Cholesky O(N^3)
            t0 = time.perf_counter()
            lra = arpcholesky(A, k=k, b=10, seed=seed)
            F = lra.get_left_factor()  # shape (N, k_eff)
            rp_factor_time = time.perf_counter() - t0

            approx_err = float(
                np.linalg.norm(K_dense - (F @ F.T), "fro") / (fro_norm_K + 1e-12)
            )

            trial_stats = run_rwm(
                factor=F,
                y=y,
                n_samples=n_samples,
                n_warmup=n_warmup,
                seed=seed,
            )
            rp_trials.append(
                {
                    "factor_time": float(rp_factor_time),
                    "per_step_time": trial_stats["per_step_time"],
                    "warmup_time": trial_stats["warmup_time"],
                    "sampling_time": trial_stats["sampling_time"],
                    "total_time": trial_stats["total_mcmc_time"],
                    "total_model_compute_time": (
                        float(rp_factor_time) + trial_stats["total_sampler_time"]
                    ),
                    "accept_rate": trial_stats["accept_rate"],
                    "approx_error": approx_err,
                    "accepted_rank": float(F.shape[1]),
                    "final_step_size": trial_stats["final_step_size"],
                }
            )

        if not rp_trials:
            raise RuntimeError(f"RPChol k={k} did not run any trials.")

        rp_result = {
            "method": "RPChol",
            "k": int(k),
            "accepted_rank": summarize_trial_metric(rp_trials, "accepted_rank"),
            "factor_time": summarize_trial_metric(rp_trials, "factor_time"),
            "per_step_time": summarize_trial_metric(rp_trials, "per_step_time"),
            "warmup_time": summarize_trial_metric(rp_trials, "warmup_time"),
            "sampling_time": summarize_trial_metric(rp_trials, "sampling_time"),
            "total_time": summarize_trial_metric(rp_trials, "total_time"),
            "total_model_compute_time": summarize_trial_metric(
                rp_trials, "total_model_compute_time"
            ),
            "accept_rate": summarize_trial_metric(rp_trials, "accept_rate"),
            "approx_error": summarize_trial_metric(rp_trials, "approx_error"),
            "final_step_size": summarize_trial_metric(rp_trials, "final_step_size"),
        }
        results.append(rp_result)

    # Print clean, aligned results table.
    row_fmt = (
        "{:<10} {:>6} {:>18} {:>18} {:>22} {:>22} {:>17} {:>22}"
    )
    print(
        row_fmt.format(
            "Method",
            "k",
            "Rank",
            "Factor time(s)",
            "Per-step time(s)",
            "Total time(s)",
            "Accept rate",
            "Approx error",
        )
    )
    print("-" * 137)
    for row in results:
        print(
            row_fmt.format(
                row["method"],
                row["k"],
                metric_text(row["accepted_rank"], 1),
                metric_text(row["factor_time"], 4),
                metric_text(row["per_step_time"], 6),
                metric_text(row["total_time"], 3),
                metric_text(row["accept_rate"], 2),
                metric_text(row["approx_error"], 3),
            )
        )

    summary_csv_path = os.path.join(data_dir, "exp1_timing_summary.csv")
    write_summary_csv(results, summary_csv_path)

    np.save(
        os.path.join(data_dir, "exp1_results.npy"),
        {
            "results": results,
            "n_samples": n_samples,
            "n_warmup": n_warmup,
            "n_repeats": N_REPEATS,
            "target_accept": 0.30,
            "adapt_interval": 50,
        },
        allow_pickle=True,
    )
    print("Saved:")
    print("- data/exp1_timing_summary.csv")
    print("- data/exp1_results.npy")


if __name__ == "__main__":
    main()

