"""
Experiment 1b: compare three samplers on the RPCholesky GP classification target.

- emcee-based RWM using a Gaussian random-walk proposal
- emcee-based MALA
- a self-contained HMC implementation

This keeps the same non-centered parameterization f = F @ nu used in the
other experiments. emcee drives the RWM and MALA runs; HMC is added directly
because emcee does not provide an HMC kernel.
"""

from __future__ import annotations

import csv
import datetime
import os
import socket
import sys
import time

import emcee
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


def log_posterior_batch(coords: np.ndarray, factor: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized log posterior over shape (nwalkers, dim)."""
    f = coords @ factor.T
    p = expit(f)
    log_lik = np.sum(y[None, :] * np.log(p + 1e-10), axis=1)
    log_lik += np.sum((1 - y)[None, :] * np.log(1 - p + 1e-10), axis=1)
    log_prior = -0.5 * np.sum(coords * coords, axis=1)
    return log_lik + log_prior


def grad_log_posterior_batch(
    coords: np.ndarray, factor: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Vectorized gradient wrt nu for shape (nwalkers, dim)."""
    f = coords @ factor.T
    p = expit(f)
    return (y[None, :] - p) @ factor - coords


def grad_log_posterior(nu: np.ndarray, factor: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient wrt a single latent vector nu."""
    f = factor @ nu
    p = expit(f)
    return factor.T @ (y - p) - nu


def compute_tau_emcee(chain: np.ndarray) -> float:
    """
    Estimate integrated autocorrelation time using emcee's implementation.

    Expects an array shaped either (nsteps,) or (nsteps, nwalkers).
    """
    try:
        tau = emcee.autocorr.integrated_time(chain, quiet=True)
    except Exception as err:
        print(f"  WARNING: emcee tau estimate failed: {err}")
        return float("nan")

    tau = np.asarray(tau, dtype=float).reshape(-1)
    
    # Filter out non-finite values
    tau = tau[np.isfinite(tau)]
    
    if tau.size == 0:
        return float("nan")
    
    # Return the maximum tau (most conservative estimate)
    return float(np.max(tau))


def compute_ess_from_tau(n_steps: int, n_walkers: int, tau: float) -> float:
    """Convert tau to an approximate ensemble ESS."""
    if not np.isfinite(tau) or tau <= 0:
        return float("nan")
    return float((n_steps * n_walkers) / tau)


def make_mala_move(
    factor: np.ndarray,
    y: np.ndarray,
    step_size: float,
) -> emcee.moves.MHMove:
    """Create an emcee MH move that uses a MALA proposal."""

    def proposal_function(coords: np.ndarray, random) -> tuple[np.ndarray, np.ndarray]:
        grads = grad_log_posterior_batch(coords, factor, y)
        noise = random.randn(*coords.shape)
        drift = 0.5 * (step_size**2) * grads
        proposals = coords + drift + step_size * noise

        prop_grads = grad_log_posterior_batch(proposals, factor, y)

        forward_diff = proposals - coords - drift
        reverse_diff = coords - proposals - 0.5 * (step_size**2) * prop_grads

        # emcee expects log q(x | x') - log q(x' | x).
        forward = -0.5 * np.sum(forward_diff * forward_diff, axis=1) / (step_size**2)
        reverse = -0.5 * np.sum(reverse_diff * reverse_diff, axis=1) / (step_size**2)
        factors = reverse - forward
        return proposals, factors

    return emcee.moves.MHMove(proposal_function)


def run_emcee_sampler(
    factor: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    n_warmup: int,
    n_walkers: int,
    seed: int,
    move,
    init_scale: float = 0.1,
) -> dict:
    """Run emcee and return timing/statistics."""
    rng = np.random.RandomState(seed)
    dim = factor.shape[1]
    initial_state = init_scale * rng.randn(n_walkers, dim)

    sampler = emcee.EnsembleSampler(
        nwalkers=n_walkers,
        ndim=dim,
        log_prob_fn=log_posterior_batch,
        args=(factor, y),
        moves=move,
        vectorize=True,
    )

    t0 = time.perf_counter()
    sampler.run_mcmc(initial_state, n_warmup, progress=False)
    warmup_time = time.perf_counter() - t0
    sampler.reset()

    t0 = time.perf_counter()
    sampler.run_mcmc(None, n_samples, progress=False)
    sample_time = time.perf_counter() - t0

    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    flat_chain = sampler.get_chain(flat=True)
    flat_log_prob = sampler.get_log_prob(flat=True)

    return {
        "per_step_time": float(sample_time / max(n_samples, 1)),
        "total_mcmc_time": float(sample_time),
        "warmup_time": float(warmup_time),
        "sampling_time": float(sample_time),
        "total_sampler_time": float(warmup_time + sample_time),
        "accept_rate": float(np.mean(sampler.acceptance_fraction)),
        "nu_samples": flat_chain,
        "chain": chain,
        "logp_trace": np.mean(log_prob, axis=1),
        "logp_by_walker": log_prob,
        "flat_logp": flat_log_prob,
    }


def run_hmc(
    factor: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    n_warmup: int,
    seed: int,
    step_size: float,
    n_leapfrog: int,
) -> dict:
    """Run a simple Euclidean HMC sampler with fixed mass matrix."""
    rng = np.random.default_rng(seed)
    dim = factor.shape[1]
    total_steps = n_warmup + n_samples

    nu = np.zeros(dim, dtype=float)
    logp = log_posterior(nu, factor, y)

    step_times = np.zeros(total_steps, dtype=float)
    logp_trace = np.zeros(total_steps, dtype=float)
    accepts = np.zeros(total_steps, dtype=bool)
    nu_samples = np.zeros((n_samples, dim), dtype=float)
    post_idx = 0

    for i in range(total_steps):
        t0 = time.perf_counter()

        current_nu = nu.copy()
        current_logp = logp
        momentum = rng.standard_normal(dim)
        current_momentum = momentum.copy()

        grad = grad_log_posterior(current_nu, factor, y)
        proposal_nu = current_nu.copy()
        proposal_p = momentum + 0.5 * step_size * grad

        for leapfrog_idx in range(n_leapfrog):
            proposal_nu = proposal_nu + step_size * proposal_p
            grad = grad_log_posterior(proposal_nu, factor, y)
            if leapfrog_idx != n_leapfrog - 1:
                proposal_p = proposal_p + step_size * grad

        proposal_p = proposal_p + 0.5 * step_size * grad
        proposal_p = -proposal_p

        proposal_logp = log_posterior(proposal_nu, factor, y)
        current_h = -current_logp + 0.5 * np.dot(current_momentum, current_momentum)
        proposal_h = -proposal_logp + 0.5 * np.dot(proposal_p, proposal_p)
        log_accept = current_h - proposal_h

        if np.log(rng.random()) < log_accept:
            nu = proposal_nu
            logp = proposal_logp
            accepts[i] = True

        step_times[i] = time.perf_counter() - t0
        logp_trace[i] = logp
        if i >= n_warmup:
            nu_samples[post_idx, :] = nu
            post_idx += 1

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
        "logp_trace": logp_trace,
        "nu_samples": nu_samples,
        "step_size": float(step_size),
        "n_leapfrog": int(n_leapfrog),
    }


def main() -> None:
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    n_samples = 5000
    n_warmup = 200
    N_SEED_REPEATS = 3
    k_values = [10, 20, 50, 100, 200]
    run_started_utc = datetime.datetime.now(datetime.timezone.utc)
    run_timestamp_utc = run_started_utc.isoformat()
    output_timestamp = run_started_utc.strftime("%Y%m%d-%H%M%S")
    hostname = socket.gethostname()
    job_id = os.environ.get("SLURM_JOB_ID", "not_set")

    X, y = make_fake_blobs(seed=42)
    A = KernelMatrix(X, kernel="gaussian", bandwidth=1.0)
    results = []

    def result_row(
        *,
        sampler: str,
        k: int,
        n_walkers: int,
        seed: int,
        step_size: float,
        n_leapfrog: int | str,
        ess_logp: float,
        ess_per_sec: float,
        tau: float,
        accept_rate: float,
        factor_time_sec: float,
        warmup_time_sec: float,
        sampling_time_sec: float,
        per_step_time_sec: float,
        total_mcmc_time_sec: float,
    ) -> dict[str, object]:
        return {
            "experiment": "exp1b",
            "dataset": "synthetic_blobs",
            "n_train": int(X.shape[0]),
            "sampler": sampler,
            "k": int(k),
            "n_walkers": int(n_walkers),
            "seed": int(seed),
            "step_size": float(step_size),
            "n_leapfrog": n_leapfrog,
            "n_samples": int(n_samples),
            "n_warmup": int(n_warmup),
            "ess_logp": float(ess_logp),
            "ess_per_sec": float(ess_per_sec),
            "tau": float(tau),
            "accept_rate": float(accept_rate),
            "factor_time_sec": float(factor_time_sec),
            "warmup_time_sec": float(warmup_time_sec),
            "sampling_time_sec": float(sampling_time_sec),
            "per_step_time_sec": float(per_step_time_sec),
            "total_mcmc_time_sec": float(total_mcmc_time_sec),
            "run_timestamp_utc": run_timestamp_utc,
            "hostname": hostname,
            "job_id": job_id,
        }

    for seed_offset in range(N_SEED_REPEATS):
        print(f"\n=== Seed repeat {seed_offset + 1}/{N_SEED_REPEATS} ===")
        for k in k_values:
            factor_start = time.perf_counter()
            lra = arpcholesky(A, k=k, b=10)
            F = lra.get_left_factor()
            factor_time = time.perf_counter() - factor_start
            dim = F.shape[1]
            n_walkers = max(2 * dim + 2, 24)

            rwm_scale = 1.2
            mala_scale = 1.5
            hmc_scale = 0.5

            gaussian_step = (rwm_scale / np.sqrt(dim)) ** 2
            mala_step = mala_scale / np.sqrt(dim)
            hmc_step = hmc_scale / np.sqrt(max(dim, 1))
            hmc_leapfrog = 12

            rwm_seed = 1000 + k + seed_offset * 10000
            mala_seed = 2000 + k + seed_offset * 10000
            hmc_seed = 3000 + k + seed_offset * 10000

            print(
                f"k={k} actual_rank={dim} seed_offset={seed_offset}: "
                f"RWM step={np.sqrt(gaussian_step):.4f}, MALA step={mala_step:.4f}, "
                f"HMC step={hmc_step:.4f}"
            )

            # emcee's GaussianMove is exactly a Gaussian random-walk MH proposal here.
            gaussian_move = emcee.moves.GaussianMove(cov=gaussian_step, mode="vector")
            mala_move = make_mala_move(F, y, step_size=mala_step)

            gaussian_stats = run_emcee_sampler(
                F,
                y,
                n_samples=n_samples,
                n_warmup=n_warmup,
                n_walkers=n_walkers,
                seed=rwm_seed,
                move=gaussian_move,
            )
            mala_stats = run_emcee_sampler(
                F,
                y,
                n_samples=n_samples,
                n_warmup=n_warmup,
                n_walkers=n_walkers,
                seed=mala_seed,
                move=mala_move,
            )
            hmc_stats = run_hmc(
                F,
                y,
                n_samples=n_samples,
                n_warmup=n_warmup,
                seed=hmc_seed,
                step_size=hmc_step,
                n_leapfrog=hmc_leapfrog,
            )

            tau_gaussian = compute_tau_emcee(gaussian_stats["logp_by_walker"])
            tau_mala = compute_tau_emcee(mala_stats["logp_by_walker"])
            tau_hmc = compute_tau_emcee(hmc_stats["logp_trace"][n_warmup:, None])
            ess_gaussian = compute_ess_from_tau(n_samples, n_walkers, tau_gaussian)
            ess_mala = compute_ess_from_tau(n_samples, n_walkers, tau_mala)
            ess_hmc = compute_ess_from_tau(n_samples, 1, tau_hmc)

            essps_gaussian = ess_gaussian / max(gaussian_stats["total_mcmc_time"], 1e-12)
            essps_mala = ess_mala / max(mala_stats["total_mcmc_time"], 1e-12)
            essps_hmc = ess_hmc / max(hmc_stats["total_mcmc_time"], 1e-12)

            results.append(
                result_row(
                    sampler="RWM",
                    k=dim,
                    n_walkers=n_walkers,
                    seed=rwm_seed,
                    step_size=np.sqrt(gaussian_step),
                    n_leapfrog="",
                    ess_logp=ess_gaussian,
                    ess_per_sec=essps_gaussian,
                    tau=tau_gaussian,
                    accept_rate=gaussian_stats["accept_rate"],
                    factor_time_sec=factor_time,
                    warmup_time_sec=gaussian_stats["warmup_time"],
                    sampling_time_sec=gaussian_stats["sampling_time"],
                    per_step_time_sec=gaussian_stats["per_step_time"],
                    total_mcmc_time_sec=gaussian_stats["total_mcmc_time"],
                )
            )
            results.append(
                result_row(
                    sampler="MALA",
                    k=dim,
                    n_walkers=n_walkers,
                    seed=mala_seed,
                    step_size=mala_step,
                    n_leapfrog="",
                    ess_logp=ess_mala,
                    ess_per_sec=essps_mala,
                    tau=tau_mala,
                    accept_rate=mala_stats["accept_rate"],
                    factor_time_sec=factor_time,
                    warmup_time_sec=mala_stats["warmup_time"],
                    sampling_time_sec=mala_stats["sampling_time"],
                    per_step_time_sec=mala_stats["per_step_time"],
                    total_mcmc_time_sec=mala_stats["total_mcmc_time"],
                )
            )
            results.append(
                result_row(
                    sampler="HMC",
                    k=dim,
                    n_walkers=1,
                    seed=hmc_seed,
                    step_size=hmc_stats["step_size"],
                    n_leapfrog=hmc_leapfrog,
                    ess_logp=ess_hmc,
                    ess_per_sec=essps_hmc,
                    tau=tau_hmc,
                    accept_rate=hmc_stats["accept_rate"],
                    factor_time_sec=factor_time,
                    warmup_time_sec=hmc_stats["warmup_time"],
                    sampling_time_sec=hmc_stats["sampling_time"],
                    per_step_time_sec=hmc_stats["per_step_time"],
                    total_mcmc_time_sec=hmc_stats["total_mcmc_time"],
                )
            )

    available_k_values = sorted({int(r["k"]) for r in results})

    fmt = "{:<8} {:>7} {:>8} {:>10} {:>8} {:>12} {:>10} {:>10} {:>8}"
    for k in available_k_values:
        print(f"k={k}")
        print(
            fmt.format(
                "Sampler",
                "Seed",
                "Walkers",
                "Step size",
                "Accept",
                "Per-step(s)",
                "ESS",
                "ESS/sec",
                "tau",
            )
        )
        for row in [r for r in results if r["k"] == k]:
            print(
                fmt.format(
                    row["sampler"],
                    f"{row['seed']}",
                    f"{row['n_walkers']}",
                    f"{row['step_size']:.4f}",
                    f"{row['accept_rate']:.3f}",
                    f"{row['per_step_time_sec']:.6f}",
                    f"{row['ess_logp']:.1f}",
                    f"{row['ess_per_sec']:.2f}",
                    f"{row['tau']:.2f}",
                )
            )
        print()

    print("Mean ± std ESS/sec across seed repeats:")
    for k in available_k_values:
        parts = []
        for sampler in ["RWM", "MALA", "HMC"]:
            values = [
                float(r["ess_per_sec"])
                for r in results
                if int(r["k"]) == k and r["sampler"] == sampler
            ]
            if values:
                arr = np.asarray(values, dtype=float)
                if sampler == "RWM":
                    parts.append(f"{sampler} ess/sec = {np.mean(arr):.1f} ± {np.std(arr):.1f}")
                else:
                    parts.append(f"{sampler} = {np.mean(arr):.1f} ± {np.std(arr):.1f}")
        print(f"k={k}: " + ", ".join(parts))

    csv_columns = [
        "experiment",
        "dataset",
        "n_train",
        "sampler",
        "k",
        "n_walkers",
        "seed",
        "step_size",
        "n_leapfrog",
        "n_samples",
        "n_warmup",
        "ess_logp",
        "ess_per_sec",
        "tau",
        "accept_rate",
        "factor_time_sec",
        "warmup_time_sec",
        "sampling_time_sec",
        "per_step_time_sec",
        "total_mcmc_time_sec",
        "run_timestamp_utc",
        "hostname",
        "job_id",
    ]
    csv_path = os.path.join(data_dir, f"exp1b_emcee_results_{output_timestamp}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    main()
