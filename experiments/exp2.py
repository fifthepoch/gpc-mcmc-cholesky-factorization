"""
Real-data low-rank GP classification predictive sampling using:
1) precomputed RP-Cholesky factor F for K_XX,
2) HMC in the non-centered training coordinates f = F nu,
3) pivot points from RP-Cholesky as inducing points for prediction,
4) per-point marginal predictive sampling for classification metrics.

Example:
    python experiments/exp2.py \
        --factor-dir data/rpchol_test \
        --test-embeddings datasets/pcam_test_embedding.npy \
        --test-labels datasets/pcam_test_label.npy \
        --k 100
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import emcee
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import expit

# Allow direct script execution without package install.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from my_cholesky.kernels import GaussianKernel_mtx
from my_cholesky.result_logging import append_result_row
from predictive_metrics2 import (
    evaluate_binary_probabilistic_predictions,
    print_metric_table,
    summarize_predictive_distribution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-data low-rank HMC GP classification prediction."
    )
    parser.add_argument(
        "--factor-dir",
        type=str,
        required=True,
        help="Directory containing factor_k{k}.npy and labels.npy.",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Path to training embeddings .npy. If omitted, loads from factor_dir/embeddings.npy.",
    )
    parser.add_argument(
        "--test-embeddings",
        type=str,
        required=True,
        help="Test embeddings .npy file of shape (n_test, d).",
    )
    parser.add_argument(
        "--test-labels",
        type=str,
        required=True,
        help="Test labels .npy file of shape (n_test,).",
    )
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=None,
        help=(
            "Gaussian kernel bandwidth. If omitted, loads bandwidth_actual from "
            "factor_dir/summary.json (the value used when the factor was built)."
        ),
    )
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-warmup", type=int, default=200)
    parser.add_argument("--n-leapfrog", type=int, default=12)
    parser.add_argument(
        "--metric",
        choices=["laplace", "identity"],
        default="laplace",
        help=(
            "HMC mass matrix. 'laplace' uses M = I + F^T W F at the Laplace "
            "mode and starts the chain there, which removes the anisotropy "
            "that pins the step size. 'identity' reproduces the unpreconditioned "
            "sampler."
        ),
    )
    parser.add_argument(
        "--prediction-batch-size",
        type=int,
        default=512,
        help="Number of posterior samples to process per batched predictive solve.",
    )
    parser.add_argument("--hmc-step-constant", type=float, default=0.05)
    parser.add_argument("--hmc-target-accept", type=float, default=0.8)
    parser.add_argument(
        "--no-adapt-step-size",
        action="store_false",
        dest="adapt_step_size",
        help="Disable HMC dual averaging step-size adaptation during warmup.",
    )
    parser.add_argument(
        "--subsample-test",
        type=int,
        default=None,
        help="Randomly subsample test data to this size for quick tests.",
    )
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def log_posterior(nu, factor, y):
    """
    Approximate low-rank posterior with f = factor @ nu and nu ~ N(0, I).
    """
    f = factor @ nu
    p = expit(f)
    return log_posterior_from_cached_forward(nu, p, y)


def log_posterior_from_cached_forward(nu, p, y):
    eps = 1e-10
    log_lik = np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    log_prior = -0.5 * np.dot(nu, nu)
    return float(log_lik + log_prior)


def grad_log_posterior(nu, factor, y):
    grad, _ = grad_log_posterior_with_cache(nu, factor, y)
    return grad


def grad_log_posterior_with_cache(nu, factor, y):
    f = factor @ nu
    p = expit(f)
    return factor.T @ (y - p) - nu, p


def compute_tau_emcee(chain):
    """Estimate integrated autocorrelation time with emcee's implementation."""
    try:
        tau = emcee.autocorr.integrated_time(chain, quiet=True)
    except Exception as err:
        print(f"WARNING: emcee autocorr.integrated_time failed: {err}")
        return float("nan")

    tau = np.asarray(tau, dtype=float).reshape(-1)
    if tau.size == 0:
        return float("nan")
    return float(np.nanmean(tau))


def sample_predictive_probabilities_pivots(
    F,
    X_train,
    X_test,
    pivot_indices,
    K_pp,
    nu_samples,
    bandwidth,
    batch_size=512,
    seed=0,
):
    """
    Option B predictive sampler using RPCholesky pivots as inducing points.

    This computes per-test-point marginal predictive distributions without
    building n_test x n_train or n_test x n_test kernel matrices. The joint
    predictive covariance is not sampled; add a pivot-based joint sampler later
    if a downstream diagnostic needs correlations across test points.
    """
    rng = np.random.default_rng(seed)

    pivot_indices = np.asarray(pivot_indices, dtype=np.int64)
    X_pivots = X_train[pivot_indices].astype(np.float32, copy=False)
    F_pivots = F[pivot_indices, :].astype(np.float32, copy=False)

    K_pp = np.asarray(K_pp, dtype=np.float64)
    # Numerical jitter for Cholesky stability. This is NOT a model nugget --
    # the model is K_PP exactly. The jitter only protects the float64 Cholesky
    # from breakdown when K_PP is borderline-singular due to highly correlated pivots.
    k_size = K_pp.shape[0]
    jitter = 1e-10 * np.trace(K_pp) / k_size
    K_pp_jittered = K_pp + jitter * np.eye(k_size)
    L_pp = np.linalg.cholesky(K_pp_jittered)

    K_test_pivots = GaussianKernel_mtx(X_test, X_pivots, bandwidth=bandwidth).astype(
        np.float32,
        copy=False,
    )

    K_tp_t = K_test_pivots.T.astype(np.float64, copy=False)
    V = solve_triangular(L_pp, K_tp_t, lower=True, check_finite=False)
    V = solve_triangular(L_pp.T, V, lower=False, check_finite=False)
    var_f64 = 1.0 - np.sum(
        K_test_pivots.astype(np.float64, copy=False)
        * V.T.astype(np.float64, copy=False),
        axis=1,
    )
    var = np.clip(var_f64, 1e-8, None).astype(np.float32, copy=False)
    std = np.sqrt(var, dtype=np.float32)

    n_samples = nu_samples.shape[0]
    n_test = X_test.shape[0]
    latent_samples = np.zeros((n_samples, n_test), dtype=np.float32)
    p_samples = np.zeros((n_samples, n_test), dtype=np.float32)

    batch_size = max(1, int(batch_size))
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        nu_batch = nu_samples[start:end].T
        f_pivots = F_pivots @ nu_batch
        rhs = f_pivots.astype(np.float64, copy=False)
        alpha = solve_triangular(L_pp, rhs, lower=True, check_finite=False)
        alpha = solve_triangular(L_pp.T, alpha, lower=False, check_finite=False)
        alpha = alpha.astype(np.float32, copy=False)

        mean = K_test_pivots @ alpha
        xi = rng.standard_normal(mean.shape).astype(np.float32, copy=False)
        f_star = mean + std[:, None] * xi
        latent_samples[start:end] = f_star.T
        p_samples[start:end] = expit(f_star).T.astype(np.float32, copy=False)

    return {
        "p_samples": p_samples,
        "latent_samples": latent_samples,
        "predictive_cov_approx_factor": None,
    }


def laplace_metric(factor, y, newton_iters=8, tol=1e-8):
    """Posterior mode and precision from a Newton (Laplace) approximation.

    The target is log p(nu) = sum_i log Bernoulli(y_i | sigmoid(F nu)_i)
    - 0.5||nu||^2, whose Hessian is -(I + F^T W F) with W = diag(p(1-p)).
    That precision is only k x k, so forming and factoring it is cheap even
    when n is large, and it captures the *rotational* anisotropy that a
    diagonal metric cannot.
    """
    n, k = factor.shape
    nu = np.zeros(k, dtype=np.float64)
    eye = np.eye(k, dtype=np.float64)
    hess = eye.copy()
    for _ in range(newton_iters):
        prob = expit(factor @ nu)
        w = np.maximum(prob * (1.0 - prob), 1e-10)
        grad = factor.T @ (y - prob) - nu
        hess = eye + factor.T @ (factor * w[:, None])
        step = np.linalg.solve(hess, grad)
        nu = nu + step
        if np.max(np.abs(step)) < tol:
            break
    prob = expit(factor @ nu)
    w = np.maximum(prob * (1.0 - prob), 1e-10)
    hess = eye + factor.T @ (factor * w[:, None])
    return nu, hess


def run_hmc(
    factor,
    y,
    n_samples,
    n_warmup,
    seed,
    initial_step_size,
    n_leapfrog,
    target_accept=0.8,
    adapt_step_size=True,
    metric="laplace",
):
    """HMC with dual-averaging step size and an optional Laplace metric.

    With the identity metric the sampler mixes badly here: the low-rank factor
    has a decaying spectrum, so the posterior precision I + F^T W F is highly
    anisotropic (condition numbers of 1e3-1e4 are typical). The step size is
    then pinned by the tightest direction while the trajectory cannot cross the
    widest one, which shows up as a collapsed step size and a large integrated
    autocorrelation time.

    metric="laplace" uses M = I + F^T W F evaluated at the Laplace mode, which
    makes the target approximately isotropic in the transformed coordinates,
    and starts the chain at that mode so there is no burn-in transient.
    metric="identity" reproduces the previous behaviour.
    """
    rng = np.random.default_rng(seed)
    if n_leapfrog < 1:
        raise ValueError("n_leapfrog must be at least 1")
    if metric not in ("identity", "laplace"):
        raise ValueError(f"Unknown metric {metric!r}")

    dim = factor.shape[1]
    total_steps = n_warmup + n_samples

    nu = np.zeros(dim, dtype=np.float64)
    metric_chol = None
    if metric == "laplace":
        t_metric = time.perf_counter()
        mode, hess = laplace_metric(factor, y)
        # Symmetrize and jitter for a safe Cholesky.
        hess = 0.5 * (hess + hess.T)
        jitter = 1e-10 * float(np.trace(hess)) / dim
        metric_chol = np.linalg.cholesky(hess + jitter * np.eye(dim))
        nu = mode.copy()
        cond = float(np.linalg.cond(hess))
        print(
            f"Laplace metric: built in {time.perf_counter() - t_metric:.2f}s, "
            f"posterior precision condition number {cond:.3g}, "
            f"chain initialized at the mode",
            flush=True,
        )

    logp = log_posterior(nu, factor, y)

    nu_samples = np.zeros((n_samples, dim), dtype=np.float64)
    logp_trace = np.zeros((n_samples,), dtype=np.float64)
    step_times = np.zeros(total_steps, dtype=np.float64)
    accepts = np.zeros(total_steps, dtype=bool)
    post_idx = 0

    def draw_momentum():
        z = rng.standard_normal(dim)
        # p ~ N(0, M). With M = L L^T that is p = L z.
        return z if metric_chol is None else metric_chol @ z

    def apply_inv_mass(p_vec):
        # M^{-1} p via the Cholesky factor.
        if metric_chol is None:
            return p_vec
        tmp = solve_triangular(metric_chol, p_vec, lower=True, check_finite=False)
        return solve_triangular(metric_chol.T, tmp, lower=False, check_finite=False)

    step_size = initial_step_size
    log_step = np.log(initial_step_size)
    mu = np.log(10 * initial_step_size)
    log_step_bar = 0.0
    H_bar = 0.0
    gamma, t0_dual, kappa = 0.05, 10, 0.75

    for i in range(total_steps):
        t0 = time.perf_counter()
        if adapt_step_size and n_warmup > 0:
            if i < n_warmup:
                step_size = np.exp(log_step)
            elif i == n_warmup:
                step_size = np.exp(log_step_bar)

        current_nu = nu.copy()
        current_logp = logp
        momentum = draw_momentum()

        grad, _ = grad_log_posterior_with_cache(current_nu, factor, y)
        proposal_nu = current_nu.copy()
        proposal_p = momentum + 0.5 * step_size * grad

        for leapfrog_idx in range(n_leapfrog):
            proposal_nu = proposal_nu + step_size * apply_inv_mass(proposal_p)
            grad, proposal_prob = grad_log_posterior_with_cache(
                proposal_nu,
                factor,
                y,
            )
            if leapfrog_idx != n_leapfrog - 1:
                proposal_p = proposal_p + step_size * grad

        proposal_p = proposal_p + 0.5 * step_size * grad
        proposal_p = -proposal_p

        proposal_logp = log_posterior_from_cached_forward(
            proposal_nu,
            proposal_prob,
            y,
        )
        current_h = -current_logp + 0.5 * np.dot(momentum, apply_inv_mass(momentum))
        proposal_h = -proposal_logp + 0.5 * np.dot(
            proposal_p, apply_inv_mass(proposal_p)
        )
        log_accept = current_h - proposal_h
        if not np.isfinite(log_accept):
            log_accept = -np.inf
        accept_prob = 1.0 if log_accept >= 0.0 else float(np.exp(log_accept))

        if np.log(rng.random()) < log_accept:
            nu = proposal_nu
            logp = proposal_logp
            accepts[i] = True

        if adapt_step_size and i < n_warmup:
            eta = 1.0 / (i + 1 + t0_dual)
            H_bar = (1 - eta) * H_bar + eta * (target_accept - accept_prob)
            log_step = mu - np.sqrt(i + 1) / gamma * H_bar
            eta_bar = (i + 1) ** (-kappa)
            log_step_bar = eta_bar * log_step + (1 - eta_bar) * log_step_bar

        step_times[i] = time.perf_counter() - t0
        if i >= n_warmup:
            nu_samples[post_idx] = nu
            logp_trace[post_idx] = logp
            post_idx += 1

    post = slice(n_warmup, total_steps)
    warmup = slice(0, n_warmup)
    warmup_time = float(np.sum(step_times[warmup]))
    sampling_time = float(np.sum(step_times[post]))
    return {
        "nu_samples": nu_samples,
        "logp_trace": logp_trace,
        "accept_rate": float(np.mean(accepts[post])),
        "step_size": float(step_size),
        "metric": metric,
        "per_step_time": float(np.mean(step_times[post])),
        # Standard timing convention: total_mcmc_time is post-warmup sampling only.
        "warmup_time": warmup_time,
        "sampling_time": sampling_time,
        "total_mcmc_time": sampling_time,
        "total_sampler_time": warmup_time + sampling_time,
    }


def _subsample_rows(rng, size, *arrays):
    n_rows = arrays[0].shape[0]
    if size is None or size >= n_rows:
        return arrays
    idx = np.sort(rng.choice(n_rows, size=size, replace=False))
    return tuple(arr[idx] for arr in arrays)


def main():
    wall_t0 = time.perf_counter()
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_load_start = time.perf_counter()
    factor_dir = Path(args.factor_dir)
    if args.bandwidth is None:
        summary_path = factor_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"No --bandwidth given and {summary_path} not found. "
                "Either pass --bandwidth explicitly, or re-run "
                "exp_rpcholesky_embeddings.py to regenerate the factor directory "
                "with summary.json included."
            )
        with open(summary_path, "r") as f:
            summary = json.load(f)
        if "bandwidth_actual" not in summary:
            raise KeyError(
                f"{summary_path} does not contain 'bandwidth_actual'. "
                "Pass --bandwidth explicitly."
            )
        bandwidth = float(summary["bandwidth_actual"])
        print(f"Loaded bandwidth from {summary_path}: {bandwidth:.6f}")
    else:
        bandwidth = float(args.bandwidth)
        print(f"Using --bandwidth: {bandwidth:.6f}")

    if args.embeddings is None:
        emb_path = factor_dir / "embeddings.npy"
        if not emb_path.exists():
            raise FileNotFoundError(
                f"No --embeddings given and {emb_path} not found. "
                "Re-run exp_rpcholesky_embeddings.py to regenerate the factor directory "
                "with embeddings.npy included, or pass --embeddings explicitly."
            )
    else:
        emb_path = Path(args.embeddings)
    X_train = np.load(emb_path).astype(np.float32, copy=False)
    print(f"Loaded training embeddings from: {emb_path}  shape={X_train.shape}")
    y_train = np.load(factor_dir / "labels.npy").astype(np.float32, copy=False).squeeze()
    X_test = np.load(args.test_embeddings).astype(np.float32, copy=False)
    y_test = np.load(args.test_labels).astype(np.float32, copy=False).squeeze()
    F = np.load(factor_dir / f"factor_k{args.k}.npy").astype(np.float64, copy=False)
    pivots_path = factor_dir / f"pivots_k{args.k}.npy"
    if not pivots_path.exists():
        raise FileNotFoundError(
            f"{pivots_path} not found. Re-run exp_rpcholesky_embeddings.py "
            "to regenerate the factor directory with pivot indices included."
        )
    pivot_indices = np.load(pivots_path).astype(np.int64, copy=False)
    kpp_path = factor_dir / f"kernel_submatrix_k{args.k}.npy"
    if not kpp_path.exists():
        raise FileNotFoundError(
            f"{kpp_path} not found. Re-run exp_rpcholesky_embeddings.py "
            "to regenerate the factor directory with the pivot kernel submatrix included."
        )
    K_pivots = np.load(kpp_path).astype(np.float64, copy=False)
    data_loading_time = time.perf_counter() - data_load_start

    assert X_train.shape[0] == F.shape[0] == y_train.shape[0], (
        f"Misaligned training arrays: X_train={X_train.shape[0]}, "
        f"F={F.shape[0]}, y_train={y_train.shape[0]}. "
        f"All three must come from the same factor directory."
    )
    assert pivot_indices.shape[0] == F.shape[1], (
        f"Misaligned pivot indices: pivots={pivot_indices.shape[0]}, "
        f"F rank={F.shape[1]}."
    )
    assert K_pivots.shape == (F.shape[1], F.shape[1]), (
        f"Misaligned pivot kernel submatrix: K_pivots={K_pivots.shape}, "
        f"expected {(F.shape[1], F.shape[1])}."
    )
    F_P_check = F[pivot_indices, :]
    recon = F_P_check @ F_P_check.T
    abs_err = float(np.max(np.abs(recon - K_pivots)))
    rel_err = abs_err / float(np.max(np.abs(K_pivots)) + 1e-30)
    print(f"Pivot consistency check: |F_P F_P^T - K_PP| max={abs_err:.2e}, rel={rel_err:.2e}")
    if rel_err > 1e-3:
        print(
            "  WARNING: F[P,:] F[P,:]^T does not closely match K_PP. "
            "The pivots-as-inducing-points prediction path assumes equality. "
            "If your RPCholesky variant doesn't preserve this property exactly, "
            "predictions may be inconsistent with the HMC training model."
        )

    rng = np.random.default_rng(args.seed)
    if args.subsample_test is not None:
        X_test, y_test = _subsample_rows(rng, args.subsample_test, X_test, y_test)

    assert X_test.shape[0] == y_test.shape[0]

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    actual_rank = F.shape[1]
    F_hmc = F.astype(np.float64, copy=False)
    print(f"HMC factor dtype: {F_hmc.dtype}, prediction factor dtype: {F.dtype}")

    hmc_step = args.hmc_step_constant / (args.k ** 0.25)
    print(
        f"Data: X_train={X_train.shape}, F={F.shape}, X_test={X_test.shape}, "
        f"y_test={y_test.shape}"
    )
    print(
        f"HMC: k={args.k}, actual_rank={actual_rank}, initial_step={hmc_step:.6f}, "
        f"target_accept={args.hmc_target_accept:.3f}, adapt={args.adapt_step_size}"
    )

    hmc_stats = run_hmc(
        factor=F_hmc,
        y=y_train,
        n_samples=args.n_samples,
        n_warmup=args.n_warmup,
        seed=args.seed,
        initial_step_size=hmc_step,
        n_leapfrog=args.n_leapfrog,
        target_accept=args.hmc_target_accept,
        adapt_step_size=args.adapt_step_size,
        metric=args.metric,
    )

    nu_samples = hmc_stats["nu_samples"]
    tau_nu = compute_tau_emcee(nu_samples)
    tau_logp = compute_tau_emcee(hmc_stats["logp_trace"])
    print(f"HMC warmup time: {hmc_stats['warmup_time']:.2f}s")
    print(f"HMC sampling time: {hmc_stats['sampling_time']:.2f}s")
    print(f"HMC acceptance rate: {hmc_stats['accept_rate']:.3f}")
    print(f"HMC final step size: {hmc_stats['step_size']:.6f}")
    def _ess(tau_value):
        """ESS from tau. emcee can return tau <= 0 when the chain is
        effectively uncorrelated; treat that as tau = 1 (independent draws)."""
        if tau_value is None or not np.isfinite(tau_value):
            return float("nan")
        return args.n_samples / max(float(tau_value), 1.0)

    ess_nu = _ess(tau_nu)
    ess_logp = _ess(tau_logp)
    print(f"HMC tau (nu mean): {tau_nu:.2f}   ESS ~ {ess_nu:.1f} / {args.n_samples}")
    print(f"HMC tau (logp): {tau_logp:.2f}   ESS ~ {ess_logp:.1f} / {args.n_samples}")
    print(f"HMC metric: {hmc_stats.get('metric')}")
    if np.isfinite(ess_nu) and ess_nu < 50:
        print(
            f"WARNING: effective sample size {ess_nu:.1f} is far below the "
            f"{args.n_samples} nominal draws. Posterior summaries and the "
            "predictive distribution are unreliable; raise --n-samples/--n-warmup "
            "or reduce --n-leapfrog.",
            flush=True,
        )

    print("Sampling test predictive distribution...")
    t_pred_start = time.perf_counter()
    pred_test = sample_predictive_probabilities_pivots(
        F=F,
        X_train=X_train,
        X_test=X_test,
        pivot_indices=pivot_indices,
        K_pp=K_pivots,
        nu_samples=nu_samples,
        bandwidth=bandwidth,
        batch_size=args.prediction_batch_size,
        seed=999,
    )
    inference_time = time.perf_counter() - t_pred_start
    print(f"Predictive sampling time: {inference_time:.2f}s")

    evaluation_start = time.perf_counter()
    pred_summary = summarize_predictive_distribution(
        p_samples=pred_test["p_samples"],
        latent_samples=pred_test["latent_samples"],
    )
    predictive_prob = pred_summary["prob_mean"]

    test_metrics = evaluate_binary_probabilistic_predictions(
        y_true=y_test,
        p_pred=predictive_prob,
        threshold=0.5,
        n_bins=15,
        p_samples=pred_test["p_samples"],
    )
    print_metric_table(test_metrics, title="Exp2 low-rank HMC GP test metrics")
    evaluation_time = time.perf_counter() - evaluation_start

    total_wall = time.perf_counter() - wall_t0

    output_path = output_dir / f"exp2_k{args.k}_results.npz"
    np.savez(
        output_path,
        predictive_prob=pred_summary["prob_mean"],
        predictive_std=pred_summary["prob_std"],
        predictive_var=pred_summary["prob_variance"],
        predictive_latent_mean=pred_summary["latent_mean"],
        predictive_latent_std=pred_summary["latent_std"],
        predictive_latent_var=pred_summary["latent_variance"],
        prob_q05=pred_summary["prob_q05"],
        prob_q50=pred_summary["prob_q50"],
        prob_q95=pred_summary["prob_q95"],
        latent_q05=pred_summary["latent_q05"],
        latent_q50=pred_summary["latent_q50"],
        latent_q95=pred_summary["latent_q95"],
        y_test=y_test,
        accept_rate=hmc_stats["accept_rate"],
        tau_nu=tau_nu,
        tau_logp=tau_logp,
        final_step_size=hmc_stats["step_size"],
        nu_samples=hmc_stats["nu_samples"],
        k=args.k,
        bandwidth=bandwidth,
        pivot_indices=pivot_indices,
        n_train=n_train,
        n_test=n_test,
        timing_scope="precomputed_factor_load, hmc_warmup, hmc_sampling, predictive_sampling, evaluation",
        data_loading_time_sec=data_loading_time,
        factor_time_sec=0.0,
        warmup_time_sec=hmc_stats["warmup_time"],
        sampling_time_sec=hmc_stats["sampling_time"],
        per_step_time_sec=hmc_stats["per_step_time"],
        total_mcmc_time_sec=hmc_stats["total_mcmc_time"],
        inference_time_sec=inference_time,
        evaluation_time_sec=evaluation_time,
        total_pipeline_time_sec=total_wall,
        total_model_compute_time_sec=(
            hmc_stats["total_sampler_time"] + inference_time + evaluation_time
        ),
        **test_metrics,
    )

    print(f"Total wall-clock time: {total_wall:.2f}s")
    print(f"Saved results to {output_path}")
    csv_path = append_result_row(
        {
            "experiment": "exp2",
            "script_path": "experiments/exp2.py",
            "artifacts": str(output_path),
            "dataset": Path(args.test_embeddings).stem,
            "seed": args.seed,
            "k": args.k,
            "kernel": "gaussian",
            "kernel_bandwidth": bandwidth,
            "n_train": n_train,
            "n_test": n_test,
            "n_samples": args.n_samples,
            "n_warmup": args.n_warmup,
            "n_leapfrog": args.n_leapfrog,
            "sampler": "hmc",
            "timing_scope": "precomputed_factor_load, hmc_warmup, hmc_sampling, predictive_sampling, evaluation",
            "data_loading_time_sec": data_loading_time,
            "factor_time_sec": 0.0,
            "warmup_time_sec": hmc_stats["warmup_time"],
            "sampling_time_sec": hmc_stats["sampling_time"],
            "per_step_time_sec": hmc_stats["per_step_time"],
            "total_mcmc_time_sec": hmc_stats["total_mcmc_time"],
            "inference_time_sec": inference_time,
            "evaluation_time_sec": evaluation_time,
            "total_pipeline_time_sec": total_wall,
            "accept_rate": hmc_stats["accept_rate"],
            "final_step_size": hmc_stats["step_size"],
            "tau": tau_logp,
            **test_metrics,
        }
    )
    print(f"Appended CSV metrics to {csv_path}")


if __name__ == "__main__":
    main()
