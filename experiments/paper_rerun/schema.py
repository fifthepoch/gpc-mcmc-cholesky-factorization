"""
Shared schema and helpers for the paper-format rerun harness.

Emits rows in the exact 146-column layout of
    experiment_results_pcam.csv
    experiment_results_camelyon17.csv
so fresh runs can be diffed directly against the recorded results.

Nothing here modifies the original experiment scripts; exp2 logic is imported
from experiments/exp2.py and reused as-is.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Placeholder strings used by the original CSVs. Kept verbatim so a diff against
# the recorded rows only shows genuine numeric drift.
NOT_APPLICABLE = "not applicable"
NOT_COMPUTED = "not computed"

COLUMNS = [
    "record_id",
    "experiment",
    "dataset",
    "method_name",
    "sampler",
    "model_architecture",
    "script_path",
    "run_timestamp_utc",
    "job_id",
    "hostname",
    "node_gpu",
    "code_ref",
    "notes",
    "dataset_sources",
    "embedding_source",
    "embedding_variant",
    "feature_dim",
    "n_train",
    "n_val",
    "n_test",
    "max_train_samples",
    "max_valid_samples",
    "max_test_samples",
    "seed",
    "standardize",
    "kernel",
    "kernel_bandwidth",
    "lengthscale_initial",
    "outputscale_initial",
    "k",
    "arpcholesky_b",
    "actual_rank",
    "relative_trace_error",
    "kernel_entries_queried",
    "factor_MB",
    "trace_K",
    "trace_FF_T",
    "pivot_consistency_max",
    "n_samples",
    "n_warmup",
    "n_walkers",
    "n_leapfrog",
    "n_conditional_draws",
    "step_size",
    "final_step_size",
    "initial_step",
    "target_accept",
    "adapt",
    "jitter",
    "num_inducing",
    "hidden_dim",
    "num_layers",
    "dropout",
    "attn_dropout",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "epochs_requested",
    "epochs_ran",
    "best_epoch",
    "snapshot_epoch",
    "trainable_parameters",
    "image_size",
    "patch_size",
    "embed_dim",
    "depth",
    "num_heads",
    "mlp_ratio",
    "mask_ratio",
    "best_val_auroc",
    "train_loss_final",
    "valid_loss_final",
    "test_loss",
    "test_acc",
    "test_nll",
    "log_likelihood_mean",
    "negative_log_likelihood_mean",
    "elpd",
    "elpd_mean",
    "pell",
    "pell_mean",
    "mean_predictive_log_likelihood",
    "predictive_likelihood",
    "posterior_expected_log_loss",
    "posterior_log_loss_mean",
    "posterior_total_log_loss",
    "loss",
    "negative_log_loss",
    "auroc",
    "auprc",
    "accuracy",
    "precision",
    "recall",
    "sensitivity",
    "brier",
    "ece",
    "number_errors",
    "tp",
    "fp",
    "tn",
    "fn",
    "sensitivity_tpr",
    "specificity_tnr",
    "true_positive_rate",
    "true_negative_rate",
    "false_positive_rate",
    "false_negative_rate",
    "positive_rate",
    "target_positive_rate",
    "accept_rate",
    "tau_logp",
    "tau_nu",
    "tau_nu_mean",
    "prob_min",
    "prob_mean",
    "prob_max",
    "ess_logp",
    "ess_per_sec",
    "latent_mean_min",
    "latent_mean_mean",
    "latent_mean_max",
    "latent_var_min",
    "latent_var_mean",
    "latent_var_max",
    "rbf_train_offdiag_mean",
    "rbf_train_offdiag_median",
    "rbf_test_train_mean",
    "rbf_test_train_median",
    "kernel_signal_std",
    "data_loading_time_sec",
    "embedding_time_sec",
    "kernel_time_sec",
    "cholesky_time_sec",
    "factor_time_sec",
    "warmup_time_sec",
    "sampling_time_sec",
    "total_mcmc_time_sec",
    "predictive_sampling_time_sec",
    "fit_or_train_time_sec",
    "train_time_sec",
    "inference_time_sec",
    "prediction_time_sec",
    "evaluation_time_sec",
    "total_pipeline_time_sec",
    "total_runtime_sec",
    "timing_scope",
]

# Columns that are labels/provenance rather than measurements. The aggregator
# never averages these; it carries them through when every run agrees and
# flags them when they diverge.
NON_NUMERIC_COLUMNS = {
    "record_id",
    "experiment",
    "dataset",
    "method_name",
    "sampler",
    "model_architecture",
    "script_path",
    "run_timestamp_utc",
    "job_id",
    "hostname",
    "node_gpu",
    "code_ref",
    "notes",
    "dataset_sources",
    "embedding_source",
    "embedding_variant",
    "max_train_samples",
    "max_valid_samples",
    "max_test_samples",
    "kernel",
    "adapt",
    "timing_scope",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit(default: str = NOT_COMPUTED) -> str:
    """Short commit hash of the working tree, for provenance."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return default


def slurm_context() -> dict:
    """Job id / host / GPU, filled in when running under SLURM."""
    return {
        "job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "hostname": socket.gethostname(),
        "node_gpu": os.environ.get("SLURM_JOB_GPUS", NOT_APPLICABLE),
    }


def blank_row() -> dict:
    """A full 146-column row with every field defaulted to 'not applicable'."""
    return {column: NOT_APPLICABLE for column in COLUMNS}


def write_run(row: dict, run_dir: str | Path, dataset: str, experiment: str, seed: int) -> Path:
    """Persist one run's row as JSON under run_dir/<dataset>/<experiment>/."""
    missing = set(row) - set(COLUMNS)
    if missing:
        raise KeyError(f"row has fields outside the schema: {sorted(missing)}")

    out_dir = Path(run_dir) / dataset / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"seed{seed:03d}.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2, default=str)
    return path
