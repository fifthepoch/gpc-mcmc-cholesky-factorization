"""
Experiment 2 (GPyTorch): SVGP binary GP classification on real PCam-HG embeddings.

This version:
1. Does NOT use fake/generated blob data.
2. Loads real embeddings from:

   datasets/pcam-hg/{train,valid,test}/embeddings/

3. Adds:
   - elpd = sum_i log p(y_i | x_i, D)
   - pell = mean_i log p(y_i | x_i, D)
   - posterior_expected_log_loss = mean_i[-log p(y_i | x_i, D)]
   - posterior_total_log_loss = sum_i[-log p(y_i | x_i, D)]

4. Saves only numerical output to .npz.
   No figures/pictures are generated.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from predictive_metrics import (
        evaluate_binary_probabilistic_predictions,
        print_metric_table,
    )
except Exception:
    evaluate_binary_probabilistic_predictions = None
    print_metric_table = None


EPS = 1e-12


@dataclass
class SplitData:
    X: np.ndarray
    y: np.ndarray
    embedding_path: str
    label_path: str


def _import_torch_stack():
    try:
        torch = importlib.import_module("torch")
        gpytorch = importlib.import_module("gpytorch")
        return torch, gpytorch
    except Exception as err:
        raise ImportError(
            "Could not import torch/gpytorch. Install them first, e.g. "
            "pip install torch gpytorch"
        ) from err


def _first_existing(paths):
    for path in paths:
        if path is not None and os.path.exists(path):
            return path

    raise FileNotFoundError(
        "None of these files exist:\n" + "\n".join(map(str, paths))
    )


def _read_labels_csv(path: str) -> np.ndarray:
    """
    Robust labels.csv reader.

    Supported cases:
    - a column named label/y/target/tumor/class
    - otherwise, the last numeric column
    - if no header exists, a plain one-column CSV
    """
    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames
    except Exception:
        arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
        return np.asarray(arr).reshape(-1).astype(np.int64)

    if not rows or not fieldnames:
        arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
        return np.asarray(arr).reshape(-1).astype(np.int64)

    lower_to_name = {name.lower(): name for name in fieldnames}
    preferred = ["label", "y", "target", "tumor", "class"]

    chosen = None
    for key in preferred:
        if key in lower_to_name:
            chosen = lower_to_name[key]
            break

    if chosen is None:
        for name in reversed(fieldnames):
            try:
                _ = [float(row[name]) for row in rows]
                chosen = name
                break
            except Exception:
                continue

    if chosen is None:
        raise ValueError(
            f"Could not infer label column from {path}; columns={fieldnames}"
        )

    y = np.asarray([float(row[chosen]) for row in rows], dtype=np.float32)
    return y.reshape(-1).astype(np.int64)


def _stratified_subsample_indices(
    y: np.ndarray,
    max_n: int,
    seed: int,
) -> np.ndarray:
    """
    Return sorted indices for a stratified subsample without replacement.

    The requested sample size is allocated across classes in proportion to the
    original split, with leftover slots assigned to classes with the largest
    fractional remainder.
    """
    y = np.asarray(y).reshape(-1)
    rng = np.random.default_rng(seed)

    classes, counts = np.unique(y, return_counts=True)
    proportions = counts / counts.sum()
    raw_alloc = proportions * max_n
    alloc = np.floor(raw_alloc).astype(int)

    remaining = max_n - int(alloc.sum())
    if remaining > 0:
        fractional_order = np.argsort(-(raw_alloc - alloc))
        for class_pos in fractional_order[:remaining]:
            alloc[class_pos] += 1

    alloc = np.minimum(alloc, counts)

    while int(alloc.sum()) < max_n:
        capacity = counts - alloc
        if np.all(capacity <= 0):
            break
        class_pos = int(np.argmax(capacity))
        alloc[class_pos] += 1

    chosen_indices = []
    for class_value, class_n in zip(classes, alloc):
        class_indices = np.flatnonzero(y == class_value)
        chosen = rng.choice(class_indices, size=int(class_n), replace=False)
        chosen_indices.append(chosen)

    idx = np.concatenate(chosen_indices)
    idx.sort()
    return idx


def _find_embedding_path(
    dataset_root: str,
    split: str,
    embedding_file: Optional[str],
) -> str:
    emb_root = os.path.join(dataset_root, split, "embeddings")

    if embedding_file is not None:
        candidate = os.path.join(emb_root, embedding_file)
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(f"Embedding file not found: {candidate}")

    return _first_existing(
        [
            os.path.join(emb_root, "projected_512.npy"),
            os.path.join(emb_root, "embeddings.npy"),
        ]
    )


def _find_label_path(
    dataset_root: str,
    split: str,
    label_file: Optional[str],
) -> str:
    split_root = os.path.join(dataset_root, split)
    emb_root = os.path.join(split_root, "embeddings")

    if label_file is not None:
        candidates = [
            os.path.join(emb_root, label_file),
            os.path.join(split_root, label_file),
        ]
        return _first_existing(candidates)

    return _first_existing(
        [
            os.path.join(emb_root, "y_embeddings.npy"),
            os.path.join(split_root, "labels.csv"),
        ]
    )


def load_split_raw(
    dataset_root: str,
    split: str,
    embedding_file: Optional[str] = None,
    label_file: Optional[str] = None,
    max_n: Optional[int] = None,
    seed: int = 42,
) -> SplitData:
    embedding_path = _find_embedding_path(dataset_root, split, embedding_file)
    label_path = _find_label_path(dataset_root, split, label_file)

    X = np.load(embedding_path).astype(np.float32)

    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    if label_path.endswith(".npy"):
        y = np.load(label_path).reshape(-1).astype(np.int64)
    elif label_path.endswith(".csv"):
        y = _read_labels_csv(label_path)
    else:
        raise ValueError(f"Unsupported label file type: {label_path}")

    if len(X) != len(y):
        n = min(len(X), len(y))
        print(
            f"WARNING: split={split} has len(X)={len(X)} but len(y)={len(y)}. "
            f"Using first {n} rows."
        )
        X = X[:n]
        y = y[:n]

    unique = np.unique(y)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(
            f"Expected binary labels 0/1 for split={split}, got {unique[:20]}"
        )

    if max_n is not None and max_n > 0 and len(X) > max_n:
        idx = _stratified_subsample_indices(y, max_n=max_n, seed=seed)
        X = X[idx]
        y = y[idx]

    return SplitData(
        X=X.astype(np.float32),
        y=y.astype(np.int64),
        embedding_path=embedding_path,
        label_path=label_path,
    )


def standardize_split(
    split_data: SplitData,
    mean: np.ndarray,
    std: np.ndarray,
) -> SplitData:
    X_std = ((split_data.X - mean) / std).astype(np.float32)
    return SplitData(
        X=X_std,
        y=split_data.y,
        embedding_path=split_data.embedding_path,
        label_path=split_data.label_path,
    )


def binary_metrics(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    threshold: float = 0.5,
    n_bins: int = 15,
) -> Dict[str, float]:
    """
    Compute binary probabilistic prediction metrics from posterior predictive
    probabilities p_pred = P(y=1 | x, D).

    Important posterior quantities:

        log p(y_i | x_i, D)
        =
        y_i log p_i + (1 - y_i) log(1 - p_i)

    where p_i = P(y_i = 1 | x_i, D).

    elpd:
        sum_i log p(y_i | x_i, D)

    pell:
        mean_i log p(y_i | x_i, D)

    posterior_expected_log_loss:
        mean_i[-log p(y_i | x_i, D)]

    posterior_total_log_loss:
        sum_i[-log p(y_i | x_i, D)]
    """
    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)
    p_pred = np.asarray(p_pred).reshape(-1).astype(np.float64)
    p_clip = np.clip(p_pred, EPS, 1.0 - EPS)

    log_prob_i = (
        y_true * np.log(p_clip)
        + (1 - y_true) * np.log(1.0 - p_clip)
    )

    elpd = float(np.sum(log_prob_i))
    pell = float(np.mean(log_prob_i))

    posterior_log_loss_i = -log_prob_i
    posterior_expected_log_loss = float(np.mean(posterior_log_loss_i))
    posterior_total_log_loss = float(np.sum(posterior_log_loss_i))

    metrics: Dict[str, float] = {}

    if evaluate_binary_probabilistic_predictions is not None:
        metrics.update(
            evaluate_binary_probabilistic_predictions(
                y_true=y_true,
                p_pred=p_clip,
                threshold=threshold,
                n_bins=n_bins,
            )
        )
    else:
        y_pred = (p_clip >= threshold).astype(np.int64)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        metrics.update(
            {
                "log_likelihood_mean": pell,
                "negative_log_likelihood_mean": posterior_expected_log_loss,
                "brier": float(np.mean((p_clip - y_true) ** 2)),
                "accuracy": float(np.mean(y_pred == y_true)),
                "number_errors": int(np.sum(y_pred != y_true)),
                "sensitivity_TPR": float(tp / max(tp + fn, 1)),
                "FNR": float(fn / max(tp + fn, 1)),
                "specificity_TNR": float(tn / max(tn + fp, 1)),
                "FPR": float(fp / max(tn + fp, 1)),
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
            }
        )

    metrics["elpd"] = elpd
    metrics["pell"] = pell
    metrics["mean_predictive_log_likelihood"] = pell

    metrics["posterior_expected_log_loss"] = posterior_expected_log_loss
    metrics["posterior_log_loss_mean"] = posterior_expected_log_loss
    metrics["posterior_total_log_loss"] = posterior_total_log_loss

    return metrics


def print_metrics(metrics: Dict[str, float], title: str) -> None:
    if print_metric_table is not None:
        print_metric_table(metrics, title=title)
        return

    print("\n" + title)
    print("-" * len(title))

    for key, value in metrics.items():
        if isinstance(value, (int, np.integer)):
            print(f"{key:35s}: {value:d}")
        else:
            print(f"{key:35s}: {float(value):.6f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SVGP binary classifier on HG embedding datasets."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="pcam",
        choices=["pcam", "camelyon17"],
        help="Dataset name used for default root inference and output metadata.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help=(
            "Path to datasets/<dataset>-hg. If omitted, inferred from --dataset."
        ),
    )

    parser.add_argument(
        "--embedding-file",
        type=str,
        default="projected_512.npy",
        help=(
            "Embedding file inside each split's embeddings directory. "
            "Use projected_512.npy or embeddings.npy."
        ),
    )

    parser.add_argument(
        "--label-file",
        type=str,
        default=None,
        help=(
            "Optional label file. If omitted, tries embeddings/y_embeddings.npy "
            "then labels.csv."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data"),
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="exp2_gpytorch_svgp_pcam_results.npz",
    )

    parser.add_argument("--num-inducing", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--lengthscale",
        type=float,
        default=None,
        help=(
            "Initial RBF lengthscale. If omitted, uses sqrt(feature_dim), "
            "which is usually more stable for standardized high-dimensional embeddings."
        ),
    )
    parser.add_argument(
        "--outputscale",
        type=float,
        default=1.0,
        help="Initial RBF outputscale.",
    )

    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-valid", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument(
        "--no-standardize",
        action="store_false",
        dest="standardize",
        help="Disable train-set feature standardization.",
    )
    parser.set_defaults(standardize=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--cholesky-jitter",
        type=float,
        default=1e-4,
        help=(
            "Jitter added by GPyTorch Cholesky routines. Increase if the "
            "inducing-point covariance is reported as not positive definite."
        ),
    )

    parser.add_argument(
        "--save-model",
        action="store_true",
        help="If passed, also save model state_dict to .pt.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    torch, gpytorch = _import_torch_stack()

    if args.dataset_root is None:
        args.dataset_root = os.path.join(PROJECT_ROOT, "datasets", f"{args.dataset}-hg")

    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_raw = load_split_raw(
        dataset_root=args.dataset_root,
        split="train",
        embedding_file=args.embedding_file,
        label_file=args.label_file,
        max_n=args.max_train,
        seed=args.seed,
    )

    valid_raw = load_split_raw(
        dataset_root=args.dataset_root,
        split="valid",
        embedding_file=args.embedding_file,
        label_file=args.label_file,
        max_n=args.max_valid,
        seed=args.seed + 1,
    )

    test_raw = load_split_raw(
        dataset_root=args.dataset_root,
        split="test",
        embedding_file=args.embedding_file,
        label_file=args.label_file,
        max_n=args.max_test,
        seed=args.seed + 2,
    )

    if args.standardize:
        train_mean = train_raw.X.mean(axis=0, keepdims=True)
        train_std = train_raw.X.std(axis=0, keepdims=True)
        train_std = np.where(train_std < 1e-6, 1.0, train_std)
        train = standardize_split(train_raw, train_mean, train_std)
        valid = standardize_split(valid_raw, train_mean, train_std)
        test = standardize_split(test_raw, train_mean, train_std)
    else:
        train_mean = np.zeros((1, train_raw.X.shape[1]), dtype=np.float32)
        train_std = np.ones((1, train_raw.X.shape[1]), dtype=np.float32)
        train = train_raw
        valid = valid_raw
        test = test_raw

    print(f"Loaded {args.dataset} HG embeddings:")
    print(f"  train: X={train.X.shape}, y={train.y.shape}, pos_rate={train.y.mean():.4f}")
    print(f"  valid: X={valid.X.shape}, y={valid.y.shape}, pos_rate={valid.y.mean():.4f}")
    print(f"  test : X={test.X.shape}, y={test.y.shape}, pos_rate={test.y.mean():.4f}")
    print(f"  train embeddings: {train.embedding_path}")
    print(f"  valid embeddings: {valid.embedding_path}")
    print(f"  test  embeddings: {test.embedding_path}")
    print(f"  train labels    : {train.label_path}")
    print(f"  valid labels    : {valid.label_path}")
    print(f"  test  labels    : {test.label_path}")
    print(f"  standardize     : {args.standardize}")

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu"
    )

    X_cpu = torch.tensor(train.X, dtype=torch.float32)
    y_cpu = torch.tensor(train.y, dtype=torch.float32)

    input_dim = int(train.X.shape[1])
    initial_lengthscale = (
        float(args.lengthscale)
        if args.lengthscale is not None
        else float(np.sqrt(input_dim))
    )

    print(f"  initial lengthscale: {initial_lengthscale:.6f}")
    print(f"  initial outputscale : {float(args.outputscale):.6f}")

    class SVGPBinaryClassifier(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points):
            variational_distribution = (
                gpytorch.variational.CholeskyVariationalDistribution(
                    inducing_points.size(0)
                )
            )

            variational_strategy = gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            )

            super().__init__(variational_strategy)

            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )
            self.covar_module.base_kernel.initialize(
                lengthscale=inducing_points.new_tensor([initial_lengthscale])
            )
            self.covar_module.initialize(
                outputscale=inducing_points.new_tensor([float(args.outputscale)])
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    num_inducing = min(args.num_inducing, X_cpu.size(0))

    perm = torch.randperm(X_cpu.size(0))
    inducing_points = X_cpu[perm[:num_inducing]].clone().to(device)

    model = SVGPBinaryClassifier(inducing_points=inducing_points).to(device)
    likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device)

    model.train()
    likelihood.train()

    dataset = torch.utils.data.TensorDataset(X_cpu, y_cpu)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda" and args.num_workers > 0),
    )

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": likelihood.parameters()},
        ],
        lr=args.learning_rate,
    )

    mll = gpytorch.mlls.VariationalELBO(
        likelihood,
        model,
        num_data=X_cpu.size(0),
    )

    t0 = time.perf_counter()
    epoch_losses = []

    for epoch in range(args.num_epochs):
        running_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            yb = yb.to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad(set_to_none=True)

            with gpytorch.settings.cholesky_jitter(args.cholesky_jitter):
                output = model(xb)
                loss = -mll(output, yb)

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)

        mean_loss = running_loss / X_cpu.size(0)
        epoch_losses.append(mean_loss)

        print_every = max(1, args.num_epochs // 10)
        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:>4}/{args.num_epochs}: loss={mean_loss:.6f}")

    train_time = time.perf_counter() - t0

    model.eval()
    likelihood.eval()

    def predict_numpy(X_np: np.ndarray, batch_size: int = 8192):
        mus = []
        vars_ = []
        probs = []

        with (
            torch.no_grad(),
            gpytorch.settings.fast_pred_var(),
            gpytorch.settings.cholesky_jitter(args.cholesky_jitter),
        ):
            for start in range(0, len(X_np), batch_size):
                xb = torch.tensor(
                    X_np[start : start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )

                q_f = model(xb)
                p_y = likelihood(q_f)

                mus.append(q_f.mean.detach().cpu().numpy())
                vars_.append(q_f.variance.detach().cpu().numpy())
                probs.append(p_y.probs.detach().cpu().numpy())

        return (
            np.concatenate(mus, axis=0),
            np.concatenate(vars_, axis=0),
            np.concatenate(probs, axis=0),
        )

    pred_t0 = time.perf_counter()

    mu_train, var_train, p_train = predict_numpy(train.X)
    mu_valid, var_valid, p_valid = predict_numpy(valid.X)
    mu_test, var_test, p_test = predict_numpy(test.X)

    pred_time = time.perf_counter() - pred_t0

    train_metrics = binary_metrics(
        train.y,
        p_train,
        threshold=args.threshold,
        n_bins=args.n_bins,
    )

    valid_metrics = binary_metrics(
        valid.y,
        p_valid,
        threshold=args.threshold,
        n_bins=args.n_bins,
    )

    test_metrics = binary_metrics(
        test.y,
        p_test,
        threshold=args.threshold,
        n_bins=args.n_bins,
    )

    print_metrics(train_metrics, title=f"GPyTorch SVGP {args.dataset} train metrics")
    print_metrics(valid_metrics, title=f"GPyTorch SVGP {args.dataset} valid metrics")
    print_metrics(test_metrics, title=f"GPyTorch SVGP {args.dataset} test metrics")

    results_path = os.path.join(args.output_dir, args.output_name)

    np.savez(
        results_path,
        num_inducing=num_inducing,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lengthscale=float(initial_lengthscale),
        outputscale=float(args.outputscale),
        standardize=bool(args.standardize),
        train_mean=train_mean.astype(np.float32, copy=False),
        train_std=train_std.astype(np.float32, copy=False),
        train_time=float(train_time),
        prediction_time=float(pred_time),
        final_loss=float(epoch_losses[-1]) if epoch_losses else float("nan"),
        device=str(device),
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        train_embedding_path=train.embedding_path,
        valid_embedding_path=valid.embedding_path,
        test_embedding_path=test.embedding_path,
        train_label_path=train.label_path,
        valid_label_path=valid.label_path,
        test_label_path=test.label_path,
        train_metrics=json.dumps(train_metrics),
        valid_metrics=json.dumps(valid_metrics),
        test_metrics=json.dumps(test_metrics),
        epoch_losses=np.asarray(epoch_losses, dtype=np.float64),
        latent_mean_train=mu_train,
        latent_var_train=var_train,
        prob_train=p_train,
        y_train=train.y,
        latent_mean_valid=mu_valid,
        latent_var_valid=var_valid,
        prob_valid=p_valid,
        y_valid=valid.y,
        latent_mean_test=mu_test,
        latent_var_test=var_test,
        prob_test=p_test,
        y_test=test.y,
    )

    if args.save_model:
        model_path = os.path.join(
            args.output_dir,
            "exp2_gpytorch_svgp_pcam_model.pt",
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "likelihood_state_dict": likelihood.state_dict(),
                "train_mean": train_mean,
                "train_std": train_std,
                "args": vars(args),
            },
            model_path,
        )

        print(f"Saved model: {model_path}")

    print(f"\nTraining done in {train_time:.2f}s on {device}.")
    print(f"Prediction done in {pred_time:.2f}s.")
    print("Saved:")
    print(f"- {results_path}")


if __name__ == "__main__":
    main()
