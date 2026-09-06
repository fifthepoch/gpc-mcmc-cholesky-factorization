"""
Reliability diagrams from the *_preds.npz files written by the rerun scripts.

A reliability curve bins test points by predicted probability and plots the
observed positive frequency in each bin against the mean prediction. A
perfectly calibrated model sits on the diagonal. Below the diagonal at the high
end means overconfident; a curve flatter than the diagonal means underconfident.

Usage:
    python scripts/paper_rerun_reliability.py --seed 1
    python scripts/paper_rerun_reliability.py --run-dir ... --out-dir ...

Reads   <run-dir>/<dataset>/exp{1,2}_seed<N>_preds.npz
Writes  <out-dir>/reliability_<dataset>.png   (SVGP vs RPChol+HMC overlaid)
        <out-dir>/reliability_summary.csv     (per-bin numbers behind the plots)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments" / "paper_rerun"))
import runpaths  # noqa: E402

METHODS = {"exp1": "SVGP", "exp2": "RPChol+HMC"}
COLORS = {"exp1": "tab:orange", "exp2": "tab:blue"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    runpaths.add_arguments(p)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-bins", type=int, default=15)
    p.add_argument(
        "--strategy",
        choices=["uniform", "quantile"],
        default="uniform",
        help="uniform: equal-width bins. quantile: equal-count bins.",
    )
    return p.parse_args()


def reliability_curve(y_true, p_pred, n_bins, strategy):
    """Return per-bin (count, mean prediction, observed frequency)."""
    if strategy == "quantile":
        edges = np.quantile(p_pred, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    idx = np.clip(np.digitize(p_pred, edges[1:-1], right=False), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        rows.append((b, edges[b], edges[b + 1], n,
                     float(p_pred[m].mean()), float(y_true[m].mean())))
    return rows


def expected_calibration_error(rows, total):
    """Sum of |confidence - accuracy| weighted by bin population."""
    return sum(n * abs(conf - freq) for _, _, _, n, conf, freq in rows) / total


def main() -> None:
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir, out_dir = runpaths.resolve(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(d.name for d in run_dir.iterdir() if d.is_dir())
    if not datasets:
        raise SystemExit(f"No dataset directories under {run_dir}")

    summary_rows = []
    for dataset in datasets:
        found = {}
        for exp in METHODS:
            path = run_dir / dataset / f"{exp}_seed{args.seed}_preds.npz"
            if path.exists():
                d = np.load(path)
                found[exp] = (np.asarray(d["y_test"]).squeeze().astype(int),
                              np.asarray(d["predictive_prob"]).squeeze().astype(float))
            else:
                print(f"  missing: {path}")
        if not found:
            continue

        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")

        for exp, (y, p) in sorted(found.items()):
            rows = reliability_curve(y, p, args.n_bins, args.strategy)
            ece = expected_calibration_error(rows, len(y))
            conf = [r[4] for r in rows]
            freq = [r[5] for r in rows]
            ax.plot(conf, freq, "o-", color=COLORS[exp], lw=1.6, ms=4,
                    label=f"{METHODS[exp]}  (ECE={ece:.4f})")
            for b, lo, hi, n, c, f in rows:
                summary_rows.append({
                    "dataset": dataset, "experiment": exp, "method": METHODS[exp],
                    "seed": args.seed, "bin": b, "bin_lo": f"{lo:.4f}",
                    "bin_hi": f"{hi:.4f}", "n": n,
                    "mean_predicted": f"{c:.6f}", "observed_frequency": f"{f:.6f}",
                    "gap": f"{f - c:+.6f}", "ece": f"{ece:.6f}",
                })
            print(f"{dataset:11s} {METHODS[exp]:12s} ECE={ece:.4f}  n={len(y)}")

        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency of class 1")
        ax.set_title(f"Reliability diagram — {dataset} (seed {args.seed})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()

        png = out_dir / f"reliability_{dataset}.png"
        fig.savefig(png, dpi=180)
        plt.close(fig)
        print(f"  -> {png}")

    if summary_rows:
        csv_path = out_dir / "reliability_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
