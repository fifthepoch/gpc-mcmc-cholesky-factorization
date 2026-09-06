"""
Average the per-seed runs written by experiments/paper_rerun/run_exp{1,2}.py
and emit one CSV per dataset in the original 146-column paper schema.

Outputs, under --out-dir:
    experiment_results_<dataset>_rerun.csv       <- the deliverable: mean of all seeds
    experiment_results_<dataset>_rerun_std.csv   <- sample std of the same fields
    experiment_results_<dataset>_allruns.csv     <- every individual seed, for traceability

Averaging policy:
  * a field is averaged only if EVERY run reports a finite number for it;
  * otherwise the placeholder is preserved, so a diff against the recorded
    CSVs only ever shows genuine numeric drift;
  * label/provenance fields are never averaged -- identical values are carried
    through, and differing ones are collapsed to a "varies(...)" marker.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments" / "paper_rerun"))

import runpaths  # noqa: E402
import schema  # noqa: E402

EXPERIMENT_ORDER = ["exp1", "exp2"]

# Averaging a seed is meaningless -- list them instead.
SEED_COLUMN = "seed"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    runpaths.add_arguments(p)
    p.add_argument(
        "--expect-runs",
        type=int,
        default=None,
        help="Fail if any group has a different number of runs (e.g. 5).",
    )
    return p.parse_args()


def as_number(value):
    """Return a finite float, or None if the value is not a usable number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str):
        text = value.strip()
        if not text or text in (schema.NOT_APPLICABLE, schema.NOT_COMPUTED):
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def load_runs(run_dir: Path):
    """-> {dataset: {experiment: [row, ...]}}, sorted by seed."""
    grouped = defaultdict(lambda: defaultdict(list))
    for path in sorted(run_dir.glob("*/*/seed*.json")):
        with open(path, encoding="utf-8") as handle:
            row = json.load(handle)
        grouped[path.parent.parent.name][path.parent.name].append(row)
    return grouped


def reduce_group(rows: list[dict], stat: str) -> dict:
    """Collapse N runs into one row using mean or sample std."""
    out = schema.blank_row()
    n = len(rows)

    for column in schema.COLUMNS:
        values = [row.get(column, schema.NOT_APPLICABLE) for row in rows]

        if column in schema.NON_NUMERIC_COLUMNS:
            distinct = {str(v) for v in values}
            if stat == "std":
                out[column] = values[0] if len(distinct) == 1 else schema.NOT_APPLICABLE
            else:
                out[column] = (
                    values[0] if len(distinct) == 1 else f"varies({len(distinct)})"
                )
            continue

        if column == SEED_COLUMN:
            seeds = [as_number(v) for v in values]
            out[column] = (
                schema.NOT_APPLICABLE
                if stat == "std" or any(x is None for x in seeds)
                else ";".join(str(int(x)) for x in sorted(seeds))
            )
            continue

        # A field that never varied is a config constant, not a measurement.
        # Carry it through verbatim so ints stay ints and the diff stays clean.
        if len({str(v) for v in values}) == 1:
            out[column] = 0.0 if stat == "std" and as_number(values[0]) is not None \
                else values[0]
            continue

        numbers = [as_number(v) for v in values]
        if any(x is None for x in numbers):
            # Not numeric in every run -- keep whatever the runs agreed on.
            distinct = {str(v) for v in values}
            out[column] = values[0] if len(distinct) == 1 else schema.NOT_COMPUTED
            continue

        if stat == "mean":
            out[column] = statistics.fmean(numbers)
        else:
            out[column] = statistics.stdev(numbers) if n > 1 else 0.0

    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=schema.COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, schema.NOT_APPLICABLE) for c in schema.COLUMNS})


def main() -> None:
    args = parse_args()
    run_dir, out_dir = runpaths.resolve(args)
    if not run_dir.exists():
        raise SystemExit(f"No run directory at {run_dir} -- run the experiments first.")

    grouped = load_runs(run_dir)
    if not grouped:
        raise SystemExit(f"No seed*.json files found under {run_dir}.")

    problems = []

    for dataset in sorted(grouped):
        out_rows = []

        for experiment in EXPERIMENT_ORDER:
            rows = grouped[dataset].get(experiment)
            if not rows:
                problems.append(f"{dataset}/{experiment}: no runs found")
                continue

            rows = sorted(rows, key=lambda r: as_number(r.get("seed")) or 0)
            n = len(rows)
            if args.expect_runs is not None and n != args.expect_runs:
                problems.append(
                    f"{dataset}/{experiment}: {n} runs, expected {args.expect_runs}"
                )

            seeds = [int(as_number(r.get("seed")) or -1) for r in rows]
            seed_list = ", ".join(str(s) for s in seeds)

            # The individual runs first, then their mean, then their spread.
            out_rows.extend(rows)

            mean_row = reduce_group(rows, "mean")
            mean_row["record_id"] = f"rerun-{dataset}-{experiment}-MEAN-n{n}"
            mean_row["notes"] = f"MEAN of {n} runs (seeds {seed_list})."
            out_rows.append(mean_row)

            std_row = reduce_group(rows, "std")
            std_row["record_id"] = f"rerun-{dataset}-{experiment}-STD-n{n}"
            std_row["notes"] = f"Sample standard deviation over {n} runs (seeds {seed_list})."
            out_rows.append(std_row)

            acc, auroc = mean_row.get("accuracy"), mean_row.get("auroc")
            if isinstance(acc, float) and isinstance(auroc, float):
                print(f"{dataset:11s} {experiment}  n={n}  "
                      f"accuracy={acc:.6f}  auroc={auroc:.6f}")
            else:
                print(f"{dataset:11s} {experiment}  n={n}")

        if out_rows:
            path = out_dir / f"experiment_results_{dataset}_rerun.csv"
            write_csv(path, out_rows)
            print(f"  -> {path}  ({len(out_rows)} rows)")

    if problems:
        print("\nWARNINGS:")
        for problem in problems:
            print(f"  - {problem}")


if __name__ == "__main__":
    main()
