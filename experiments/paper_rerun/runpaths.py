"""Resolve which batch directory a post-processing script should read.

Each batch of runs lives in data/paper_rerun/<run_id>/ so a new submission
never overwrites an earlier one. Scripts accept --run-id to pick one
explicitly, or fall back to the most recently modified batch.
"""

from __future__ import annotations

from pathlib import Path


def add_arguments(parser) -> None:
    parser.add_argument(
        "--base-dir",
        default="data/paper_rerun",
        help="Directory holding the per-batch <run_id> subdirectories.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Batch to read. Defaults to the most recently modified one.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Explicit path to a runs/ directory, bypassing --run-id.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Where to write outputs. Defaults next to the runs/ directory.",
    )


def list_batches(base_dir: str | Path) -> list[Path]:
    """Batch directories, newest first."""
    base = Path(base_dir)
    if not base.exists():
        return []
    found = [d for d in base.iterdir() if d.is_dir() and (d / "runs").is_dir()]
    return sorted(found, key=lambda d: d.stat().st_mtime, reverse=True)


def resolve(args) -> tuple[Path, Path]:
    """Return (run_dir, out_dir), honouring the flags in precedence order."""
    if args.run_dir:
        run_dir = Path(args.run_dir)
        out_dir = Path(args.out_dir) if args.out_dir else run_dir.parent
        return run_dir, out_dir

    base = Path(args.base_dir)
    if args.run_id:
        batch = base / args.run_id
        if not batch.is_dir():
            available = ", ".join(d.name for d in list_batches(base)) or "none"
            raise SystemExit(
                f"No batch '{args.run_id}' under {base}. Available: {available}"
            )
    else:
        batches = list_batches(base)
        if not batches:
            # Tolerate the old flat layout so earlier results still aggregate.
            if (base / "runs").is_dir():
                print(f"Using legacy flat layout at {base}/runs")
                return base / "runs", Path(args.out_dir) if args.out_dir else base
            raise SystemExit(f"No batch directories found under {base}.")
        batch = batches[0]
        if len(batches) > 1:
            print(f"Using most recent batch: {batch.name} "
                  f"({len(batches) - 1} older, pass --run-id to pick another)")
        else:
            print(f"Using batch: {batch.name}")

    return batch / "runs", Path(args.out_dir) if args.out_dir else batch
