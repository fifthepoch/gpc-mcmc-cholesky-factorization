#!/usr/bin/env python3
"""
Turn the flat EMBED embedding outputs into the per-split layout the
experiment scripts expect.

Input (produced by scripts/create_embed_embeddings.py):
    <embeddings-dir>/
        manifest.csv          row_index,image
        embeddings.npy        (N, 768)
        projected_512.npy     (N, 512)

Output:
    <output-root>/<split>/embeddings/
        embeddings.npy        (n_split, 768)
        projected_512.npy     (n_split, 512)
        y_embeddings.npy      (n_split,)   int64 binary labels
        manifest.csv          row_index,source_row,image,label
    <output-root>/split_metadata.json

Row alignment guarantee:
    row i of every array in a split corresponds to data row i of that split's
    manifest.csv, which records the original row in the flat manifest.

Labels come from one of two sources:

  1. --label-csv: a metadata CSV joined against the manifest image paths.
     The join key is configurable (full relative path, basename, or stem).

  2. --positive-pattern: a regex matched against the manifest image path.
     Rows that match get label 1, the rest get 0. This is the zero-metadata
     option (e.g. ffdm_diagnostic vs ffdm_screening).

Splits are random and stratified by label. Pass --group-regex to keep all
rows sharing a captured id (patient, study) inside one split, which avoids
leaking the same subject across train and test.

Example:
    python scripts/create_embed_splits.py \
        --embeddings-dir /gpfs/scratch/sd6701/EMBED_embeddings \
        --output-root datasets/embed \
        --positive-pattern 'ffdm_diagnostic' \
        --group-regex '([0-9]{6,})' \
        --train-frac 0.7 --valid-frac 0.1
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build train/valid/test splits from flat EMBED embeddings."
    )
    p.add_argument("--embeddings-dir", type=Path, required=True,
                   help="Directory holding manifest.csv and embeddings.npy.")
    p.add_argument("--output-root", type=Path, required=True,
                   help="Destination root, e.g. datasets/embed.")
    p.add_argument("--arrays", nargs="+", default=["embeddings.npy", "projected_512.npy"],
                   help="Which flat arrays to slice into each split.")

    # Label source: exactly one of these two.
    p.add_argument("--label-csv", type=Path, default=None,
                   help="Metadata CSV mapping images to labels.")
    p.add_argument("--label-csv-key-column", type=str, default=None,
                   help="Column in --label-csv holding the image identifier.")
    p.add_argument("--label-column", type=str, default="label",
                   help="Column in --label-csv holding the label.")
    p.add_argument("--label-positive-values", nargs="+", default=None,
                   help="Label values mapped to 1; everything else maps to 0. "
                        "Omit for a column that is already 0/1.")
    p.add_argument("--join-on", choices=["path", "basename", "stem"], default="stem",
                   help="How to match manifest images to --label-csv keys.")
    p.add_argument("--drop-unlabeled", action="store_true",
                   help="Drop manifest rows with no CSV match instead of failing.")

    p.add_argument("--positive-pattern", type=str, default=None,
                   help="Regex on the image path; matches get label 1.")

    p.add_argument("--group-regex", type=str, default=None,
                   help="Regex with one capture group on the image path. Rows "
                        "sharing a captured value stay in the same split.")
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--valid-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None,
                   help="Use only the first N manifest rows (smoke tests).")
    p.add_argument("--chunk-rows", type=int, default=8192,
                   help="Rows copied per write when slicing the flat arrays.")
    p.add_argument("--overwrite", action="store_true")

    args = p.parse_args()
    if (args.label_csv is None) == (args.positive_pattern is None):
        p.error("Pass exactly one of --label-csv or --positive-pattern.")
    if args.label_csv is not None and args.label_csv_key_column is None:
        p.error("--label-csv requires --label-csv-key-column.")
    if not 0.0 < args.train_frac < 1.0 or not 0.0 <= args.valid_frac < 1.0:
        p.error("--train-frac and --valid-frac must lie in [0, 1).")
    if args.train_frac + args.valid_frac >= 1.0:
        p.error("--train-frac + --valid-frac must leave room for a test split.")
    return args


def read_manifest(manifest_path: Path, limit: int | None) -> list[str]:
    with manifest_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if "image" not in (reader.fieldnames or []):
            raise ValueError(f"{manifest_path} has no 'image' column.")
        images = [row["image"] for row in reader]
    if not images:
        raise RuntimeError(f"{manifest_path} has no data rows.")
    if limit is not None:
        images = images[:limit]
    print(f"[manifest] {len(images)} rows from {manifest_path}")
    return images


def join_key(image: str, mode: str) -> str:
    path = Path(image)
    if mode == "path":
        return image
    if mode == "basename":
        return path.name
    return path.stem


def labels_from_pattern(images: list[str], pattern: str) -> np.ndarray:
    regex = re.compile(pattern)
    labels = np.fromiter(
        (1 if regex.search(image) else 0 for image in images),
        dtype=np.int64,
        count=len(images),
    )
    print(f"[labels] pattern {pattern!r}: "
          f"{int(labels.sum())} positive / {len(labels)} total")
    return labels


def labels_from_csv(
    images: list[str], args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, kept_row_indices) for rows found in the label CSV."""
    positives = set(args.label_positive_values) if args.label_positive_values else None
    lookup: dict[str, int] = {}
    with args.label_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        for column in (args.label_csv_key_column, args.label_column):
            if column not in fields:
                raise ValueError(
                    f"{args.label_csv} has no column {column!r}; columns: {fields}"
                )
        for row in reader:
            key = join_key(row[args.label_csv_key_column].strip(), args.join_on)
            raw = row[args.label_column].strip()
            if positives is not None:
                lookup[key] = int(raw in positives)
            else:
                lookup[key] = int(float(raw) > 0.5)
    print(f"[labels] {len(lookup)} keys from {args.label_csv}")

    kept: list[int] = []
    values: list[int] = []
    missing = 0
    for row_index, image in enumerate(images):
        label = lookup.get(join_key(image, args.join_on))
        if label is None:
            missing += 1
            continue
        kept.append(row_index)
        values.append(label)

    if missing and not args.drop_unlabeled:
        raise RuntimeError(
            f"{missing} manifest rows had no match in {args.label_csv}. "
            "Check --join-on / --label-csv-key-column, or pass --drop-unlabeled."
        )
    if missing:
        print(f"[labels] dropped {missing} unlabeled rows")
    labels = np.asarray(values, dtype=np.int64)
    print(f"[labels] {int(labels.sum())} positive / {len(labels)} labeled")
    return labels, np.asarray(kept, dtype=np.int64)


def stratified_split(
    labels: np.ndarray, train_frac: float, valid_frac: float, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Split row positions, keeping the class balance of each split equal."""
    splits: dict[str, list[np.ndarray]] = {"train": [], "valid": [], "test": []}
    for value in np.unique(labels):
        positions = np.flatnonzero(labels == value)
        rng.shuffle(positions)
        n_train = int(round(train_frac * len(positions)))
        n_valid = int(round(valid_frac * len(positions)))
        n_valid = min(n_valid, len(positions) - n_train)
        splits["train"].append(positions[:n_train])
        splits["valid"].append(positions[n_train:n_train + n_valid])
        splits["test"].append(positions[n_train + n_valid:])
    return {name: np.sort(np.concatenate(parts)) for name, parts in splits.items()}


def grouped_split(
    images: list[str],
    labels: np.ndarray,
    group_regex: str,
    train_frac: float,
    valid_frac: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Assign whole groups to splits so no group spans two splits."""
    regex = re.compile(group_regex)
    members: dict[str, list[int]] = defaultdict(list)
    for position, image in enumerate(images):
        match = regex.search(image)
        # Rows with no group id are their own singleton group.
        key = match.group(1) if match else f"__ungrouped_{position}"
        members[key].append(position)

    group_keys = np.array(sorted(members))
    # Stratify groups by their majority label so class balance survives.
    group_label = np.array(
        [int(round(float(labels[members[key]].mean()))) for key in group_keys]
    )
    print(f"[split] {len(group_keys)} groups from regex {group_regex!r}")

    assignment = stratified_split(group_label, train_frac, valid_frac, rng)
    return {
        name: np.sort(
            np.concatenate(
                [np.asarray(members[key], dtype=np.int64) for key in group_keys[idx]]
            )
        )
        for name, idx in assignment.items()
    }


def write_split_array(
    source_path: Path,
    dest_path: Path,
    source_rows: np.ndarray,
    chunk_rows: int,
    overwrite: bool,
) -> tuple[int, int]:
    if dest_path.exists() and not overwrite:
        raise FileExistsError(f"{dest_path} exists; pass --overwrite to rebuild.")
    source = np.load(source_path, mmap_mode="r")
    dest = open_memmap(
        dest_path,
        mode="w+",
        dtype=source.dtype,
        shape=(len(source_rows), int(source.shape[1])),
    )
    for start in range(0, len(source_rows), chunk_rows):
        stop = min(start + chunk_rows, len(source_rows))
        dest[start:stop] = source[source_rows[start:stop]]
    dest.flush()
    del dest
    return len(source_rows), int(source.shape[1])


def main() -> None:
    args = parse_args()
    emb_dir: Path = args.embeddings_dir
    images = read_manifest(emb_dir / "manifest.csv", args.limit)

    if args.positive_pattern is not None:
        labels = labels_from_pattern(images, args.positive_pattern)
        source_rows_all = np.arange(len(images), dtype=np.int64)
    else:
        labels, source_rows_all = labels_from_csv(images, args)
        images = [images[i] for i in source_rows_all]

    if labels.size == 0:
        raise RuntimeError("No labeled rows remain; nothing to split.")
    if len(np.unique(labels)) < 2:
        raise RuntimeError(
            "Labels are single-class after joining. Binary GPC needs both "
            "classes; check the label source."
        )

    rng = np.random.default_rng(args.seed)
    if args.group_regex:
        split_positions = grouped_split(
            images, labels, args.group_regex, args.train_frac, args.valid_frac, rng
        )
    else:
        split_positions = stratified_split(
            labels, args.train_frac, args.valid_frac, rng
        )

    summary: dict[str, dict] = {}
    for split, positions in split_positions.items():
        if positions.size == 0:
            raise RuntimeError(f"Split {split!r} is empty; adjust the fractions.")
        out_dir = args.output_root / split / "embeddings"
        out_dir.mkdir(parents=True, exist_ok=True)
        source_rows = source_rows_all[positions]

        shapes = {}
        for array_name in args.arrays:
            source_path = emb_dir / array_name
            if not source_path.exists():
                print(f"[{split}] skipping missing {source_path}")
                continue
            n_rows, dim = write_split_array(
                source_path, out_dir / array_name, source_rows,
                args.chunk_rows, args.overwrite,
            )
            shapes[array_name] = [n_rows, dim]
            print(f"[{split}] wrote {out_dir / array_name}  ({n_rows}, {dim})")

        split_labels = labels[positions]
        np.save(out_dir / "y_embeddings.npy", split_labels)

        with (out_dir / "manifest.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row_index", "source_row", "image", "label"])
            for row_index, position in enumerate(positions):
                writer.writerow(
                    [row_index, int(source_rows_all[position]),
                     images[position], int(labels[position])]
                )

        summary[split] = {
            "n_rows": int(positions.size),
            "n_positive": int(split_labels.sum()),
            "positive_rate": float(split_labels.mean()),
            "arrays": shapes,
        }
        print(f"[{split}] labels: {int(split_labels.sum())} positive "
              f"/ {positions.size} ({split_labels.mean():.4f})")

    metadata = {
        "dataset": "embed",
        "source_embeddings_dir": str(emb_dir),
        "label_source": (
            {"type": "path_pattern", "pattern": args.positive_pattern}
            if args.positive_pattern
            else {
                "type": "csv",
                "csv": str(args.label_csv),
                "key_column": args.label_csv_key_column,
                "label_column": args.label_column,
                "positive_values": args.label_positive_values,
                "join_on": args.join_on,
            }
        ),
        "group_regex": args.group_regex,
        "train_frac": args.train_frac,
        "valid_frac": args.valid_frac,
        "seed": args.seed,
        "splits": summary,
        "alignment": (
            "row i of each split array matches data row i of that split's "
            "manifest.csv (0-indexed)."
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "split_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"\nWrote {args.output_root / 'split_metadata.json'}")


if __name__ == "__main__":
    main()
