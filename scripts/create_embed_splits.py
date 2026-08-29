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
    p.add_argument("--join-on", choices=["path", "basename", "stem", "regex"],
                   default="stem",
                   help="How to match manifest images to --label-csv keys. Use "
                        "'regex' to join on an id captured from the path (e.g. "
                        "the patient id), which is what EMBED needs when the "
                        "label table is per-patient rather than per-image.")
    p.add_argument("--join-regex", type=str, default=None,
                   help="Regex with one capture group, applied to the manifest "
                        "image path to build the join key. Required for "
                        "--join-on regex.")
    p.add_argument("--label-key-regex", type=str, default=None,
                   help="Regex with one capture group, applied to the "
                        "--label-csv key column. Defaults to --join-regex.")
    p.add_argument("--drop-unlabeled", action="store_true",
                   help="Drop manifest rows with no CSV match instead of failing.")

    p.add_argument("--inspect", action="store_true",
                   help="Report the join instead of building splits: prints the "
                        "CSV columns, sample keys from both sides, and how many "
                        "manifest rows would match. Writes nothing.")

    # Reuse an externally computed split assignment (e.g. a collaborator's
    # patient-stratified split) instead of drawing a fresh random one.
    p.add_argument("--split-csv", type=Path, default=None,
                   help="CSV assigning ids to train/valid/test. Overrides "
                        "--train-frac/--valid-frac/--group-regex.")
    p.add_argument("--split-key-column", type=str, default=None,
                   help="Column in --split-csv holding the id.")
    p.add_argument("--split-column", type=str, default="split",
                   help="Column in --split-csv holding the split name.")
    p.add_argument("--split-key-regex", type=str, default=None,
                   help="Regex with one capture group applied to the manifest "
                        "image path to build the --split-csv key. Defaults to "
                        "--join-regex.")

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

    if args.join_on == "regex" and not args.join_regex:
        p.error("--join-on regex requires --join-regex.")
    if args.label_key_regex is None:
        args.label_key_regex = args.join_regex
    if args.split_key_regex is None:
        args.split_key_regex = args.join_regex

    # --inspect only reads; it does not need a complete build configuration.
    if args.inspect:
        if args.label_csv is not None and args.label_csv_key_column is None:
            p.error("--inspect with --label-csv requires --label-csv-key-column.")
        return args

    if args.split_csv is not None and args.split_key_column is None:
        p.error("--split-csv requires --split-key-column.")

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


def make_key_fn(mode: str, pattern: str | None = None):
    """Return a callable mapping a path (or CSV cell) to a join key.

    Returns None for values the key cannot be derived from, which callers
    treat as 'no match' rather than as a key of their own.
    """
    if mode == "regex":
        if not pattern:
            raise ValueError("regex join mode requires a pattern")
        regex = re.compile(pattern)

        def from_regex(value: str) -> str | None:
            match = regex.search(value)
            return match.group(1) if match else None

        return from_regex

    def from_path(value: str) -> str | None:
        path = Path(value)
        if mode == "path":
            return value
        if mode == "basename":
            return path.name
        return path.stem

    return from_path


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
    manifest_key = make_key_fn(args.join_on, args.join_regex)
    csv_key = make_key_fn(args.join_on, args.label_key_regex)

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
            key = csv_key(row[args.label_csv_key_column].strip())
            if key is None:
                continue
            raw = row[args.label_column].strip()
            if positives is not None:
                label = int(raw in positives)
            else:
                label = int(float(raw) > 0.5)
            # A per-patient table repeats one id across many images. Keep the
            # positive if any row for that id is positive.
            lookup[key] = max(lookup.get(key, 0), label)
    print(f"[labels] {len(lookup)} keys from {args.label_csv}")

    kept: list[int] = []
    values: list[int] = []
    missing = 0
    for row_index, image in enumerate(images):
        key = manifest_key(image)
        label = lookup.get(key) if key is not None else None
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


def inspect_join(images: list[str], args: argparse.Namespace) -> None:
    """Report whether a candidate label CSV can actually be joined.

    Cheap dry run: prints both sides of the key and the overlap, so a bad
    --join-on is caught before a full pass over 293k rows.
    """
    manifest_key = make_key_fn(args.join_on, args.join_regex)
    manifest_keys = [manifest_key(image) for image in images]
    derived = [key for key in manifest_keys if key is not None]

    print(f"\n=== manifest ({len(images)} rows) ===")
    print(f"  sample paths : {images[:3]}")
    print(f"  join mode    : {args.join_on}"
          + (f"  regex={args.join_regex!r}" if args.join_on == "regex" else ""))
    print(f"  sample keys  : {derived[:5]}")
    print(f"  derived keys : {len(derived)} / {len(images)} rows "
          f"({len(set(derived))} distinct)")
    if len(derived) < len(images):
        print(f"  WARNING: {len(images) - len(derived)} rows yielded no key")

    if args.label_csv is None:
        print("\nNo --label-csv given; nothing to join against.")
        return

    csv_key = make_key_fn(args.join_on, args.label_key_regex)
    with args.label_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        print(f"\n=== {args.label_csv} ===")
        print(f"  columns: {fields}")
        if args.label_csv_key_column not in fields:
            print(f"  ERROR: no key column {args.label_csv_key_column!r}")
            return
        raw_samples: list[str] = []
        keys: set[str] = set()
        label_values: dict[str, int] = defaultdict(int)
        for row in reader:
            raw = row[args.label_csv_key_column].strip()
            if len(raw_samples) < 3:
                raw_samples.append(raw)
            key = csv_key(raw)
            if key is not None:
                keys.add(key)
            if args.label_column in fields:
                label_values[row[args.label_column].strip()] += 1

    print(f"  sample {args.label_csv_key_column}: {raw_samples}")
    print(f"  distinct join keys: {len(keys)}")
    print(f"  sample keys       : {sorted(keys)[:5]}")
    if label_values:
        top = sorted(label_values.items(), key=lambda kv: -kv[1])[:12]
        print(f"  {args.label_column} value counts: {top}")

    matched = sum(1 for key in manifest_keys if key is not None and key in keys)
    print(f"\n=== overlap ===")
    print(f"  manifest rows matched: {matched} / {len(images)} "
          f"({matched / max(len(images), 1):.1%})")
    if matched == 0:
        print("  The two key spaces do not intersect. Change --join-on / "
              "--join-regex / --label-key-regex before building splits.")


def splits_from_csv(
    images: list[str], args: argparse.Namespace
) -> dict[str, np.ndarray]:
    """Assign splits from an external assignment file rather than at random."""
    key_fn = make_key_fn(
        "regex" if args.split_key_regex else args.join_on, args.split_key_regex
    )
    aliases = {"val": "valid", "validation": "valid", "dev": "valid"}

    assignment: dict[str, str] = {}
    with args.split_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        for column in (args.split_key_column, args.split_column):
            if column not in fields:
                raise ValueError(
                    f"{args.split_csv} has no column {column!r}; columns: {fields}"
                )
        for row in reader:
            key = key_fn(row[args.split_key_column].strip())
            if key is None:
                continue
            name = row[args.split_column].strip().lower()
            assignment[key] = aliases.get(name, name)

    print(f"[split] {len(assignment)} ids from {args.split_csv}")

    positions: dict[str, list[int]] = {"train": [], "valid": [], "test": []}
    unassigned = 0
    unknown: set[str] = set()
    for position, image in enumerate(images):
        key = key_fn(image)
        name = assignment.get(key) if key is not None else None
        if name is None:
            unassigned += 1
            continue
        if name not in positions:
            unknown.add(name)
            continue
        positions[name].append(position)

    if unknown:
        raise ValueError(
            f"{args.split_csv} has unrecognized split names {sorted(unknown)}; "
            "expected train/valid/test."
        )
    if unassigned:
        print(f"[split] WARNING: {unassigned} rows had no split assignment "
              "and were dropped")
    if all(len(v) == 0 for v in positions.values()):
        raise RuntimeError(
            f"No manifest row matched an id in {args.split_csv}. Check "
            "--split-key-column / --split-key-regex."
        )
    return {
        name: np.asarray(sorted(rows), dtype=np.int64)
        for name, rows in positions.items()
    }


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

    if args.inspect:
        inspect_join(images, args)
        return

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
    if args.split_csv is not None:
        split_positions = splits_from_csv(images, args)
    elif args.group_regex:
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
                "join_regex": args.join_regex,
                "label_key_regex": args.label_key_regex,
            }
        ),
        "split_source": (
            {
                "type": "external_csv",
                "csv": str(args.split_csv),
                "key_column": args.split_key_column,
                "split_column": args.split_column,
                "key_regex": args.split_key_regex,
            }
            if args.split_csv is not None
            else {"type": "grouped" if args.group_regex else "stratified"}
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
