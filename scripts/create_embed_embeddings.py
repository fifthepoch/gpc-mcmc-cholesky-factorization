#!/usr/bin/env python3
"""
Create Phikon embeddings (768-d) and PCA-projected embeddings (e.g. 512-d)
for the EMBED mammography dataset.

Unlike create_phikon_embeddings.py, this script does not assume a
<split>/images + labels.csv layout. It recursively discovers image files
(.png/.jpg/.jpeg/.dcm) under --data-root, freezes their order into a
manifest.csv, and writes:

    <output-dir>/
        manifest.csv                  row i -> image path (relative to data-root)
        embeddings.npy                (N, 768) Phikon features
        metadata.json
        progress.json                 resume checkpoint
        projected_<D>.npy             (N, D) PCA projection (if --project-dim)
        projected_<D>_metadata.json
        projected_<D>_progress.json
        pca_<D>.npz                   fitted PCA components/mean

Row alignment guarantee:
    embeddings.npy[i] and projected_<D>.npy[i] correspond to row i+1 of
    manifest.csv (excluding header).

Usage:
    python scripts/create_embed_embeddings.py \
        --data-root /gpfs/scratch/wh2757/EMBED \
        --output-dir /gpfs/scratch/wh2757/EMBED/embeddings \
        --project-dim 512
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap
from PIL import Image
from tqdm.auto import tqdm

import torch


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm", ".hdf5", ".h5"}

HDF5_PREFERRED_KEYS = ("image", "img", "data", "pixels", "pixel_array", "x", "scan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Phikon embeddings for the EMBED dataset."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing the EMBED images (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write embeddings (default: <data-root>/embeddings).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="owkin/phikon",
        help="Hugging Face model name to use for feature extraction.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of images to encode per forward pass.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Storage dtype for embeddings.npy.",
    )
    parser.add_argument(
        "--feature-pooling",
        choices=["cls", "mean"],
        default="cls",
        help="How to reduce token features into one vector per image.",
    )
    parser.add_argument(
        "--project-dim",
        type=int,
        default=512,
        help=(
            "Projected embedding size (PCA fitted on the extracted embeddings). "
            "Pass 0 to skip projection and keep only the 768-d embeddings."
        ),
    )
    parser.add_argument(
        "--projection-batch-size",
        type=int,
        default=2048,
        help="Batch size while fitting and applying the PCA projection.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=sorted(IMAGE_EXTENSIONS),
        help="Image file extensions to include during discovery.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N discovered images (smoke tests).",
    )
    parser.add_argument(
        "--hdf5-key",
        type=str,
        default=None,
        help=(
            "Dataset key holding the image inside .hdf5/.h5 files. "
            "Default: auto-detect (common key names, then the first 2-D dataset)."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory override.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Flush arrays and update progress.json every N rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate the manifest and all embedding outputs from scratch.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return json.load(handle)


def discover_images(
    data_root: Path, output_dir: Path, extensions: list[str], limit: int | None
) -> list[str]:
    """Recursively find image files under data_root, sorted for determinism."""
    wanted = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    found: list[str] = []
    output_resolved = output_dir.resolve()
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in wanted:
            continue
        # Never re-embed our own outputs if they live inside data_root.
        if output_resolved in path.resolve().parents:
            continue
        found.append(str(path.relative_to(data_root)))
        if limit is not None and len(found) >= limit:
            break
    return found


def load_or_create_manifest(
    manifest_path: Path,
    data_root: Path,
    output_dir: Path,
    extensions: list[str],
    limit: int | None,
    overwrite: bool,
) -> list[str]:
    if manifest_path.exists() and not overwrite:
        with manifest_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            if "image" not in (reader.fieldnames or []):
                raise ValueError(f"{manifest_path} is missing an 'image' column.")
            images = [row["image"] for row in reader]
        if not images:
            raise RuntimeError(f"No rows found in existing manifest {manifest_path}")
        print(f"[manifest] reusing {manifest_path} with {len(images)} rows")
        return images

    print(f"[manifest] scanning {data_root} for images ...")
    images = discover_images(data_root, output_dir, extensions, limit)
    if not images:
        raise RuntimeError(
            f"No image files with extensions {sorted(extensions)} found under {data_root}"
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "image"])
        for row_index, image_rel in enumerate(images):
            writer.writerow([row_index, image_rel])
    print(f"[manifest] wrote {manifest_path} with {len(images)} rows")
    return images


def array_to_rgb(arr: np.ndarray) -> Image.Image:
    """Min-max normalize a raw pixel array to uint8 and convert to RGB."""
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim == 3:
        # Mammograms are grayscale: reduce channels-last (H, W, C) to the
        # first channel, otherwise assume a leading stack/channel dim.
        if arr.shape[-1] in (3, 4):
            arr = arr[..., 0]
        else:
            arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Cannot interpret array of shape {arr.shape} as an image.")
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)
    arr8 = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr8).convert("RGB")


def load_hdf5_as_rgb(path: Path, hdf5_key: str | None) -> Image.Image:
    import h5py

    with h5py.File(path, "r") as handle:
        if hdf5_key:
            if hdf5_key not in handle:
                raise KeyError(
                    f"--hdf5-key '{hdf5_key}' not found in {path}; "
                    f"top-level keys: {list(handle.keys())}"
                )
            return array_to_rgb(handle[hdf5_key][()])
        for key in HDF5_PREFERRED_KEYS:
            if key in handle and isinstance(handle[key], h5py.Dataset):
                return array_to_rgb(handle[key][()])
        candidates: list[str] = []

        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
                candidates.append(name)

        handle.visititems(visit)
        if not candidates:
            raise ValueError(
                f"No 2-D dataset found in {path}; top-level keys: {list(handle.keys())}"
            )
        return array_to_rgb(handle[candidates[0]][()])


def load_image_as_rgb(path: Path, hdf5_key: str | None = None) -> Image.Image:
    """Load a PNG/JPEG, DICOM, or single-image HDF5 file as an RGB PIL image."""
    suffix = path.suffix.lower()
    if suffix in (".hdf5", ".h5"):
        return load_hdf5_as_rgb(path, hdf5_key)

    if suffix == ".dcm":
        try:
            import pydicom
            from pydicom.pixel_data_handlers.util import apply_voi_lut
        except Exception as exc:
            raise RuntimeError(
                "Reading DICOM files requires `pydicom` (pip install pydicom)."
            ) from exc
        ds = pydicom.dcmread(path)
        arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            arr = arr.max() - arr
        return array_to_rgb(arr)

    with Image.open(path) as image:
        return image.convert("RGB")


def load_processor_and_model(model_name: str, cache_dir: Path | None):
    from transformers import AutoImageProcessor, ViTModel

    processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    model = ViTModel.from_pretrained(
        model_name,
        add_pooling_layer=False,
        cache_dir=cache_dir,
    )
    return processor, model


def get_embedding_dtype(dtype_name: str) -> np.dtype:
    return np.float16 if dtype_name == "float16" else np.float32


def prepare_embedding_array(
    array_path: Path,
    rows_total: int,
    embedding_dim: int,
    dtype: np.dtype,
    overwrite: bool,
) -> np.memmap:
    mode = "r+"
    if overwrite or not array_path.exists():
        mode = "w+"
    return open_memmap(
        array_path,
        mode=mode,
        dtype=dtype,
        shape=(rows_total, embedding_dim),
    )


def initial_completed_rows(progress_path: Path, rows_total: int, overwrite: bool) -> int:
    if overwrite or not progress_path.exists():
        return 0
    progress = load_json(progress_path)
    completed_rows = int(progress.get("rows_completed", 0))
    return max(0, min(completed_rows, rows_total))


def validate_or_write_metadata(
    metadata_path: Path, payload: dict[str, Any], overwrite: bool
) -> None:
    if metadata_path.exists() and not overwrite:
        existing = load_json(metadata_path)
        for key in ["model_name", "feature_pooling", "embedding_dim", "dtype", "rows_total"]:
            if key in payload and existing.get(key) != payload.get(key):
                raise RuntimeError(
                    f"Existing metadata at {metadata_path} does not match current "
                    f"run for key '{key}'. Use --overwrite to rebuild."
                )
        return
    save_json(metadata_path, payload)


@torch.no_grad()
def extract_batch_embeddings(
    images: list[Image.Image],
    processor,
    model,
    device: torch.device,
    feature_pooling: str,
) -> np.ndarray:
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    token_features = outputs.last_hidden_state
    if feature_pooling == "cls":
        batch_embeddings = token_features[:, 0, :]
    elif token_features.shape[1] > 1:
        batch_embeddings = token_features[:, 1:, :].mean(dim=1)
    else:
        batch_embeddings = token_features[:, 0, :]
    return batch_embeddings.detach().cpu().numpy()


def extract_embeddings(
    images: list[str],
    data_root: Path,
    output_dir: Path,
    processor,
    model,
    embedding_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    array_path = output_dir / "embeddings.npy"
    metadata_path = output_dir / "metadata.json"
    progress_path = output_dir / "progress.json"

    validate_or_write_metadata(
        metadata_path,
        {
            "dataset": "embed",
            "data_root": str(data_root),
            "model_name": args.model_name,
            "feature_pooling": args.feature_pooling,
            "embedding_dim": embedding_dim,
            "dtype": args.dtype,
            "rows_total": len(images),
            "alignment": "embeddings.npy row i matches manifest.csv data row i (0-indexed).",
            "manifest_file": "manifest.csv",
            "image_column": "image",
        },
        overwrite=args.overwrite,
    )

    embeddings = prepare_embedding_array(
        array_path,
        rows_total=len(images),
        embedding_dim=embedding_dim,
        dtype=get_embedding_dtype(args.dtype),
        overwrite=args.overwrite,
    )
    completed_rows = initial_completed_rows(
        progress_path, rows_total=len(images), overwrite=args.overwrite
    )

    if completed_rows >= len(images):
        print(f"[embed] embeddings already complete at {array_path}")
        return array_path

    if completed_rows > 0:
        print(f"[embed] resuming from row {completed_rows + 1} / {len(images)}")
    else:
        print(f"[embed] starting from row 1 / {len(images)}")

    progress = tqdm(
        total=len(images),
        initial=completed_rows,
        desc="embed:extract",
        unit="img",
        file=sys.stdout,
        dynamic_ncols=True,
    )
    for start in range(completed_rows, len(images), args.batch_size):
        stop = min(start + args.batch_size, len(images))
        batch_images = [
            load_image_as_rgb(data_root / rel, args.hdf5_key)
            for rel in images[start:stop]
        ]
        batch_embeddings = extract_batch_embeddings(
            batch_images, processor, model, device, args.feature_pooling
        ).astype(get_embedding_dtype(args.dtype), copy=False)
        embeddings[start:stop] = batch_embeddings
        progress.update(stop - start)
        if stop % args.checkpoint_every < args.batch_size or stop == len(images):
            embeddings.flush()
            save_json(
                progress_path,
                {
                    "dataset": "embed",
                    "model_name": args.model_name,
                    "rows_completed": stop,
                    "rows_total": len(images),
                },
            )
    progress.close()
    print(f"[embed] saved embeddings to {array_path}")
    return array_path


def fit_projection(
    array_path: Path, output_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    matrix_path = output_dir / f"pca_{args.project_dim}.npz"
    matrix_meta_path = output_dir / f"pca_{args.project_dim}_metadata.json"

    if matrix_path.exists() and matrix_meta_path.exists() and not args.overwrite:
        metadata = load_json(matrix_meta_path)
        projection = np.load(matrix_path)
        print(
            f"[embed] reusing existing PCA projection "
            f"{metadata['source_dim']} -> {metadata['projected_dim']} from {matrix_path}"
        )
        return {
            "source_dim": int(metadata["source_dim"]),
            "projected_dim": int(metadata["projected_dim"]),
            "components": projection["components"],
            "mean": projection["mean"],
        }

    from sklearn.decomposition import IncrementalPCA

    base_embeddings = np.load(array_path, mmap_mode="r")
    source_dim = int(base_embeddings.shape[1])
    rows_total = int(base_embeddings.shape[0])
    if args.project_dim <= 0 or args.project_dim > source_dim:
        raise ValueError(
            f"--project-dim must be between 1 and {source_dim}, got {args.project_dim}."
        )
    if rows_total < args.project_dim:
        raise ValueError(
            f"Only {rows_total} embeddings available, fewer than "
            f"--project-dim={args.project_dim}."
        )

    print(f"[embed] fitting PCA projection {source_dim} -> {args.project_dim}")
    fit_batch_size = max(args.projection_batch_size, args.project_dim)
    pca = IncrementalPCA(n_components=args.project_dim, batch_size=fit_batch_size)

    progress = tqdm(
        total=rows_total, desc="embed:fit-pca", unit="row",
        file=sys.stdout, dynamic_ncols=True,
    )
    start = 0
    while start < rows_total:
        stop = min(start + fit_batch_size, rows_total)
        remaining = rows_total - stop
        if 0 < remaining < args.project_dim:
            stop = rows_total
        batch = np.asarray(base_embeddings[start:stop], dtype=np.float32)
        pca.partial_fit(batch)
        progress.update(stop - start)
        start = stop
    progress.close()

    np.savez(
        matrix_path,
        components=pca.components_.astype(np.float32),
        mean=pca.mean_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
    )
    save_json(
        matrix_meta_path,
        {
            "dataset": "embed",
            "method": "pca",
            "fitted_on": "all extracted embeddings",
            "source_dim": source_dim,
            "projected_dim": args.project_dim,
            "projection_batch_size": args.projection_batch_size,
        },
    )
    return {
        "source_dim": source_dim,
        "projected_dim": args.project_dim,
        "components": pca.components_.astype(np.float32),
        "mean": pca.mean_.astype(np.float32),
    }


def apply_projection(
    array_path: Path,
    output_dir: Path,
    projection: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    projected_dim = int(projection["projected_dim"])
    out_array_path = output_dir / f"projected_{projected_dim}.npy"
    out_meta_path = output_dir / f"projected_{projected_dim}_metadata.json"
    out_progress_path = output_dir / f"projected_{projected_dim}_progress.json"

    base_embeddings = np.load(array_path, mmap_mode="r")
    rows_total = int(base_embeddings.shape[0])

    validate_or_write_metadata(
        out_meta_path,
        {
            "dataset": "embed",
            "method": "pca",
            "source_array": array_path.name,
            "source_dim": int(projection["source_dim"]),
            "projected_dim": projected_dim,
            "dtype": args.dtype,
            "rows_total": rows_total,
            "alignment": (
                f"projected_{projected_dim}.npy row i matches embeddings.npy row i "
                "and manifest.csv data row i (0-indexed)."
            ),
        },
        overwrite=args.overwrite,
    )

    projected = prepare_embedding_array(
        out_array_path,
        rows_total=rows_total,
        embedding_dim=projected_dim,
        dtype=get_embedding_dtype(args.dtype),
        overwrite=args.overwrite,
    )
    completed_rows = initial_completed_rows(
        out_progress_path, rows_total=rows_total, overwrite=args.overwrite
    )
    if completed_rows >= rows_total:
        print(f"[embed] projected embeddings already complete at {out_array_path}")
        return

    components = np.asarray(projection["components"], dtype=np.float32)
    mean = np.asarray(projection["mean"], dtype=np.float32)

    progress = tqdm(
        total=rows_total, initial=completed_rows,
        desc=f"embed:project_{projected_dim}", unit="row",
        file=sys.stdout, dynamic_ncols=True,
    )
    for start in range(completed_rows, rows_total, args.projection_batch_size):
        stop = min(start + args.projection_batch_size, rows_total)
        batch = np.asarray(base_embeddings[start:stop], dtype=np.float32)
        projected[start:stop] = ((batch - mean) @ components.T).astype(
            get_embedding_dtype(args.dtype), copy=False
        )
        progress.update(stop - start)
        projected.flush()
        save_json(
            out_progress_path,
            {"dataset": "embed", "rows_completed": stop, "rows_total": rows_total},
        )
    progress.close()
    print(f"[embed] saved projected embeddings to {out_array_path}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    data_root: Path = args.data_root
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    output_dir: Path = args.output_dir or (data_root / "embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    images = load_or_create_manifest(
        output_dir / "manifest.csv",
        data_root,
        output_dir,
        args.extensions,
        args.limit,
        args.overwrite,
    )

    progress_path = output_dir / "progress.json"
    already_done = (
        not args.overwrite
        and (output_dir / "embeddings.npy").exists()
        and initial_completed_rows(progress_path, len(images), False) >= len(images)
    )

    processor = model = None
    if already_done:
        embedding_dim = int(np.load(output_dir / "embeddings.npy", mmap_mode="r").shape[1])
    else:
        processor, model = load_processor_and_model(args.model_name, args.cache_dir)
        model = model.to(device)
        model.eval()
        embedding_dim = int(model.config.hidden_size)

    print("==============================================================================")
    print("EMBED Phikon Embedding Extraction")
    print("==============================================================================")
    print(f"Data root       : {data_root}")
    print(f"Output dir      : {output_dir}")
    print(f"Images          : {len(images)}")
    print(f"Model           : {args.model_name}")
    print(f"Embedding dim   : {embedding_dim}")
    print(f"Projected dim   : {args.project_dim if args.project_dim else 'disabled'}")
    print(f"Storage dtype   : {args.dtype}")
    print(f"Device          : {device}")
    print("==============================================================================")

    array_path = extract_embeddings(
        images, data_root, output_dir, processor, model, embedding_dim, args, device
    )

    if args.project_dim:
        projection = fit_projection(array_path, output_dir, args)
        apply_projection(array_path, output_dir, projection, args)


if __name__ == "__main__":
    main()
