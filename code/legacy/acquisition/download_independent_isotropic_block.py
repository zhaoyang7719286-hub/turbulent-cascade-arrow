#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from giverny.turbulence_dataset import turb_dataset
from giverny.turbulence_toolkit import getCutout


ROOT = Path("legacy_workspace")

DATASET = "isotropic1024coarse"
VARIABLE = "velocity"

X_RANGE = [385, 448]
Y_RANGE = [385, 448]
Z_RANGE = [385, 448]
T_START = 2501
T_END = 2600

OUT_NAME = "velocity_64cube_rep2_xyz385_t2501_2600"
OUT = ROOT / "datasets" / DATASET / OUT_NAME
OUT_FRAMES = OUT / "frames_npz"
OUT_META = OUT / "metadata"

EXPECTED_SHAPE = (64, 64, 64, 3)
DX = float(2.0 * np.pi / 1024.0)

MANIFEST = OUT_META / "manifest.csv"
SUMMARY = OUT_META / "summary.json"
DOWNLOAD_METADATA = OUT_META / "download_metadata.json"


def validate_velocity_file(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        with np.load(path, allow_pickle=False) as z:
            if "velocity" not in z.files:
                return False, "missing velocity"
            velocity = z["velocity"]
            if tuple(velocity.shape) != EXPECTED_SHAPE:
                return False, f"shape={velocity.shape}"
            if velocity.dtype != np.float32:
                return False, f"dtype={velocity.dtype}"
            if not np.all(np.isfinite(velocity)):
                return False, "nonfinite"
        return True, "ok"
    except Exception as exc:
        return False, repr(exc)


def normalize_cutout_array(array: np.ndarray) -> np.ndarray:
    out = np.asarray(array)

    # A one-time cutout should normally already be (z,y,x,component).
    # Remove only singleton time dimensions, never spatial dimensions.
    while out.ndim > 4 and 1 in out.shape[:-1]:
        axis = next(i for i, size in enumerate(out.shape[:-1]) if size == 1)
        out = np.squeeze(out, axis=axis)

    if tuple(out.shape) != EXPECTED_SHAPE:
        raise RuntimeError(
            f"Unexpected cutout shape {out.shape}; expected {EXPECTED_SHAPE}."
        )

    out = np.asarray(out, dtype=np.float32)
    if not np.all(np.isfinite(out)):
        raise RuntimeError("Cutout contains NaN or Inf.")
    return out


def rewrite_manifest(rows: list[dict]) -> None:
    fields = [
        "frame_index",
        "status",
        "file",
        "shape",
        "dtype",
        "min",
        "max",
        "mean",
        "std",
        "elapsed_sec",
        "bytes",
        "message",
    ]
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    token = os.environ.get("JHTDB_TOKEN", "").strip()
    if not token:
        raise RuntimeError("JHTDB_TOKEN is not set.")

    OUT_FRAMES.mkdir(parents=True, exist_ok=True)
    OUT_META.mkdir(parents=True, exist_ok=True)

    metadata = {
        "dataset": DATASET,
        "variable": VARIABLE,
        "x_range": X_RANGE,
        "y_range": Y_RANGE,
        "z_range": Z_RANGE,
        "t_start": T_START,
        "t_end": T_END,
        "n_frames": T_END - T_START + 1,
        "expected_shape_per_frame": list(EXPECTED_SHAPE),
        "dx": DX,
        "storage_axis_order": "z,y,x,component",
        "purpose": (
            "Independent spatial-time replication block for the "
            "phase-organization finite-time cascade-arrow experiment."
        ),
        "independence_from_primary_block": {
            "primary_xyz": [[1, 64], [1, 64], [1, 64]],
            "primary_t": [1, 100],
            "replication_xyz": [X_RANGE, Y_RANGE, Z_RANGE],
            "replication_t": [T_START, T_END],
        },
    }
    DOWNLOAD_METADATA.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 100)
    print("DOWNLOAD INDEPENDENT ISOTROPIC BLOCK")
    print("=" * 100)
    print("dataset =", DATASET)
    print("xyz     =", X_RANGE, Y_RANGE, Z_RANGE)
    print("time    =", T_START, T_END)
    print("output  =", OUT)
    print("token   = [hidden]")

    cube = turb_dataset(
        dataset_title=DATASET,
        output_path=str(OUT),
        auth_token=token,
    )
    strides = np.array([1, 1, 1, 1], dtype=np.int32)

    rows: list[dict] = []
    failures: list[int] = []

    for frame_index in range(T_START, T_END + 1):
        destination = OUT_FRAMES / f"velocity_t{frame_index:05d}.npz"
        valid, message = validate_velocity_file(destination)
        if valid:
            with np.load(destination, allow_pickle=False) as z:
                velocity = z["velocity"]
                rows.append(
                    {
                        "frame_index": frame_index,
                        "status": "existing",
                        "file": destination.name,
                        "shape": "x".join(map(str, velocity.shape)),
                        "dtype": str(velocity.dtype),
                        "min": float(velocity.min()),
                        "max": float(velocity.max()),
                        "mean": float(velocity.mean(dtype=np.float64)),
                        "std": float(velocity.std(dtype=np.float64)),
                        "elapsed_sec": 0.0,
                        "bytes": destination.stat().st_size,
                        "message": message,
                    }
                )
            print(f"[skip] t={frame_index}: existing valid frame")
            continue

        started = time.perf_counter()
        try:
            ranges = np.array(
                [X_RANGE, Y_RANGE, Z_RANGE, [frame_index, frame_index]],
                dtype=np.int32,
            )
            result = getCutout(
                cube,
                VARIABLE,
                ranges,
                strides,
                trace_memory=False,
                verbose=False,
            )
            if not hasattr(result, "data_vars"):
                raise RuntimeError("getCutout returned no data_vars.")
            data_vars = list(result.data_vars)
            if not data_vars:
                raise RuntimeError("getCutout returned an empty dataset.")

            data_array = result[data_vars[0]]
            velocity = normalize_cutout_array(data_array.values)

            np.savez_compressed(
                destination,
                velocity=velocity,
                frame_index=np.int32(frame_index),
                dataset=np.array(DATASET),
                variable=np.array(VARIABLE),
                x_range=np.asarray(X_RANGE, dtype=np.int32),
                y_range=np.asarray(Y_RANGE, dtype=np.int32),
                z_range=np.asarray(Z_RANGE, dtype=np.int32),
                t_range=np.asarray([frame_index, frame_index], dtype=np.int32),
                dx=np.float64(DX),
            )

            valid, message = validate_velocity_file(destination)
            if not valid:
                raise RuntimeError(f"Saved-frame validation failed: {message}")

            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "frame_index": frame_index,
                    "status": "downloaded",
                    "file": destination.name,
                    "shape": "x".join(map(str, velocity.shape)),
                    "dtype": str(velocity.dtype),
                    "min": float(velocity.min()),
                    "max": float(velocity.max()),
                    "mean": float(velocity.mean(dtype=np.float64)),
                    "std": float(velocity.std(dtype=np.float64)),
                    "elapsed_sec": elapsed,
                    "bytes": destination.stat().st_size,
                    "message": "ok",
                }
            )
            print(
                f"[ok] t={frame_index} "
                f"elapsed={elapsed:.2f}s "
                f"mean={rows[-1]['mean']:.6e}"
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            failures.append(frame_index)
            rows.append(
                {
                    "frame_index": frame_index,
                    "status": "failed",
                    "file": destination.name,
                    "shape": "",
                    "dtype": "",
                    "min": "",
                    "max": "",
                    "mean": "",
                    "std": "",
                    "elapsed_sec": elapsed,
                    "bytes": 0,
                    "message": repr(exc),
                }
            )
            print(f"[failed] t={frame_index}: {exc}")
            traceback.print_exc()

        rewrite_manifest(rows)

    rewrite_manifest(rows)

    valid_count = 0
    for frame_index in range(T_START, T_END + 1):
        path = OUT_FRAMES / f"velocity_t{frame_index:05d}.npz"
        valid_count += int(validate_velocity_file(path)[0])

    summary = {
        **metadata,
        "valid_frame_count": valid_count,
        "failed_frame_indices": failures,
        "complete": valid_count == metadata["n_frames"],
        "numpy_version": np.__version__,
        "numpy_path": np.__file__,
    }
    SUMMARY.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 100)
    print("DOWNLOAD RESULT")
    print("=" * 100)
    print("valid frames =", valid_count)
    print("expected     =", metadata["n_frames"])
    print("failures     =", failures)
    print("summary      =", SUMMARY)

    if valid_count != metadata["n_frames"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
