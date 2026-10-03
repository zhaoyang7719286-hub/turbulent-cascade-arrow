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
DATASET = "channel"
VARIABLE = "velocity"

# Held-out confirmation block, fixed before formal analysis.
X_RANGE = [1153, 1216]
Y_RANGE = [225, 288]
Z_RANGE = [961, 1024]
T_START = 3001
T_END = 3100

OUT_NAME = "heldout_x1153_1216_y225_288_z961_1024_t3001_3100"
OUT = ROOT / "datasets" / DATASET / OUT_NAME
OUT_FRAMES = OUT / "frames_npz"
OUT_META = OUT / "metadata"

EXPECTED_SHAPE = (64, 64, 64, 3)  # z,y,x,component
MANIFEST = OUT_META / "manifest.csv"
SUMMARY = OUT_META / "summary.json"


def coordinate(data_array, candidates: tuple[str, ...]) -> np.ndarray:
    for name in candidates:
        if name in data_array.coords:
            return np.asarray(
                data_array.coords[name].values,
                dtype=np.float64,
            ).reshape(-1)
    raise RuntimeError(
        f"Missing coordinate {candidates}; available={list(data_array.coords)}"
    )


def normalize_velocity(array: np.ndarray) -> np.ndarray:
    out = np.asarray(array)
    while out.ndim > 4 and 1 in out.shape[:-1]:
        axis = next(
            i for i, size in enumerate(out.shape[:-1])
            if size == 1
        )
        out = np.squeeze(out, axis=axis)

    if tuple(out.shape) != EXPECTED_SHAPE:
        raise RuntimeError(
            f"Unexpected shape {out.shape}; expected {EXPECTED_SHAPE}."
        )

    out = np.asarray(out, dtype=np.float32)
    if not np.all(np.isfinite(out)):
        raise RuntimeError("Velocity contains NaN or Inf.")
    return out


def validate(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        with np.load(path, allow_pickle=False) as z:
            required = {"velocity", "xcoor", "ycoor", "zcoor"}
            missing = required.difference(z.files)
            if missing:
                return False, f"missing keys {sorted(missing)}"
            if tuple(z["velocity"].shape) != EXPECTED_SHAPE:
                return False, f"shape={z['velocity'].shape}"
            if z["velocity"].dtype != np.float32:
                return False, f"dtype={z['velocity'].dtype}"
            for key in required:
                if not np.all(np.isfinite(z[key])):
                    return False, f"nonfinite {key}"
        return True, "ok"
    except Exception as exc:
        return False, repr(exc)


def write_manifest(rows: list[dict]) -> None:
    fields = [
        "frame_index", "status", "file", "shape", "dtype",
        "min", "max", "mean", "std", "elapsed_sec",
        "bytes", "message",
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
        "expected_shape": list(EXPECTED_SHAPE),
        "storage_axis_order": "z,y,x,component",
        "status": "pre-registered held-out channel confirmation block",
        "pilot_block": {
            "x_range": [769, 832],
            "y_range": [225, 288],
            "z_range": [577, 640],
            "t_range": [2001, 2010],
        },
    }
    (OUT_META / "download_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print("DOWNLOAD HELD-OUT CHANNEL BLOCK")
    print("=" * 100)
    print("xyz    =", X_RANGE, Y_RANGE, Z_RANGE)
    print("time   =", T_START, T_END)
    print("output =", OUT)
    print("token  = [hidden]")

    cube = turb_dataset(
        dataset_title=DATASET,
        output_path=str(OUT),
        auth_token=token,
    )
    strides = np.array([1, 1, 1, 1], dtype=np.int32)

    rows: list[dict] = []
    failures: list[int] = []

    for t in range(T_START, T_END + 1):
        destination = OUT_FRAMES / f"velocity_t{t:05d}.npz"
        valid, message = validate(destination)

        if valid:
            with np.load(destination, allow_pickle=False) as z:
                velocity = z["velocity"]
            rows.append({
                "frame_index": t,
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
            })
            print(f"[skip] t={t}")
            continue

        started = time.perf_counter()
        try:
            ranges = np.array(
                [X_RANGE, Y_RANGE, Z_RANGE, [t, t]],
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

            variables = list(result.data_vars)
            if not variables:
                raise RuntimeError("getCutout returned no data variables.")

            data_array = result[variables[0]]
            velocity = normalize_velocity(data_array.values)
            xcoor = coordinate(data_array, ("xcoor", "x", "X"))
            ycoor = coordinate(data_array, ("ycoor", "y", "Y"))
            zcoor = coordinate(data_array, ("zcoor", "z", "Z"))

            if not (
                len(xcoor) == len(ycoor) == len(zcoor) == 64
            ):
                raise RuntimeError(
                    f"Coordinate lengths: "
                    f"x={len(xcoor)}, y={len(ycoor)}, z={len(zcoor)}"
                )

            np.savez_compressed(
                destination,
                velocity=velocity,
                frame_index=np.int32(t),
                dataset=np.array(DATASET),
                variable=np.array(VARIABLE),
                x_range=np.asarray(X_RANGE, dtype=np.int32),
                y_range=np.asarray(Y_RANGE, dtype=np.int32),
                z_range=np.asarray(Z_RANGE, dtype=np.int32),
                t_range=np.asarray([t, t], dtype=np.int32),
                xcoor=xcoor,
                ycoor=ycoor,
                zcoor=zcoor,
            )

            valid, message = validate(destination)
            if not valid:
                raise RuntimeError(f"Saved-frame validation failed: {message}")

            elapsed = time.perf_counter() - started
            rows.append({
                "frame_index": t,
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
            })
            print(f"[ok] t={t} elapsed={elapsed:.2f}s")
        except Exception as exc:
            elapsed = time.perf_counter() - started
            failures.append(t)
            rows.append({
                "frame_index": t,
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
            })
            print(f"[failed] t={t}: {exc}")
            traceback.print_exc()

        write_manifest(rows)

    write_manifest(rows)

    valid_count = sum(
        validate(OUT_FRAMES / f"velocity_t{t:05d}.npz")[0]
        for t in range(T_START, T_END + 1)
    )

    summary = {
        **metadata,
        "valid_frame_count": int(valid_count),
        "failed_frame_indices": failures,
        "complete": bool(valid_count == metadata["n_frames"]),
        "numpy_version": np.__version__,
        "numpy_path": np.__file__,
    }
    SUMMARY.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print("HELD-OUT CHANNEL DOWNLOAD RESULT")
    print("=" * 100)
    print("valid frames =", valid_count)
    print("failures     =", failures)
    print("summary      =", SUMMARY)

    if valid_count != metadata["n_frames"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
