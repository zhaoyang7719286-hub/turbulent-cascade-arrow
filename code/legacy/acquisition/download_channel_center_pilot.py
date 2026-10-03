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

X_RANGE = [769, 832]
Y_RANGE = [225, 288]
Z_RANGE = [577, 640]
T_START = 2001
T_END = 2010

OUT_NAME = "center_pilot_x769_832_y225_288_z577_640_t2001_2010"
OUT = ROOT / "datasets" / DATASET / OUT_NAME
OUT_FRAMES = OUT / "frames_npz"
OUT_META = OUT / "metadata"

EXPECTED_SHAPE = (64, 64, 64, 3)
MANIFEST = OUT_META / "manifest.csv"
SUMMARY = OUT_META / "summary.json"


def coordinate_from_data_array(data_array, names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        if name in data_array.coords:
            values = np.asarray(data_array.coords[name].values, dtype=np.float64)
            return values.reshape(-1)
    raise RuntimeError(
        f"None of the coordinate names {names} are present. "
        f"Available coordinates: {list(data_array.coords)}"
    )


def normalize_cutout_array(array: np.ndarray) -> np.ndarray:
    out = np.asarray(array)
    while out.ndim > 4 and 1 in out.shape[:-1]:
        axis = next(i for i, size in enumerate(out.shape[:-1]) if size == 1)
        out = np.squeeze(out, axis=axis)
    if tuple(out.shape) != EXPECTED_SHAPE:
        raise RuntimeError(
            f"Unexpected cutout shape {out.shape}; expected {EXPECTED_SHAPE}."
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
            if not required.issubset(z.files):
                return False, f"missing keys {sorted(required - set(z.files))}"
            if tuple(z["velocity"].shape) != EXPECTED_SHAPE:
                return False, f"shape={z['velocity'].shape}"
            for key in required:
                if not np.all(np.isfinite(z[key])):
                    return False, f"nonfinite {key}"
        return True, "ok"
    except Exception as exc:
        return False, repr(exc)


def write_manifest(rows: list[dict]) -> None:
    fields = [
        "frame_index", "status", "file", "shape", "dtype",
        "min", "max", "mean", "std", "elapsed_sec", "bytes", "message",
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
        "storage_axis_order": "z,y,x,component",
        "purpose": (
            "Ten-frame center-channel pilot before the full "
            "cross-flow phase-intervention ensemble."
        ),
    }
    (OUT_META / "download_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print("DOWNLOAD CHANNEL CENTER PILOT")
    print("=" * 100)
    print("xyz    =", X_RANGE, Y_RANGE, Z_RANGE)
    print("time   =", T_START, T_END)
    print("output =", OUT)

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
        valid, message = validate(destination)
        if valid:
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
            variables = list(result.data_vars)
            if not variables:
                raise RuntimeError("getCutout returned an empty dataset.")

            data_array = result[variables[0]]
            velocity = normalize_cutout_array(data_array.values)

            xcoor = coordinate_from_data_array(
                data_array, ("xcoor", "x", "X")
            )
            ycoor = coordinate_from_data_array(
                data_array, ("ycoor", "y", "Y")
            )
            zcoor = coordinate_from_data_array(
                data_array, ("zcoor", "z", "Z")
            )

            if not (
                len(xcoor) == len(ycoor) == len(zcoor) == 64
            ):
                raise RuntimeError(
                    f"Coordinate lengths are "
                    f"x={len(xcoor)}, y={len(ycoor)}, z={len(zcoor)}."
                )

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
                xcoor=xcoor,
                ycoor=ycoor,
                zcoor=zcoor,
            )

            valid, message = validate(destination)
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
                f"[ok] t={frame_index} elapsed={elapsed:.2f}s "
                f"dx={np.mean(np.diff(xcoor)):.8e} "
                f"dy_center={np.mean(np.diff(ycoor)):.8e} "
                f"dz={np.mean(np.diff(zcoor)):.8e}"
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

        write_manifest(rows)

    valid_count = sum(
        validate(OUT_FRAMES / f"velocity_t{t:05d}.npz")[0]
        for t in range(T_START, T_END + 1)
    )

    summary = {
        **metadata,
        "valid_frame_count": int(valid_count),
        "failed_frame_indices": failures,
        "complete": valid_count == metadata["n_frames"],
        "numpy_version": np.__version__,
        "numpy_path": np.__file__,
    }
    SUMMARY.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print("CHANNEL DOWNLOAD RESULT")
    print("=" * 100)
    print("valid frames =", valid_count)
    print("failures     =", failures)
    print("summary      =", SUMMARY)

    if valid_count != metadata["n_frames"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
