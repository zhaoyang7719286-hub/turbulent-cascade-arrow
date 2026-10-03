#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


def numeric_key(path: Path):
    values = re.findall(r"\d+", path.stem)
    return tuple(int(v) for v in values) if values else (path.name,)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate the center-channel block onto an exactly "
            "uniform y grid and compute the held-out mean profile."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path(
            "datasets/channel/"
            "heldout_x1153_1216_y225_288_z961_1024_t3001_3100/"
            "frames_npz"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(
            "datasets/channel/"
            "heldout_x1153_1216_y225_288_z961_1024_t3001_3100/"
            "uniform_y"
        ),
    )
    parser.add_argument("--expected_frames", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.outdir.exists() and args.force:
        shutil.rmtree(args.outdir)

    frames_out = args.outdir / "frames_npz"
    metadata_out = args.outdir / "metadata"
    frames_out.mkdir(parents=True, exist_ok=True)
    metadata_out.mkdir(parents=True, exist_ok=True)

    files = sorted(args.input_dir.glob("*.npz"), key=numeric_key)
    if len(files) != args.expected_frames:
        raise RuntimeError(
            f"Expected {args.expected_frames} raw frames; found {len(files)}."
        )

    x_ref = y_ref = z_ref = None
    profile_sum = None
    profile_count = 0
    rows = []

    for index, source in enumerate(files, start=1):
        with np.load(source, allow_pickle=False) as z:
            velocity = np.asarray(z["velocity"], dtype=np.float64)
            x = np.asarray(z["xcoor"], dtype=np.float64)
            y = np.asarray(z["ycoor"], dtype=np.float64)
            zz = np.asarray(z["zcoor"], dtype=np.float64)
            frame_index = (
                int(z["frame_index"])
                if "frame_index" in z.files
                else index
            )

        if x_ref is None:
            x_ref, y_ref, z_ref = x, y, zz
        elif not (
            np.array_equal(x_ref, x)
            and np.array_equal(y_ref, y)
            and np.array_equal(z_ref, zz)
        ):
            raise RuntimeError(
                f"Coordinate mismatch in {source.name}."
            )

        y_uniform = np.linspace(y_ref[0], y_ref[-1], len(y_ref))
        interpolator = PchipInterpolator(
            y_ref,
            velocity,
            axis=1,
        )
        uniform = np.asarray(
            interpolator(y_uniform),
            dtype=np.float32,
        )

        if not np.all(np.isfinite(uniform)):
            raise RuntimeError(
                f"Nonfinite values after interpolation: {source.name}"
            )

        destination = frames_out / source.name
        np.savez_compressed(
            destination,
            velocity=uniform,
            frame_index=np.int32(frame_index),
            xcoor=x_ref,
            ycoor=y_uniform,
            zcoor=z_ref,
        )

        frame_profile_sum = np.sum(
            np.asarray(uniform, dtype=np.float64),
            axis=(0, 2),
            dtype=np.float64,
        )  # y,component
        if profile_sum is None:
            profile_sum = np.zeros_like(frame_profile_sum)
        profile_sum += frame_profile_sum
        profile_count += uniform.shape[0] * uniform.shape[2]

        # Interpolation displacement relative to raw data is expected
        # because y coordinates change; report it rather than threshold it.
        raw64 = np.asarray(velocity, dtype=np.float64)
        uniform64 = np.asarray(uniform, dtype=np.float64)
        rel_l2 = float(
            np.linalg.norm(uniform64 - raw64)
            / max(np.linalg.norm(raw64), np.finfo(float).tiny)
        )

        rows.append({
            "frame_number": index,
            "source_file": source.name,
            "output_file": destination.name,
            "relative_l2_uniform_minus_raw_indexwise": rel_l2,
            "finite": True,
        })
        print(
            f"[uniform-y] {index}/{len(files)} "
            f"rel_l2_indexwise={rel_l2:.6e}",
            flush=True,
        )

    mean_profile = profile_sum / float(profile_count)
    np.save(args.outdir / "mean_profile_y_component.npy", mean_profile)

    dx = float(np.mean(np.diff(x_ref)))
    dy = float(np.mean(np.diff(y_uniform)))
    dz = float(np.mean(np.diff(z_ref)))

    pd.DataFrame(rows).to_csv(
        metadata_out / "uniform_y_manifest.csv",
        index=False,
    )

    summary = {
        "input_dir": str(args.input_dir),
        "outdir": str(args.outdir),
        "n_frames": len(files),
        "shape_per_frame": [64, 64, 64, 3],
        "storage_axis_order": "z,y,x,component",
        "spacings_xyz": [dx, dy, dz],
        "y_raw_min": float(y_ref[0]),
        "y_raw_max": float(y_ref[-1]),
        "y_raw_spacing_min": float(np.min(np.diff(y_ref))),
        "y_raw_spacing_max": float(np.max(np.diff(y_ref))),
        "y_uniform_spacing": dy,
        "mean_profile_path": str(
            args.outdir / "mean_profile_y_component.npy"
        ),
        "complete": True,
    }
    (metadata_out / "uniform_y_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print("CHANNEL UNIFORM-Y PREPARATION COMPLETE")
    print("=" * 100)
    print("frames       =", len(files))
    print("spacings xyz =", dx, dy, dz)
    print("output       =", args.outdir)


if __name__ == "__main__":
    main()
