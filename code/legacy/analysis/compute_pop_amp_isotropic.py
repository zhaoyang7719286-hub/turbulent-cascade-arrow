#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
V3_SCRIPT = (
    SCRIPT_DIR
    / "eulerian_finite_time_arrow_axisfixed_64cube_v3.py"
)

spec = importlib.util.spec_from_file_location(
    "finite_time_v3",
    V3_SCRIPT,
)
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

old = v3.old
DX_DEFAULT = v3.DX_DEFAULT


def pop_amp_statistics(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]

    if x.size == 0:
        raise RuntimeError("No finite values.")

    positive = x > 0.0
    negative = x < 0.0
    zero = x == 0.0

    n = int(x.size)
    n_pos = int(np.count_nonzero(positive))
    n_neg = int(np.count_nonzero(negative))
    n_zero = int(np.count_nonzero(zero))

    p_pos = n_pos / n
    p_neg = n_neg / n
    p_zero = n_zero / n

    if n_pos == 0 or n_neg == 0:
        raise RuntimeError(
            f"Both signs are required: n_pos={n_pos}, n_neg={n_neg}"
        )

    mu_pos = float(np.mean(x[positive], dtype=np.float64))
    mu_neg = float(np.mean(-x[negative], dtype=np.float64))

    mean_signed = float(np.mean(x, dtype=np.float64))
    mean_absolute = float(np.mean(np.abs(x), dtype=np.float64))

    a_pop = float(p_pos - p_neg)
    a_amp = float(
        (mu_pos - mu_neg)
        / (mu_pos + mu_neg)
    )

    normalized_mean_lhs = float(
        mean_signed / mean_absolute
    )

    normalized_mean_rhs = float(
        (a_pop + a_amp)
        / (1.0 + a_pop * a_amp)
    )

    return {
        "n_samples": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_zero": n_zero,
        "p_positive": p_pos,
        "p_negative": p_neg,
        "p_zero": p_zero,
        "mu_positive": mu_pos,
        "mu_negative_abs": mu_neg,
        "A_pop": a_pop,
        "A_amp": a_amp,
        "mean_signed": mean_signed,
        "mean_absolute": mean_absolute,
        "normalized_mean_lhs": normalized_mean_lhs,
        "normalized_mean_rhs": normalized_mean_rhs,
        "identity_abs_error": float(
            abs(normalized_mean_lhs - normalized_mean_rhs)
        ),
    }


def read_seed(input_dir: Path) -> int | None:
    metadata = (
        input_dir
        / "df_phase_randomized_dataset_metadata.json"
    )

    if not metadata.exists():
        return None

    content = json.loads(
        metadata.read_text(encoding="utf-8")
    )
    seed = content.get("seed")
    return None if seed is None else int(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute exact population-amplitude decomposition "
            "for instantaneous and finite-time SGS flux."
        )
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--block",
        required=True,
    )
    parser.add_argument(
        "--case",
        choices=("reference", "surrogate", "dns"),
        required=True,
    )
    parser.add_argument(
        "--label",
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[4, 8],
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[5, 10, 20],
    )
    parser.add_argument(
        "--dx",
        type=float,
        default=DX_DEFAULT,
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--crop_factor",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    output_csv = args.outdir / "pop_amp_metrics.csv"

    if output_csv.exists() and not args.force:
        raise FileExistsError(
            f"{output_csv} already exists. Use --force."
        )

    files = sorted(args.input_dir.glob("*.npz"))[
        :args.max_frames
    ]

    if len(files) != args.max_frames:
        raise RuntimeError(
            f"Expected {args.max_frames} frames, found {len(files)} "
            f"in {args.input_dir}"
        )

    source_seed = (
        args.seed
        if args.seed is not None
        else read_seed(args.input_dir)
    )

    rows: list[dict] = []

    for scale in args.scales:
        crop_margin = args.crop_factor * scale + 4
        pi_frames: list[np.ndarray] = []

        for frame_index, frame_path in enumerate(files):
            print(
                f"[{args.label}] scale={scale} "
                f"frame={frame_index + 1}/{len(files)}",
                flush=True,
            )

            velocity = old.load_velocity_npz(frame_path)

            pi, _, rtheta = old.compute_flux_and_ksgs(
                velocity,
                scale,
                args.dx,
                crop_margin,
            )

            if not np.isfinite(pi).all():
                raise RuntimeError(
                    f"Non-finite Pi at {frame_path}"
                )

            pi_frames.append(
                pi.ravel().astype(np.float32)
            )

        pi_array = np.stack(
            pi_frames,
            axis=0,
        ).astype(np.float32)

        instant = pop_amp_statistics(pi_array)
        instant.update({
            "block": args.block,
            "case": args.case,
            "label": args.label,
            "seed": source_seed,
            "variable": "Pi",
            "scale": scale,
            "window_frames": 0,
            "window_physical_time": 0.0,
            "n_frames": len(files),
            "n_spatial_points": int(pi_array.shape[1]),
            "crop_margin": crop_margin,
        })
        rows.append(instant)

        for window in args.windows:
            accumulated = v3.nonoverlap_window_sums(
                pi_array,
                win=window,
                dt=args.dt,
            )

            finite = pop_amp_statistics(accumulated)
            finite.update({
                "block": args.block,
                "case": args.case,
                "label": args.label,
                "seed": source_seed,
                "variable": "I",
                "scale": scale,
                "window_frames": window,
                "window_physical_time": float(
                    window * args.dt
                ),
                "n_frames": len(files),
                "n_spatial_points": int(pi_array.shape[1]),
                "crop_margin": crop_margin,
            })
            rows.append(finite)

        del pi_array
        del pi_frames

    frame = pd.DataFrame(rows)

    ordered = [
        "block",
        "case",
        "label",
        "seed",
        "variable",
        "scale",
        "window_frames",
        "window_physical_time",
        "n_frames",
        "n_spatial_points",
        "crop_margin",
        "n_samples",
        "n_positive",
        "n_negative",
        "n_zero",
        "p_positive",
        "p_negative",
        "p_zero",
        "mu_positive",
        "mu_negative_abs",
        "A_pop",
        "A_amp",
        "mean_signed",
        "mean_absolute",
        "normalized_mean_lhs",
        "normalized_mean_rhs",
        "identity_abs_error",
    ]

    frame = frame[ordered]
    frame.to_csv(output_csv, index=False)

    max_error = float(
        frame["identity_abs_error"].max()
    )

    audit = {
        "input_dir": str(args.input_dir),
        "output_csv": str(output_csv),
        "block": args.block,
        "case": args.case,
        "label": args.label,
        "seed": source_seed,
        "n_frames": len(files),
        "scales": args.scales,
        "windows": args.windows,
        "dt": args.dt,
        "dx": args.dx,
        "max_identity_abs_error": max_error,
        "identity_tolerance": 1.0e-12,
        "identity_pass": bool(max_error < 1.0e-12),
    }

    (
        args.outdir / "pop_amp_audit.json"
    ).write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    if not audit["identity_pass"]:
        raise RuntimeError(
            f"Identity audit failed: {max_error}"
        )

    (
        args.outdir / "DONE"
    ).write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("POP-AMP CASE COMPLETE")
    print("output =", output_csv)
    print("max identity error =", max_error)


if __name__ == "__main__":
    main()
