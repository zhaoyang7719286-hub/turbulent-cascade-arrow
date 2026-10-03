from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


# 复用已经校核过的轴顺序、D4导数、滤波和通量计算。
SCRIPT_DIR = Path(__file__).resolve().parent
OLD_SCRIPT = SCRIPT_DIR / "eulerian_finite_time_arrow_axisfixed_64cube.py"

spec = importlib.util.spec_from_file_location("old_ft_arrow", OLD_SCRIPT)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

AXIS_PERM = old.AXIS_PERM
DX_DEFAULT = old.DX_DEFAULT


def nonoverlap_window_sums(
    q: np.ndarray,
    win: int,
    dt: float,
) -> np.ndarray:
    """
    非重叠时间窗口累计。

    q shape = (n_frames, n_spatial_points)

    例如 win=10:
        0--9, 10--19, ..., 90--99
    """
    if q.ndim != 2:
        raise ValueError(f"Expected 2-D array, got {q.shape}")

    n_frames = q.shape[0]
    n_blocks = n_frames // win

    if n_blocks < 1:
        raise ValueError(
            f"Window {win} exceeds available frames {n_frames}"
        )

    trimmed = q[:n_blocks * win]

    accumulated = trimmed.reshape(
        n_blocks,
        win,
        q.shape[1],
    ).sum(axis=1, dtype=np.float64)

    return accumulated * dt


def make_ratio_curve_fixed(
    accumulated: np.ndarray,
    mode: str,
    nbins: int,
    zmax: float,
    min_count_each: int,
) -> tuple[pd.DataFrame, dict]:
    """
    使用固定、严格对称的bin计算：

        R(z) = log[P(z) / P(-z)]

    raw_zero:
        z = I / std(I)

    centered:
        s = [I - mean(I)] / std(I)
    """
    if nbins % 2 != 0:
        raise ValueError("nbins must be even for exact symmetric pairing.")

    x0 = np.asarray(accumulated, dtype=np.float64).ravel()
    x0 = x0[np.isfinite(x0)]

    if x0.size < 1000:
        return pd.DataFrame(), {
            "n_samples": int(x0.size),
            "error": "insufficient_samples",
        }

    mean_i = float(np.mean(x0))
    std_i = float(np.std(x0))

    if std_i <= 0:
        return pd.DataFrame(), {
            "n_samples": int(x0.size),
            "error": "zero_standard_deviation",
        }

    if mode == "raw_zero":
        x = x0 / std_i
    elif mode == "centered":
        x = (x0 - mean_i) / std_i
    else:
        raise ValueError(f"Unknown mode: {mode}")

    edges = np.linspace(-zmax, zmax, nbins + 1)
    counts, _ = np.histogram(x, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    rows = []

    # 后半部分为正值bin，镜像bin索引为 nbins - 1 - i。
    for i in range(nbins // 2, nbins):
        j = nbins - 1 - i

        z_pos = centers[i]
        z_neg = centers[j]

        if not np.isclose(z_neg, -z_pos):
            raise RuntimeError("Histogram bins are not exactly symmetric.")

        count_pos = int(counts[i])
        count_neg = int(counts[j])

        # 正负两侧分别达到阈值，而不是二者之和达到阈值。
        if (
            count_pos < min_count_each
            or count_neg < min_count_each
        ):
            continue

        ratio = np.log(count_pos / count_neg)

        rows.append({
            "z": float(z_pos),
            "count_pos": count_pos,
            "count_neg": count_neg,
            "R": float(ratio),
        })

    curve = pd.DataFrame(rows)

    n_tail_pos = int(np.sum(x >= 2.0))
    n_tail_neg = int(np.sum(x <= -2.0))

    if n_tail_pos > 0 and n_tail_neg > 0:
        tail_log_ratio = float(
            np.log(n_tail_pos / n_tail_neg)
        )
    else:
        tail_log_ratio = np.nan

    stats = {
        "n_samples": int(x0.size),
        "mean_I": mean_i,
        "std_I": std_i,
        "normalized_mean_I": float(mean_i / std_i),
        "positive_fraction_I": float(np.mean(x0 > 0)),
        "sign_bias_I": float(np.mean(x0 > 0) - 0.5),
        "skew_centered_I": float(
            np.mean(((x0 - mean_i) / std_i) ** 3)
        ),
        "n_tail_pos_z2": n_tail_pos,
        "n_tail_neg_z2": n_tail_neg,
        "tail_log_ratio_z2": tail_log_ratio,
        "mode": mode,
        "zmax": float(zmax),
        "nbins": int(nbins),
        "min_count_each": int(min_count_each),
        "valid_bin_count": int(len(curve)),
    }

    if len(curve) >= 4:
        z = curve["z"].to_numpy()
        r = curve["R"].to_numpy()
        weights = (
            curve["count_pos"].to_numpy()
            + curve["count_neg"].to_numpy()
        )

        fit_mask = (z >= 0.2) & (z <= 2.0)

        if np.sum(fit_mask) >= 4:
            coef = np.polyfit(
                z[fit_mask],
                r[fit_mask],
                deg=1,
                w=weights[fit_mask],
            )

            prediction = (
                coef[0] * z[fit_mask] + coef[1]
            )

            weighted_mean = np.average(
                r[fit_mask],
                weights=weights[fit_mask],
            )

            ss_res = np.sum(
                weights[fit_mask]
                * (r[fit_mask] - prediction) ** 2
            )

            ss_tot = np.sum(
                weights[fit_mask]
                * (r[fit_mask] - weighted_mean) ** 2
            )

            stats["alpha"] = float(coef[0])
            stats["intercept"] = float(coef[1])
            stats["fit_r2"] = float(
                1.0 - ss_res / (ss_tot + 1e-30)
            )
            stats["n_fit_bins"] = int(
                np.sum(fit_mask)
            )
        else:
            stats["alpha"] = np.nan
            stats["intercept"] = np.nan
            stats["fit_r2"] = np.nan
            stats["n_fit_bins"] = int(
                np.sum(fit_mask)
            )
    else:
        stats["alpha"] = np.nan
        stats["intercept"] = np.nan
        stats["fit_r2"] = np.nan
        stats["n_fit_bins"] = int(len(curve))

    return curve, stats


def read_source_metadata(input_dir: Path) -> dict:
    metadata_path = (
        input_dir
        / "df_phase_randomized_dataset_metadata.json"
    )

    if metadata_path.exists():
        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Finite-time SGS-flux asymmetry using "
            "non-overlapping time windows."
        )
    )

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--label", required=True)

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
        "--nbins",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--zmax",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--min_count_each",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.npz"))[
        :args.max_frames
    ]

    if not files:
        raise FileNotFoundError(
            f"No NPZ files found in {input_dir}"
        )

    source_metadata = read_source_metadata(input_dir)
    source_seed = source_metadata.get("seed", None)

    frame_rows = []
    instantaneous_rows = []
    finite_rows = []
    all_curves = []

    for scale in args.scales:
        print(f"\n=== scale={scale} ===", flush=True)

        crop_margin = (
            args.crop_factor * scale + 4
        )

        pi_frames = []

        for frame_index, frame_path in enumerate(files):
            print(
                f"[scale={scale}] "
                f"{frame_index + 1}/{len(files)} "
                f"{frame_path.name}",
                flush=True,
            )

            velocity = old.load_velocity_npz(
                frame_path
            )

            pi, ksgs, rtheta = (
                old.compute_flux_and_ksgs(
                    velocity,
                    scale,
                    args.dx,
                    crop_margin,
                )
            )

            pi_flat = pi.ravel().astype(
                np.float32
            )

            pi_frames.append(pi_flat)

            frame_rows.append({
                "label": args.label,
                "source_seed": source_seed,
                "scale": scale,
                "frame_index": frame_index,
                "file": frame_path.name,
                "core_npoints": int(pi.size),
                "crop_margin": crop_margin,
                "rtheta": float(rtheta),
                "mean_Pi": float(np.mean(pi)),
                "std_Pi": float(np.std(pi)),
                "positive_fraction_Pi": float(
                    np.mean(pi > 0)
                ),
                "sign_bias_Pi": float(
                    np.mean(pi > 0) - 0.5
                ),
                "mean_ksgs": float(
                    np.mean(ksgs)
                ),
                "min_ksgs": float(
                    np.min(ksgs)
                ),
            })

        pi_array = np.stack(
            pi_frames,
            axis=0,
        ).astype(np.float32)

        mean_pi = float(np.mean(pi_array))
        std_pi = float(np.std(pi_array))

        instantaneous_rows.append({
            "label": args.label,
            "source_seed": source_seed,
            "scale": scale,
            "n_frames": int(pi_array.shape[0]),
            "n_spatial_points": int(
                pi_array.shape[1]
            ),
            "mean_Pi": mean_pi,
            "std_Pi": std_pi,
            "normalized_mean_Pi": float(
                mean_pi / (std_pi + 1e-30)
            ),
            "positive_fraction_Pi": float(
                np.mean(pi_array > 0)
            ),
            "sign_bias_Pi": float(
                np.mean(pi_array > 0) - 0.5
            ),
        })

        for window_frames in args.windows:
            accumulated = nonoverlap_window_sums(
                pi_array,
                win=window_frames,
                dt=args.dt,
            )

            for mode in ("raw_zero", "centered"):
                curve, stats = (
                    make_ratio_curve_fixed(
                        accumulated,
                        mode=mode,
                        nbins=args.nbins,
                        zmax=args.zmax,
                        min_count_each=(
                            args.min_count_each
                        ),
                    )
                )

                stats.update({
                    "label": args.label,
                    "source_seed": source_seed,
                    "scale": scale,
                    "window_frames": (
                        window_frames
                    ),
                    "window_physical_time": float(
                        window_frames * args.dt
                    ),
                    "n_nonoverlap_windows": int(
                        pi_array.shape[0]
                        // window_frames
                    ),
                })

                finite_rows.append(stats)

                if len(curve):
                    curve["label"] = args.label
                    curve["source_seed"] = (
                        source_seed
                    )
                    curve["scale"] = scale
                    curve["window_frames"] = (
                        window_frames
                    )
                    curve[
                        "window_physical_time"
                    ] = window_frames * args.dt
                    curve["mode"] = mode

                    all_curves.append(curve)

    frame_df = pd.DataFrame(frame_rows)
    instantaneous_df = pd.DataFrame(
        instantaneous_rows
    )
    finite_df = pd.DataFrame(finite_rows)

    if all_curves:
        curves_df = pd.concat(
            all_curves,
            ignore_index=True,
        )
    else:
        curves_df = pd.DataFrame()

    frame_df.to_csv(
        outdir / "framewise_flux_stats_v3.csv",
        index=False,
    )

    instantaneous_df.to_csv(
        outdir / "instantaneous_arrow_metrics_v3.csv",
        index=False,
    )

    finite_df.to_csv(
        outdir / "finite_time_arrow_summary_v3.csv",
        index=False,
    )

    curves_df.to_csv(
        outdir / "finite_time_FR_curves_v3.csv",
        index=False,
    )

    metadata = {
        "input_dir": str(input_dir),
        "label": args.label,
        "source_seed": source_seed,
        "n_files": len(files),
        "scales": args.scales,
        "windows": args.windows,
        "dx": args.dx,
        "dt": args.dt,
        "axis_perm": str(AXIS_PERM),
        "crop_factor": args.crop_factor,
        "nbins": args.nbins,
        "zmax": args.zmax,
        "min_count_each": args.min_count_each,
        "window_protocol": "nonoverlapping",
        "primary_accumulated_variable": (
            "I = dt * sum(Pi)"
        ),
        "modes": ["raw_zero", "centered"],
        "note": (
            "raw_zero is the main finite-time "
            "flux-asymmetry statistic; centered is "
            "retained as a supplementary shape check."
        ),
    }

    with (
        outdir / "run_metadata_v3.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nDone.")
    print("Output directory:", outdir)


if __name__ == "__main__":
    main()
