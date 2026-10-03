from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy


SCRIPT_DIR = Path(__file__).resolve().parent

OLD_SCRIPT = (
    SCRIPT_DIR
    / "eulerian_finite_time_arrow_axisfixed_64cube.py"
)

V3_SCRIPT = (
    SCRIPT_DIR
    / "eulerian_finite_time_arrow_axisfixed_64cube_v3.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot load module from {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


old = load_module("arrow_axisfixed_old", OLD_SCRIPT)
v3 = load_module("arrow_axisfixed_v3", V3_SCRIPT)

AXIS_PERM = old.AXIS_PERM


@dataclass
class Moments:
    n: int = 0
    sum_x: float = 0.0
    sum_x2: float = 0.0
    n_positive: int = 0

    def update(self, x: np.ndarray) -> None:
        values = np.asarray(
            x,
            dtype=np.float64,
        ).ravel()

        finite = np.isfinite(values)
        values = values[finite]

        self.n += int(values.size)
        self.sum_x += float(
            np.sum(values, dtype=np.float64)
        )
        self.sum_x2 += float(
            np.dot(values, values)
        )
        self.n_positive += int(
            np.count_nonzero(values > 0.0)
        )

    def result(self, prefix: str) -> dict:
        if self.n == 0:
            return {
                f"{prefix}_n": 0,
                f"{prefix}_mean": np.nan,
                f"{prefix}_std": np.nan,
                f"{prefix}_normalized_mean": np.nan,
                f"{prefix}_positive_fraction": np.nan,
                f"{prefix}_sign_bias": np.nan,
            }

        mean = self.sum_x / self.n

        variance = (
            self.sum_x2 / self.n
            - mean * mean
        )
        variance = max(variance, 0.0)

        std = float(np.sqrt(variance))
        positive_fraction = (
            self.n_positive / self.n
        )

        return {
            f"{prefix}_n": self.n,
            f"{prefix}_mean": mean,
            f"{prefix}_std": std,
            f"{prefix}_normalized_mean": (
                mean / std
                if std > 0.0
                else np.nan
            ),
            f"{prefix}_positive_fraction": (
                positive_fraction
            ),
            f"{prefix}_sign_bias": (
                positive_fraction - 0.5
            ),
        }


@dataclass
class BivariateMoments:
    n: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0

    def update(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        xv = np.asarray(
            x,
            dtype=np.float64,
        ).ravel()

        yv = np.asarray(
            y,
            dtype=np.float64,
        ).ravel()

        finite = (
            np.isfinite(xv)
            & np.isfinite(yv)
        )

        xv = xv[finite]
        yv = yv[finite]

        if xv.size == 0:
            return

        self.n += int(xv.size)
        self.sum_x += float(np.sum(xv))
        self.sum_y += float(np.sum(yv))
        self.sum_x2 += float(np.dot(xv, xv))
        self.sum_y2 += float(np.dot(yv, yv))
        self.sum_xy += float(np.dot(xv, yv))

    def correlation(self) -> float:
        if self.n < 3:
            return np.nan

        cov = (
            self.sum_xy
            - self.sum_x * self.sum_y / self.n
        )

        var_x = (
            self.sum_x2
            - self.sum_x * self.sum_x / self.n
        )

        var_y = (
            self.sum_y2
            - self.sum_y * self.sum_y / self.n
        )

        if var_x <= 0.0 or var_y <= 0.0:
            return np.nan

        return float(
            cov / np.sqrt(var_x * var_y)
        )


def safe_corr(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    acc = BivariateMoments()
    acc.update(x, y)
    return acc.correlation()


def basic_stats(
    x: np.ndarray,
    prefix: str,
) -> dict:
    acc = Moments()
    acc.update(x)
    return acc.result(prefix)


def top_positive_z(
    pi: np.ndarray,
    field: np.ndarray,
    fraction: float = 0.05,
) -> float:
    pi_flat = np.asarray(
        pi,
        dtype=np.float64,
    ).ravel()

    field_flat = np.asarray(
        field,
        dtype=np.float64,
    ).ravel()

    positive = pi_flat[pi_flat > 0.0]

    if positive.size < 10:
        return np.nan

    k = max(
        1,
        int(np.ceil(
            positive.size * fraction
        )),
    )

    threshold = np.partition(
        positive,
        -k,
    )[-k]

    mask = pi_flat >= threshold

    mean_all = float(np.mean(field_flat))
    std_all = float(np.std(field_flat))

    if std_all <= 0.0:
        return np.nan

    return float(
        (
            np.mean(field_flat[mask])
            - mean_all
        )
        / std_all
    )


def crop_all(
    fields: dict[str, np.ndarray],
    margin: int,
) -> dict[str, np.ndarray]:
    return {
        key: old.crop_core(value, margin)
        for key, value in fields.items()
    }


def compute_fields_v3(
    velocity: np.ndarray,
    scale: int,
    dx: float,
    crop_margin: int,
) -> dict[str, np.ndarray]:
    ubar = old.box_filter_velocity(
        velocity,
        scale,
    )

    tau = np.empty(
        velocity.shape[:3] + (3, 3),
        dtype=np.float64,
    )

    for i in range(3):
        for j in range(3):
            tau[..., i, j] = (
                old.box_filter_scalar(
                    velocity[..., i]
                    * velocity[..., j],
                    scale,
                )
                - ubar[..., i] * ubar[..., j]
            )

    gradient = np.empty(
        velocity.shape[:3] + (3, 3),
        dtype=np.float64,
    )

    for i in range(3):
        for j in range(3):
            gradient[..., i, j] = old.diff4(
                ubar[..., i],
                axis=AXIS_PERM[j],
                dx=dx,
            )

    strain = 0.5 * (
        gradient
        + np.swapaxes(
            gradient,
            -1,
            -2,
        )
    )

    pi = -np.einsum(
        "...ij,...ij->...",
        tau,
        strain,
    )

    omega = np.empty(
        velocity.shape[:3] + (3,),
        dtype=np.float64,
    )

    omega[..., 0] = (
        gradient[..., 2, 1]
        - gradient[..., 1, 2]
    )

    omega[..., 1] = (
        gradient[..., 0, 2]
        - gradient[..., 2, 0]
    )

    omega[..., 2] = (
        gradient[..., 1, 0]
        - gradient[..., 0, 1]
    )

    w = np.einsum(
        "...i,...ij,...j->...",
        omega,
        strain,
        omega,
    )

    tr_s3 = np.einsum(
        "...ij,...jk,...ki->...",
        strain,
        strain,
        strain,
    )

    t = -tr_s3
    quarter_w = 0.25 * w
    m = t + quarter_w

    theta = np.trace(
        gradient,
        axis1=-2,
        axis2=-1,
    )

    identity = np.eye(
        3,
        dtype=np.float64,
    )

    deviatoric_strain = (
        strain
        - theta[..., None, None]
        * identity
        / 3.0
    )

    pi_dev = -np.einsum(
        "...ij,...ij->...",
        tau,
        deviatoric_strain,
    )

    tau_trace = np.trace(
        tau,
        axis1=-2,
        axis2=-1,
    )

    pi_vol = (
        -(theta / 3.0)
        * tau_trace
    )

    return crop_all(
        {
            "Pi": pi,
            "M": m,
            "T": t,
            "quarter_W": quarter_w,
            "Pi_dev": pi_dev,
            "Pi_vol": pi_vol,
            "theta": theta,
        },
        crop_margin,
    )


def v3_array_stats(
    x: np.ndarray,
    prefix: str,
) -> dict:
    """
    Reproduce the float-reduction convention used by
    eulerian_finite_time_arrow_axisfixed_64cube_v3.py.

    In particular, do not cast a float32 array to float64
    before np.mean and np.std.
    """
    values = np.asarray(x)

    n = int(values.size)

    if n == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_normalized_mean": np.nan,
            f"{prefix}_positive_fraction": np.nan,
            f"{prefix}_sign_bias": np.nan,
        }

    mean = float(np.mean(values))
    std = float(np.std(values))

    positive_fraction = float(
        np.mean(values > 0.0)
    )

    return {
        f"{prefix}_n": n,
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_normalized_mean": (
            mean / (std + 1.0e-30)
        ),
        f"{prefix}_positive_fraction": (
            positive_fraction
        ),
        f"{prefix}_sign_bias": (
            positive_fraction - 0.5
        ),
    }



def numeric_difference(
    computed: float,
    reference: float,
) -> float:
    if (
        np.isnan(computed)
        and np.isnan(reference)
    ):
        return 0.0

    return float(
        abs(computed - reference)
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        required=True,
    )

    parser.add_argument(
        "--outdir",
        required=True,
    )

    parser.add_argument(
        "--reference_v3_dir",
        required=True,
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--label",
        default=None,
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
        default=2.0 * np.pi / 1024.0,
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
        "--qc_tolerance",
        type=float,
        default=1.0e-12,
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    reference_dir = Path(
        args.reference_v3_dir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label = (
        args.label
        if args.label is not None
        else f"surrogate_{args.seed}"
    )

    files = sorted(
        input_dir.glob("velocity_t*.npz")
    )[:args.max_frames]

    if not files:
        raise FileNotFoundError(
            f"No velocity files in {input_dir}"
        )

    reference_frame = pd.read_csv(
        reference_dir
        / "framewise_flux_stats_v3.csv"
    )

    reference_inst = pd.read_csv(
        reference_dir
        / "instantaneous_arrow_metrics_v3.csv"
    )

    reference_finite = pd.read_csv(
        reference_dir
        / "finite_time_arrow_summary_v3.csv"
    )

    reference_finite = reference_finite[
        reference_finite["mode"]
        == "raw_zero"
    ].copy()

    frame_rows = []
    summary_rows = []
    finite_rows = []
    qc_rows = []

    for scale in args.scales:
        print(
            f"\n=== seed={args.seed}, "
            f"scale={scale} ===",
            flush=True,
        )

        crop_margin = (
            args.crop_factor * scale + 4
        )

        arrays = {
            "Pi": [],
            "M": [],
            "T": [],
            "quarter_W": [],
        }

        pooled_fields = {
            name: Moments()
            for name in [
                "M",
                "T",
                "quarter_W",
            ]
        }

        pooled_corr = {
            name: BivariateMoments()
            for name in [
                "M",
                "T",
                "quarter_W",
                "Pi_dev",
                "Pi_vol",
            ]
        }

        decomposition_residual_max = 0.0

        for frame_index, path in enumerate(
            files,
            start=1,
        ):
            velocity = old.load_velocity_npz(
                path
            )

            fields = compute_fields_v3(
                velocity,
                scale=scale,
                dx=args.dx,
                crop_margin=crop_margin,
            )

            # Keep the float64 field for the exact
            # deviatoric/volumetric decomposition, but use
            # the float32 Pi convention returned by the
            # official v3 flux routine for all Pi-dependent
            # statistics, signs, masks and correlations.
            pi_float64 = fields["Pi"]
            pi = pi_float64.astype(
                np.float32,
                copy=False,
            )

            decomposition_residual = (
                pi_float64
                - fields["Pi_dev"]
                - fields["Pi_vol"]
            )

            decomposition_residual_max = max(
                decomposition_residual_max,
                float(np.max(np.abs(
                    decomposition_residual
                ))),
            )

            row = {
                "label": label,
                "source_seed": args.seed,
                "scale": scale,
                "frame_index": frame_index,
                "file": path.name,
                "crop_margin": crop_margin,
                "n_points": int(pi.size),
                "Pi_mean": float(np.mean(pi)),
                "Pi_std": float(np.std(pi)),
                "Pi_positive_fraction": float(
                    np.mean(pi > 0.0)
                ),
                "Pi_sign_bias": float(
                    np.mean(pi > 0.0) - 0.5
                ),
                "corr_Pi_M": safe_corr(
                    pi,
                    fields["M"],
                ),
                "corr_Pi_T": safe_corr(
                    pi,
                    fields["T"],
                ),
                "corr_Pi_quarter_W": safe_corr(
                    pi,
                    fields["quarter_W"],
                ),
                "corr_Pi_Pi_dev": safe_corr(
                    pi,
                    fields["Pi_dev"],
                ),
                "corr_Pi_Pi_vol": safe_corr(
                    pi,
                    fields["Pi_vol"],
                ),
                "top5_positive_M_z": (
                    top_positive_z(
                        pi,
                        fields["M"],
                    )
                ),
                "top5_positive_T_z": (
                    top_positive_z(
                        pi,
                        fields["T"],
                    )
                ),
                "top5_positive_quarter_W_z": (
                    top_positive_z(
                        pi,
                        fields["quarter_W"],
                    )
                ),
                "Pi_decomposition_max_abs": (
                    float(np.max(np.abs(
                        decomposition_residual
                    )))
                ),
            }

            for name in [
                "M",
                "T",
                "quarter_W",
            ]:
                row.update(
                    basic_stats(
                        fields[name],
                        name,
                    )
                )

                pooled_fields[name].update(
                    fields[name]
                )

            for name in pooled_corr:
                pooled_corr[name].update(
                    pi,
                    fields[name],
                )

            frame_rows.append(row)

            for name in arrays:
                arrays[name].append(
                    np.asarray(
                        fields[name],
                        dtype=np.float32,
                    ).ravel()
                )

            ref = reference_frame[
                (
                    reference_frame["scale"]
                    == scale
                )
                & (
                    reference_frame[
                        "frame_index"
                    ]
                    == frame_index - 1
                )
            ]

            if len(ref) != 1:
                raise RuntimeError(
                    "Reference frame row not unique: "
                    f"scale={scale}, "
                    f"frame_index={frame_index}"
                )

            ref = ref.iloc[0]

            qc_rows.append({
                "qc_type": "framewise_Pi",
                "source_seed": args.seed,
                "scale": scale,
                "frame_index": frame_index,
                "window_frames": np.nan,
                "metric": "mean_Pi",
                "computed": row["Pi_mean"],
                "reference": float(
                    ref["mean_Pi"]
                ),
                "abs_difference": (
                    numeric_difference(
                        row["Pi_mean"],
                        float(ref["mean_Pi"]),
                    )
                ),
            })

            qc_rows.append({
                "qc_type": "framewise_Pi",
                "source_seed": args.seed,
                "scale": scale,
                "frame_index": frame_index,
                "window_frames": np.nan,
                "metric": "std_Pi",
                "computed": row["Pi_std"],
                "reference": float(
                    ref["std_Pi"]
                ),
                "abs_difference": (
                    numeric_difference(
                        row["Pi_std"],
                        float(ref["std_Pi"]),
                    )
                ),
            })

            qc_rows.append({
                "qc_type": "framewise_Pi",
                "source_seed": args.seed,
                "scale": scale,
                "frame_index": frame_index,
                "window_frames": np.nan,
                "metric": (
                    "positive_fraction_Pi"
                ),
                "computed": (
                    row[
                        "Pi_positive_fraction"
                    ]
                ),
                "reference": float(
                    ref[
                        "positive_fraction_Pi"
                    ]
                ),
                "abs_difference": (
                    numeric_difference(
                        row[
                            "Pi_positive_fraction"
                        ],
                        float(
                            ref[
                                "positive_fraction_Pi"
                            ]
                        ),
                    )
                ),
            })

            if frame_index % 10 == 0:
                print(
                    f"processed "
                    f"{frame_index}/{len(files)}",
                    flush=True,
                )

        stacked = {
            name: np.stack(
                values,
                axis=0,
            ).astype(
                np.float32,
                copy=False,
            )
            for name, values in arrays.items()
        }

        pi_array = stacked["Pi"]

        summary = {
            "label": label,
            "source_seed": args.seed,
            "scale": scale,
            "n_frames": len(files),
            "n_spatial_points": (
                pi_array.shape[1]
            ),
            "crop_margin": crop_margin,
            "corr_Pi_M_pooled": (
                pooled_corr["M"].correlation()
            ),
            "corr_Pi_T_pooled": (
                pooled_corr["T"].correlation()
            ),
            "corr_Pi_quarter_W_pooled": (
                pooled_corr[
                    "quarter_W"
                ].correlation()
            ),
            "corr_Pi_Pi_dev_pooled": (
                pooled_corr[
                    "Pi_dev"
                ].correlation()
            ),
            "corr_Pi_Pi_vol_pooled": (
                pooled_corr[
                    "Pi_vol"
                ].correlation()
            ),
            "Pi_decomposition_max_abs": (
                decomposition_residual_max
            ),
        }

        frame_scale = pd.DataFrame(
            [
                row
                for row in frame_rows
                if row["scale"] == scale
            ]
        )

        for column in [
            "corr_Pi_M",
            "corr_Pi_T",
            "corr_Pi_quarter_W",
            "corr_Pi_Pi_dev",
            "corr_Pi_Pi_vol",
            "top5_positive_M_z",
            "top5_positive_T_z",
            "top5_positive_quarter_W_z",
        ]:
            summary[
                f"{column}_frame_mean"
            ] = float(
                frame_scale[column].mean()
            )

            summary[
                f"{column}_frame_std"
            ] = float(
                frame_scale[column].std(
                    ddof=1
                )
            )

        for name, accumulator in (
            pooled_fields.items()
        ):
            summary.update(
                accumulator.result(name)
            )

        pi_inst = v3_array_stats(
            pi_array,
            "Pi",
        )

        summary.update(pi_inst)

        ref_inst = reference_inst[
            reference_inst["scale"]
            == scale
        ]

        if len(ref_inst) != 1:
            raise RuntimeError(
                "Reference instantaneous row "
                f"not unique for scale={scale}"
            )

        ref_inst = ref_inst.iloc[0]

        for computed_name, ref_name in [
            ("Pi_mean", "mean_Pi"),
            ("Pi_std", "std_Pi"),
            (
                "Pi_normalized_mean",
                "normalized_mean_Pi",
            ),
            (
                "Pi_positive_fraction",
                "positive_fraction_Pi",
            ),
            (
                "Pi_sign_bias",
                "sign_bias_Pi",
            ),
        ]:
            computed = float(
                summary[computed_name]
            )
            reference = float(
                ref_inst[ref_name]
            )

            qc_rows.append({
                "qc_type": (
                    "instantaneous_Pi"
                ),
                "source_seed": args.seed,
                "scale": scale,
                "frame_index": np.nan,
                "window_frames": np.nan,
                "metric": ref_name,
                "computed": computed,
                "reference": reference,
                "abs_difference": (
                    numeric_difference(
                        computed,
                        reference,
                    )
                ),
            })

        summary_rows.append(summary)

        for window_frames in args.windows:
            accumulated = {
                name: (
                    v3.nonoverlap_window_sums(
                        array,
                        win=window_frames,
                        dt=args.dt,
                    )
                )
                for name, array
                in stacked.items()
            }

            finite = {
                "label": label,
                "source_seed": args.seed,
                "scale": scale,
                "window_frames": (
                    window_frames
                ),
                "window_physical_time": (
                    window_frames * args.dt
                ),
                "n_nonoverlap_windows": (
                    pi_array.shape[0]
                    // window_frames
                ),
                "n_samples": int(
                    accumulated["Pi"].size
                ),
                "corr_I_Pi_I_M": safe_corr(
                    accumulated["Pi"],
                    accumulated["M"],
                ),
                "corr_I_Pi_I_T": safe_corr(
                    accumulated["Pi"],
                    accumulated["T"],
                ),
                "corr_I_Pi_I_quarter_W": (
                    safe_corr(
                        accumulated["Pi"],
                        accumulated[
                            "quarter_W"
                        ],
                    )
                ),
            }

            for name in [
                "Pi",
                "M",
                "T",
                "quarter_W",
            ]:
                if name == "Pi":
                    stats = v3_array_stats(
                        accumulated[name],
                        f"I_{name}",
                    )
                else:
                    stats = basic_stats(
                        accumulated[name],
                        f"I_{name}",
                    )

                finite.update(stats)

            ref_finite = reference_finite[
                (
                    reference_finite["scale"]
                    == scale
                )
                & (
                    reference_finite[
                        "window_frames"
                    ]
                    == window_frames
                )
            ]

            if len(ref_finite) != 1:
                raise RuntimeError(
                    "Reference finite-time row "
                    "not unique for "
                    f"scale={scale}, "
                    f"window={window_frames}"
                )

            ref_finite = ref_finite.iloc[0]

            for computed_name, ref_name in [
                ("I_Pi_mean", "mean_I"),
                ("I_Pi_std", "std_I"),
                (
                    "I_Pi_normalized_mean",
                    "normalized_mean_I",
                ),
                (
                    "I_Pi_positive_fraction",
                    "positive_fraction_I",
                ),
                (
                    "I_Pi_sign_bias",
                    "sign_bias_I",
                ),
            ]:
                computed = float(
                    finite[computed_name]
                )
                reference = float(
                    ref_finite[ref_name]
                )

                qc_rows.append({
                    "qc_type": (
                        "finite_time_Pi"
                    ),
                    "source_seed": args.seed,
                    "scale": scale,
                    "frame_index": np.nan,
                    "window_frames": (
                        window_frames
                    ),
                    "metric": ref_name,
                    "computed": computed,
                    "reference": reference,
                    "abs_difference": (
                        numeric_difference(
                            computed,
                            reference,
                        )
                    ),
                })

            finite_rows.append(finite)

    frame_df = pd.DataFrame(frame_rows)
    summary_df = pd.DataFrame(summary_rows)
    finite_df = pd.DataFrame(finite_rows)
    qc_df = pd.DataFrame(qc_rows)

    frame_path = (
        outdir
        / "mechanism_frame_stats_v3.csv"
    )

    summary_path = (
        outdir
        / "mechanism_summary_v3.csv"
    )

    finite_path = (
        outdir
        / "mechanism_finite_time_v3.csv"
    )

    qc_path = (
        outdir
        / "mechanism_qc_v3.csv"
    )

    frame_df.to_csv(
        frame_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    finite_df.to_csv(
        finite_path,
        index=False,
    )

    qc_df.to_csv(
        qc_path,
        index=False,
    )

    max_qc_difference = float(
        qc_df["abs_difference"].max()
    )

    metadata = {
        "input_dir": str(input_dir),
        "reference_v3_dir": str(
            reference_dir
        ),
        "outdir": str(outdir),
        "label": label,
        "source_seed": args.seed,
        "n_files": len(files),
        "scales": args.scales,
        "windows": args.windows,
        "dx": args.dx,
        "dt": args.dt,
        "axis_perm": str(AXIS_PERM),
        "filter_boundary_mode": (
            "nearest"
        ),
        "crop_margin_rule": (
            "crop_factor * scale + 4"
        ),
        "crop_factor": args.crop_factor,
        "derivative_scheme": "D4",
        "top_region_definition": (
            "top 5 percent of positive Pi "
            "within each frame"
        ),
        "phase_environment_required": (
            "default user-site environment "
            "used for original ensemble"
        ),
        "python": sys.version,
        "python_executable": (
            sys.executable
        ),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "max_v3_qc_abs_difference": (
            max_qc_difference
        ),
        "qc_tolerance": (
            args.qc_tolerance
        ),
        "qc_pass": bool(
            max_qc_difference
            <= args.qc_tolerance
        ),
    }

    metadata_path = (
        outdir
        / "mechanism_run_metadata_v3.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(frame_path)
    print(summary_path)
    print(finite_path)
    print(qc_path)
    print(metadata_path)

    print(
        "\nmax v3 QC absolute difference =",
        max_qc_difference,
    )

    print(
        "QC result =",
        "PASS"
        if metadata["qc_pass"]
        else "FAIL",
    )

    print("\nMechanism summary:")
    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\nFinite-time mechanism summary:")
    print(
        finite_df.to_string(
            index=False
        )
    )

    if not metadata["qc_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
