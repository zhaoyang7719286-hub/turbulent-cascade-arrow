#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

SCALES = (4, 8)
WINDOWS = (5, 10, 20)
PRIMARY_METRICS = (
    "mean_Pi",
    "sign_bias_Pi",
    "sign_bias_I_w10",
)


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def require_one(frame: pd.DataFrame, description: str) -> pd.Series:
    if len(frame) != 1:
        raise RuntimeError(
            f"Expected exactly one row for {description}; found {len(frame)}."
        )
    return frame.iloc[0]


def load_case_metrics(
    v3_dir: Path,
    case_type: str,
    case_label: str,
    seed: Optional[int],
) -> pd.DataFrame:
    inst_path = v3_dir / "instantaneous_arrow_metrics_v3.csv"
    finite_path = v3_dir / "finite_time_arrow_summary_v3.csv"
    if not inst_path.exists():
        raise FileNotFoundError(inst_path)
    if not finite_path.exists():
        raise FileNotFoundError(finite_path)

    inst = pd.read_csv(inst_path)
    finite = pd.read_csv(finite_path)
    rows: list[dict[str, Any]] = []

    for scale in SCALES:
        inst_row = require_one(
            inst.loc[inst["scale"] == scale],
            f"{case_label}: instantaneous scale={scale}",
        )
        row: dict[str, Any] = {
            "case_type": case_type,
            "case_label": case_label,
            "seed": seed,
            "scale": scale,
            "n_frames": int(inst_row["n_frames"]),
            "n_spatial_points": int(inst_row["n_spatial_points"]),
            "mean_Pi": float(inst_row["mean_Pi"]),
            "std_Pi": float(inst_row["std_Pi"]),
            "normalized_mean_Pi": float(inst_row["normalized_mean_Pi"]),
            "positive_fraction_Pi": float(inst_row["positive_fraction_Pi"]),
            "sign_bias_Pi": float(inst_row["sign_bias_Pi"]),
        }
        for window in WINDOWS:
            finite_row = require_one(
                finite.loc[
                    (finite["scale"] == scale)
                    & (finite["window_frames"] == window)
                    & (finite["mode"] == "raw_zero")
                ],
                f"{case_label}: scale={scale}, window={window}, raw_zero",
            )
            row[f"n_samples_I_w{window}"] = int(finite_row["n_samples"])
            row[f"mean_I_w{window}"] = float(finite_row["mean_I"])
            row[f"std_I_w{window}"] = float(finite_row["std_I"])
            row[f"normalized_mean_I_w{window}"] = float(
                finite_row["normalized_mean_I"]
            )
            row[f"positive_fraction_I_w{window}"] = float(
                finite_row["positive_fraction_I"]
            )
            row[f"sign_bias_I_w{window}"] = float(
                finite_row["sign_bias_I"]
            )
        rows.append(row)

    return pd.DataFrame(rows)


def metric_metadata(metric: str) -> dict[str, Any]:
    if metric in {"mean_Pi", "normalized_mean_Pi", "sign_bias_Pi"}:
        return {
            "statistic_type": "instantaneous",
            "window_frames": 0,
            "mode": "raw_zero",
        }
    for prefix in ("mean_I_w", "normalized_mean_I_w", "sign_bias_I_w"):
        if metric.startswith(prefix):
            return {
                "statistic_type": "finite_time",
                "window_frames": int(metric.split("_w")[-1]),
                "mode": "raw_zero",
            }
    raise ValueError(f"Unsupported metric: {metric}")


def summarize_comparator(
    comparator: pd.DataFrame,
    surrogates: pd.DataFrame,
    comparator_name: str,
    metric_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scale in SCALES:
        comp_row = require_one(
            comparator.loc[comparator["scale"] == scale],
            f"{comparator_name}: scale={scale}",
        )
        sur_rows = surrogates.loc[surrogates["scale"] == scale]
        if sur_rows["seed"].nunique() != 24 or len(sur_rows) != 24:
            raise RuntimeError(
                f"Expected 24 surrogate rows for scale={scale}; "
                f"found rows={len(sur_rows)}, unique seeds={sur_rows['seed'].nunique()}."
            )

        for metric in metric_names:
            values = sur_rows[metric].to_numpy(dtype=float)
            comp = float(comp_row[metric])
            null_mean = float(np.mean(values))
            null_std = float(np.std(values, ddof=1))
            n_ge = int(np.count_nonzero(values >= comp))
            rows.append({
                "comparator": comparator_name,
                "scale": scale,
                "metric": metric,
                **metric_metadata(metric),
                "n_surrogates": len(values),
                "comparator_value": comp,
                "surrogate_mean": null_mean,
                "surrogate_std": null_std,
                "surrogate_median": float(np.median(values)),
                "surrogate_min": float(np.min(values)),
                "surrogate_max": float(np.max(values)),
                "surrogate_q05": float(np.quantile(values, 0.05)),
                "surrogate_q95": float(np.quantile(values, 0.95)),
                "comparator_minus_surrogate_mean": float(comp - null_mean),
                "z_effect_size": float(
                    (comp - null_mean) / max(null_std, np.finfo(float).tiny)
                ),
                "n_surrogate_ge_comparator": n_ge,
                "empirical_p_greater": float((n_ge + 1) / (len(values) + 1)),
                "comparator_above_all": bool(n_ge == 0),
            })
    return pd.DataFrame(rows)


def flatten_qc(seed: int, qc: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "seed": seed,
        "qc_pass": bool(qc.get("qc_pass", False)),
        "errors": " | ".join(map(str, qc.get("errors", []))),
    }
    checks = qc.get("checks", {})
    if isinstance(checks, dict):
        row.update(checks)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the held-out 24-seed isotropic replication and "
            "apply the pre-specified six-endpoint criterion."
        )
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(
            "phase_ensemble_v1/runs/64_rep2_xyz385_t2501_2600"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(
            "phase_ensemble_v1/summaries/isotropic_rep2_24"
        ),
    )
    parser.add_argument("--seed-start", type=int, default=2026072001)
    parser.add_argument("--seed-end", type=int, default=2026072024)
    parser.add_argument(
        "--strict-exit",
        action="store_true",
        help="Exit with status 2 if the formal result is FAIL.",
    )
    args = parser.parse_args()

    expected_seeds = list(range(args.seed_start, args.seed_end + 1))
    if len(expected_seeds) != 24:
        raise RuntimeError(
            f"The formal protocol expects 24 seeds; got {len(expected_seeds)}."
        )
    args.outdir.mkdir(parents=True, exist_ok=True)

    reference = load_case_metrics(
        args.base / "reference_star32" / "finite_time_v3",
        case_type="reference_star32",
        case_label="reference_star32_rep2",
        seed=None,
    )
    dns = load_case_metrics(
        args.base / "dns_raw" / "finite_time_v3",
        case_type="dns_raw",
        case_label="dns_rep2_raw",
        seed=None,
    )

    surrogate_frames: list[pd.DataFrame] = []
    qc_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for seed in expected_seeds:
        run_dir = args.base / f"seed_{seed}"
        v3_dir = run_dir / "finite_time_v3"
        required = [
            run_dir / "DONE",
            run_dir / "qc_status.json",
            run_dir / "phase_hash.json",
            v3_dir / "instantaneous_arrow_metrics_v3.csv",
            v3_dir / "finite_time_arrow_summary_v3.csv",
        ]
        for path in required:
            if not path.exists():
                missing.append(str(path))
        if (run_dir / "FAILED").exists():
            missing.append(f"FAILED marker present: {run_dir / 'FAILED'}")
        if any(not path.exists() for path in required):
            continue

        surrogate_frames.append(
            load_case_metrics(
                v3_dir,
                case_type="surrogate",
                case_label=f"rep2_surrogate_{seed}",
                seed=seed,
            )
        )
        qc = json.loads((run_dir / "qc_status.json").read_text(encoding="utf-8"))
        qc_rows.append(flatten_qc(seed, qc))
        phase = json.loads((run_dir / "phase_hash.json").read_text(encoding="utf-8"))
        phase_rows.append({
            "seed": seed,
            "phase_sha256": phase.get("phase_sha256"),
            "hermitian_error_max": phase.get("hermitian_error_max"),
            "unit_modulus_error_max": phase.get("unit_modulus_error_max"),
            "inverse_imaginary_ratio": phase.get("inverse_imaginary_ratio"),
        })

    if missing:
        raise RuntimeError(
            "Independent replication is incomplete:\n" + "\n".join(missing)
        )

    surrogates = pd.concat(surrogate_frames, ignore_index=True)
    qc_df = pd.DataFrame(qc_rows)
    phase_df = pd.DataFrame(phase_rows)

    if surrogates["seed"].nunique() != 24:
        raise RuntimeError(
            "Expected 24 unique surrogate seeds; found "
            f"{surrogates['seed'].nunique()}."
        )
    if len(qc_df) != 24 or len(phase_df) != 24:
        raise RuntimeError("Expected 24 QC and 24 phase-hash records.")

    unique_phase_hashes = int(phase_df["phase_sha256"].dropna().nunique())

    all_cases = pd.concat([dns, reference, surrogates], ignore_index=True)
    all_cases.to_csv(
        args.outdir / "isotropic_rep2_all_case_metrics.csv", index=False
    )
    qc_df.to_csv(args.outdir / "isotropic_rep2_seed_qc.csv", index=False)
    phase_df.to_csv(
        args.outdir / "isotropic_rep2_phase_hashes.csv", index=False
    )

    metric_names = ["mean_Pi", "normalized_mean_Pi", "sign_bias_Pi"]
    for window in WINDOWS:
        metric_names.extend([
            f"mean_I_w{window}",
            f"normalized_mean_I_w{window}",
            f"sign_bias_I_w{window}",
        ])

    reference_summary = summarize_comparator(
        reference,
        surrogates,
        comparator_name="reference_star32",
        metric_names=metric_names,
    )
    dns_summary = summarize_comparator(
        dns,
        surrogates,
        comparator_name="dns_raw",
        metric_names=metric_names,
    )
    comparison = pd.concat([reference_summary, dns_summary], ignore_index=True)
    comparison.to_csv(
        args.outdir / "isotropic_rep2_comparator_vs_24.csv", index=False
    )
    reference_summary.to_csv(
        args.outdir / "isotropic_rep2_reference_vs_24.csv", index=False
    )
    dns_summary.to_csv(
        args.outdir / "isotropic_rep2_dns_vs_24.csv", index=False
    )

    fidelity_rows: list[dict[str, Any]] = []
    for scale in SCALES:
        ref_row = require_one(
            reference.loc[reference["scale"] == scale],
            f"reference fidelity scale={scale}",
        )
        dns_row = require_one(
            dns.loc[dns["scale"] == scale],
            f"DNS fidelity scale={scale}",
        )
        sur_rows = surrogates.loc[surrogates["scale"] == scale]
        for metric in metric_names:
            ref_value = float(ref_row[metric])
            dns_value = float(dns_row[metric])
            null_mean = float(sur_rows[metric].mean())
            dns_null_gap = dns_value - null_mean
            fidelity_rows.append({
                "scale": scale,
                "metric": metric,
                **metric_metadata(metric),
                "dns_value": dns_value,
                "reference_value": ref_value,
                "reference_minus_dns": ref_value - dns_value,
                "absolute_relative_difference_vs_dns": float(
                    abs(ref_value - dns_value)
                    / max(abs(dns_value), np.finfo(float).tiny)
                ),
                "surrogate_mean": null_mean,
                "retained_dns_null_separation": float(
                    (ref_value - null_mean)
                    / max(abs(dns_null_gap), np.finfo(float).tiny)
                ),
            })

    fidelity_df = pd.DataFrame(fidelity_rows)
    fidelity_df.to_csv(
        args.outdir / "isotropic_rep2_reference_dns_fidelity.csv", index=False
    )

    primary = reference_summary.loc[
        reference_summary["metric"].isin(PRIMARY_METRICS)
    ].copy()
    primary_pass = bool(
        len(primary) == 6 and primary["comparator_above_all"].all()
    )
    qc_pass = bool(
        len(qc_df) == 24
        and qc_df["qc_pass"].all()
        and unique_phase_hashes == 24
        and (args.base / "ENSEMBLE_24_DONE").exists()
    )
    status = "PASS" if primary_pass and qc_pass else "FAIL"

    primary_output = primary[[
        "scale",
        "metric",
        "comparator_value",
        "surrogate_mean",
        "surrogate_std",
        "surrogate_max",
        "z_effect_size",
        "n_surrogate_ge_comparator",
        "empirical_p_greater",
        "comparator_above_all",
    ]].sort_values(["scale", "metric"])
    primary_output.to_csv(
        args.outdir / "isotropic_rep2_primary_six.csv", index=False
    )

    report = {
        "status": status,
        "base": str(args.base),
        "outdir": str(args.outdir),
        "formal_comparator": (
            "reference_star32 is the identity-phase output of the same "
            "projection/rescaling operator used for the phase surrogates. "
            "dns_raw is reported as a fidelity and robustness comparator."
        ),
        "expected_seeds": expected_seeds,
        "n_unique_surrogate_seeds": int(surrogates["seed"].nunique()),
        "n_qc_pass": int(qc_df["qc_pass"].sum()),
        "n_unique_phase_hashes": unique_phase_hashes,
        "ensemble_marker_present": bool(
            (args.base / "ENSEMBLE_24_DONE").exists()
        ),
        "primary_definition": {
            "scales": list(SCALES),
            "metrics": list(PRIMARY_METRICS),
            "finite_time_mode": "raw_zero",
            "criterion": (
                "reference_star32 strictly exceeds all 24 surrogates for all "
                "six pre-specified comparisons"
            ),
        },
        "minimum_empirical_p_if_no_exceedance": 1.0 / 25.0,
        "primary_pass": primary_pass,
        "qc_pass": qc_pass,
        "primary_results": primary_output.to_dict(orient="records"),
    }

    for marker in (
        args.outdir / "ISOTROPIC_REP2_FORMAL_PASS",
        args.outdir / "ISOTROPIC_REP2_FORMAL_FAIL",
    ):
        if marker.exists():
            marker.unlink()

    summary_path = args.outdir / "isotropic_rep2_formal_summary.json"
    summary_path.write_text(
        json.dumps(native(report), indent=2), encoding="utf-8"
    )
    (args.outdir / f"ISOTROPIC_REP2_FORMAL_{status}").write_text(
        json.dumps(native(report), indent=2), encoding="utf-8"
    )

    print("=" * 110)
    print("ISOTROPIC HELD-OUT 24-SEED FORMAL RESULT")
    print("=" * 110)
    print("status                 =", status)
    print("formal comparator      = reference_star32")
    print("unique surrogate seeds =", surrogates["seed"].nunique())
    print("QC pass                =", qc_pass)
    print("unique phase hashes    =", unique_phase_hashes)
    print("primary pass           =", primary_pass)
    print()
    print(primary_output.to_string(index=False))
    print()
    print("DNS/reference primary fidelity:")
    primary_fidelity = fidelity_df.loc[
        fidelity_df["metric"].isin(PRIMARY_METRICS)
    ][[
        "scale",
        "metric",
        "dns_value",
        "reference_value",
        "reference_minus_dns",
        "absolute_relative_difference_vs_dns",
        "retained_dns_null_separation",
    ]].sort_values(["scale", "metric"])
    print(primary_fidelity.to_string(index=False))
    print()
    print("output =", args.outdir)

    if status != "PASS" and args.strict_exit:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
