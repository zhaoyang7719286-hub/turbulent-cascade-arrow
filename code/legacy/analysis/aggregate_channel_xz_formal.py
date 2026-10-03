#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(
            "phase_ensemble_v1/runs/"
            "channel_xz_heldout_t3001_3100"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(
            "phase_ensemble_v1/summaries/"
            "channel_xz_heldout_24"
        ),
    )
    parser.add_argument(
        "--seed_start",
        type=int,
        default=2026072201,
    )
    parser.add_argument(
        "--seed_end",
        type=int,
        default=2026072224,
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    expected_seeds = list(
        range(args.seed_start, args.seed_end + 1)
    )

    reference_path = (
        args.base / "reference" / "case_metrics.csv"
    )
    reference_qc_path = (
        args.base / "reference" / "case_qc_summary.json"
    )

    if not reference_path.exists():
        raise FileNotFoundError(reference_path)

    reference = pd.read_csv(reference_path)
    reference_qc = json.loads(
        reference_qc_path.read_text(encoding="utf-8")
    )

    surrogate_frames = []
    qc_rows = []
    missing = []

    for seed in expected_seeds:
        run_dir = args.base / f"seed_{seed}"
        metrics_path = run_dir / "case_metrics.csv"
        qc_path = run_dir / "case_qc_summary.json"
        done_path = run_dir / "DONE"

        if not (
            metrics_path.exists()
            and qc_path.exists()
            and done_path.exists()
        ):
            missing.append(seed)
            continue

        frame = pd.read_csv(metrics_path)
        frame["seed"] = seed
        surrogate_frames.append(frame)

        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        qc_rows.append(qc)

    if missing:
        raise RuntimeError(f"Missing/incomplete seeds: {missing}")

    surrogates = pd.concat(
        surrogate_frames,
        ignore_index=True,
    )
    qc_df = pd.DataFrame(qc_rows)

    all_metrics = pd.concat(
        [reference, surrogates],
        ignore_index=True,
    )
    all_metrics.to_csv(
        args.outdir / "channel_xz_all_case_metrics.csv",
        index=False,
    )
    qc_df.to_csv(
        args.outdir / "channel_xz_seed_qc.csv",
        index=False,
    )

    metric_names = [
        "mean_Pi",
        "normalized_mean_Pi",
        "sign_bias_Pi",
        "sign_bias_I_w10",
    ]

    rows = []
    for branch in ("profile_removed", "total_velocity"):
        for scale in (4, 8):
            ref_row = reference.loc[
                (reference["branch"] == branch)
                & (reference["scale"] == scale)
            ]
            sur_rows = surrogates.loc[
                (surrogates["branch"] == branch)
                & (surrogates["scale"] == scale)
            ]

            if len(ref_row) != 1:
                raise RuntimeError(
                    f"Reference row count={len(ref_row)} "
                    f"for {branch}, scale={scale}"
                )
            if len(sur_rows) != len(expected_seeds):
                raise RuntimeError(
                    f"Surrogate row count={len(sur_rows)} "
                    f"for {branch}, scale={scale}"
                )

            ref_row = ref_row.iloc[0]

            for metric in metric_names:
                values = sur_rows[metric].to_numpy(dtype=float)
                ref = float(ref_row[metric])
                n_ge = int(np.count_nonzero(values >= ref))
                std = float(np.std(values, ddof=1))
                rows.append({
                    "branch": branch,
                    "scale": scale,
                    "metric": metric,
                    "n_surrogates": len(values),
                    "reference": ref,
                    "surrogate_mean": float(np.mean(values)),
                    "surrogate_std": std,
                    "surrogate_min": float(np.min(values)),
                    "surrogate_max": float(np.max(values)),
                    "reference_minus_surrogate_mean": float(
                        ref - np.mean(values)
                    ),
                    "z_reference_vs_surrogate": float(
                        (ref - np.mean(values))
                        / max(std, np.finfo(float).tiny)
                    ),
                    "n_surrogate_ge_reference": n_ge,
                    "empirical_p_greater": float(
                        (n_ge + 1) / (len(values) + 1)
                    ),
                    "reference_above_all": bool(n_ge == 0),
                })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(
        args.outdir / "channel_xz_reference_vs_24.csv",
        index=False,
    )

    primary_metrics = (
        "mean_Pi",
        "sign_bias_Pi",
        "sign_bias_I_w10",
    )
    primary = summary_df.loc[
        (summary_df["branch"] == "profile_removed")
        & summary_df["metric"].isin(primary_metrics)
    ]

    primary_pass = bool(
        len(primary) == 6
        and primary["reference_above_all"].all()
    )
    qc_pass = bool(
        reference_qc.get("qc_pass", False)
        and len(qc_df) == len(expected_seeds)
        and qc_df["qc_pass"].all()
    )
    status = (
        "PASS"
        if primary_pass and qc_pass
        else "FAIL"
    )

    report = {
        "status": status,
        "base": str(args.base),
        "expected_seeds": expected_seeds,
        "n_unique_seeds": int(
            surrogates["seed"].nunique()
        ),
        "primary_definition": {
            "branch": "profile_removed",
            "scales": [4, 8],
            "metrics": list(primary_metrics),
            "criterion": (
                "reference exceeds all 24 surrogates "
                "for all six pre-registered comparisons"
            ),
        },
        "primary_pass": primary_pass,
        "qc_pass": qc_pass,
        "minimum_empirical_p_if_no_exceedance": 1.0 / 25.0,
    }

    (args.outdir / "channel_xz_formal_summary.json").write_text(
        json.dumps(native(report), indent=2),
        encoding="utf-8",
    )
    (args.outdir / f"CHANNEL_XZ_FORMAL_{status}").write_text(
        json.dumps(native(report), indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print("CHANNEL X-Z HELD-OUT FORMAL RESULT")
    print("=" * 100)
    print("status =", status)
    print("unique seeds =", surrogates["seed"].nunique())
    print("primary pass =", primary_pass)
    print("QC pass      =", qc_pass)
    print()
    print(
        primary[
            [
                "scale",
                "metric",
                "reference",
                "surrogate_mean",
                "surrogate_max",
                "n_surrogate_ge_reference",
                "empirical_p_greater",
            ]
        ].to_string(index=False)
    )
    print()
    print("output =", args.outdir)

    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
