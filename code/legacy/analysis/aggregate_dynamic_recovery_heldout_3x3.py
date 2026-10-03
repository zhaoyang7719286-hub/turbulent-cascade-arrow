#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(
    "phase_ensemble_v1/summaries/"
    "dynamic_recovery/heldout_3x3"
)

OUTDIR = Path(
    "phase_ensemble_v1/summaries/"
    "dynamic_recovery/heldout_3x3_aggregate"
)

METRICS = [
    "Pi_normalized_mean",
    "Pi_A_pop",
    "Pi_A_amp",
]

PRIMARY_METRIC = "Pi_normalized_mean"

EXPECTED_CHECKPOINTS = [
    1750,
    2250,
    2750,
]

EXPECTED_SEEDS = [
    2026074101,
    2026074102,
    2026074103,
]

EARLY_HORIZON = 1.0
FINAL_HORIZON = 3.0
TIME_TOL = 1e-8


def parse_case(path):
    checkpoint_match = re.search(
        r"checkpoint_(\d+)",
        str(path),
    )

    seed_match = re.search(
        r"seed_(\d+)",
        str(path),
    )

    if checkpoint_match is None:
        raise RuntimeError(
            "Cannot parse checkpoint from {}".format(path)
        )

    if seed_match is None:
        raise RuntimeError(
            "Cannot parse seed from {}".format(path)
        )

    return (
        int(checkpoint_match.group(1)),
        int(seed_match.group(1)),
    )


def interpolate_value(
    time_values,
    values,
    target_time,
):
    time_values = np.asarray(
        time_values,
        dtype=float,
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    order = np.argsort(time_values)

    return float(
        np.interp(
            target_time,
            time_values[order],
            values[order],
        )
    )


def first_crossing_time(
    time_values,
    recovery_values,
    threshold,
):
    time_values = np.asarray(
        time_values,
        dtype=float,
    )

    recovery_values = np.asarray(
        recovery_values,
        dtype=float,
    )

    valid = (
        np.isfinite(time_values)
        & np.isfinite(recovery_values)
    )

    time_values = time_values[valid]
    recovery_values = recovery_values[valid]

    order = np.argsort(time_values)
    time_values = time_values[order]
    recovery_values = recovery_values[order]

    indices = np.flatnonzero(
        recovery_values >= threshold
    )

    if len(indices) == 0:
        return np.nan

    index = int(indices[0])

    if index == 0:
        return float(time_values[0])

    t0 = float(time_values[index - 1])
    t1 = float(time_values[index])

    r0 = float(recovery_values[index - 1])
    r1 = float(recovery_values[index])

    if abs(r1 - r0) < 1e-14:
        return t1

    fraction = (
        threshold - r0
    ) / (
        r1 - r0
    )

    fraction = float(
        np.clip(
            fraction,
            0.0,
            1.0,
        )
    )

    return float(
        t0 + fraction * (t1 - t0)
    )


def normalized_auc(
    time_values,
    recovery_values,
    horizon,
):
    time_values = np.asarray(
        time_values,
        dtype=float,
    )

    recovery_values = np.asarray(
        recovery_values,
        dtype=float,
    )

    valid = (
        np.isfinite(time_values)
        & np.isfinite(recovery_values)
        & (time_values >= -TIME_TOL)
        & (time_values <= horizon + TIME_TOL)
    )

    time_values = time_values[valid]
    recovery_values = recovery_values[valid]

    order = np.argsort(time_values)
    time_values = time_values[order]
    recovery_values = recovery_values[order]

    if len(time_values) < 2:
        raise RuntimeError(
            "Insufficient points for AUC"
        )

    if time_values[0] > TIME_TOL:
        raise RuntimeError(
            "Recovery curve does not begin at t=0"
        )

    if time_values[-1] < horizon - TIME_TOL:
        raise RuntimeError(
            "Recovery curve does not reach requested horizon"
        )

    if abs(time_values[-1] - horizon) > TIME_TOL:
        recovery_at_horizon = interpolate_value(
            time_values,
            recovery_values,
            horizon,
        )

        inside = time_values < horizon

        time_values = np.concatenate([
            time_values[inside],
            [horizon],
        ])

        recovery_values = np.concatenate([
            recovery_values[inside],
            [recovery_at_horizon],
        ])

    integral = np.trapezoid(
        recovery_values,
        time_values,
    )

    return float(
        integral / horizon
    )


def quantile_summary(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean": float(np.mean(values)),
        "std": float(
            np.std(values, ddof=1)
        ) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "q25": float(
            np.quantile(values, 0.25)
        ),
        "median": float(
            np.median(values)
        ),
        "q75": float(
            np.quantile(values, 0.75)
        ),
        "max": float(np.max(values)),
    }


def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = sorted(
        BASE.glob(
            "checkpoint_*/seed_*/"
            "recovery_diagnostics_all_branches.csv"
        )
    )

    if len(paths) != 9:
        raise RuntimeError(
            "Expected 9 diagnostic files, found {}"
            .format(len(paths))
        )

    raw_frames = []
    audit_records = []

    for path in paths:
        checkpoint, seed = parse_case(path)

        df = pd.read_csv(path)

        required_columns = {
            "branch",
            "relative_time",
            *METRICS,
            "corr_Pi_M",
            "energy",
            "dissipation_rate",
            "divergence_relative_error",
            "dealias_leakage",
            "hermitian_error",
        }

        missing = required_columns - set(df.columns)

        if missing:
            raise RuntimeError(
                "{} missing columns: {}"
                .format(path, sorted(missing))
            )

        df["checkpoint"] = checkpoint
        df["phase_seed"] = seed
        df["source_file"] = str(path)

        df["relative_time"] = pd.to_numeric(
            df["relative_time"],
            errors="raise",
        ).round(10)

        raw_frames.append(df)

        audit_path = (
            path.parent
            / "recovery_smoke_audit.json"
        )

        if not audit_path.is_file():
            raise FileNotFoundError(audit_path)

        audit = json.loads(
            audit_path.read_text()
        )

        perturbation = audit[
            "perturbation_audit"
        ]

        branch_audits = audit[
            "branch_audits"
        ]

        audit_records.append({
            "checkpoint":
                checkpoint,
            "phase_seed":
                seed,
            "numerical_pass":
                bool(audit["numerical_pass"]),
            "perturbation_pass":
                bool(perturbation["pass"]),
            "modal_amplitude_error":
                float(
                    perturbation[
                        "modal_vector_amplitude_relative_max_error"
                    ]
                ),
            "energy_relative_error":
                float(
                    perturbation[
                        "energy_relative_error"
                    ]
                ),
            "initial_divergence_error":
                float(
                    perturbation[
                        "divergence_relative_error"
                    ]
                ),
            "initial_dealias_leakage":
                float(
                    perturbation[
                        "dealias_leakage"
                    ]
                ),
            "initial_hermitian_error":
                float(
                    perturbation[
                        "hermitian_error"
                    ]
                ),
            "natural_pass":
                bool(
                    branch_audits[
                        "natural"
                    ]["pass"]
                ),
            "nonlinear_pass":
                bool(
                    branch_audits[
                        "nonlinear_recovery"
                    ]["pass"]
                ),
            "linear_pass":
                bool(
                    branch_audits[
                        "linear_control"
                    ]["pass"]
                ),
        })

    raw = pd.concat(
        raw_frames,
        ignore_index=True,
    )

    audit_df = pd.DataFrame(
        audit_records
    ).sort_values([
        "checkpoint",
        "phase_seed",
    ])

    observed_checkpoints = sorted(
        raw["checkpoint"].unique().tolist()
    )

    observed_seeds = sorted(
        raw["phase_seed"].unique().tolist()
    )

    if observed_checkpoints != EXPECTED_CHECKPOINTS:
        raise RuntimeError(
            "Unexpected checkpoints: {}"
            .format(observed_checkpoints)
        )

    if observed_seeds != EXPECTED_SEEDS:
        raise RuntimeError(
            "Unexpected seeds: {}"
            .format(observed_seeds)
        )

    expected_branches = {
        "natural",
        "nonlinear_recovery",
        "linear_control",
    }

    recovery_rows = []
    case_rows = []

    initial_branch_match_errors = []

    for checkpoint in EXPECTED_CHECKPOINTS:
        for seed in EXPECTED_SEEDS:
            case = raw[
                (raw["checkpoint"] == checkpoint)
                &
                (raw["phase_seed"] == seed)
            ].copy()

            if set(case["branch"].unique()) != expected_branches:
                raise RuntimeError(
                    "Branch mismatch for checkpoint={} seed={}"
                    .format(checkpoint, seed)
                )

            branch_data = {}

            for branch in expected_branches:
                sub = (
                    case[
                        case["branch"] == branch
                    ]
                    .sort_values("relative_time")
                    .reset_index(drop=True)
                )

                branch_data[branch] = sub

            natural_times = branch_data[
                "natural"
            ]["relative_time"].to_numpy(dtype=float)

            for branch in (
                "nonlinear_recovery",
                "linear_control",
            ):
                branch_times = branch_data[
                    branch
                ]["relative_time"].to_numpy(dtype=float)

                if not np.allclose(
                    natural_times,
                    branch_times,
                    rtol=0.0,
                    atol=TIME_TOL,
                ):
                    raise RuntimeError(
                        "Time-grid mismatch for "
                        "checkpoint={} seed={} branch={}"
                        .format(
                            checkpoint,
                            seed,
                            branch,
                        )
                    )

            for metric in METRICS:
                natural_values = branch_data[
                    "natural"
                ][metric].to_numpy(dtype=float)

                nonlinear_values = branch_data[
                    "nonlinear_recovery"
                ][metric].to_numpy(dtype=float)

                linear_values = branch_data[
                    "linear_control"
                ][metric].to_numpy(dtype=float)

                perturbed_initial_nl = float(
                    nonlinear_values[0]
                )

                perturbed_initial_lin = float(
                    linear_values[0]
                )

                initial_match_error = abs(
                    perturbed_initial_nl
                    - perturbed_initial_lin
                )

                initial_branch_match_errors.append(
                    initial_match_error
                )

                natural_initial = float(
                    natural_values[0]
                )

                perturbed_initial = 0.5 * (
                    perturbed_initial_nl
                    + perturbed_initial_lin
                )

                initial_gap = abs(
                    natural_initial
                    - perturbed_initial
                )

                if initial_gap <= 1e-10:
                    raise RuntimeError(
                        "Initial gap too small for "
                        "checkpoint={} seed={} metric={}"
                        .format(
                            checkpoint,
                            seed,
                            metric,
                        )
                    )

                nonlinear_recovery = (
                    1.0
                    - np.abs(
                        nonlinear_values
                        - natural_values
                    ) / initial_gap
                )

                linear_recovery = (
                    1.0
                    - np.abs(
                        linear_values
                        - natural_values
                    ) / initial_gap
                )

                for branch, values, recovery in (
                    (
                        "nonlinear_recovery",
                        nonlinear_values,
                        nonlinear_recovery,
                    ),
                    (
                        "linear_control",
                        linear_values,
                        linear_recovery,
                    ),
                ):
                    for time_value, raw_value, recovery_value in zip(
                        natural_times,
                        values,
                        recovery,
                    ):
                        recovery_rows.append({
                            "checkpoint":
                                checkpoint,
                            "phase_seed":
                                seed,
                            "metric":
                                metric,
                            "branch":
                                branch,
                            "relative_time":
                                float(time_value),
                            "raw_value":
                                float(raw_value),
                            "natural_value":
                                float(
                                    interpolate_value(
                                        natural_times,
                                        natural_values,
                                        time_value,
                                    )
                                ),
                            "natural_initial":
                                natural_initial,
                            "perturbed_initial":
                                perturbed_initial,
                            "initial_gap":
                                initial_gap,
                            "recovery":
                                float(recovery_value),
                        })

                nonlinear_auc = normalized_auc(
                    natural_times,
                    nonlinear_recovery,
                    EARLY_HORIZON,
                )

                linear_auc = normalized_auc(
                    natural_times,
                    linear_recovery,
                    EARLY_HORIZON,
                )

                nonlinear_r_t1 = interpolate_value(
                    natural_times,
                    nonlinear_recovery,
                    EARLY_HORIZON,
                )

                linear_r_t1 = interpolate_value(
                    natural_times,
                    linear_recovery,
                    EARLY_HORIZON,
                )

                nonlinear_r_t3 = interpolate_value(
                    natural_times,
                    nonlinear_recovery,
                    FINAL_HORIZON,
                )

                linear_r_t3 = interpolate_value(
                    natural_times,
                    linear_recovery,
                    FINAL_HORIZON,
                )

                case_rows.append({
                    "checkpoint":
                        checkpoint,
                    "phase_seed":
                        seed,
                    "metric":
                        metric,
                    "natural_initial":
                        natural_initial,
                    "perturbed_initial":
                        perturbed_initial,
                    "initial_gap":
                        initial_gap,
                    "auc_recovery_nonlinear_0_1":
                        nonlinear_auc,
                    "auc_recovery_linear_0_1":
                        linear_auc,
                    "delta_auc_nonlinear_minus_linear":
                        nonlinear_auc - linear_auc,
                    "R_nonlinear_t1":
                        nonlinear_r_t1,
                    "R_linear_t1":
                        linear_r_t1,
                    "delta_R_t1":
                        nonlinear_r_t1 - linear_r_t1,
                    "R_nonlinear_t3":
                        nonlinear_r_t3,
                    "R_linear_t3":
                        linear_r_t3,
                    "delta_R_t3":
                        nonlinear_r_t3 - linear_r_t3,
                    "t50_nonlinear":
                        first_crossing_time(
                            natural_times,
                            nonlinear_recovery,
                            0.5,
                        ),
                    "t90_nonlinear":
                        first_crossing_time(
                            natural_times,
                            nonlinear_recovery,
                            0.9,
                        ),
                    "t50_linear":
                        first_crossing_time(
                            natural_times,
                            linear_recovery,
                            0.5,
                        ),
                    "t90_linear":
                        first_crossing_time(
                            natural_times,
                            linear_recovery,
                            0.9,
                        ),
                    "nonlinear_auc_exceeds_linear":
                        bool(
                            nonlinear_auc
                            > linear_auc
                        ),
                })

    recovery = pd.DataFrame(
        recovery_rows
    )

    case_summary = pd.DataFrame(
        case_rows
    ).sort_values([
        "metric",
        "checkpoint",
        "phase_seed",
    ])

    raw_output = (
        OUTDIR
        / "heldout_3x3_raw_long.csv"
    )

    recovery_output = (
        OUTDIR
        / "heldout_3x3_recovery_long.csv"
    )

    case_output = (
        OUTDIR
        / "heldout_3x3_case_summary.csv"
    )

    audit_table_output = (
        OUTDIR
        / "heldout_3x3_numerical_audit.csv"
    )

    raw.to_csv(
        raw_output,
        index=False,
    )

    recovery.to_csv(
        recovery_output,
        index=False,
    )

    case_summary.to_csv(
        case_output,
        index=False,
    )

    audit_df.to_csv(
        audit_table_output,
        index=False,
    )

    time_summary_rows = []

    for (
        metric,
        branch,
        relative_time,
    ), group in recovery.groupby([
        "metric",
        "branch",
        "relative_time",
    ]):
        values = group[
            "recovery"
        ].to_numpy(dtype=float)

        summary = quantile_summary(values)

        time_summary_rows.append({
            "metric": metric,
            "branch": branch,
            "relative_time":
                float(relative_time),
            "n_cases":
                int(len(values)),
            **summary,
        })

    time_summary = pd.DataFrame(
        time_summary_rows
    ).sort_values([
        "metric",
        "branch",
        "relative_time",
    ])

    time_summary_output = (
        OUTDIR
        / "heldout_3x3_recovery_time_summary.csv"
    )

    time_summary.to_csv(
        time_summary_output,
        index=False,
    )

    metric_summary_rows = []

    for metric, group in case_summary.groupby(
        "metric"
    ):
        delta_auc = group[
            "delta_auc_nonlinear_minus_linear"
        ].to_numpy(dtype=float)

        nonlinear_auc = group[
            "auc_recovery_nonlinear_0_1"
        ].to_numpy(dtype=float)

        linear_auc = group[
            "auc_recovery_linear_0_1"
        ].to_numpy(dtype=float)

        nonlinear_t1 = group[
            "R_nonlinear_t1"
        ].to_numpy(dtype=float)

        linear_t1 = group[
            "R_linear_t1"
        ].to_numpy(dtype=float)

        metric_summary_rows.append({
            "metric": metric,
            "n_cases":
                int(len(group)),
            "positive_delta_auc_count":
                int(
                    np.count_nonzero(
                        delta_auc > 0.0
                    )
                ),
            "positive_delta_auc_fraction":
                float(
                    np.mean(
                        delta_auc > 0.0
                    )
                ),
            "median_auc_nonlinear":
                float(
                    np.median(
                        nonlinear_auc
                    )
                ),
            "median_auc_linear":
                float(
                    np.median(
                        linear_auc
                    )
                ),
            "median_delta_auc":
                float(
                    np.median(
                        delta_auc
                    )
                ),
            "minimum_delta_auc":
                float(
                    np.min(
                        delta_auc
                    )
                ),
            "median_R_nonlinear_t1":
                float(
                    np.median(
                        nonlinear_t1
                    )
                ),
            "median_R_linear_t1":
                float(
                    np.median(
                        linear_t1
                    )
                ),
            "minimum_R_nonlinear_t1":
                float(
                    np.min(
                        nonlinear_t1
                    )
                ),
            "maximum_R_linear_t1":
                float(
                    np.max(
                        linear_t1
                    )
                ),
            "median_t50_nonlinear":
                float(
                    np.nanmedian(
                        group[
                            "t50_nonlinear"
                        ]
                    )
                ),
            "median_t90_nonlinear":
                float(
                    np.nanmedian(
                        group[
                            "t90_nonlinear"
                        ]
                    )
                ),
        })

    metric_summary = pd.DataFrame(
        metric_summary_rows
    ).sort_values("metric")

    metric_summary_output = (
        OUTDIR
        / "heldout_3x3_metric_summary.csv"
    )

    metric_summary.to_csv(
        metric_summary_output,
        index=False,
    )

    primary = case_summary[
        case_summary["metric"]
        == PRIMARY_METRIC
    ].copy()

    checkpoint_rows = []

    for checkpoint, group in primary.groupby(
        "checkpoint"
    ):
        delta = group[
            "delta_auc_nonlinear_minus_linear"
        ].to_numpy(dtype=float)

        checkpoint_rows.append({
            "checkpoint":
                int(checkpoint),
            "n_phase_seeds":
                int(len(group)),
            "positive_count":
                int(
                    np.count_nonzero(
                        delta > 0.0
                    )
                ),
            "minimum_delta_auc":
                float(np.min(delta)),
            "mean_delta_auc":
                float(np.mean(delta)),
            "median_delta_auc":
                float(np.median(delta)),
        })

    checkpoint_robustness = pd.DataFrame(
        checkpoint_rows
    ).sort_values("checkpoint")

    checkpoint_output = (
        OUTDIR
        / "heldout_3x3_checkpoint_robustness.csv"
    )

    checkpoint_robustness.to_csv(
        checkpoint_output,
        index=False,
    )

    seed_rows = []

    for seed, group in primary.groupby(
        "phase_seed"
    ):
        delta = group[
            "delta_auc_nonlinear_minus_linear"
        ].to_numpy(dtype=float)

        seed_rows.append({
            "phase_seed":
                int(seed),
            "n_checkpoints":
                int(len(group)),
            "positive_count":
                int(
                    np.count_nonzero(
                        delta > 0.0
                    )
                ),
            "minimum_delta_auc":
                float(np.min(delta)),
            "mean_delta_auc":
                float(np.mean(delta)),
            "median_delta_auc":
                float(np.median(delta)),
        })

    seed_robustness = pd.DataFrame(
        seed_rows
    ).sort_values("phase_seed")

    seed_output = (
        OUTDIR
        / "heldout_3x3_seed_robustness.csv"
    )

    seed_robustness.to_csv(
        seed_output,
        index=False,
    )

    natural_repeat_spreads = []

    for (
        checkpoint,
        relative_time,
        metric,
    ), group in (
        raw[
            raw["branch"] == "natural"
        ]
        .melt(
            id_vars=[
                "checkpoint",
                "phase_seed",
                "relative_time",
            ],
            value_vars=METRICS,
            var_name="metric",
            value_name="value",
        )
        .groupby([
            "checkpoint",
            "relative_time",
            "metric",
        ])
    ):
        values = group[
            "value"
        ].to_numpy(dtype=float)

        natural_repeat_spreads.append(
            float(
                np.max(values)
                - np.min(values)
            )
        )

    primary_delta = primary[
        "delta_auc_nonlinear_minus_linear"
    ].to_numpy(dtype=float)

    primary_nonlinear_t1 = primary[
        "R_nonlinear_t1"
    ].to_numpy(dtype=float)

    primary_linear_t1 = primary[
        "R_linear_t1"
    ].to_numpy(dtype=float)

    gate_checks = {
        "all_9_cases_present": bool(
            len(primary) == 9
        ),
        "at_least_8_of_9_positive_delta_auc": bool(
            np.count_nonzero(
                primary_delta > 0.0
            ) >= 8
        ),
        "median_delta_auc_exceeds_0_4": bool(
            np.median(
                primary_delta
            ) > 0.4
        ),
        "median_nonlinear_R_t1_at_least_0_8": bool(
            np.median(
                primary_nonlinear_t1
            ) >= 0.8
        ),
        "median_linear_R_t1_at_most_0_4": bool(
            np.median(
                primary_linear_t1
            ) <= 0.4
        ),
        "every_checkpoint_direction_positive": bool(
            (
                checkpoint_robustness[
                    "minimum_delta_auc"
                ] > 0.0
            ).all()
        ),
        "every_phase_seed_direction_positive": bool(
            (
                seed_robustness[
                    "minimum_delta_auc"
                ] > 0.0
            ).all()
        ),
        "all_numerical_audits_pass": bool(
            audit_df[
                [
                    "numerical_pass",
                    "perturbation_pass",
                    "natural_pass",
                    "nonlinear_pass",
                    "linear_pass",
                ]
            ].all().all()
        ),
        "natural_replay_consistency": bool(
            max(
                natural_repeat_spreads
            ) < 1e-12
        ),
        "identical_perturbed_initial_branches": bool(
            max(
                initial_branch_match_errors
            ) < 1e-12
        ),
    }

    gate_pass = bool(
        all(gate_checks.values())
    )

    gate_audit = {
        "design": {
            "checkpoints":
                EXPECTED_CHECKPOINTS,
            "phase_seeds":
                EXPECTED_SEEDS,
            "n_crossed_cases":
                9,
            "filter_scale":
                8,
            "early_horizon":
                EARLY_HORIZON,
            "final_horizon":
                FINAL_HORIZON,
            "primary_metric":
                PRIMARY_METRIC,
            "recovery_definition": (
                "R_b(t)=1-|X_b(t)-X_natural(t)|/"
                "|X_natural(0)-X_perturbed(0)|"
            ),
            "early_auc_definition": (
                "integral_0^1 R_b(t) dt"
            ),
        },
        "primary_results": {
            "positive_delta_auc_count":
                int(
                    np.count_nonzero(
                        primary_delta > 0.0
                    )
                ),
            "median_auc_nonlinear":
                float(
                    np.median(
                        primary[
                            "auc_recovery_nonlinear_0_1"
                        ]
                    )
                ),
            "median_auc_linear":
                float(
                    np.median(
                        primary[
                            "auc_recovery_linear_0_1"
                        ]
                    )
                ),
            "median_delta_auc":
                float(
                    np.median(
                        primary_delta
                    )
                ),
            "minimum_delta_auc":
                float(
                    np.min(
                        primary_delta
                    )
                ),
            "median_R_nonlinear_t1":
                float(
                    np.median(
                        primary_nonlinear_t1
                    )
                ),
            "median_R_linear_t1":
                float(
                    np.median(
                        primary_linear_t1
                    )
                ),
            "minimum_R_nonlinear_t1":
                float(
                    np.min(
                        primary_nonlinear_t1
                    )
                ),
            "maximum_R_linear_t1":
                float(
                    np.max(
                        primary_linear_t1
                    )
                ),
            "median_t50_nonlinear":
                float(
                    np.nanmedian(
                        primary[
                            "t50_nonlinear"
                        ]
                    )
                ),
            "median_t90_nonlinear":
                float(
                    np.nanmedian(
                        primary[
                            "t90_nonlinear"
                        ]
                    )
                ),
        },
        "maximum_errors": {
            "modal_amplitude_error":
                float(
                    audit_df[
                        "modal_amplitude_error"
                    ].max()
                ),
            "energy_relative_error":
                float(
                    audit_df[
                        "energy_relative_error"
                    ].max()
                ),
            "initial_divergence_error":
                float(
                    audit_df[
                        "initial_divergence_error"
                    ].max()
                ),
            "initial_dealias_leakage":
                float(
                    audit_df[
                        "initial_dealias_leakage"
                    ].max()
                ),
            "initial_hermitian_error":
                float(
                    audit_df[
                        "initial_hermitian_error"
                    ].max()
                ),
            "natural_repeat_spread":
                float(
                    max(
                        natural_repeat_spreads
                    )
                ),
            "perturbed_initial_branch_mismatch":
                float(
                    max(
                        initial_branch_match_errors
                    )
                ),
        },
        "checks":
            gate_checks,
        "gate_pass":
            gate_pass,
        "interpretation_boundary": (
            "The nine observations form a crossed "
            "3-checkpoint by 3-phase-seed design and "
            "must not be treated as nine independent "
            "identically distributed realizations."
        ),
    }

    audit_output = (
        OUTDIR
        / "heldout_3x3_gate_audit.json"
    )

    audit_output.write_text(
        json.dumps(
            gate_audit,
            indent=2,
        )
    )

    print("=" * 100)
    print("HELDOUT 3x3 PRIMARY GATE")
    print("=" * 100)

    print(
        primary[[
            "checkpoint",
            "phase_seed",
            "auc_recovery_nonlinear_0_1",
            "auc_recovery_linear_0_1",
            "delta_auc_nonlinear_minus_linear",
            "R_nonlinear_t1",
            "R_linear_t1",
            "t50_nonlinear",
            "t90_nonlinear",
        ]].to_string(index=False)
    )

    print("\nMETRIC SUMMARY")
    print(
        metric_summary.to_string(
            index=False
        )
    )

    print("\nCHECKPOINT ROBUSTNESS")
    print(
        checkpoint_robustness.to_string(
            index=False
        )
    )

    print("\nPHASE-SEED ROBUSTNESS")
    print(
        seed_robustness.to_string(
            index=False
        )
    )

    print("\nGATE AUDIT")
    print(
        json.dumps(
            gate_audit,
            indent=2,
        )
    )

    print("\nsaved:", raw_output)
    print("saved:", recovery_output)
    print("saved:", case_output)
    print("saved:", time_summary_output)
    print("saved:", metric_summary_output)
    print("saved:", checkpoint_output)
    print("saved:", seed_output)
    print("saved:", audit_table_output)
    print("saved:", audit_output)

    if not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
