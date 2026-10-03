#!/usr/bin/env python3
from pathlib import Path
import json

import numpy as np
import pandas as pd


BASE = Path(
    "phase_ensemble_v1/summaries/"
    "dynamic_recovery/amplitude_matched_3x3"
)

PRIOR = Path(
    "phase_ensemble_v1/summaries/"
    "dynamic_recovery/heldout_3x3_aggregate/"
    "heldout_3x3_case_summary.csv"
)

OUTDIR = Path(
    "phase_ensemble_v1/summaries/"
    "dynamic_recovery/amplitude_matched_3x3_aggregate"
)

METRICS = [
    "Pi_normalized_mean",
    "Pi_A_pop",
    "Pi_A_amp",
]

PRIMARY = "Pi_normalized_mean"

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


def auc_0_1(frame, column):
    frame = (
        frame[
            frame["relative_time"] <= 1.0 + 1e-10
        ]
        .sort_values("relative_time")
    )

    time = frame[
        "relative_time"
    ].to_numpy(dtype=float)

    values = frame[
        column
    ].to_numpy(dtype=float)

    if (
        abs(time[0]) > 1e-10
        or abs(time[-1] - 1.0) > 1e-10
    ):
        raise RuntimeError(
            "AUC interval is not exactly [0,1]."
        )

    return float(
        np.trapezoid(values, time)
    )


def at_time(frame, column, target):
    selected = frame[
        np.isclose(
            frame["relative_time"],
            target,
            rtol=0.0,
            atol=1e-10,
        )
    ]

    if len(selected) != 1:
        raise RuntimeError(
            "Expected one row at t={}".format(
                target
            )
        )

    return float(
        selected.iloc[0][column]
    )


def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_paths = sorted(
        BASE.glob(
            "checkpoint_*/seed_*/"
            "amplitude_matched_counterfactual_audit.json"
        )
    )

    if len(audit_paths) != 9:
        raise RuntimeError(
            "Expected 9 audit files, found {}."
            .format(len(audit_paths))
        )

    audit_rows = []
    metric_rows = []
    long_frames = []

    for audit_path in audit_paths:
        audit = json.loads(
            audit_path.read_text()
        )

        configuration = audit[
            "configuration"
        ]

        checkpoint = int(
            configuration[
                "checkpoint_step"
            ]
        )

        phase_seed = int(
            configuration[
                "phase_seed"
            ]
        )

        csv_path = (
            audit_path.parent
            / "amplitude_matched_counterfactual.csv"
        )

        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)

        data = pd.read_csv(csv_path)

        data["checkpoint"] = checkpoint
        data["phase_seed"] = phase_seed
        data["source_file"] = str(csv_path)

        long_frames.append(data)

        errors = audit[
            "maximum_errors"
        ]

        audit_rows.append({
            "checkpoint":
                checkpoint,
            "phase_seed":
                phase_seed,
            "pass":
                bool(audit["pass"]),
            "modal_amplitude_error":
                float(
                    errors[
                        "modal_amplitude_relative_max_error"
                    ]
                ),
            "energy_relative_error":
                float(
                    errors[
                        "energy_relative_error"
                    ]
                ),
            "divergence_relative_error":
                float(
                    errors[
                        "divergence_relative_error"
                    ]
                ),
            "dealias_leakage":
                float(
                    errors[
                        "dealias_leakage"
                    ]
                ),
            "hermitian_error":
                float(
                    errors[
                        "hermitian_error"
                    ]
                ),
            "nonlinear_replay_error":
                float(
                    errors[
                        "nonlinear_replay_max_abs_error"
                    ]
                ),
        })

        for metric in METRICS:
            recovery_column = (
                "R_" + metric
            )

            nonlinear = data[
                data["branch"]
                == "nonlinear_recovery"
            ].copy()

            frozen = data[
                data["branch"]
                == "amplitude_matched_frozen"
            ].copy()

            auc_nonlinear = auc_0_1(
                nonlinear,
                recovery_column,
            )

            auc_frozen = auc_0_1(
                frozen,
                recovery_column,
            )

            metric_rows.append({
                "checkpoint":
                    checkpoint,
                "phase_seed":
                    phase_seed,
                "metric":
                    metric,
                "auc_nonlinear_0_1":
                    auc_nonlinear,
                "auc_frozen_0_1":
                    auc_frozen,
                "delta_auc_nonlinear_minus_frozen":
                    auc_nonlinear
                    - auc_frozen,
                "R_nonlinear_t1":
                    at_time(
                        nonlinear,
                        recovery_column,
                        1.0,
                    ),
                "R_frozen_t1":
                    at_time(
                        frozen,
                        recovery_column,
                        1.0,
                    ),
                "R_nonlinear_t3":
                    at_time(
                        nonlinear,
                        recovery_column,
                        3.0,
                    ),
                "R_frozen_t3":
                    at_time(
                        frozen,
                        recovery_column,
                        3.0,
                    ),
            })

    audit_table = pd.DataFrame(
        audit_rows
    ).sort_values([
        "checkpoint",
        "phase_seed",
    ])

    case_summary = pd.DataFrame(
        metric_rows
    ).sort_values([
        "metric",
        "checkpoint",
        "phase_seed",
    ])

    long_table = pd.concat(
        long_frames,
        ignore_index=True,
    )

    observed_checkpoints = sorted(
        audit_table[
            "checkpoint"
        ].unique().tolist()
    )

    observed_seeds = sorted(
        audit_table[
            "phase_seed"
        ].unique().tolist()
    )

    if observed_checkpoints != EXPECTED_CHECKPOINTS:
        raise RuntimeError(
            "Unexpected checkpoints: {}"
            .format(observed_checkpoints)
        )

    if observed_seeds != EXPECTED_SEEDS:
        raise RuntimeError(
            "Unexpected phase seeds: {}"
            .format(observed_seeds)
        )

    summary_rows = []

    for metric, group in case_summary.groupby(
        "metric"
    ):
        delta = group[
            "delta_auc_nonlinear_minus_frozen"
        ].to_numpy(dtype=float)

        summary_rows.append({
            "metric":
                metric,
            "n_cases":
                int(len(group)),
            "positive_delta_count":
                int(
                    np.count_nonzero(
                        delta > 0.0
                    )
                ),
            "median_auc_nonlinear":
                float(
                    group[
                        "auc_nonlinear_0_1"
                    ].median()
                ),
            "median_auc_frozen":
                float(
                    group[
                        "auc_frozen_0_1"
                    ].median()
                ),
            "median_delta_auc":
                float(np.median(delta)),
            "minimum_delta_auc":
                float(np.min(delta)),
            "median_R_nonlinear_t1":
                float(
                    group[
                        "R_nonlinear_t1"
                    ].median()
                ),
            "median_R_frozen_t1":
                float(
                    group[
                        "R_frozen_t1"
                    ].median()
                ),
            "minimum_R_nonlinear_t1":
                float(
                    group[
                        "R_nonlinear_t1"
                    ].min()
                ),
            "maximum_R_frozen_t1":
                float(
                    group[
                        "R_frozen_t1"
                    ].max()
                ),
        })

    metric_summary = pd.DataFrame(
        summary_rows
    ).sort_values("metric")

    primary = case_summary[
        case_summary["metric"] == PRIMARY
    ].copy()

    checkpoint_robustness = (
        primary
        .groupby(
            "checkpoint",
            as_index=False,
        )
        .agg(
            n_phase_seeds=(
                "phase_seed",
                "size",
            ),
            minimum_delta_auc=(
                "delta_auc_nonlinear_minus_frozen",
                "min",
            ),
            mean_delta_auc=(
                "delta_auc_nonlinear_minus_frozen",
                "mean",
            ),
            median_delta_auc=(
                "delta_auc_nonlinear_minus_frozen",
                "median",
            ),
        )
    )

    seed_robustness = (
        primary
        .groupby(
            "phase_seed",
            as_index=False,
        )
        .agg(
            n_checkpoints=(
                "checkpoint",
                "size",
            ),
            minimum_delta_auc=(
                "delta_auc_nonlinear_minus_frozen",
                "min",
            ),
            mean_delta_auc=(
                "delta_auc_nonlinear_minus_frozen",
                "mean",
            ),
            median_delta_auc=(
                "delta_auc_nonlinear_minus_frozen",
                "median",
            ),
        )
    )

    # Merge the previous linear-control results to form
    # a three-control hierarchy.
    prior = pd.read_csv(PRIOR)

    comparison = case_summary.merge(
        prior[[
            "checkpoint",
            "phase_seed",
            "metric",
            "auc_recovery_nonlinear_0_1",
            "auc_recovery_linear_0_1",
            "R_nonlinear_t1",
            "R_linear_t1",
        ]],
        on=[
            "checkpoint",
            "phase_seed",
            "metric",
        ],
        suffixes=(
            "_amplitude_matched",
            "_prior",
        ),
        validate="one_to_one",
    )

    nonlinear_consistency = float(
        np.max(
            np.abs(
                comparison[
                    "auc_nonlinear_0_1"
                ]
                -
                comparison[
                    "auc_recovery_nonlinear_0_1"
                ]
            )
        )
    )

    hierarchy_rows = []

    for metric, group in comparison.groupby(
        "metric"
    ):
        hierarchy_rows.append({
            "metric":
                metric,
            "median_auc_nonlinear":
                float(
                    group[
                        "auc_nonlinear_0_1"
                    ].median()
                ),
            "median_auc_amplitude_matched_frozen":
                float(
                    group[
                        "auc_frozen_0_1"
                    ].median()
                ),
            "median_auc_linear_control":
                float(
                    group[
                        "auc_recovery_linear_0_1"
                    ].median()
                ),
            "median_R_nonlinear_t1":
                float(
                    group[
                        "R_nonlinear_t1_amplitude_matched"
                    ].median()
                ),
            "median_R_amplitude_matched_frozen_t1":
                float(
                    group[
                        "R_frozen_t1"
                    ].median()
                ),
            "median_R_linear_t1":
                float(
                    group[
                        "R_linear_t1"
                    ].median()
                ),
        })

    hierarchy = pd.DataFrame(
        hierarchy_rows
    ).sort_values("metric")

    delta = primary[
        "delta_auc_nonlinear_minus_frozen"
    ].to_numpy(dtype=float)

    gate_checks = {
        "all_9_cases_present": bool(
            len(primary) == 9
        ),
        "all_9_numerical_audits_pass": bool(
            audit_table["pass"].all()
        ),
        "all_9_positive_delta_auc": bool(
            np.count_nonzero(
                delta > 0.0
            ) == 9
        ),
        "median_delta_auc_exceeds_0_4": bool(
            np.median(delta) > 0.4
        ),
        "median_nonlinear_R_t1_at_least_0_8": bool(
            primary[
                "R_nonlinear_t1"
            ].median() >= 0.8
        ),
        "median_frozen_R_t1_below_0_4": bool(
            primary[
                "R_frozen_t1"
            ].median() < 0.4
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
        "modal_amplitude_matching": bool(
            audit_table[
                "modal_amplitude_error"
            ].max() < 1e-12
        ),
        "energy_matching": bool(
            audit_table[
                "energy_relative_error"
            ].max() < 1e-12
        ),
        "divergence_control": bool(
            audit_table[
                "divergence_relative_error"
            ].max() < 1e-12
        ),
        "dealias_control": bool(
            audit_table[
                "dealias_leakage"
            ].max() < 1e-12
        ),
        "hermitian_control": bool(
            audit_table[
                "hermitian_error"
            ].max() < 1e-8
        ),
        "nonlinear_replay_consistency": bool(
            nonlinear_consistency < 1e-12
        ),
    }

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
            "primary_metric":
                PRIMARY,
            "counterfactual": (
                "Exact vector-modal amplitude matching "
                "with the initial randomized normalized "
                "complex vector direction frozen."
            ),
        },
        "primary_results": {
            "median_auc_nonlinear":
                float(
                    primary[
                        "auc_nonlinear_0_1"
                    ].median()
                ),
            "median_auc_frozen":
                float(
                    primary[
                        "auc_frozen_0_1"
                    ].median()
                ),
            "median_delta_auc":
                float(np.median(delta)),
            "minimum_delta_auc":
                float(np.min(delta)),
            "median_R_nonlinear_t1":
                float(
                    primary[
                        "R_nonlinear_t1"
                    ].median()
                ),
            "median_R_frozen_t1":
                float(
                    primary[
                        "R_frozen_t1"
                    ].median()
                ),
            "minimum_R_nonlinear_t1":
                float(
                    primary[
                        "R_nonlinear_t1"
                    ].min()
                ),
            "maximum_R_frozen_t1":
                float(
                    primary[
                        "R_frozen_t1"
                    ].max()
                ),
        },
        "maximum_errors": {
            "modal_amplitude_error":
                float(
                    audit_table[
                        "modal_amplitude_error"
                    ].max()
                ),
            "energy_relative_error":
                float(
                    audit_table[
                        "energy_relative_error"
                    ].max()
                ),
            "divergence_relative_error":
                float(
                    audit_table[
                        "divergence_relative_error"
                    ].max()
                ),
            "hermitian_error":
                float(
                    audit_table[
                        "hermitian_error"
                    ].max()
                ),
            "nonlinear_replay_error":
                float(
                    audit_table[
                        "nonlinear_replay_error"
                    ].max()
                ),
            "cross_experiment_nonlinear_auc_error":
                nonlinear_consistency,
        },
        "checks":
            gate_checks,
        "gate_pass": bool(
            all(gate_checks.values())
        ),
        "interpretation_boundary": (
            "The counterfactual matches scalar vector-modal "
            "energy at every wavevector but does not match "
            "the component-resolved spectral tensor, because "
            "polarization is intentionally frozen. The nine "
            "cases form a crossed 3-by-3 design and are not "
            "nine independent identically distributed runs."
        ),
    }

    long_table.to_csv(
        OUTDIR
        / "amplitude_matched_3x3_long.csv",
        index=False,
    )

    audit_table.to_csv(
        OUTDIR
        / "amplitude_matched_3x3_numerical_audit.csv",
        index=False,
    )

    case_summary.to_csv(
        OUTDIR
        / "amplitude_matched_3x3_case_summary.csv",
        index=False,
    )

    metric_summary.to_csv(
        OUTDIR
        / "amplitude_matched_3x3_metric_summary.csv",
        index=False,
    )

    checkpoint_robustness.to_csv(
        OUTDIR
        / "amplitude_matched_checkpoint_robustness.csv",
        index=False,
    )

    seed_robustness.to_csv(
        OUTDIR
        / "amplitude_matched_seed_robustness.csv",
        index=False,
    )

    comparison.to_csv(
        OUTDIR
        / "three_control_case_comparison.csv",
        index=False,
    )

    hierarchy.to_csv(
        OUTDIR
        / "three_control_metric_hierarchy.csv",
        index=False,
    )

    (
        OUTDIR
        / "amplitude_matched_3x3_gate_audit.json"
    ).write_text(
        json.dumps(
            gate_audit,
            indent=2,
        )
    )

    print("=" * 100)
    print("AMPLITUDE-MATCHED 3x3 PRIMARY GATE")
    print("=" * 100)

    print(
        primary[[
            "checkpoint",
            "phase_seed",
            "auc_nonlinear_0_1",
            "auc_frozen_0_1",
            "delta_auc_nonlinear_minus_frozen",
            "R_nonlinear_t1",
            "R_frozen_t1",
        ]].to_string(index=False)
    )

    print("\nMETRIC SUMMARY")
    print(
        metric_summary.to_string(
            index=False
        )
    )

    print("\nTHREE-CONTROL HIERARCHY")
    print(
        hierarchy.to_string(
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

    if not gate_audit["gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
