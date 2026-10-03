#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = load_module(
    "dynamic_recovery_smoke",
    SCRIPT_DIR / "run_dynamic_recovery_smoke.py",
)

dns = smoke.dns


METRICS = [
    "Pi_normalized_mean",
    "Pi_A_pop",
    "Pi_A_amp",
]

AUDIT_METRICS = [
    "Pi_normalized_mean",
    "Pi_A_pop",
    "Pi_A_amp",
    "corr_Pi_M",
    "energy",
    "dissipation_rate",
]


def modal_vector_amplitude(uhat):
    return np.sqrt(
        np.sum(
            np.abs(uhat) ** 2,
            axis=0,
        )
    )


def normalized_complex_direction(uhat):
    amplitude = modal_vector_amplitude(
        uhat
    )

    direction = np.zeros_like(
        uhat,
        dtype=np.complex128,
    )

    threshold = (
        1e-14
        * max(
            float(np.max(amplitude)),
            1.0,
        )
    )

    active = amplitude > threshold

    direction[:, active] = (
        uhat[:, active]
        / amplitude[active]
    )

    return direction, active


def construct_amplitude_matched_field(
    nonlinear_uhat,
    frozen_direction,
    operators,
):
    target_amplitude = modal_vector_amplitude(
        nonlinear_uhat
    )

    frozen = (
        frozen_direction
        * target_amplitude[None, ...]
    )

    frozen = dns.project_dealias(
        frozen,
        operators,
        zero_mean=True,
    )

    reconstructed_amplitude = (
        modal_vector_amplitude(
            frozen
        )
    )

    relevant = target_amplitude > (
        1e-13
        * max(
            float(
                np.max(target_amplitude)
            ),
            1.0,
        )
    )

    if np.any(relevant):
        amplitude_relative_error = float(
            np.max(
                np.abs(
                    reconstructed_amplitude[
                        relevant
                    ]
                    - target_amplitude[
                        relevant
                    ]
                )
                / target_amplitude[
                    relevant
                ]
            )
        )
    else:
        amplitude_relative_error = 0.0

    energy_nonlinear = (
        dns.kinetic_energy_hat(
            nonlinear_uhat
        )
    )

    energy_frozen = (
        dns.kinetic_energy_hat(
            frozen
        )
    )

    energy_relative_error = float(
        abs(
            energy_frozen
            - energy_nonlinear
        )
        / max(
            abs(energy_nonlinear),
            1e-30,
        )
    )

    audit = {
        "modal_amplitude_relative_max_error":
            amplitude_relative_error,
        "energy_relative_error":
            energy_relative_error,
        "divergence_relative_error":
            float(
                dns.divergence_relative_error(
                    frozen,
                    operators,
                )
            ),
        "dealias_leakage":
            float(
                dns.dealiased_leakage(
                    frozen,
                    operators,
                )
            ),
        "hermitian_error":
            float(
                dns.hermitian_error(
                    frozen
                )
            ),
    }

    return frozen, audit


def normalized_auc(
    time_values,
    recovery_values,
    horizon=1.0,
):
    time_values = np.asarray(
        time_values,
        dtype=float,
    )

    recovery_values = np.asarray(
        recovery_values,
        dtype=float,
    )

    order = np.argsort(time_values)

    time_values = time_values[order]
    recovery_values = recovery_values[order]

    inside = time_values <= horizon + 1e-10

    time_values = time_values[inside]
    recovery_values = recovery_values[inside]

    if time_values[-1] < horizon - 1e-10:
        raise RuntimeError(
            "Curve does not reach horizon."
        )

    if abs(time_values[-1] - horizon) > 1e-10:
        value_at_horizon = np.interp(
            horizon,
            time_values,
            recovery_values,
        )

        retain = time_values < horizon

        time_values = np.concatenate([
            time_values[retain],
            [horizon],
        ])

        recovery_values = np.concatenate([
            recovery_values[retain],
            [value_at_horizon],
        ])

    return float(
        np.trapezoid(
            recovery_values,
            time_values,
        ) / horizon
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--reference_csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--phase_seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--viscosity",
        type=float,
        default=0.015,
    )

    parser.add_argument(
        "--epsilon_input",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--n_steps",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--output_every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--scale",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    natural_uhat, initial_time, initial_step = (
        smoke.load_checkpoint(
            args.checkpoint
        )
    )

    n = int(
        natural_uhat.shape[1]
    )

    operators = dns.make_operators(
        n=n,
        box_length=2.0 * np.pi,
    )

    randomized_uhat, perturbation_audit = (
        smoke.phase_randomize_spectral(
            natural_uhat,
            operators,
            args.phase_seed,
        )
    )

    perturbation_audit["checks"] = {
        "modal_amplitude_preserved": bool(
            perturbation_audit[
                "modal_vector_amplitude_relative_max_error"
            ] < 1e-12
        ),
        "energy_preserved": bool(
            perturbation_audit[
                "energy_relative_error"
            ] < 1e-12
        ),
        "divergence_preserved": bool(
            perturbation_audit[
                "divergence_relative_error"
            ] < 1e-12
        ),
        "dealias_preserved": bool(
            perturbation_audit[
                "dealias_leakage"
            ] < 1e-12
        ),
        "hermitian_preserved": bool(
            perturbation_audit[
                "hermitian_error"
            ] < 1e-9
        ),
    }

    perturbation_audit["pass"] = bool(
        all(
            perturbation_audit[
                "checks"
            ].values()
        )
    )

    frozen_direction, active_modes = (
        normalized_complex_direction(
            randomized_uhat
        )
    )

    reference = pd.read_csv(
        args.reference_csv
    )

    reference[
        "relative_time_key"
    ] = reference[
        "relative_time"
    ].round(10)

    natural_reference = (
        reference[
            reference["branch"] == "natural"
        ]
        .copy()
        .set_index("relative_time_key")
    )

    nonlinear_reference = (
        reference[
            reference["branch"]
            == "nonlinear_recovery"
        ]
        .copy()
        .set_index("relative_time_key")
    )

    expected_times = np.arange(
        0,
        args.n_steps + 1,
        args.output_every,
        dtype=float,
    ) * args.dt

    expected_time_keys = np.round(
        expected_times,
        10,
    )

    for key in expected_time_keys:
        if key not in natural_reference.index:
            raise RuntimeError(
                "Natural reference missing time {}"
                .format(key)
            )

        if key not in nonlinear_reference.index:
            raise RuntimeError(
                "Nonlinear reference missing time {}"
                .format(key)
            )

    initial_actual_metrics = smoke.arrow_metrics(
        randomized_uhat,
        operators,
        args.scale,
        args.viscosity,
        args.workers,
    )

    natural_initial_row = (
        natural_reference.loc[0.0]
    )

    initial_gaps = {}

    for metric in METRICS:
        initial_gaps[metric] = abs(
            float(
                natural_initial_row[metric]
            )
            - float(
                initial_actual_metrics[metric]
            )
        )

        if initial_gaps[metric] <= 1e-10:
            raise RuntimeError(
                "Initial gap too small for {}"
                .format(metric)
            )

    rows = []
    audit_rows = []
    uhat = randomized_uhat.copy()

    wall_start = time.perf_counter()
    replay_errors = []

    for local_step in range(
        0,
        args.n_steps + 1,
    ):
        output_due = (
            local_step == 0
            or local_step % args.output_every == 0
            or local_step == args.n_steps
        )

        if output_due:
            relative_time = (
                local_step * args.dt
            )

            time_key = round(
                relative_time,
                10,
            )

            actual_metrics = smoke.arrow_metrics(
                uhat,
                operators,
                args.scale,
                args.viscosity,
                args.workers,
            )

            frozen_uhat, frozen_audit = (
                construct_amplitude_matched_field(
                    uhat,
                    frozen_direction,
                    operators,
                )
            )

            frozen_metrics = smoke.arrow_metrics(
                frozen_uhat,
                operators,
                args.scale,
                args.viscosity,
                args.workers,
            )

            natural_row = (
                natural_reference.loc[
                    time_key
                ]
            )

            prior_nonlinear_row = (
                nonlinear_reference.loc[
                    time_key
                ]
            )

            for metric in AUDIT_METRICS:
                replay_errors.append(
                    abs(
                        float(
                            actual_metrics[
                                metric
                            ]
                        )
                        - float(
                            prior_nonlinear_row[
                                metric
                            ]
                        )
                    )
                )

            for branch, metrics in (
                (
                    "natural_reference",
                    {
                        key: float(
                            natural_row[key]
                        )
                        for key in AUDIT_METRICS
                    },
                ),
                (
                    "nonlinear_recovery",
                    actual_metrics,
                ),
                (
                    "amplitude_matched_frozen",
                    frozen_metrics,
                ),
            ):
                row = {
                    "branch": branch,
                    "phase_seed":
                        int(args.phase_seed),
                    "checkpoint_step":
                        int(initial_step),
                    "local_step":
                        int(local_step),
                    "relative_time":
                        float(relative_time),
                    "absolute_time":
                        float(
                            initial_time
                            + relative_time
                        ),
                }

                for key in AUDIT_METRICS:
                    row[key] = float(
                        metrics[key]
                    )

                for metric in METRICS:
                    natural_value = float(
                        natural_row[metric]
                    )

                    if branch == "natural_reference":
                        recovery = 1.0
                    else:
                        recovery = (
                            1.0
                            - abs(
                                float(
                                    metrics[
                                        metric
                                    ]
                                )
                                - natural_value
                            )
                            / initial_gaps[
                                metric
                            ]
                        )

                    row[
                        "R_" + metric
                    ] = float(recovery)

                rows.append(row)

            audit_rows.append({
                "local_step":
                    int(local_step),
                "relative_time":
                    float(relative_time),
                **frozen_audit,
            })

            elapsed = (
                time.perf_counter()
                - wall_start
            )

            print(
                "step={:4d}/{:4d} "
                "t={:.2f} "
                "A_actual={:+.6f} "
                "A_frozen={:+.6f} "
                "A_natural={:+.6f} "
                "R_actual={:+.6f} "
                "R_frozen={:+.6f} "
                "amp_err={:.3e} "
                "wall={:.1f}s".format(
                    local_step,
                    args.n_steps,
                    relative_time,
                    actual_metrics[
                        "Pi_normalized_mean"
                    ],
                    frozen_metrics[
                        "Pi_normalized_mean"
                    ],
                    float(
                        natural_row[
                            "Pi_normalized_mean"
                        ]
                    ),
                    rows[-2][
                        "R_Pi_normalized_mean"
                    ],
                    rows[-1][
                        "R_Pi_normalized_mean"
                    ],
                    frozen_audit[
                        "modal_amplitude_relative_max_error"
                    ],
                    elapsed,
                ),
                flush=True,
            )

        if local_step == args.n_steps:
            break

        uhat, _ = dns.ssprk3_step(
            uhat,
            dt=args.dt,
            operators=operators,
            viscosity=args.viscosity,
            epsilon_input=args.epsilon_input,
            workers=args.workers,
            nonlinear=True,
        )

    diagnostics = pd.DataFrame(
        rows
    )

    audit_table = pd.DataFrame(
        audit_rows
    )

    diagnostics_path = (
        args.outdir
        / "amplitude_matched_counterfactual.csv"
    )

    audit_table_path = (
        args.outdir
        / "amplitude_match_audit.csv"
    )

    diagnostics.to_csv(
        diagnostics_path,
        index=False,
    )

    audit_table.to_csv(
        audit_table_path,
        index=False,
    )

    primary = "Pi_normalized_mean"

    actual = diagnostics[
        diagnostics["branch"]
        == "nonlinear_recovery"
    ].sort_values(
        "relative_time"
    )

    frozen = diagnostics[
        diagnostics["branch"]
        == "amplitude_matched_frozen"
    ].sort_values(
        "relative_time"
    )

    actual_auc = normalized_auc(
        actual["relative_time"],
        actual["R_" + primary],
        horizon=min(1.0, args.n_steps * args.dt),
    )

    frozen_auc = normalized_auc(
        frozen["relative_time"],
        frozen["R_" + primary],
        horizon=min(1.0, args.n_steps * args.dt),
    )

    checks = {
        "perturbation_pass":
            bool(perturbation_audit["pass"]),
        "modal_amplitude_match": bool(
            audit_table[
                "modal_amplitude_relative_max_error"
            ].max() < 1e-12
        ),
        "energy_match": bool(
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
            max(replay_errors) < 1e-9
        ),
    }

    audit = {
        "configuration": {
            "checkpoint":
                str(args.checkpoint),
            "reference_csv":
                str(args.reference_csv),
            "checkpoint_step":
                int(initial_step),
            "phase_seed":
                int(args.phase_seed),
            "N":
                n,
            "scale":
                int(args.scale),
            "n_steps":
                int(args.n_steps),
            "dt":
                float(args.dt),
            "active_modes":
                int(
                    np.count_nonzero(
                        active_modes
                    )
                ),
        },
        "primary_results": ({
            "auc_recovery_actual_0_1":
                float(actual_auc),
            "auc_recovery_frozen_0_1":
                float(frozen_auc),
            "delta_auc_actual_minus_frozen":
                float(
                    actual_auc
                    - frozen_auc
                ),
            "R_actual_t1":
                float(
                    actual[
                        np.isclose(
                            actual[
                                "relative_time"
                            ],
                            1.0,
                        )
                    ][
                        "R_" + primary
                    ].iloc[0]
                ),
            "R_frozen_t1":
                float(
                    frozen[
                        np.isclose(
                            frozen[
                                "relative_time"
                            ],
                            1.0,
                        )
                    ][
                        "R_" + primary
                    ].iloc[0]
                ),
        } if args.n_steps * args.dt >= 1.0 - 1e-10 else {
            "scope": "Numerical smoke only; no t=1 endpoint or 0-to-1 AUC reported"
        }),
        "maximum_errors": {
            "modal_amplitude_relative_max_error":
                float(
                    audit_table[
                        "modal_amplitude_relative_max_error"
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
            "dealias_leakage":
                float(
                    audit_table[
                        "dealias_leakage"
                    ].max()
                ),
            "hermitian_error":
                float(
                    audit_table[
                        "hermitian_error"
                    ].max()
                ),
            "nonlinear_replay_max_abs_error":
                float(
                    max(replay_errors)
                ),
        },
        "checks": checks,
        "pass": bool(
            all(checks.values())
        ),
        "interpretation": (
            "The frozen counterfactual matches the "
            "nonlinear vector-modal amplitude at every "
            "wavevector while retaining the initial "
            "randomized normalized complex direction."
        ),
    }

    audit_path = (
        args.outdir
        / "amplitude_matched_counterfactual_audit.json"
    )

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
        )
    )

    print("\n" + "=" * 96)
    print("AMPLITUDE-MATCHED COUNTERFACTUAL AUDIT")
    print("=" * 96)
    print(
        json.dumps(
            audit,
            indent=2,
        )
    )

    print("\nsaved:", diagnostics_path)
    print("saved:", audit_table_path)
    print("saved:", audit_path)

    if not audit["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
