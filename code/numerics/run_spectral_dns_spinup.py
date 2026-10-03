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
CORE_PATH = SCRIPT_DIR / "spectral_dns_core.py"

spec = importlib.util.spec_from_file_location(
    "spectral_dns_core",
    CORE_PATH,
)
dns = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dns)


DEFAULT_INPUT = Path(
    "phase_ensemble_v1/summaries/"
    "dynamic_recovery/short_forced_audit_N64/"
    "final_checkpoint.npz"
)

DEFAULT_OUTDIR = Path(
    "phase_ensemble_v1/summaries/"
    "dynamic_recovery/spinup_N64"
)


def load_checkpoint(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path) as z:
        uhat = np.asarray(
            z["uhat"],
            dtype=np.complex128,
        )
        simulation_time = float(
            z["simulation_time"]
        )
        step = int(z["step"])

    if (
        uhat.ndim != 4
        or uhat.shape[0] != 3
        or uhat.shape[1] != uhat.shape[2]
        or uhat.shape[1] != uhat.shape[3]
    ):
        raise RuntimeError(
            "Unexpected checkpoint shape: {}".format(
                uhat.shape
            )
        )

    return uhat, simulation_time, step


def save_checkpoint(
    path,
    uhat,
    simulation_time,
    step,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.stem + ".tmp.npz"
    )

    np.savez_compressed(
        temporary,
        uhat=np.asarray(
            uhat,
            dtype=np.complex128,
        ),
        simulation_time=np.float64(
            simulation_time
        ),
        step=np.int64(step),
    )

    temporary.replace(path)


def weighted_stage_quantity(
    rk_diagnostics,
    key,
):
    return float(
        (
            rk_diagnostics["stage0"][key]
            + rk_diagnostics["stage1"][key]
        ) / 6.0
        +
        (
            2.0 / 3.0
        ) * rk_diagnostics["stage2"][key]
    )


def turbulence_diagnostics(
    uhat,
    operators,
    viscosity,
    workers,
):
    n = operators["n"]

    energy = dns.kinetic_energy_hat(
        uhat
    )

    dissipation = dns.dissipation_rate(
        uhat,
        operators,
        viscosity,
    )

    one_component_rms = np.sqrt(
        2.0 * energy / 3.0
    )

    three_component_rms = np.sqrt(
        2.0 * energy
    )

    kmagnitude = np.sqrt(
        operators["k2"]
    )

    shell_index = np.rint(
        kmagnitude
    ).astype(np.int64)

    modal_energy = (
        0.5
        * np.sum(
            np.abs(uhat) ** 2,
            axis=0,
        )
        / n**6
    )

    shell_energy = np.bincount(
        shell_index.ravel(),
        weights=modal_energy.ravel(),
    )

    shell_wavenumber = np.arange(
        len(shell_energy),
        dtype=np.float64,
    )

    positive_shell = (
        shell_wavenumber > 0.0
    )

    if energy > 0.0:
        integral_length = float(
            (
                3.0 * np.pi
                / (4.0 * energy)
            )
            * np.sum(
                shell_energy[positive_shell]
                / shell_wavenumber[
                    positive_shell
                ]
            )
        )
    else:
        integral_length = np.nan

    if one_component_rms > 0.0:
        large_eddy_time = float(
            integral_length
            / one_component_rms
        )
    else:
        large_eddy_time = np.nan

    if dissipation > 0.0:
        kolmogorov_length = float(
            (
                viscosity**3
                / dissipation
            ) ** 0.25
        )

        taylor_microscale = float(
            np.sqrt(
                15.0
                * viscosity
                * one_component_rms**2
                / dissipation
            )
        )

        reynolds_lambda = float(
            one_component_rms
            * taylor_microscale
            / viscosity
        )
    else:
        kolmogorov_length = np.nan
        taylor_microscale = np.nan
        reynolds_lambda = np.nan

    kmax = float(
        (n // 3)
        * 2.0 * np.pi
        / operators["box_length"]
    )

    maximum_speed = dns.maximum_speed(
        uhat,
        workers=workers,
    )

    return {
        "energy": float(energy),
        "dissipation_rate":
            float(dissipation),
        "u_rms_one_component":
            float(one_component_rms),
        "u_rms_three_component":
            float(three_component_rms),
        "integral_length":
            float(integral_length),
        "large_eddy_time":
            float(large_eddy_time),
        "kolmogorov_length":
            float(kolmogorov_length),
        "taylor_microscale":
            float(taylor_microscale),
        "reynolds_lambda":
            float(reynolds_lambda),
        "kmax":
            float(kmax),
        "kmax_eta":
            float(
                kmax * kolmogorov_length
            ),
        "maximum_speed":
            float(maximum_speed),
        "divergence_relative_error":
            float(
                dns.divergence_relative_error(
                    uhat,
                    operators,
                )
            ),
        "dealias_leakage":
            float(
                dns.dealiased_leakage(
                    uhat,
                    operators,
                )
            ),
        "hermitian_error":
            float(
                dns.hermitian_error(
                    uhat
                )
            ),
    }


def stationarity_audit(
    diagnostics,
    epsilon_input,
    window_time,
):
    if diagnostics.empty:
        return {
            "ready": False,
            "pass": False,
            "reason": "no diagnostics",
        }

    final_time = float(
        diagnostics[
            "simulation_time"
        ].max()
    )

    window = diagnostics[
        diagnostics["simulation_time"]
        >= final_time - window_time
    ].copy()

    if len(window) < 20:
        return {
            "ready": False,
            "pass": False,
            "reason": (
                "fewer than 20 diagnostic samples"
            ),
            "n_window_samples":
                int(len(window)),
        }

    span = float(
        window["simulation_time"].max()
        - window["simulation_time"].min()
    )

    if span < 0.8 * window_time:
        return {
            "ready": False,
            "pass": False,
            "reason": (
                "insufficient time span"
            ),
            "window_span": span,
            "n_window_samples":
                int(len(window)),
        }

    time_values = window[
        "simulation_time"
    ].to_numpy(dtype=float)

    energy_values = window[
        "energy"
    ].to_numpy(dtype=float)

    dissipation_values = window[
        "dissipation_rate"
    ].to_numpy(dtype=float)

    mean_energy = float(
        np.mean(energy_values)
    )

    mean_dissipation = float(
        np.mean(dissipation_values)
    )

    slope = float(
        np.polyfit(
            time_values,
            energy_values,
            deg=1,
        )[0]
    )

    normalized_energy_drift = float(
        abs(slope)
        * span
        / max(
            abs(mean_energy),
            1e-30,
        )
    )

    midpoint = len(window) // 2

    first_half_energy = float(
        np.mean(
            energy_values[:midpoint]
        )
    )

    second_half_energy = float(
        np.mean(
            energy_values[midpoint:]
        )
    )

    half_window_shift = float(
        abs(
            second_half_energy
            - first_half_energy
        )
        / max(
            abs(mean_energy),
            1e-30,
        )
    )

    dissipation_ratio = float(
        mean_dissipation
        / epsilon_input
    )

    energy_cv = float(
        np.std(
            energy_values,
            ddof=1,
        )
        / max(
            abs(mean_energy),
            1e-30,
        )
    )

    dissipation_cv = float(
        np.std(
            dissipation_values,
            ddof=1,
        )
        / max(
            abs(mean_dissipation),
            1e-30,
        )
    )

    minimum_kmax_eta = float(
        window["kmax_eta"].min()
    )

    checks = {
        "mean_dissipation_balance": bool(
            0.90
            <= dissipation_ratio
            <= 1.10
        ),
        "energy_drift_small": bool(
            normalized_energy_drift
            <= 0.08
        ),
        "half_window_shift_small": bool(
            half_window_shift
            <= 0.08
        ),
        "resolution_adequate": bool(
            minimum_kmax_eta
            >= 1.30
        ),
    }

    return {
        "ready": True,
        "pass": bool(
            all(checks.values())
        ),
        "window_time_requested":
            float(window_time),
        "window_span":
            float(span),
        "n_window_samples":
            int(len(window)),
        "mean_energy":
            mean_energy,
        "mean_dissipation":
            mean_dissipation,
        "dissipation_to_input_ratio":
            dissipation_ratio,
        "energy_linear_slope":
            slope,
        "normalized_energy_drift":
            normalized_energy_drift,
        "half_window_energy_shift":
            half_window_shift,
        "energy_coefficient_of_variation":
            energy_cv,
        "dissipation_coefficient_of_variation":
            dissipation_cv,
        "minimum_kmax_eta":
            minimum_kmax_eta,
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Restartable N=64 forced spectral DNS "
            "spin-up toward statistical stationarity."
        )
    )

    parser.add_argument(
        "--checkpoint_in",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
    )

    parser.add_argument(
        "--target_step",
        type=int,
        default=3000,
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
        "--cfl_abort",
        type=float,
        default=0.55,
    )

    parser.add_argument(
        "--diag_every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--print_every",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--stationarity_window_time",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
    )

    args = parser.parse_args()

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir = (
        args.outdir
        / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_checkpoint = (
        checkpoint_dir
        / "latest_checkpoint.npz"
    )

    diagnostics_path = (
        args.outdir
        / "spinup_diagnostics.csv"
    )

    audit_path = (
        args.outdir
        / "spinup_audit.json"
    )

    if (
        latest_checkpoint.exists()
        and not args.fresh
    ):
        source_checkpoint = (
            latest_checkpoint
        )
    else:
        source_checkpoint = (
            args.checkpoint_in
        )

    uhat, simulation_time, start_step = (
        load_checkpoint(
            source_checkpoint
        )
    )

    n = int(
        uhat.shape[1]
    )

    operators = dns.make_operators(
        n=n,
        box_length=2.0 * np.pi,
    )

    if (
        diagnostics_path.exists()
        and not args.fresh
    ):
        diagnostics = pd.read_csv(
            diagnostics_path
        )
    else:
        diagnostics = pd.DataFrame()

    existing_rows = (
        diagnostics.to_dict("records")
        if not diagnostics.empty
        else []
    )

    print("=" * 104)
    print("FORCED SPECTRAL DNS SPIN-UP")
    print("=" * 104)
    print("source checkpoint =", source_checkpoint)
    print("N                 =", n)
    print("start step        =", start_step)
    print("start time        =", simulation_time)
    print("target step       =", args.target_step)
    print("dt                =", args.dt)
    print("workers           =", args.workers)
    print("viscosity         =", args.viscosity)
    print("epsilon_input     =", args.epsilon_input)
    print("=" * 104)

    if start_step >= args.target_step:
        print(
            "Target step already reached."
        )

    wall_start = time.perf_counter()
    numerical_failure = None

    rows = list(existing_rows)

    for step in range(
        start_step + 1,
        args.target_step + 1,
    ):
        energy_before = (
            dns.kinetic_energy_hat(
                uhat
            )
        )

        new_uhat, rk_diagnostics = (
            dns.ssprk3_step(
                uhat,
                dt=args.dt,
                operators=operators,
                viscosity=args.viscosity,
                epsilon_input=args.epsilon_input,
                workers=args.workers,
                nonlinear=True,
            )
        )

        energy_after = (
            dns.kinetic_energy_hat(
                new_uhat
            )
        )

        injection = weighted_stage_quantity(
            rk_diagnostics,
            "energy_injection",
        )

        dissipation = weighted_stage_quantity(
            rk_diagnostics,
            "dissipation_rate",
        )

        nonlinear_rate = (
            weighted_stage_quantity(
                rk_diagnostics,
                "nonlinear_energy_rate",
            )
        )

        observed_energy_rate = (
            energy_after
            - energy_before
        ) / args.dt

        expected_energy_rate = (
            injection
            - dissipation
            + nonlinear_rate
        )

        budget_scale = max(
            abs(injection)
            + abs(dissipation)
            + abs(nonlinear_rate),
            1e-14,
        )

        budget_relative_residual = float(
            abs(
                observed_energy_rate
                - expected_energy_rate
            )
            / budget_scale
        )

        simulation_time += args.dt
        uhat = new_uhat

        finite = bool(
            np.isfinite(uhat.real).all()
            and np.isfinite(uhat.imag).all()
            and np.isfinite(energy_after)
        )

        diagnostic_due = bool(
            step == start_step + 1
            or step % args.diag_every == 0
            or step == args.target_step
        )

        if diagnostic_due:
            turbulent = (
                turbulence_diagnostics(
                    uhat,
                    operators,
                    viscosity=args.viscosity,
                    workers=args.workers,
                )
            )

            actual_cfl = float(
                turbulent["maximum_speed"]
                * args.dt
                / operators["dx"]
            )

            row = {
                "step": int(step),
                "simulation_time":
                    float(simulation_time),
                "dt": float(args.dt),
                "actual_cfl":
                    actual_cfl,
                "energy_injection":
                    float(injection),
                "rk_weighted_dissipation":
                    float(dissipation),
                "nonlinear_energy_rate":
                    float(nonlinear_rate),
                "observed_energy_rate":
                    float(observed_energy_rate),
                "expected_energy_rate":
                    float(expected_energy_rate),
                "budget_relative_residual":
                    budget_relative_residual,
                **turbulent,
            }

            rows.append(row)

            diagnostics = (
                pd.DataFrame(rows)
                .drop_duplicates(
                    subset=["step"],
                    keep="last",
                )
                .sort_values("step")
                .reset_index(drop=True)
            )

            diagnostics.to_csv(
                diagnostics_path,
                index=False,
            )

            if actual_cfl > args.cfl_abort:
                numerical_failure = (
                    "actual CFL {:.6f} exceeded "
                    "abort threshold {:.6f}"
                ).format(
                    actual_cfl,
                    args.cfl_abort,
                )
                break

            if (
                turbulent[
                    "divergence_relative_error"
                ] > 1e-12
            ):
                numerical_failure = (
                    "divergence control failed"
                )
                break

            if not finite:
                numerical_failure = (
                    "non-finite state"
                )
                break

        if (
            step % args.checkpoint_every == 0
            or step == args.target_step
            or numerical_failure is not None
        ):
            numbered_checkpoint = (
                checkpoint_dir
                / "checkpoint_step_{:07d}.npz"
                .format(step)
            )

            save_checkpoint(
                numbered_checkpoint,
                uhat,
                simulation_time,
                step,
            )

            save_checkpoint(
                latest_checkpoint,
                uhat,
                simulation_time,
                step,
            )

        if (
            step == start_step + 1
            or step % args.print_every == 0
            or step == args.target_step
            or numerical_failure is not None
        ):
            elapsed = (
                time.perf_counter()
                - wall_start
            )

            if diagnostics.empty:
                cfl_display = np.nan
                kmax_eta_display = np.nan
                re_lambda_display = np.nan
            else:
                last = diagnostics.iloc[-1]
                cfl_display = float(
                    last["actual_cfl"]
                )
                kmax_eta_display = float(
                    last["kmax_eta"]
                )
                re_lambda_display = float(
                    last["reynolds_lambda"]
                )

            print(
                "step={:5d}/{:5d} "
                "t={:8.3f} "
                "E={:.6f} "
                "eps={:.6f} "
                "CFL={:.3f} "
                "kmax_eta={:.3f} "
                "Re_lambda={:.2f} "
                "wall={:.1f}s".format(
                    step,
                    args.target_step,
                    simulation_time,
                    energy_after,
                    dissipation,
                    cfl_display,
                    kmax_eta_display,
                    re_lambda_display,
                    elapsed,
                ),
                flush=True,
            )

        if numerical_failure is not None:
            print(
                "NUMERICAL FAILURE:",
                numerical_failure,
            )
            break

    diagnostics = (
        pd.DataFrame(rows)
        .drop_duplicates(
            subset=["step"],
            keep="last",
        )
        .sort_values("step")
        .reset_index(drop=True)
    )

    diagnostics.to_csv(
        diagnostics_path,
        index=False,
    )

    final_step = int(
        diagnostics["step"].max()
    )

    final_time = float(
        diagnostics.iloc[-1][
            "simulation_time"
        ]
    )

    stationarity = stationarity_audit(
        diagnostics,
        epsilon_input=args.epsilon_input,
        window_time=args.stationarity_window_time,
    )

    maximum_cfl = float(
        diagnostics[
            "actual_cfl"
        ].max()
    )

    maximum_divergence = float(
        diagnostics[
            "divergence_relative_error"
        ].max()
    )

    maximum_budget_residual = float(
        diagnostics[
            "budget_relative_residual"
        ].max()
    )

    minimum_kmax_eta = float(
        diagnostics[
            "kmax_eta"
        ].min()
    )

    numerical_checks = {
        "target_step_reached": bool(
            final_step >= args.target_step
        ),
        "no_recorded_failure": bool(
            numerical_failure is None
        ),
        "cfl_control": bool(
            maximum_cfl
            <= args.cfl_abort
        ),
        "divergence_control": bool(
            maximum_divergence < 1e-12
        ),
        "dealias_control": bool(
            diagnostics[
                "dealias_leakage"
            ].max() < 1e-12
        ),
        "hermitian_control": bool(
            diagnostics[
                "hermitian_error"
            ].max() < 1e-9
        ),
        "energy_budget_control": bool(
            maximum_budget_residual
            < 1e-3
        ),
        "resolution_control": bool(
            minimum_kmax_eta >= 1.30
        ),
    }

    audit = {
        "configuration": {
            "source_checkpoint":
                str(source_checkpoint),
            "N": n,
            "start_step":
                int(start_step),
            "target_step":
                int(args.target_step),
            "workers":
                int(args.workers),
            "viscosity":
                float(args.viscosity),
            "epsilon_input":
                float(args.epsilon_input),
            "dt":
                float(args.dt),
            "cfl_abort":
                float(args.cfl_abort),
            "stationarity_window_time":
                float(
                    args.stationarity_window_time
                ),
        },
        "final_state": {
            "final_step":
                final_step,
            "final_simulation_time":
                final_time,
            "final_energy":
                float(
                    diagnostics.iloc[-1][
                        "energy"
                    ]
                ),
            "final_dissipation":
                float(
                    diagnostics.iloc[-1][
                        "dissipation_rate"
                    ]
                ),
            "final_reynolds_lambda":
                float(
                    diagnostics.iloc[-1][
                        "reynolds_lambda"
                    ]
                ),
            "final_kmax_eta":
                float(
                    diagnostics.iloc[-1][
                        "kmax_eta"
                    ]
                ),
            "maximum_cfl":
                maximum_cfl,
            "maximum_divergence_error":
                maximum_divergence,
            "maximum_budget_relative_residual":
                maximum_budget_residual,
        },
        "numerical_failure":
            numerical_failure,
        "numerical_checks":
            numerical_checks,
        "numerical_pass": bool(
            all(
                numerical_checks.values()
            )
        ),
        "stationarity":
            stationarity,
        "stationarity_pass": bool(
            stationarity.get(
                "pass",
                False,
            )
        ),
    }

    audit["pass"] = bool(
        audit["numerical_pass"]
    )

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
        )
    )

    print("\n" + "=" * 104)
    print("SPIN-UP AUDIT")
    print("=" * 104)

    print(
        "numerical_pass  =",
        audit["numerical_pass"],
    )

    print(
        "stationarity_pass =",
        audit["stationarity_pass"],
    )

    print("\nSTATIONARITY")
    print(
        json.dumps(
            stationarity,
            indent=2,
        )
    )

    print("\nsaved:", diagnostics_path)
    print("saved:", audit_path)
    print("saved:", latest_checkpoint)

    if not audit["numerical_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
