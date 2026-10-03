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


DEFAULT_OUTDIR = Path(
    "phase_ensemble_v1/summaries/"
    "dynamic_recovery/short_forced_audit_N64"
)


def weighted_stage_quantity(
    rk_diagnostics,
    key,
):
    """
    SSPRK3 Butcher weights:
        b = (1/6, 1/6, 2/3)
    """
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


def save_checkpoint(
    path,
    uhat,
    simulation_time,
    step,
):
    np.savez_compressed(
        path,
        uhat=np.asarray(
            uhat,
            dtype=np.complex128,
        ),
        simulation_time=np.float64(
            simulation_time
        ),
        step=np.int64(step),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Short forced pseudo-spectral DNS run "
            "with energy-budget and constraint audit."
        )
    )

    parser.add_argument(
        "--N",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=300,
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
        "--target_energy",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--spectral_peak",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260725,
    )

    parser.add_argument(
        "--cfl",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--dt_max",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--diag_every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--print_every",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics_path = (
        args.outdir
        / "short_forced_step_diagnostics.csv"
    )

    audit_path = (
        args.outdir
        / "short_forced_audit.json"
    )

    initial_checkpoint = (
        args.outdir
        / "initial_checkpoint.npz"
    )

    final_checkpoint = (
        args.outdir
        / "final_checkpoint.npz"
    )

    existing = [
        diagnostics_path,
        audit_path,
        initial_checkpoint,
        final_checkpoint,
    ]

    if (
        any(path.exists() for path in existing)
        and not args.force
    ):
        raise FileExistsError(
            "Output already exists. "
            "Use --force to overwrite."
        )

    operators = dns.make_operators(
        n=args.N,
        box_length=2.0 * np.pi,
    )

    uhat = dns.random_initial_condition(
        operators,
        seed=args.seed,
        target_energy=args.target_energy,
        spectral_peak=args.spectral_peak,
        workers=args.workers,
    )

    save_checkpoint(
        initial_checkpoint,
        uhat,
        simulation_time=0.0,
        step=0,
    )

    rows = []
    simulation_time = 0.0
    wall_start = time.perf_counter()

    initial_energy = dns.kinetic_energy_hat(
        uhat
    )

    print("=" * 100)
    print("SHORT FORCED SPECTRAL DNS AUDIT")
    print("=" * 100)
    print("N                =", args.N)
    print("workers          =", args.workers)
    print("steps            =", args.steps)
    print("viscosity        =", args.viscosity)
    print("epsilon_input    =", args.epsilon_input)
    print("target_energy    =", args.target_energy)
    print("spectral_peak    =", args.spectral_peak)
    print("dt_max           =", args.dt_max)
    print("initial energy   =", initial_energy)
    print("=" * 100)

    finite_pass = True

    for step in range(
        1,
        args.steps + 1,
    ):
        energy_before = (
            dns.kinetic_energy_hat(
                uhat
            )
        )

        timestep = dns.stable_timestep(
            uhat,
            operators,
            viscosity=args.viscosity,
            workers=args.workers,
            cfl=args.cfl,
            viscous_safety=0.5,
            dt_max=args.dt_max,
        )

        dt = float(
            timestep["dt"]
        )

        new_uhat, rk_diagnostics = (
            dns.ssprk3_step(
                uhat,
                dt=dt,
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

        nonlinear_rate = weighted_stage_quantity(
            rk_diagnostics,
            "nonlinear_energy_rate",
        )

        expected_energy_rate = (
            injection
            - dissipation
            + nonlinear_rate
        )

        observed_energy_rate = (
            energy_after
            - energy_before
        ) / dt

        budget_residual = (
            observed_energy_rate
            - expected_energy_rate
        )

        budget_scale = max(
            abs(injection)
            + abs(dissipation)
            + abs(nonlinear_rate),
            1e-14,
        )

        budget_relative_residual = (
            abs(budget_residual)
            / budget_scale
        )

        injection_relative_error = (
            abs(
                injection
                - args.epsilon_input
            )
            / max(
                abs(args.epsilon_input),
                1e-30,
            )
        )

        simulation_time += dt
        uhat = new_uhat

        finite = bool(
            np.isfinite(uhat.real).all()
            and np.isfinite(uhat.imag).all()
            and np.isfinite(energy_after)
        )

        finite_pass = (
            finite_pass and finite
        )

        row = {
            "step": step,
            "simulation_time":
                simulation_time,
            "dt": dt,
            "dt_advective":
                timestep["dt_advective"],
            "dt_viscous":
                timestep["dt_viscous"],
            "maximum_speed":
                timestep["maximum_speed"],
            "energy_before":
                energy_before,
            "energy_after":
                energy_after,
            "observed_energy_rate":
                observed_energy_rate,
            "expected_energy_rate":
                expected_energy_rate,
            "energy_injection":
                injection,
            "dissipation_rate":
                dissipation,
            "nonlinear_energy_rate":
                nonlinear_rate,
            "budget_residual":
                budget_residual,
            "budget_relative_residual":
                budget_relative_residual,
            "injection_relative_error":
                injection_relative_error,
            "finite":
                finite,
            "divergence_relative_error":
                np.nan,
            "dealias_leakage":
                np.nan,
            "hermitian_error":
                np.nan,
        }

        if (
            step == 1
            or step % args.diag_every == 0
            or step == args.steps
        ):
            row[
                "divergence_relative_error"
            ] = (
                dns.divergence_relative_error(
                    uhat,
                    operators,
                )
            )

            row[
                "dealias_leakage"
            ] = dns.dealiased_leakage(
                uhat,
                operators,
            )

            row[
                "hermitian_error"
            ] = dns.hermitian_error(
                uhat
            )

        rows.append(row)

        if (
            step == 1
            or step % args.print_every == 0
            or step == args.steps
        ):
            elapsed = (
                time.perf_counter()
                - wall_start
            )

            print(
                "step={:4d}/{:4d} "
                "t={:9.5f} "
                "dt={:.5e} "
                "E={:.8f} "
                "eps={:.6f} "
                "inj={:.6f} "
                "budget_rel={:.3e} "
                "wall={:.1f}s".format(
                    step,
                    args.steps,
                    simulation_time,
                    dt,
                    energy_after,
                    dissipation,
                    injection,
                    budget_relative_residual,
                    elapsed,
                ),
                flush=True,
            )

        if not finite:
            print(
                "ERROR: non-finite state at step",
                step,
            )
            break

    diagnostics = pd.DataFrame(
        rows
    )

    diagnostics.to_csv(
        diagnostics_path,
        index=False,
    )

    save_checkpoint(
        final_checkpoint,
        uhat,
        simulation_time=simulation_time,
        step=len(diagnostics),
    )

    sampled_divergence = pd.to_numeric(
        diagnostics[
            "divergence_relative_error"
        ],
        errors="coerce",
    ).dropna()

    sampled_leakage = pd.to_numeric(
        diagnostics[
            "dealias_leakage"
        ],
        errors="coerce",
    ).dropna()

    sampled_hermitian = pd.to_numeric(
        diagnostics[
            "hermitian_error"
        ],
        errors="coerce",
    ).dropna()

    budget_values = pd.to_numeric(
        diagnostics[
            "budget_relative_residual"
        ],
        errors="raise",
    ).to_numpy(dtype=float)

    injection_errors = pd.to_numeric(
        diagnostics[
            "injection_relative_error"
        ],
        errors="raise",
    ).to_numpy(dtype=float)

    nonlinear_rates = np.abs(
        pd.to_numeric(
            diagnostics[
                "nonlinear_energy_rate"
            ],
            errors="raise",
        ).to_numpy(dtype=float)
    )

    final_energy = float(
        diagnostics.iloc[-1][
            "energy_after"
        ]
    )

    checks = {
        "completed_requested_steps": bool(
            len(diagnostics)
            == args.steps
        ),
        "all_states_finite":
            bool(finite_pass),
        "positive_timestep": bool(
            (
                diagnostics["dt"] > 0.0
            ).all()
        ),
        "positive_final_energy":
            bool(final_energy > 0.0),
        "divergence_control": bool(
            sampled_divergence.max()
            < 1e-12
        ),
        "dealias_control": bool(
            sampled_leakage.max()
            < 1e-12
        ),
        "hermitian_control": bool(
            sampled_hermitian.max()
            < 1e-9
        ),
        "constant_power_injection": bool(
            np.max(injection_errors)
            < 1e-12
        ),
        "nonlinear_energy_conservation": bool(
            np.max(nonlinear_rates)
            < 1e-10
        ),
        "median_energy_budget_residual": bool(
            np.median(budget_values)
            < 5e-3
        ),
        "p95_energy_budget_residual": bool(
            np.quantile(
                budget_values,
                0.95,
            )
            < 2e-2
        ),
    }

    elapsed_total = (
        time.perf_counter()
        - wall_start
    )

    audit = {
        "configuration": {
            "N": args.N,
            "workers": args.workers,
            "requested_steps":
                args.steps,
            "completed_steps":
                int(len(diagnostics)),
            "viscosity":
                args.viscosity,
            "epsilon_input":
                args.epsilon_input,
            "target_energy":
                args.target_energy,
            "spectral_peak":
                args.spectral_peak,
            "seed":
                args.seed,
            "cfl":
                args.cfl,
            "dt_max":
                args.dt_max,
        },
        "summary": {
            "initial_energy":
                initial_energy,
            "final_energy":
                final_energy,
            "final_simulation_time":
                float(simulation_time),
            "wall_seconds":
                float(elapsed_total),
            "mean_seconds_per_step":
                float(
                    elapsed_total
                    / max(
                        len(diagnostics),
                        1,
                    )
                ),
            "dt_min":
                float(
                    diagnostics["dt"].min()
                ),
            "dt_max_observed":
                float(
                    diagnostics["dt"].max()
                ),
            "energy_min":
                float(
                    diagnostics[
                        "energy_after"
                    ].min()
                ),
            "energy_max":
                float(
                    diagnostics[
                        "energy_after"
                    ].max()
                ),
            "dissipation_initial":
                float(
                    diagnostics.iloc[0][
                        "dissipation_rate"
                    ]
                ),
            "dissipation_final":
                float(
                    diagnostics.iloc[-1][
                        "dissipation_rate"
                    ]
                ),
            "budget_relative_median":
                float(
                    np.median(
                        budget_values
                    )
                ),
            "budget_relative_p95":
                float(
                    np.quantile(
                        budget_values,
                        0.95,
                    )
                ),
            "budget_relative_max":
                float(
                    np.max(
                        budget_values
                    )
                ),
            "maximum_injection_relative_error":
                float(
                    np.max(
                        injection_errors
                    )
                ),
            "maximum_abs_nonlinear_energy_rate":
                float(
                    np.max(
                        nonlinear_rates
                    )
                ),
            "maximum_divergence_relative_error":
                float(
                    sampled_divergence.max()
                ),
            "maximum_dealias_leakage":
                float(
                    sampled_leakage.max()
                ),
            "maximum_hermitian_error":
                float(
                    sampled_hermitian.max()
                ),
        },
        "checks": checks,
        "pass": bool(
            all(checks.values())
        ),
    }

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
        )
    )

    print("\n" + "=" * 100)
    print("SHORT-RUN AUDIT")
    print("=" * 100)

    for key, value in checks.items():
        print(
            "{:<42s} {}".format(
                key,
                "PASS" if value else "FAIL",
            )
        )

    print("\nSUMMARY")
    print(
        json.dumps(
            audit["summary"],
            indent=2,
        )
    )

    print("\nOVERALL:", audit["pass"])
    print("saved:", diagnostics_path)
    print("saved:", audit_path)
    print("saved:", initial_checkpoint)
    print("saved:", final_checkpoint)

    if not audit["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
