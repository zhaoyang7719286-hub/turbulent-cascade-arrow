#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.fft import fftn, ifftn
from scipy.ndimage import uniform_filter


SCRIPT_DIR = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dns = load_module(
    "spectral_dns_core",
    SCRIPT_DIR / "spectral_dns_core.py",
)

gate = load_module(
    "phase_gate",
    SCRIPT_DIR / "df_phase_randomization_gate_64cube.py",
)

popmod = load_module(
    "pop_amp",
    SCRIPT_DIR / "population_amplitude.py",
)


DEFAULT_CHECKPOINT = Path(
    "phase_ensemble_v1/summaries/"
    "dynamic_recovery/spinup_N64/checkpoints/"
    "checkpoint_step_0003000.npz"
)

DEFAULT_OUTDIR = Path(
    "phase_ensemble_v1/summaries/"
    "dynamic_recovery/recovery_smoke_checkpoint3000"
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

    return uhat, simulation_time, step


def save_checkpoint(
    path,
    uhat,
    absolute_time,
    relative_time,
    step,
):
    np.savez_compressed(
        path,
        uhat=np.asarray(
            uhat,
            dtype=np.complex128,
        ),
        absolute_time=np.float64(
            absolute_time
        ),
        relative_time=np.float64(
            relative_time
        ),
        step=np.int64(step),
    )


def safe_corr(x, y):
    x = np.asarray(
        x,
        dtype=np.float64,
    ).ravel()

    y = np.asarray(
        y,
        dtype=np.float64,
    ).ravel()

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    if x.size < 100:
        return np.nan

    x = x - np.mean(x)
    y = y - np.mean(y)

    denominator = np.sqrt(
        np.mean(x * x)
        * np.mean(y * y)
    )

    if denominator <= 0.0:
        return np.nan

    return float(
        np.mean(x * y)
        / denominator
    )


def periodic_box_filter_vector(
    u,
    width,
):
    filtered = np.empty_like(
        u,
        dtype=np.float64,
    )

    for component in range(3):
        filtered[component] = uniform_filter(
            u[component],
            size=width,
            mode="wrap",
        )

    return filtered


def coarse_grained_fields(
    uhat,
    operators,
    scale,
    workers,
):
    """
    Periodic coarse-grained SGS flux.

    The DNS state uses physical axis order:
        u[component, x, y, z].

    Filtering is a periodic box filter.
    Gradients are spectral and consistent with
    the periodic pseudo-spectral solver.
    """
    u = dns.ifft_vector_real(
        uhat,
        workers=workers,
    )

    ubar = periodic_box_filter_vector(
        u,
        width=scale,
    )

    tau = np.empty(
        (3, 3) + u.shape[1:],
        dtype=np.float64,
    )

    for i in range(3):
        for j in range(3):
            filtered_product = uniform_filter(
                u[i] * u[j],
                size=scale,
                mode="wrap",
            )

            tau[i, j] = (
                filtered_product
                - ubar[i] * ubar[j]
            )

    ubar_hat = fftn(
        ubar,
        axes=(1, 2, 3),
        workers=workers,
    )

    wavevectors = np.stack(
        [
            operators["kx"],
            operators["ky"],
            operators["kz"],
        ],
        axis=0,
    )

    gradient_hat = (
        1j
        * ubar_hat[:, None, ...]
        * wavevectors[None, ...]
    )

    gradient = ifftn(
        gradient_hat,
        axes=(2, 3, 4),
        workers=workers,
    ).real

    strain = 0.5 * (
        gradient
        + np.swapaxes(
            gradient,
            0,
            1,
        )
    )

    pi = -np.einsum(
        "ijxyz,ijxyz->xyz",
        tau,
        strain,
        optimize=True,
    )

    omega = np.empty(
        (3,) + u.shape[1:],
        dtype=np.float64,
    )

    omega[0] = (
        gradient[2, 1]
        - gradient[1, 2]
    )

    omega[1] = (
        gradient[0, 2]
        - gradient[2, 0]
    )

    omega[2] = (
        gradient[1, 0]
        - gradient[0, 1]
    )

    strain_term = -np.einsum(
        "ijxyz,jkxyz,kixyz->xyz",
        strain,
        strain,
        strain,
        optimize=True,
    )

    vortex_term = np.einsum(
        "ixyz,ijxyz,jxyz->xyz",
        omega,
        strain,
        omega,
        optimize=True,
    )

    mechanism = (
        strain_term
        + 0.25 * vortex_term
    )

    return pi, mechanism


def turbulence_metrics(
    uhat,
    operators,
    viscosity,
    workers,
):
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

    if dissipation > 0.0:
        eta = (
            viscosity**3
            / dissipation
        ) ** 0.25

        taylor_scale = np.sqrt(
            15.0
            * viscosity
            * one_component_rms**2
            / dissipation
        )

        reynolds_lambda = (
            one_component_rms
            * taylor_scale
            / viscosity
        )
    else:
        eta = np.nan
        reynolds_lambda = np.nan

    kmax = (
        operators["n"] // 3
    ) * (
        2.0 * np.pi
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
        "reynolds_lambda":
            float(reynolds_lambda),
        "kmax_eta":
            float(kmax * eta),
        "maximum_speed":
            float(maximum_speed),
        "actual_cfl": float(
            maximum_speed
            * 0.01
            / operators["dx"]
        ),
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


def arrow_metrics(
    uhat,
    operators,
    scale,
    viscosity,
    workers,
):
    pi, mechanism = coarse_grained_fields(
        uhat,
        operators,
        scale,
        workers,
    )

    pi_stats = popmod.pop_amp_statistics(
        pi
    )

    mechanism_stats = popmod.pop_amp_statistics(
        mechanism
    )

    result = {
        "Pi_A_pop":
            float(pi_stats["A_pop"]),
        "Pi_A_amp":
            float(pi_stats["A_amp"]),
        "Pi_mean_signed":
            float(pi_stats["mean_signed"]),
        "Pi_mean_absolute":
            float(pi_stats["mean_absolute"]),
        "Pi_normalized_mean":
            float(
                pi_stats[
                    "normalized_mean_lhs"
                ]
            ),
        "Pi_identity_error":
            float(
                pi_stats[
                    "identity_abs_error"
                ]
            ),
        "M_A_pop":
            float(
                mechanism_stats["A_pop"]
            ),
        "M_A_amp":
            float(
                mechanism_stats["A_amp"]
            ),
        "M_normalized_mean":
            float(
                mechanism_stats[
                    "normalized_mean_lhs"
                ]
            ),
        "corr_Pi_M":
            safe_corr(
                pi,
                mechanism,
            ),
    }

    result.update(
        turbulence_metrics(
            uhat,
            operators,
            viscosity,
            workers,
        )
    )

    return result


def phase_randomize_spectral(
    uhat,
    operators,
    phase_seed,
):
    shape = tuple(
        uhat.shape[1:]
    )

    phase = gate.make_hermitian_phase(
        shape,
        seed=phase_seed,
    )

    randomized = (
        uhat
        * phase[None, ...]
    )

    randomized = dns.project_dealias(
        randomized,
        operators,
        zero_mean=True,
    )

    amplitude_before = np.sqrt(
        np.sum(
            np.abs(uhat) ** 2,
            axis=0,
        )
    )

    amplitude_after = np.sqrt(
        np.sum(
            np.abs(randomized) ** 2,
            axis=0,
        )
    )

    relevant = amplitude_before > (
        1e-13
        * np.max(amplitude_before)
    )

    modal_amplitude_error = float(
        np.max(
            np.abs(
                amplitude_after[relevant]
                - amplitude_before[relevant]
            )
            / amplitude_before[relevant]
        )
    )

    energy_before = dns.kinetic_energy_hat(
        uhat
    )

    energy_after = dns.kinetic_energy_hat(
        randomized
    )

    audit = {
        "phase_seed":
            int(phase_seed),
        "modal_vector_amplitude_relative_max_error":
            modal_amplitude_error,
        "energy_relative_error":
            float(
                abs(
                    energy_after
                    - energy_before
                )
                / energy_before
            ),
        "divergence_relative_error":
            float(
                dns.divergence_relative_error(
                    randomized,
                    operators,
                )
            ),
        "dealias_leakage":
            float(
                dns.dealiased_leakage(
                    randomized,
                    operators,
                )
            ),
        "hermitian_error":
            float(
                dns.hermitian_error(
                    randomized
                )
            ),
    }

    return randomized, audit


def run_branch(
    branch,
    initial_uhat,
    initial_absolute_time,
    initial_step,
    operators,
    viscosity,
    epsilon_input,
    workers,
    dt,
    n_steps,
    output_every,
    scale,
    outdir,
):
    nonlinear = branch != "linear_control"

    branch_dir = outdir / branch
    branch_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics_path = (
        branch_dir
        / "diagnostics.csv"
    )

    final_checkpoint = (
        branch_dir
        / "final_checkpoint.npz"
    )

    uhat = initial_uhat.copy()
    rows = []

    wall_start = time.perf_counter()

    initial_metrics = arrow_metrics(
        uhat,
        operators,
        scale,
        viscosity,
        workers,
    )

    rows.append({
        "branch": branch,
        "local_step": 0,
        "absolute_step":
            int(initial_step),
        "relative_time": 0.0,
        "absolute_time":
            float(initial_absolute_time),
        **initial_metrics,
    })

    print(
        "\n[{}] step=0 "
        "A_pop={:.6f} "
        "A_amp={:.6f} "
        "norm={:.6f} "
        "corr={:.6f}".format(
            branch,
            initial_metrics["Pi_A_pop"],
            initial_metrics["Pi_A_amp"],
            initial_metrics[
                "Pi_normalized_mean"
            ],
            initial_metrics["corr_Pi_M"],
        ),
        flush=True,
    )

    numerical_failure = None

    for local_step in range(
        1,
        n_steps + 1,
    ):
        uhat, _ = dns.ssprk3_step(
            uhat,
            dt=dt,
            operators=operators,
            viscosity=viscosity,
            epsilon_input=epsilon_input,
            workers=workers,
            nonlinear=nonlinear,
        )

        if (
            local_step % output_every == 0
            or local_step == n_steps
        ):
            relative_time = (
                local_step * dt
            )

            metrics = arrow_metrics(
                uhat,
                operators,
                scale,
                viscosity,
                workers,
            )

            row = {
                "branch": branch,
                "local_step":
                    int(local_step),
                "absolute_step":
                    int(
                        initial_step
                        + local_step
                    ),
                "relative_time":
                    float(relative_time),
                "absolute_time":
                    float(
                        initial_absolute_time
                        + relative_time
                    ),
                **metrics,
            }

            rows.append(row)

            pd.DataFrame(rows).to_csv(
                diagnostics_path,
                index=False,
            )

            elapsed = (
                time.perf_counter()
                - wall_start
            )

            print(
                "[{}] step={:4d}/{:4d} "
                "t_rel={:.2f} "
                "A_pop={:+.5f} "
                "A_amp={:+.5f} "
                "norm={:+.5f} "
                "corr={:.5f} "
                "E={:.5f} "
                "CFL={:.3f} "
                "wall={:.1f}s".format(
                    branch,
                    local_step,
                    n_steps,
                    relative_time,
                    metrics["Pi_A_pop"],
                    metrics["Pi_A_amp"],
                    metrics[
                        "Pi_normalized_mean"
                    ],
                    metrics["corr_Pi_M"],
                    metrics["energy"],
                    metrics["actual_cfl"],
                    elapsed,
                ),
                flush=True,
            )

            if not np.isfinite(
                np.asarray(
                    list(metrics.values()),
                    dtype=np.float64,
                )
            ).all():
                numerical_failure = (
                    "non-finite diagnostic"
                )
                break

            if (
                metrics["actual_cfl"]
                > 0.55
            ):
                numerical_failure = (
                    "CFL exceeded 0.55"
                )
                break

            if (
                metrics[
                    "divergence_relative_error"
                ]
                > 1e-12
            ):
                numerical_failure = (
                    "divergence control failed"
                )
                break

    diagnostics = pd.DataFrame(
        rows
    )

    diagnostics.to_csv(
        diagnostics_path,
        index=False,
    )

    final_relative_time = float(
        diagnostics.iloc[-1][
            "relative_time"
        ]
    )

    save_checkpoint(
        final_checkpoint,
        uhat,
        absolute_time=(
            initial_absolute_time
            + final_relative_time
        ),
        relative_time=final_relative_time,
        step=(
            initial_step
            + int(
                diagnostics.iloc[-1][
                    "local_step"
                ]
            )
        ),
    )

    branch_audit = {
        "branch": branch,
        "nonlinear":
            bool(nonlinear),
        "requested_steps":
            int(n_steps),
        "completed_steps":
            int(
                diagnostics.iloc[-1][
                    "local_step"
                ]
            ),
        "numerical_failure":
            numerical_failure,
        "maximum_cfl":
            float(
                diagnostics[
                    "actual_cfl"
                ].max()
            ),
        "maximum_divergence_error":
            float(
                diagnostics[
                    "divergence_relative_error"
                ].max()
            ),
        "maximum_dealias_leakage":
            float(
                diagnostics[
                    "dealias_leakage"
                ].max()
            ),
        "maximum_hermitian_error":
            float(
                diagnostics[
                    "hermitian_error"
                ].max()
            ),
        "pass": bool(
            numerical_failure is None
            and int(
                diagnostics.iloc[-1][
                    "local_step"
                ]
            ) == n_steps
        ),
    }

    (
        branch_dir
        / "audit.json"
    ).write_text(
        json.dumps(
            branch_audit,
            indent=2,
        )
    )

    return diagnostics, branch_audit


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
    )

    parser.add_argument(
        "--phase_seed",
        type=int,
        default=2026074001,
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
        default=400,
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

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    if args.outdir.exists() and not args.force:
        existing = list(
            args.outdir.glob("**/*.csv")
        )

        if existing:
            raise FileExistsError(
                "Output exists. Use --force."
            )

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    natural_uhat, initial_time, initial_step = (
        load_checkpoint(
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
        phase_randomize_spectral(
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

    (
        args.outdir
        / "initial_perturbation_audit.json"
    ).write_text(
        json.dumps(
            perturbation_audit,
            indent=2,
        )
    )

    if not perturbation_audit["pass"]:
        raise RuntimeError(
            "Initial phase perturbation audit failed."
        )

    print("=" * 100)
    print("DYNAMIC RECOVERY SMOKE TEST")
    print("=" * 100)
    print("checkpoint       =", args.checkpoint)
    print("initial step     =", initial_step)
    print("initial time     =", initial_time)
    print("phase seed       =", args.phase_seed)
    print("N                =", n)
    print("n_steps          =", args.n_steps)
    print("horizon          =", args.n_steps * args.dt)
    print("scale            =", args.scale)
    print("=" * 100)
    print(
        "PERTURBATION AUDIT:",
        json.dumps(
            perturbation_audit,
            indent=2,
        ),
    )

    branch_inputs = {
        "natural":
            natural_uhat,
        "nonlinear_recovery":
            randomized_uhat,
        "linear_control":
            randomized_uhat,
    }

    all_diagnostics = []
    branch_audits = {}

    for branch in (
        "natural",
        "nonlinear_recovery",
        "linear_control",
    ):
        diagnostics, audit = run_branch(
            branch=branch,
            initial_uhat=branch_inputs[branch],
            initial_absolute_time=initial_time,
            initial_step=initial_step,
            operators=operators,
            viscosity=args.viscosity,
            epsilon_input=args.epsilon_input,
            workers=args.workers,
            dt=args.dt,
            n_steps=args.n_steps,
            output_every=args.output_every,
            scale=args.scale,
            outdir=args.outdir,
        )

        all_diagnostics.append(
            diagnostics
        )

        branch_audits[branch] = audit

    combined = pd.concat(
        all_diagnostics,
        ignore_index=True,
    )

    combined_path = (
        args.outdir
        / "recovery_diagnostics_all_branches.csv"
    )

    combined.to_csv(
        combined_path,
        index=False,
    )

    t0 = combined[
        combined["relative_time"] == 0.0
    ][[
        "branch",
        "Pi_A_pop",
        "Pi_A_amp",
        "Pi_normalized_mean",
        "corr_Pi_M",
    ]]

    final_rows = (
        combined
        .sort_values(
            "relative_time"
        )
        .groupby(
            "branch",
            as_index=False,
        )
        .tail(1)
    )[[
        "branch",
        "relative_time",
        "Pi_A_pop",
        "Pi_A_amp",
        "Pi_normalized_mean",
        "corr_Pi_M",
        "energy",
        "dissipation_rate",
    ]]

    overall_audit = {
        "checkpoint":
            str(args.checkpoint),
        "phase_seed":
            int(args.phase_seed),
        "N":
            n,
        "scale":
            int(args.scale),
        "n_steps":
            int(args.n_steps),
        "horizon":
            float(args.n_steps * args.dt),
        "perturbation_audit":
            perturbation_audit,
        "branch_audits":
            branch_audits,
        "numerical_pass": bool(
            perturbation_audit["pass"]
            and all(
                item["pass"]
                for item in branch_audits.values()
            )
        ),
        "scientific_outcome_not_prejudged":
            True,
    }

    (
        args.outdir
        / "recovery_smoke_audit.json"
    ).write_text(
        json.dumps(
            overall_audit,
            indent=2,
        )
    )

    print("\n" + "=" * 100)
    print("INITIAL ARROW METRICS")
    print("=" * 100)
    print(t0.to_string(index=False))

    print("\n" + "=" * 100)
    print("FINAL ARROW METRICS")
    print("=" * 100)
    print(final_rows.to_string(index=False))

    print("\nNUMERICAL PASS:",
          overall_audit["numerical_pass"])
    print("saved:", combined_path)


if __name__ == "__main__":
    main()
