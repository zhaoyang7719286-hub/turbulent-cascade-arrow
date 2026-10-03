#!/usr/bin/env python3

from pathlib import Path
import importlib.util
import json

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CORE_PATH = SCRIPT_DIR / "spectral_dns_core.py"

spec = importlib.util.spec_from_file_location(
    "spectral_dns_core",
    CORE_PATH,
)
dns = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dns)


OUTDIR = Path(__file__).resolve().parents[2] / "outputs/core_validation"


def relative_error(
    value,
    reference,
):
    return abs(value - reference) / max(
        abs(reference),
        1e-30,
    )


def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    n = 32
    workers = 1
    viscosity = 0.015
    epsilon_input = 0.1

    operators = dns.make_operators(
        n=n,
        box_length=2.0 * np.pi,
    )

    uhat = dns.random_initial_condition(
        operators,
        seed=20260725,
        target_energy=0.5,
        spectral_peak=4.0,
        workers=workers,
    )

    energy_hat = dns.kinetic_energy_hat(
        uhat
    )

    u_complex = dns.ifft_vector_complex(
        uhat,
        workers=workers,
    )

    u_real = u_complex.real

    energy_physical = (
        dns.kinetic_energy_physical(
            u_real
        )
    )

    fft_roundtrip = dns.fft_vector(
        u_real,
        workers=workers,
    )

    fft_roundtrip = dns.project_dealias(
        fft_roundtrip,
        operators,
        zero_mean=True,
    )

    roundtrip_error = float(
        np.max(
            np.abs(
                fft_roundtrip - uhat
            )
        )
        / (
            np.max(np.abs(uhat))
            + 1e-30
        )
    )

    imaginary_reconstruction_error = float(
        np.max(
            np.abs(
                u_complex.imag
            )
        )
    )

    divergence_error = (
        dns.divergence_relative_error(
            uhat,
            operators,
        )
    )

    leakage = dns.dealiased_leakage(
        uhat,
        operators,
    )

    hermitian = dns.hermitian_error(
        uhat
    )

    nhat = dns.nonlinear_hat(
        uhat,
        operators,
        workers=workers,
    )

    nonlinear_energy_rate = (
        dns.spectral_inner_product(
            uhat,
            nhat,
        )
    )

    fhat, forcing_alpha, injection = (
        dns.constant_power_forcing_hat(
            uhat,
            operators,
            epsilon_input,
        )
    )

    forcing_error = relative_error(
        injection,
        epsilon_input,
    )

    dissipation = dns.dissipation_rate(
        uhat,
        operators,
        viscosity,
    )

    rhs_decay, decay_diag = dns.rhs_hat(
        uhat,
        operators,
        viscosity=viscosity,
        epsilon_input=0.0,
        workers=workers,
        nonlinear=True,
    )

    total_decay_rate = (
        dns.spectral_inner_product(
            uhat,
            rhs_decay,
        )
    )

    decay_balance_error = relative_error(
        total_decay_rate,
        -dissipation,
    )

    timestep = dns.stable_timestep(
        uhat,
        operators,
        viscosity=viscosity,
        workers=workers,
        cfl=0.4,
        viscous_safety=0.5,
        dt_max=0.02,
    )

    unew, rk_diag = dns.ssprk3_step(
        uhat,
        dt=timestep["dt"],
        operators=operators,
        viscosity=viscosity,
        epsilon_input=epsilon_input,
        workers=workers,
        nonlinear=True,
    )

    new_energy = dns.kinetic_energy_hat(
        unew
    )

    new_divergence_error = (
        dns.divergence_relative_error(
            unew,
            operators,
        )
    )

    new_leakage = dns.dealiased_leakage(
        unew,
        operators,
    )

    new_hermitian = dns.hermitian_error(
        unew
    )

    finite_after_step = bool(
        np.isfinite(unew.real).all()
        and np.isfinite(unew.imag).all()
        and np.isfinite(new_energy)
    )

    checks = {
        "target_energy_error":
            relative_error(
                energy_hat,
                0.5,
            )
            < 1e-12,

        "parseval_energy_error":
            relative_error(
                energy_physical,
                energy_hat,
            )
            < 1e-12,

        "fft_roundtrip_error":
            roundtrip_error
            < 1e-12,

        "imaginary_reconstruction_error":
            imaginary_reconstruction_error
            < 1e-12,

        "initial_divergence_error":
            divergence_error
            < 1e-12,

        "initial_dealias_leakage":
            leakage
            < 1e-12,

        "initial_hermitian_error":
            hermitian
            < 1e-10,

        "nonlinear_energy_conservation":
            abs(nonlinear_energy_rate)
            < 1e-11,

        "constant_power_forcing":
            forcing_error
            < 1e-12,

        "decay_energy_balance":
            decay_balance_error
            < 1e-10,

        "rk3_step_finite":
            finite_after_step,

        "rk3_divergence_error":
            new_divergence_error
            < 1e-12,

        "rk3_dealias_leakage":
            new_leakage
            < 1e-12,

        "rk3_hermitian_error":
            new_hermitian
            < 1e-9,
    }

    audit = {
        "configuration": {
            "N": n,
            "workers": workers,
            "viscosity": viscosity,
            "epsilon_input":
                epsilon_input,
        },
        "metrics": {
            "energy_hat":
                energy_hat,
            "energy_physical":
                energy_physical,
            "roundtrip_relative_error":
                roundtrip_error,
            "imaginary_reconstruction_error":
                imaginary_reconstruction_error,
            "initial_divergence_relative_error":
                divergence_error,
            "initial_dealias_leakage":
                leakage,
            "initial_hermitian_error":
                hermitian,
            "nonlinear_energy_rate":
                nonlinear_energy_rate,
            "forcing_alpha":
                forcing_alpha,
            "forcing_injection":
                injection,
            "forcing_relative_error":
                forcing_error,
            "dissipation_rate":
                dissipation,
            "total_unforced_energy_rate":
                total_decay_rate,
            "decay_balance_relative_error":
                decay_balance_error,
            "selected_dt":
                timestep["dt"],
            "dt_advective":
                timestep["dt_advective"],
            "dt_viscous":
                timestep["dt_viscous"],
            "maximum_speed":
                timestep["maximum_speed"],
            "new_energy":
                new_energy,
            "new_divergence_relative_error":
                new_divergence_error,
            "new_dealias_leakage":
                new_leakage,
            "new_hermitian_error":
                new_hermitian,
        },
        "checks": checks,
        "pass": bool(
            all(checks.values())
        ),
    }

    output = (
        OUTDIR
        / "spectral_dns_unit_test_audit.json"
    )

    output.write_text(
        json.dumps(
            audit,
            indent=2,
        )
    )

    print("=" * 88)
    print("SPECTRAL DNS UNIT TEST")
    print("=" * 88)

    for name, passed in checks.items():
        print(
            "{:<42s} {}".format(
                name,
                "PASS" if passed else "FAIL",
            )
        )

    print("\nMETRICS")
    print(
        json.dumps(
            audit["metrics"],
            indent=2,
        )
    )

    print("\nOVERALL:", audit["pass"])
    print("saved:", output)

    if not audit["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
