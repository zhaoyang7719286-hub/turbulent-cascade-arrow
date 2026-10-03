#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
from scipy.fft import fftn, ifftn


SPATIAL_AXES = (1, 2, 3)


def make_operators(
    n,
    box_length=2.0 * np.pi,
):
    """
    Build Fourier wave numbers, Leray projection data,
    and a standard 2/3-rule dealiasing mask.

    Spectral vector fields use shape:
        (3, n, n, n)
    """
    dx = box_length / n

    mode = (
        np.fft.fftfreq(n) * n
    ).astype(np.float64)

    k1 = (
        2.0 * np.pi / box_length
    ) * mode

    kx, ky, kz = np.meshgrid(
        k1,
        k1,
        k1,
        indexing="ij",
    )

    k2 = (
        kx * kx
        + ky * ky
        + kz * kz
    )

    nonzero = k2 > 0.0

    inv_k2 = np.zeros_like(k2)
    inv_k2[nonzero] = 1.0 / k2[nonzero]

    cutoff_mode = n // 3

    mx, my, mz = np.meshgrid(
        mode,
        mode,
        mode,
        indexing="ij",
    )

    dealias = (
        (np.abs(mx) <= cutoff_mode)
        &
        (np.abs(my) <= cutoff_mode)
        &
        (np.abs(mz) <= cutoff_mode)
    )

    forced = (
        nonzero
        &
        (np.sqrt(k2) >= 1.0)
        &
        (np.sqrt(k2) <= 2.0)
    )

    return {
        "n": int(n),
        "box_length": float(box_length),
        "dx": float(dx),
        "kx": kx,
        "ky": ky,
        "kz": kz,
        "k2": k2,
        "inv_k2": inv_k2,
        "nonzero": nonzero,
        "dealias": dealias,
        "forced": forced,
        "k2_dealiased_max": float(
            np.max(k2[dealias])
        ),
    }


def fft_vector(u, workers=1):
    return fftn(
        u,
        axes=SPATIAL_AXES,
        workers=workers,
    )


def ifft_vector_complex(uhat, workers=1):
    return ifftn(
        uhat,
        axes=SPATIAL_AXES,
        workers=workers,
    )


def ifft_vector_real(uhat, workers=1):
    return ifft_vector_complex(
        uhat,
        workers=workers,
    ).real


def project_dealias(
    vhat,
    operators,
    zero_mean=True,
):
    """
    Leray projection:
        P_k = I - kk/k^2

    followed by 2/3-rule truncation.
    """
    kx = operators["kx"]
    ky = operators["ky"]
    kz = operators["kz"]
    inv_k2 = operators["inv_k2"]
    mask = operators["dealias"]

    out = np.asarray(
        vhat,
        dtype=np.complex128,
    ).copy()

    longitudinal = (
        kx * out[0]
        + ky * out[1]
        + kz * out[2]
    )

    out[0] -= (
        kx
        * longitudinal
        * inv_k2
    )

    out[1] -= (
        ky
        * longitudinal
        * inv_k2
    )

    out[2] -= (
        kz
        * longitudinal
        * inv_k2
    )

    out[:, ~mask] = 0.0

    if zero_mean:
        out[:, 0, 0, 0] = 0.0

    return out


def divergence_hat(
    uhat,
    operators,
):
    return 1j * (
        operators["kx"] * uhat[0]
        + operators["ky"] * uhat[1]
        + operators["kz"] * uhat[2]
    )


def curl_hat(
    uhat,
    operators,
):
    kx = operators["kx"]
    ky = operators["ky"]
    kz = operators["kz"]

    omega_hat = np.empty_like(
        uhat,
        dtype=np.complex128,
    )

    omega_hat[0] = 1j * (
        ky * uhat[2]
        - kz * uhat[1]
    )

    omega_hat[1] = 1j * (
        kz * uhat[0]
        - kx * uhat[2]
    )

    omega_hat[2] = 1j * (
        kx * uhat[1]
        - ky * uhat[0]
    )

    return omega_hat


def nonlinear_hat(
    uhat,
    operators,
    workers=1,
):
    """
    Rotational form:

        -P[(u·grad)u] = P[u × omega]

    because

        (u·grad)u
        = omega × u + grad(|u|^2/2).
    """
    u = ifft_vector_real(
        uhat,
        workers=workers,
    )

    omega = ifft_vector_real(
        curl_hat(
            uhat,
            operators,
        ),
        workers=workers,
    )

    cross = np.empty_like(
        u,
        dtype=np.float64,
    )

    cross[0] = (
        u[1] * omega[2]
        - u[2] * omega[1]
    )

    cross[1] = (
        u[2] * omega[0]
        - u[0] * omega[2]
    )

    cross[2] = (
        u[0] * omega[1]
        - u[1] * omega[0]
    )

    nhat = fft_vector(
        cross,
        workers=workers,
    )

    return project_dealias(
        nhat,
        operators,
        zero_mean=True,
    )


def kinetic_energy_hat(
    uhat,
):
    n = uhat.shape[1]

    return float(
        0.5
        * np.sum(
            np.abs(uhat) ** 2,
            dtype=np.float64,
        )
        / n**6
    )


def kinetic_energy_physical(
    u,
):
    return float(
        0.5
        * np.mean(
            np.sum(
                u * u,
                axis=0,
            ),
            dtype=np.float64,
        )
    )


def spectral_inner_product(
    ahat,
    bhat,
):
    n = ahat.shape[1]

    return float(
        np.real(
            np.vdot(
                ahat,
                bhat,
            )
        )
        / n**6
    )


def dissipation_rate(
    uhat,
    operators,
    viscosity,
):
    return float(
        viscosity
        * np.sum(
            operators["k2"][None, ...]
            * np.abs(uhat) ** 2,
            dtype=np.float64,
        )
        / uhat.shape[1]**6
    )


def constant_power_forcing_hat(
    uhat,
    operators,
    epsilon_input,
):
    """
    Negative-damping forcing in modes 1 <= |k| <= 2.

    The coefficient is chosen so that:

        <u · f> = epsilon_input
    """
    if epsilon_input <= 0.0:
        return (
            np.zeros_like(uhat),
            0.0,
            0.0,
        )

    forced = operators["forced"]

    forced_energy = float(
        0.5
        * np.sum(
            np.abs(
                uhat[:, forced]
            ) ** 2,
            dtype=np.float64,
        )
        / uhat.shape[1]**6
    )

    if forced_energy <= 1e-20:
        raise RuntimeError(
            "Forced-band energy is too small."
        )

    alpha = float(
        epsilon_input
        / (2.0 * forced_energy)
    )

    fhat = np.zeros_like(
        uhat,
        dtype=np.complex128,
    )

    fhat[:, forced] = (
        alpha
        * uhat[:, forced]
    )

    injection = spectral_inner_product(
        uhat,
        fhat,
    )

    return fhat, alpha, injection


def rhs_hat(
    uhat,
    operators,
    viscosity,
    epsilon_input,
    workers=1,
    nonlinear=True,
):
    uhat_clean = project_dealias(
        uhat,
        operators,
        zero_mean=True,
    )

    if nonlinear:
        nhat = nonlinear_hat(
            uhat_clean,
            operators,
            workers=workers,
        )
    else:
        nhat = np.zeros_like(
            uhat_clean
        )

    viscous = (
        -viscosity
        * operators["k2"][None, ...]
        * uhat_clean
    )

    fhat, forcing_alpha, injection = (
        constant_power_forcing_hat(
            uhat_clean,
            operators,
            epsilon_input,
        )
    )

    rhs = (
        nhat
        + viscous
        + fhat
    )

    rhs = project_dealias(
        rhs,
        operators,
        zero_mean=True,
    )

    diagnostics = {
        "forcing_alpha":
            float(forcing_alpha),
        "energy_injection":
            float(injection),
        "dissipation_rate":
            dissipation_rate(
                uhat_clean,
                operators,
                viscosity,
            ),
        "nonlinear_energy_rate":
            spectral_inner_product(
                uhat_clean,
                nhat,
            ),
    }

    return rhs, diagnostics


def ssprk3_step(
    uhat,
    dt,
    operators,
    viscosity,
    epsilon_input,
    workers=1,
    nonlinear=True,
):
    """
    Third-order strong-stability-preserving RK.
    """
    u0 = project_dealias(
        uhat,
        operators,
        zero_mean=True,
    )

    r0, d0 = rhs_hat(
        u0,
        operators,
        viscosity,
        epsilon_input,
        workers=workers,
        nonlinear=nonlinear,
    )

    u1 = project_dealias(
        u0 + dt * r0,
        operators,
        zero_mean=True,
    )

    r1, d1 = rhs_hat(
        u1,
        operators,
        viscosity,
        epsilon_input,
        workers=workers,
        nonlinear=nonlinear,
    )

    u2 = project_dealias(
        0.75 * u0
        + 0.25 * (
            u1 + dt * r1
        ),
        operators,
        zero_mean=True,
    )

    r2, d2 = rhs_hat(
        u2,
        operators,
        viscosity,
        epsilon_input,
        workers=workers,
        nonlinear=nonlinear,
    )

    unew = project_dealias(
        (
            1.0 / 3.0
        ) * u0
        +
        (
            2.0 / 3.0
        ) * (
            u2 + dt * r2
        ),
        operators,
        zero_mean=True,
    )

    diagnostics = {
        "stage0": d0,
        "stage1": d1,
        "stage2": d2,
    }

    return unew, diagnostics


def maximum_speed(
    uhat,
    workers=1,
):
    u = ifft_vector_real(
        uhat,
        workers=workers,
    )

    speed = np.sqrt(
        np.sum(
            u * u,
            axis=0,
        )
    )

    return float(np.max(speed))


def stable_timestep(
    uhat,
    operators,
    viscosity,
    workers=1,
    cfl=0.4,
    viscous_safety=0.5,
    dt_max=0.02,
):
    umax = maximum_speed(
        uhat,
        workers=workers,
    )

    dt_advective = float(
        cfl
        * operators["dx"]
        / max(umax, 1e-12)
    )

    if viscosity > 0.0:
        dt_viscous = float(
            viscous_safety
            / (
                viscosity
                * operators[
                    "k2_dealiased_max"
                ]
            )
        )
    else:
        dt_viscous = np.inf

    dt = min(
        dt_advective,
        dt_viscous,
        dt_max,
    )

    return {
        "dt": float(dt),
        "dt_advective":
            float(dt_advective),
        "dt_viscous":
            float(dt_viscous),
        "maximum_speed":
            float(umax),
    }


def random_initial_condition(
    operators,
    seed=20260725,
    target_energy=0.5,
    spectral_peak=4.0,
    workers=1,
):
    """
    Smooth divergence-free random initial condition.

    The rough amplitude envelope is:

        k^2 exp[-(k/k0)^2]
    """
    n = operators["n"]

    rng = np.random.default_rng(
        seed
    )

    u = rng.normal(
        size=(3, n, n, n)
    )

    uhat = fft_vector(
        u,
        workers=workers,
    )

    kmag = np.sqrt(
        operators["k2"]
    )

    envelope = (
        kmag**2
        * np.exp(
            -(
                kmag
                / spectral_peak
            ) ** 2
        )
    )

    envelope[0, 0, 0] = 0.0

    uhat *= envelope[None, ...]

    uhat = project_dealias(
        uhat,
        operators,
        zero_mean=True,
    )

    energy = kinetic_energy_hat(
        uhat
    )

    if energy <= 0.0:
        raise RuntimeError(
            "Initial condition has zero energy."
        )

    uhat *= np.sqrt(
        target_energy / energy
    )

    return project_dealias(
        uhat,
        operators,
        zero_mean=True,
    )


def hermitian_error(
    uhat,
):
    n = uhat.shape[1]

    partner_index = (
        -np.arange(n)
    ) % n

    partner = np.conj(
        uhat[
            :,
            partner_index,
            :,
            :,
        ][
            :,
            :,
            partner_index,
            :,
        ][
            :,
            :,
            :,
            partner_index,
        ]
    )

    return float(
        np.max(
            np.abs(
                uhat - partner
            )
        )
    )


def divergence_relative_error(
    uhat,
    operators,
):
    divergence = divergence_hat(
        uhat,
        operators,
    )

    numerator = float(
        np.max(
            np.abs(divergence)
        )
    )

    scale = (
        np.sqrt(
            operators["k2"]
        )[None, ...]
        * np.abs(uhat)
    )

    denominator = float(
        np.max(scale)
    )

    return numerator / max(
        denominator,
        1e-30,
    )


def dealiased_leakage(
    uhat,
    operators,
):
    outside = np.abs(
        uhat[
            :,
            ~operators["dealias"],
        ]
    )

    if outside.size == 0:
        return 0.0

    return float(
        np.max(outside)
    )
