import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter


DX_DEFAULT = 2.0 * np.pi / 1024.0


def load_velocity_npz(path):
    z = np.load(path)
    for key in ["velocity", "u", "U", "vel"]:
        if key in z:
            arr = z[key]
            break
    else:
        arr = None
        for key in z.keys():
            a = z[key]
            if a.ndim == 4 and a.shape[-1] == 3:
                arr = a
                break
        if arr is None:
            raise ValueError(f"No velocity array found in {path}, keys={list(z.keys())}")

    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected velocity shape (...,3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite values in {path}")
    return arr


def storage_to_phys(U):
    # JHTDB cutout storage is interpreted as (z, y, x, component).
    # Convert to physical (x, y, z, component).
    return np.transpose(U, (2, 1, 0, 3))


def phys_to_storage(Uphys):
    return np.transpose(Uphys, (2, 1, 0, 3))


def make_hermitian_phase(shape, seed):
    """
    Generate a random Fourier phase factor with Hermitian symmetry by taking
    the phase of the FFT of a real Gaussian field. The same phase factor will
    be applied to all velocity components and all time frames.
    """
    rng = np.random.default_rng(seed)
    eta = rng.normal(size=shape)
    ehat = np.fft.fftn(eta, axes=(0, 1, 2))
    amp = np.abs(ehat)
    phase = np.ones_like(ehat, dtype=np.complex128)
    mask = amp > 1e-30
    phase[mask] = ehat[mask] / amp[mask]

    # Preserve the mean mode.
    phase[0, 0, 0] = 1.0 + 0.0j
    return phase


def modified_wavenumbers_D4(shape, dx):
    nx, ny, nz = shape

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dx)
    kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=dx)

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")

    qx = (8.0 * np.sin(KX * dx) - np.sin(2.0 * KX * dx)) / (6.0 * dx)
    qy = (8.0 * np.sin(KY * dx) - np.sin(2.0 * KY * dx)) / (6.0 * dx)
    qz = (8.0 * np.sin(KZ * dx) - np.sin(2.0 * KZ * dx)) / (6.0 * dx)

    return qx, qy, qz


def project_solenoidal_D4_hat(Uhat, dx):
    """
    Discrete Helmholtz projection consistent with fourth-order central difference.
    Uhat is in physical axis order (x,y,z,component).
    """
    nx, ny, nz, _ = Uhat.shape
    qx, qy, qz = modified_wavenumbers_D4((nx, ny, nz), dx)
    q2 = qx*qx + qy*qy + qz*qz
    mask = q2 > 0.0

    div = qx * Uhat[..., 0] + qy * Uhat[..., 1] + qz * Uhat[..., 2]

    out = Uhat.copy()
    out[..., 0][mask] -= qx[mask] * div[mask] / q2[mask]
    out[..., 1][mask] -= qy[mask] * div[mask] / q2[mask]
    out[..., 2][mask] -= qz[mask] * div[mask] / q2[mask]

    return out


def divergence_free_phase_randomize(U_storage, phase, dx, rescale_energy=True):
    """
    1. Convert to physical order.
    2. Apply the same scalar Fourier phase to all components.
    3. Project to D4-solenoidal subspace.
    4. Optionally rescale each mode to preserve its vector amplitude.
    5. Project again for numerical cleanliness.
    6. Convert back to storage order.
    """
    Uphys = storage_to_phys(U_storage)
    Uhat = np.fft.fftn(Uphys, axes=(0, 1, 2))

    Uhat_r = Uhat * phase[..., None]
    Uhat_p = project_solenoidal_D4_hat(Uhat_r, dx)

    if rescale_energy:
        amp_before = np.sqrt(np.sum(np.abs(Uhat_r)**2, axis=-1))
        amp_after = np.sqrt(np.sum(np.abs(Uhat_p)**2, axis=-1))
        scale = np.ones_like(amp_before, dtype=np.float64)
        mask = amp_after > 1e-30
        scale[mask] = amp_before[mask] / amp_after[mask]
        Uhat_p = Uhat_p * scale[..., None]
        Uhat_p = project_solenoidal_D4_hat(Uhat_p, dx)

    Uphys_r = np.fft.ifftn(Uhat_p, axes=(0, 1, 2)).real
    return phys_to_storage(Uphys_r)


def box_filter_scalar(a, width):
    if width == 1:
        return a.copy()
    return uniform_filter(a, size=width, mode="nearest")


def box_filter_velocity(U, width):
    Ub = np.empty_like(U)
    for i in range(3):
        Ub[..., i] = box_filter_scalar(U[..., i], width)
    return Ub


def compute_tau(U, Ub, width):
    tau = np.empty(U.shape[:3] + (3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            tau[..., i, j] = box_filter_scalar(U[..., i] * U[..., j], width) - Ub[..., i] * Ub[..., j]
    return tau


def diff4(f, axis, dx):
    return (
        -np.roll(f, -2, axis=axis)
        + 8.0 * np.roll(f, -1, axis=axis)
        - 8.0 * np.roll(f, 1, axis=axis)
        + np.roll(f, 2, axis=axis)
    ) / (12.0 * dx)


def compute_A_storage(Ub, dx):
    """
    A[...,i,j] = partial_j u_i.
    Physical x,y,z correspond to storage axes 2,1,0.
    """
    axis_perm = (2, 1, 0)
    A = np.empty(Ub.shape[:3] + (3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            A[..., i, j] = diff4(Ub[..., i], axis=axis_perm[j], dx=dx)
    return A


def crop_core(a, margin):
    if margin <= 0:
        return a
    if a.ndim == 3:
        return a[margin:-margin, margin:-margin, margin:-margin]
    return a[margin:-margin, margin:-margin, margin:-margin, ...]


def safe_corr(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 100:
        return np.nan

    a = a - a.mean()
    b = b - b.mean()
    sa = np.sqrt(np.mean(a*a))
    sb = np.sqrt(np.mean(b*b))
    if sa <= 0 or sb <= 0:
        return np.nan
    return float(np.mean(a*b) / (sa*sb))


def compute_metrics(U, scale, dx, crop_margin):
    Ub = box_filter_velocity(U, scale)
    tau = compute_tau(U, Ub, scale)
    A = compute_A_storage(Ub, dx)

    S = 0.5 * (A + np.swapaxes(A, -1, -2))
    theta = S[..., 0, 0] + S[..., 1, 1] + S[..., 2, 2]
    normS = np.sqrt(np.einsum("...ij,...ij->...", S, S))

    D = S.copy()
    D[..., 0, 0] -= theta / 3.0
    D[..., 1, 1] -= theta / 3.0
    D[..., 2, 2] -= theta / 3.0

    omega = np.empty(U.shape[:3] + (3,), dtype=np.float64)
    omega[..., 0] = A[..., 2, 1] - A[..., 1, 2]
    omega[..., 1] = A[..., 0, 2] - A[..., 2, 0]
    omega[..., 2] = A[..., 1, 0] - A[..., 0, 1]

    Pi = -np.einsum("...ij,...ij->...", tau, S)
    Pi_dev = -np.einsum("...ij,...ij->...", tau, D)
    taukk = tau[..., 0, 0] + tau[..., 1, 1] + tau[..., 2, 2]
    Pi_vol = -(theta / 3.0) * taukk

    T = -np.einsum("...ij,...jk,...ki->...", S, S, S)
    W = np.einsum("...i,...ij,...j->...", omega, S, omega)
    M = T + 0.25 * W

    Dc = crop_core(D, crop_margin)
    Sc = crop_core(S, crop_margin)
    Pic = crop_core(Pi, crop_margin)
    Pi_devc = crop_core(Pi_dev, crop_margin)
    Pi_volc = crop_core(Pi_vol, crop_margin)
    Tc = crop_core(T, crop_margin)
    Wc = crop_core(W, crop_margin)
    Mc = crop_core(M, crop_margin)
    thetac = crop_core(theta, crop_margin)
    normSc = crop_core(normS, crop_margin)

    rtheta = np.sqrt(np.mean(thetac*thetac)) / (np.sqrt(np.mean(normSc*normSc)) + 1e-30)

    return {
        "rtheta": float(rtheta),
        "mean_Pi": float(np.mean(Pic)),
        "std_Pi": float(np.std(Pic)),
        "positive_fraction_Pi": float(np.mean(Pic > 0)),
        "rms_Pi_vol_over_Pi": float(np.sqrt(np.mean(Pi_volc**2)) / (np.sqrt(np.mean(Pic**2)) + 1e-30)),
        "mean_abs_Pi_vol_over_abs_Pi": float(np.mean(np.abs(Pi_volc)) / (np.mean(np.abs(Pic)) + 1e-30)),
        "corr_Pi_T": safe_corr(Pic, Tc),
        "corr_Pi_W": safe_corr(Pic, Wc),
        "corr_Pi_M": safe_corr(Pic, Mc),
        "corr_Pi_Pi_dev": safe_corr(Pic, Pi_devc),
    }


def shell_spectrum(U_storage):
    Uphys = storage_to_phys(U_storage)
    nx, ny, nz, _ = Uphys.shape

    Uhat = np.fft.fftn(Uphys, axes=(0, 1, 2))
    E = 0.5 * np.sum(np.abs(Uhat)**2, axis=-1)

    ix = np.fft.fftfreq(nx) * nx
    iy = np.fft.fftfreq(ny) * ny
    iz = np.fft.fftfreq(nz) * nz
    IX, IY, IZ = np.meshgrid(ix, iy, iz, indexing="ij")
    shell = np.floor(np.sqrt(IX*IX + IY*IY + IZ*IZ)).astype(int)

    max_shell = int(shell.max())
    spec = np.bincount(shell.ravel(), weights=E.ravel(), minlength=max_shell+1)
    return spec


def spectrum_compare(U_real, U_rand):
    E0 = shell_spectrum(U_real)
    Er = shell_spectrum(U_rand)
    n = min(len(E0), len(Er))
    E0 = E0[:n]
    Er = Er[:n]

    # Ignore the mean shell for the shape error.
    start = 1 if n > 1 else 0
    denom = np.sqrt(np.sum(E0[start:]**2)) + 1e-30
    rel_l2 = np.sqrt(np.sum((Er[start:] - E0[start:])**2)) / denom

    return {
        "spectrum_rel_l2_no_mean": float(rel_l2),
        "total_energy_ratio": float(np.sum(Er) / (np.sum(E0) + 1e-30)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--outdir", default="output/df_phase_randomization_gate_64cube")
    parser.add_argument("--scales", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--max_frames", type=int, default=10)
    parser.add_argument("--dx", type=float, default=DX_DEFAULT)
    parser.add_argument("--crop_factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--no_rescale_energy", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.npz"))[:args.max_frames]
    if not files:
        raise FileNotFoundError(f"No npz files found in {input_dir}")

    U0 = load_velocity_npz(files[0])
    U0phys = storage_to_phys(U0)
    phase = make_hermitian_phase(U0phys.shape[:3], seed=args.seed)

    rows = []
    erows = []

    for idx, fp in enumerate(files):
        print(f"frame {idx+1}/{len(files)} {fp.name}", flush=True)

        U = load_velocity_npz(fp)
        Ur = divergence_free_phase_randomize(
            U,
            phase=phase,
            dx=args.dx,
            rescale_energy=(not args.no_rescale_energy),
        )

        ec = spectrum_compare(U, Ur)
        ec.update({
            "frame_index": idx,
            "file": fp.name,
        })
        erows.append(ec)

        for scale in args.scales:
            crop_margin = args.crop_factor * scale + 4

            m_real = compute_metrics(U, scale, args.dx, crop_margin)
            m_real.update({
                "frame_index": idx,
                "file": fp.name,
                "scale": scale,
                "case": "real",
            })
            rows.append(m_real)

            m_rand = compute_metrics(Ur, scale, args.dx, crop_margin)
            m_rand.update({
                "frame_index": idx,
                "file": fp.name,
                "scale": scale,
                "case": "phase_randomized_D4_projected",
            })
            rows.append(m_rand)

    df = pd.DataFrame(rows)
    edf = pd.DataFrame(erows)

    summary = df.groupby(["scale", "case"]).agg(
        n_frames=("frame_index", "count"),
        rtheta_mean=("rtheta", "mean"),
        rtheta_max=("rtheta", "max"),
        rms_Pi_vol_over_Pi_mean=("rms_Pi_vol_over_Pi", "mean"),
        rms_Pi_vol_over_Pi_max=("rms_Pi_vol_over_Pi", "max"),
        mean_abs_Pi_vol_over_abs_Pi_mean=("mean_abs_Pi_vol_over_abs_Pi", "mean"),
        positive_fraction_Pi_mean=("positive_fraction_Pi", "mean"),
        corr_Pi_T_mean=("corr_Pi_T", "mean"),
        corr_Pi_W_mean=("corr_Pi_W", "mean"),
        corr_Pi_M_mean=("corr_Pi_M", "mean"),
        corr_Pi_Pi_dev_mean=("corr_Pi_Pi_dev", "mean"),
        mean_Pi_mean=("mean_Pi", "mean"),
        std_Pi_mean=("std_Pi", "mean"),
    ).reset_index()

    esummary = edf.agg({
        "spectrum_rel_l2_no_mean": ["mean", "max"],
        "total_energy_ratio": ["mean", "min", "max"],
    })

    df.to_csv(outdir / "df_phase_randomization_gate_frame_stats.csv", index=False)
    summary.to_csv(outdir / "df_phase_randomization_gate_summary.csv", index=False)
    edf.to_csv(outdir / "df_phase_randomization_energy_frame_stats.csv", index=False)
    esummary.to_csv(outdir / "df_phase_randomization_energy_summary.csv")

    meta = {
        "input_dir": str(input_dir),
        "n_files": len(files),
        "scales": args.scales,
        "dx": args.dx,
        "crop_factor": args.crop_factor,
        "seed": args.seed,
        "phase_protocol": "same Hermitian scalar Fourier phase applied to all velocity components and all time frames",
        "projection": "D4 discrete Helmholtz projection using modified wavenumber q(k)",
        "energy_rescale_per_mode": bool(not args.no_rescale_energy),
        "outputs": [
            "df_phase_randomization_gate_frame_stats.csv",
            "df_phase_randomization_gate_summary.csv",
            "df_phase_randomization_energy_frame_stats.csv",
            "df_phase_randomization_energy_summary.csv",
        ],
    }
    with open(outdir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print("Output directory:", outdir)


if __name__ == "__main__":
    main()
