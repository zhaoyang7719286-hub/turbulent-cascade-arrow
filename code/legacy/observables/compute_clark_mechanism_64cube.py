from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "datasets/isotropic1024coarse/velocity_64cube_100frames/frames_npz"
OUT_DIR = ROOT / "output/clark_mechanism_64cube"
FIG_DIR = OUT_DIR / "figs"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DX = 2 * np.pi / 1024.0
SCALES = [4, 8]
TOP_PS = [1, 5, 10]

# 为避免 CCDF 保存过多点，每个尺度最多保留这些 Pi+ 样本
MAX_CCDF_SAMPLES_PER_SCALE = 2_000_000
RNG = np.random.default_rng(20260619)


def load_velocity(path):
    z = np.load(path)
    keys = list(z.keys())

    for k in ["U", "velocity", "vel", "data", "arr_0"]:
        if k in z:
            U = z[k]
            break
    else:
        U = z[keys[0]]

    U = np.asarray(U)

    if U.ndim == 4 and U.shape[0] == 3:
        U = np.moveaxis(U, 0, -1)

    if not (U.ndim == 4 and U.shape[-1] == 3):
        raise ValueError(f"Bad velocity shape {U.shape} in {path}; keys={keys}")

    if not np.isfinite(U).all():
        raise ValueError(f"Non-finite velocity values in {path}")

    return U.astype(np.float64, copy=False)


def box_filter(a, width):
    return uniform_filter(a, size=width, mode="wrap")


def grad_periodic(f, dx):
    dfdx = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dx)
    dfdy = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dx)
    dfdz = (np.roll(f, -1, axis=2) - np.roll(f, 1, axis=2)) / (2 * dx)
    return dfdx, dfdy, dfdz


def crop_core(A, margin):
    if margin <= 0:
        return A
    return A[margin:-margin, margin:-margin, margin:-margin]


def safe_corr(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 3:
        return np.nan

    sx = x.std()
    sy = y.std()

    if sx == 0 or sy == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def compute_fields(U, scale, dx):
    """
    返回核心区内：
    Pi_actual, W, T, M, omega2, S2
    """
    # 1. box filter velocity
    Ub = np.empty_like(U)
    for i in range(3):
        Ub[..., i] = box_filter(U[..., i], scale)

    # 2. subfilter stress tau_ij
    tau = np.zeros(U.shape[:3] + (3, 3), dtype=np.float64)

    for i in range(3):
        for j in range(3):
            tau[..., i, j] = box_filter(U[..., i] * U[..., j], scale) - Ub[..., i] * Ub[..., j]

    # 3. filtered velocity gradient A_ij = partial_j u_i
    A = np.zeros(U.shape[:3] + (3, 3), dtype=np.float64)

    for i in range(3):
        gx, gy, gz = grad_periodic(Ub[..., i], dx)
        A[..., i, 0] = gx
        A[..., i, 1] = gy
        A[..., i, 2] = gz

    # 4. strain S_ij
    S = 0.5 * (A + np.swapaxes(A, -1, -2))

    # 5. exact coarse-grained flux Pi_actual = - tau_ij S_ij
    Pi = -np.einsum("...ij,...ij->...", tau, S)

    # 6. vorticity omega_i = curl u
    # A_ij = partial_j u_i
    omega = np.empty(U.shape[:3] + (3,), dtype=np.float64)
    omega[..., 0] = A[..., 2, 1] - A[..., 1, 2]  # dw/dy - dv/dz
    omega[..., 1] = A[..., 0, 2] - A[..., 2, 0]  # du/dz - dw/dx
    omega[..., 2] = A[..., 1, 0] - A[..., 0, 1]  # dv/dx - du/dy

    # 7. W = omega_i S_ij omega_j
    W = np.einsum("...i,...ij,...j->...", omega, S, omega)

    # 8. T = - tr(S^3) = - S_ij S_jk S_ki
    trS3 = np.einsum("...ij,...jk,...ki->...", S, S, S)
    T = -trS3

    # 9. Clark mechanism combination M = 1/4 W + T
    M = 0.25 * W + T

    # 10. auxiliary intensities
    omega2 = np.einsum("...i,...i->...", omega, omega)
    S2 = np.einsum("...ij,...ij->...", S, S)

    # 与已有 Pi 计算保持一致：scale=4 -> core 48^3, scale=8 -> core 32^3
    margin = 2 * scale

    return {
        "Pi": crop_core(Pi, margin),
        "W": crop_core(W, margin),
        "T": crop_core(T, margin),
        "M": crop_core(M, margin),
        "omega2": crop_core(omega2, margin),
        "S2": crop_core(S2, margin),
        "margin": margin,
    }


def top_region_stats(Pi, field, p):
    """
    在 Pi>0 的点中取 top p% Pi+ 区域，计算 field 的 top 均值和 z-score 富集。
    """
    Pi = Pi.ravel()
    field = field.ravel()

    pos_mask = Pi > 0.0
    if pos_mask.sum() < 10:
        return np.nan, np.nan, np.nan, 0

    Pi_pos = Pi[pos_mask]
    k = max(1, int(np.ceil(Pi_pos.size * p / 100.0)))
    threshold = np.partition(Pi_pos, -k)[-k]

    top_mask = pos_mask & (Pi >= threshold)

    all_mean = float(np.nanmean(field))
    all_std = float(np.nanstd(field))
    top_mean = float(np.nanmean(field[top_mask]))

    if all_std > 0:
        z = (top_mean - all_mean) / all_std
    else:
        z = np.nan

    return top_mean, all_mean, z, int(top_mask.sum())


def contribution_concentration(Pi, p):
    """
    C(p)=sum(top p% positive Pi)/sum(positive Pi)
    """
    pos = Pi[Pi > 0.0].ravel()
    if pos.size == 0 or pos.sum() <= 0:
        return np.nan

    k = max(1, int(np.ceil(pos.size * p / 100.0)))
    pos_sorted = np.sort(pos)[::-1]
    return float(pos_sorted[:k].sum() / pos_sorted.sum())


def update_ccdf_samples(store, scale, Pi):
    pos = Pi[Pi > 0.0].ravel()

    if pos.size == 0:
        return

    if scale not in store:
        store[scale] = []

    # 每帧最多采样 30000 个，避免内存过大
    max_per_frame = 30000
    if pos.size > max_per_frame:
        idx = RNG.choice(pos.size, size=max_per_frame, replace=False)
        pos = pos[idx]

    store[scale].append(pos)


def plot_ccdf(samples_dict):
    for scale, chunks in samples_dict.items():
        if not chunks:
            continue

        x = np.concatenate(chunks)
        x = x[np.isfinite(x) & (x > 0)]

        if x.size == 0:
            continue

        # 总量仍过大则再次抽样
        if x.size > MAX_CCDF_SAMPLES_PER_SCALE:
            idx = RNG.choice(x.size, size=MAX_CCDF_SAMPLES_PER_SCALE, replace=False)
            x = x[idx]

        x = np.sort(x)
        n = x.size

        # CCDF: P(X >= x)
        ccdf = 1.0 - np.arange(n) / n

        # 保存 CSV
        out_csv = OUT_DIR / f"Pi_positive_ccdf_scale_{scale}.csv"
        pd.DataFrame({"Pi_positive": x, "CCDF": ccdf}).to_csv(out_csv, index=False)

        # 画图。用 semilogy 更稳，不强行 loglog，因为 Pi 可能跨幅有限。
        plt.figure(figsize=(7, 4.5))
        plt.semilogy(x, ccdf, linewidth=1)
        plt.xlabel(r"$\Pi_\ell^+$")
        plt.ylabel("CCDF")
        plt.title(f"CCDF of positive coarse-grained flux, scale={scale}")
        plt.tight_layout()

        out_png = FIG_DIR / f"Pi_positive_ccdf_scale_{scale}.png"
        plt.savefig(out_png, dpi=300)
        plt.close()

        print("saved:", out_csv)
        print("saved:", out_png)


def main():
    files = sorted(DATA_DIR.glob("velocity_t*.npz"))

    if not files:
        raise FileNotFoundError(f"No velocity_t*.npz found in {DATA_DIR}")

    print("data dir:", DATA_DIR)
    print("n files:", len(files))
    print("first:", files[0].name)
    print("last:", files[-1].name)

    rows = []
    ccdf_samples = {}

    for frame_index, f in enumerate(files, start=1):
        U = load_velocity(f)

        if frame_index == 1:
            print("U shape:", U.shape)
            print("U min/max:", float(U.min()), float(U.max()))

        for scale in SCALES:
            fields = compute_fields(U, scale, DX)

            Pi = fields["Pi"]
            W = fields["W"]
            T = fields["T"]
            M = fields["M"]
            omega2 = fields["omega2"]
            S2 = fields["S2"]

            # 基础统计
            row = {
                "frame_index": frame_index,
                "file": f.name,
                "scale": scale,
                "dx": DX,
                "crop_margin": fields["margin"],
                "core_shape": str(Pi.shape),
                "n_points": Pi.size,

                "Pi_mean": float(Pi.mean()),
                "Pi_std": float(Pi.std()),
                "Pi_min": float(Pi.min()),
                "Pi_max": float(Pi.max()),
                "Pi_positive_fraction": float((Pi > 0).mean()),

                "W_mean": float(W.mean()),
                "T_mean": float(T.mean()),
                "M_mean": float(M.mean()),
                "omega2_mean": float(omega2.mean()),
                "S2_mean": float(S2.mean()),

                "corr_Pi_M": safe_corr(Pi, M),
                "corr_Pi_W": safe_corr(Pi, W),
                "corr_Pi_T": safe_corr(Pi, T),
                "corr_Pi_omega2": safe_corr(Pi, omega2),
                "corr_Pi_S2": safe_corr(Pi, S2),

                "sign_agree_Pi_M": float((np.sign(Pi) == np.sign(M)).mean()),
                "sign_agree_Pi_W": float((np.sign(Pi) == np.sign(W)).mean()),
                "sign_agree_Pi_T": float((np.sign(Pi) == np.sign(T)).mean()),

                "nan_count": int(
                    np.isnan(Pi).sum()
                    + np.isnan(W).sum()
                    + np.isnan(T).sum()
                    + np.isnan(M).sum()
                ),
                "inf_count": int(
                    np.isinf(Pi).sum()
                    + np.isinf(W).sum()
                    + np.isinf(T).sum()
                    + np.isinf(M).sum()
                ),
            }

            # C(p) 和 top Pi+ 区域富集
            for p in TOP_PS:
                row[f"C_{p}"] = contribution_concentration(Pi, p)

                for name, field in [
                    ("W", W),
                    ("T", T),
                    ("M", M),
                    ("omega2", omega2),
                    ("S2", S2),
                ]:
                    top_mean, all_mean, z, n_top = top_region_stats(Pi, field, p)
                    row[f"top{p}_{name}_mean"] = top_mean
                    row[f"top{p}_{name}_all_mean"] = all_mean
                    row[f"top{p}_{name}_z"] = z
                    row[f"top{p}_{name}_n"] = n_top

            rows.append(row)

            update_ccdf_samples(ccdf_samples, scale, Pi)

        if frame_index % 10 == 0:
            print(f"processed {frame_index}/{len(files)}")

    df = pd.DataFrame(rows)

    out_csv = OUT_DIR / "clark_mechanism_frame_stats.csv"
    df.to_csv(out_csv, index=False)
    print("saved:", out_csv)

    # 按尺度聚合
    agg_cols = [
        "Pi_mean",
        "Pi_positive_fraction",
        "W_mean",
        "T_mean",
        "M_mean",
        "corr_Pi_M",
        "corr_Pi_W",
        "corr_Pi_T",
        "corr_Pi_omega2",
        "corr_Pi_S2",
        "sign_agree_Pi_M",
        "sign_agree_Pi_W",
        "sign_agree_Pi_T",
        "C_1",
        "C_5",
        "C_10",
        "top1_W_z",
        "top1_T_z",
        "top1_M_z",
        "top1_omega2_z",
        "top1_S2_z",
        "top5_W_z",
        "top5_T_z",
        "top5_M_z",
        "top5_omega2_z",
        "top5_S2_z",
        "top10_W_z",
        "top10_T_z",
        "top10_M_z",
        "top10_omega2_z",
        "top10_S2_z",
    ]

    summary = df.groupby("scale")[agg_cols].agg(["mean", "std", "min", "max"])
    summary_csv = OUT_DIR / "clark_mechanism_summary_by_scale.csv"
    summary.to_csv(summary_csv)
    print("saved:", summary_csv)

    # 简明 summary，方便直接看
    simple = df.groupby("scale").agg(
        n_frames=("frame_index", "count"),
        Pi_mean=("Pi_mean", "mean"),
        Pi_mean_std=("Pi_mean", "std"),
        positive_fraction=("Pi_positive_fraction", "mean"),
        corr_Pi_M=("corr_Pi_M", "mean"),
        corr_Pi_W=("corr_Pi_W", "mean"),
        corr_Pi_T=("corr_Pi_T", "mean"),
        corr_Pi_omega2=("corr_Pi_omega2", "mean"),
        corr_Pi_S2=("corr_Pi_S2", "mean"),
        sign_agree_Pi_M=("sign_agree_Pi_M", "mean"),
        C_1=("C_1", "mean"),
        C_5=("C_5", "mean"),
        C_10=("C_10", "mean"),
        top1_M_z=("top1_M_z", "mean"),
        top5_M_z=("top5_M_z", "mean"),
        top10_M_z=("top10_M_z", "mean"),
        top5_W_z=("top5_W_z", "mean"),
        top5_T_z=("top5_T_z", "mean"),
        top5_omega2_z=("top5_omega2_z", "mean"),
        top5_S2_z=("top5_S2_z", "mean"),
    )

    simple_csv = OUT_DIR / "clark_mechanism_simple_summary.csv"
    simple.to_csv(simple_csv)
    print("\nSimple summary:")
    print(simple.to_string())
    print("saved:", simple_csv)

    # 画 CCDF
    plot_ccdf(ccdf_samples)

    # 画相关性柱状图
    plot_df = simple.reset_index()

    plt.figure(figsize=(7, 4.5))
    x = np.arange(len(plot_df))
    width = 0.16

    labels = ["corr_Pi_M", "corr_Pi_W", "corr_Pi_T", "corr_Pi_omega2", "corr_Pi_S2"]

    for n, col in enumerate(labels):
        plt.bar(x + (n - 2) * width, plot_df[col], width=width, label=col)

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(x, [f"scale={s}" for s in plot_df["scale"]])
    plt.ylabel("Pearson correlation")
    plt.title("Correlation between actual Pi and mechanism variables")
    plt.legend(fontsize=8)
    plt.tight_layout()

    out_corr = FIG_DIR / "correlation_summary_by_scale.png"
    plt.savefig(out_corr, dpi=300)
    plt.close()
    print("saved:", out_corr)

    # 画 top 5% 富集图
    plt.figure(figsize=(7, 4.5))
    labels2 = ["top5_M_z", "top5_W_z", "top5_T_z", "top5_omega2_z", "top5_S2_z"]

    for n, col in enumerate(labels2):
        plt.bar(x + (n - 2) * width, plot_df[col], width=width, label=col)

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(x, [f"scale={s}" for s in plot_df["scale"]])
    plt.ylabel("Z-score enrichment in top 5% Pi+ region")
    plt.title("Mechanism enrichment in top positive-flux regions")
    plt.legend(fontsize=8)
    plt.tight_layout()

    out_enrich = FIG_DIR / "top5_enrichment_summary_by_scale.png"
    plt.savefig(out_enrich, dpi=300)
    plt.close()
    print("saved:", out_enrich)

    print("\nAll outputs saved in:", OUT_DIR)


if __name__ == "__main__":
    main()
