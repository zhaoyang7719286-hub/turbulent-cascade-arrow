from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def seed_from_dir(run_dir: Path) -> int:
    return int(run_dir.name.split("_")[-1])


def find_completed_runs(roots: list[Path]) -> list[Path]:
    runs: dict[int, Path] = {}

    for root in roots:
        if not root.exists():
            continue

        for run_dir in sorted(root.glob("seed_*")):
            if not (run_dir / "DONE").exists():
                continue

            seed = seed_from_dir(run_dir)
            runs[seed] = run_dir

    return [runs[seed] for seed in sorted(runs)]


def load_surrogate_metrics(run_dirs: list[Path]) -> pd.DataFrame:
    rows = []

    for run_dir in run_dirs:
        seed = seed_from_dir(run_dir)
        finite_dir = run_dir / "finite_time_v3"

        inst_path = (
            finite_dir
            / "instantaneous_arrow_metrics_v3.csv"
        )
        ft_path = (
            finite_dir
            / "finite_time_arrow_summary_v3.csv"
        )

        if not inst_path.exists():
            raise FileNotFoundError(inst_path)

        if not ft_path.exists():
            raise FileNotFoundError(ft_path)

        inst = pd.read_csv(inst_path)

        for _, row in inst.iterrows():
            for metric in [
                "sign_bias_Pi",
                "normalized_mean_Pi",
            ]:
                rows.append({
                    "seed": seed,
                    "statistic_type": "instantaneous",
                    "scale": int(row["scale"]),
                    "window_frames": 0,
                    "metric": metric,
                    "value": float(row[metric]),
                })

        ft = pd.read_csv(ft_path)
        ft = ft[ft["mode"] == "raw_zero"].copy()

        for _, row in ft.iterrows():
            for metric in [
                "sign_bias_I",
                "normalized_mean_I",
            ]:
                rows.append({
                    "seed": seed,
                    "statistic_type": "finite_time",
                    "scale": int(row["scale"]),
                    "window_frames": int(
                        row["window_frames"]
                    ),
                    "metric": metric,
                    "value": float(row[metric]),
                })

    return pd.DataFrame(rows)


def load_dns_metrics(dns_dir: Path) -> pd.DataFrame:
    rows = []

    inst = pd.read_csv(
        dns_dir
        / "instantaneous_arrow_metrics_v3.csv"
    )

    for _, row in inst.iterrows():
        for metric in [
            "sign_bias_Pi",
            "normalized_mean_Pi",
        ]:
            rows.append({
                "statistic_type": "instantaneous",
                "scale": int(row["scale"]),
                "window_frames": 0,
                "metric": metric,
                "dns_value": float(row[metric]),
            })

    ft = pd.read_csv(
        dns_dir
        / "finite_time_arrow_summary_v3.csv"
    )

    ft = ft[ft["mode"] == "raw_zero"].copy()

    for _, row in ft.iterrows():
        for metric in [
            "sign_bias_I",
            "normalized_mean_I",
        ]:
            rows.append({
                "statistic_type": "finite_time",
                "scale": int(row["scale"]),
                "window_frames": int(
                    row["window_frames"]
                ),
                "metric": metric,
                "dns_value": float(row[metric]),
            })

    return pd.DataFrame(rows)


def load_qc(run_dirs: list[Path]) -> pd.DataFrame:
    rows = []

    for run_dir in run_dirs:
        seed = seed_from_dir(run_dir)
        qc_path = run_dir / "qc_status.json"

        if not qc_path.exists():
            rows.append({
                "seed": seed,
                "qc_pass": False,
                "errors": "missing qc_status.json",
            })
            continue

        qc = json.loads(
            qc_path.read_text(encoding="utf-8")
        )

        row = {
            "seed": seed,
            "qc_pass": bool(qc["qc_pass"]),
            "errors": "; ".join(qc.get("errors", [])),
        }

        row.update(qc.get("checks", {}))
        rows.append(row)

    return pd.DataFrame(rows)


def summarize(values: np.ndarray, dns_value: float) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n = len(values)
    mean = float(np.mean(values))

    if n > 1:
        std = float(np.std(values, ddof=1))
    else:
        std = np.nan

    if np.isfinite(std) and std > 0:
        z_dns = float((dns_value - mean) / std)
    else:
        z_dns = np.nan

    # 单侧经验随机化p值：surrogate >= DNS
    empirical_p = float(
        (1 + np.sum(values >= dns_value))
        / (n + 1)
    )

    return {
        "n_seeds": n,
        "surrogate_mean": mean,
        "surrogate_std": std,
        "surrogate_median": float(
            np.median(values)
        ),
        "surrogate_min": float(np.min(values)),
        "surrogate_max": float(np.max(values)),
        "surrogate_q025": float(
            np.quantile(values, 0.025)
        ),
        "surrogate_q25": float(
            np.quantile(values, 0.25)
        ),
        "surrogate_q75": float(
            np.quantile(values, 0.75)
        ),
        "surrogate_q975": float(
            np.quantile(values, 0.975)
        ),
        "dns_value": float(dns_value),
        "dns_minus_surrogate_mean": float(
            dns_value - mean
        ),
        "z_dns_vs_surrogate": z_dns,
        "n_surrogate_ge_dns": int(
            np.sum(values >= dns_value)
        ),
        "empirical_p_greater": empirical_p,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_roots",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--dns_dir",
        required=True,
    )
    parser.add_argument(
        "--outdir",
        required=True,
    )

    args = parser.parse_args()

    run_roots = [
        Path(path) for path in args.run_roots
    ]
    dns_dir = Path(args.dns_dir)
    outdir = Path(args.outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    run_dirs = find_completed_runs(run_roots)

    if not run_dirs:
        raise RuntimeError(
            "No completed seed directories found."
        )

    surrogate = load_surrogate_metrics(run_dirs)
    dns = load_dns_metrics(dns_dir)
    qc = load_qc(run_dirs)

    if not qc["qc_pass"].all():
        failed = qc[~qc["qc_pass"]]
        raise RuntimeError(
            "Some seeds failed QC:\n"
            + failed.to_string(index=False)
        )

    merged = surrogate.merge(
        dns,
        on=[
            "statistic_type",
            "scale",
            "window_frames",
            "metric",
        ],
        how="left",
        validate="many_to_one",
    )

    group_columns = [
        "statistic_type",
        "scale",
        "window_frames",
        "metric",
    ]

    summary_rows = []

    for keys, group in merged.groupby(
        group_columns,
        sort=True,
    ):
        dns_value = float(
            group["dns_value"].iloc[0]
        )

        row = dict(zip(group_columns, keys))
        row.update(
            summarize(
                group["value"].to_numpy(),
                dns_value,
            )
        )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    surrogate.to_csv(
        outdir / "seed_metrics_64.csv",
        index=False,
    )

    qc.to_csv(
        outdir / "qc_seed_summary_64.csv",
        index=False,
    )

    summary.to_csv(
        outdir / "ensemble_summary_64.csv",
        index=False,
    )

    print("\nCompleted seeds:")
    print(
        sorted(surrogate["seed"].unique())
    )

    print("\nQC summary:")
    print(
        qc[
            ["seed", "qc_pass", "errors"]
        ].to_string(index=False)
    )

    print("\nEnsemble summary:")
    display_columns = [
        "metric",
        "scale",
        "window_frames",
        "n_seeds",
        "surrogate_mean",
        "surrogate_std",
        "surrogate_min",
        "surrogate_max",
        "dns_value",
        "z_dns_vs_surrogate",
        "empirical_p_greater",
    ]

    print(
        summary[display_columns]
        .sort_values(
            ["metric", "scale", "window_frames"]
        )
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
