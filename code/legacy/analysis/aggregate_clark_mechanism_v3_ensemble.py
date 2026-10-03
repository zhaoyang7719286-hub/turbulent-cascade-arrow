from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "legacy_workspace"
)

OUT_DIR = (
    ROOT
    / "phase_ensemble_v1"
    / "summaries"
    / "mechanism_main32_v3"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(2026071301, 2026071333))

REQUIRED_FILES = [
    "mechanism_summary_v3.csv",
    "mechanism_finite_time_v3.csv",
    "mechanism_frame_stats_v3.csv",
    "mechanism_qc_v3.csv",
    "mechanism_run_metadata_v3.json",
]


def seed_run_dir(seed: int) -> Path:
    if seed <= 2026071304:
        base = (
            ROOT
            / "phase_ensemble_v1"
            / "runs"
            / "64_smoke"
        )
    else:
        base = (
            ROOT
            / "phase_ensemble_v1"
            / "runs"
            / "64_main"
        )

    return base / f"seed_{seed}"


def compare_dns_to_surrogates(
    surrogate: pd.DataFrame,
    dns: pd.DataFrame,
    group_columns: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    rows = []

    for _, dns_row in dns.iterrows():
        mask = np.ones(
            len(surrogate),
            dtype=bool,
        )

        group_values = {}

        for column in group_columns:
            value = dns_row[column]
            mask &= surrogate[column] == value
            group_values[column] = value

        block = surrogate.loc[mask].copy()

        if len(block) != 32:
            raise RuntimeError(
                "Expected 32 surrogate rows for "
                f"{group_values}; found {len(block)}"
            )

        for metric in metrics:
            values = pd.to_numeric(
                block[metric],
                errors="coerce",
            ).to_numpy(dtype=np.float64)

            values = values[np.isfinite(values)]

            dns_value = float(dns_row[metric])

            if len(values) != 32:
                raise RuntimeError(
                    f"Non-finite surrogate data for "
                    f"{group_values}, metric={metric}"
                )

            mean = float(np.mean(values))
            median = float(np.median(values))
            std = float(np.std(values, ddof=1))

            q025, q25, q75, q975 = np.quantile(
                values,
                [0.025, 0.25, 0.75, 0.975],
            )

            n_ge = int(np.count_nonzero(
                values >= dns_value
            ))

            n_le = int(np.count_nonzero(
                values <= dns_value
            ))

            p_greater = (
                n_ge + 1
            ) / (
                len(values) + 1
            )

            p_less = (
                n_le + 1
            ) / (
                len(values) + 1
            )

            p_two_sided = min(
                1.0,
                2.0 * min(
                    p_greater,
                    p_less,
                ),
            )

            z_effect = (
                (dns_value - mean) / std
                if std > 0.0
                else np.nan
            )

            row = {
                **group_values,
                "metric": metric,
                "dns_value": dns_value,
                "surrogate_n": len(values),
                "surrogate_mean": mean,
                "surrogate_median": median,
                "surrogate_std": std,
                "surrogate_q025": float(q025),
                "surrogate_q25": float(q25),
                "surrogate_q75": float(q75),
                "surrogate_q975": float(q975),
                "surrogate_min": float(np.min(values)),
                "surrogate_max": float(np.max(values)),
                "dns_minus_surrogate_mean": (
                    dns_value - mean
                ),
                "z_effect": z_effect,
                "n_surrogate_ge_dns": n_ge,
                "n_surrogate_le_dns": n_le,
                "empirical_p_greater": p_greater,
                "empirical_p_less": p_less,
                "empirical_p_two_sided": (
                    p_two_sided
                ),
                "dns_above_all_surrogates": (
                    n_ge == 0
                ),
                "dns_below_all_surrogates": (
                    n_le == 0
                ),
                "dns_outside_surrogate_95pct": bool(
                    dns_value < q025
                    or dns_value > q975
                ),
            }

            rows.append(row)

    return pd.DataFrame(rows)


def regression_rows(
    data: pd.DataFrame,
    group_columns: list[str],
    metric_pairs: list[tuple[str, str]],
    analysis_name: str,
) -> pd.DataFrame:
    rows = []

    grouped = data.groupby(
        group_columns,
        dropna=False,
        sort=True,
    )

    for group_key, block in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_values = dict(
            zip(group_columns, group_key)
        )

        if len(block) != 32:
            raise RuntimeError(
                f"Expected 32 rows for {group_values}; "
                f"found {len(block)}"
            )

        for x_metric, y_metric in metric_pairs:
            x = pd.to_numeric(
                block[x_metric],
                errors="coerce",
            ).to_numpy(dtype=np.float64)

            y = pd.to_numeric(
                block[y_metric],
                errors="coerce",
            ).to_numpy(dtype=np.float64)

            finite = (
                np.isfinite(x)
                & np.isfinite(y)
            )

            x = x[finite]
            y = y[finite]

            if len(x) < 3:
                corr = np.nan
                slope = np.nan
                intercept = np.nan
                r2 = np.nan
            else:
                corr = float(
                    np.corrcoef(x, y)[0, 1]
                )

                slope, intercept = np.polyfit(
                    x,
                    y,
                    deg=1,
                )

                slope = float(slope)
                intercept = float(intercept)
                r2 = float(corr * corr)

            rows.append({
                "analysis": analysis_name,
                **group_values,
                "x_metric": x_metric,
                "y_metric": y_metric,
                "n_seeds": len(x),
                "correlation": corr,
                "slope": slope,
                "intercept": intercept,
                "r_squared": r2,
            })

    return pd.DataFrame(rows)


inventory_rows = []
summary_chunks = []
finite_chunks = []

for seed in SEEDS:
    run_dir = seed_run_dir(seed)
    mechanism_dir = run_dir / "mechanism_v3"

    missing = [
        name
        for name in REQUIRED_FILES
        if not (mechanism_dir / name).exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"seed={seed}: missing {missing}"
        )

    metadata_path = (
        mechanism_dir
        / "mechanism_run_metadata_v3.json"
    )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    qc_pass = metadata.get("qc_pass") is True

    max_difference = float(
        metadata.get(
            "max_v3_qc_abs_difference",
            np.inf,
        )
    )

    if metadata.get("source_seed") != seed:
        raise RuntimeError(
            f"Seed mismatch for {seed}"
        )

    if not qc_pass:
        raise RuntimeError(
            f"QC failed for seed {seed}"
        )

    if max_difference > 1.0e-12:
        raise RuntimeError(
            f"QC tolerance exceeded for seed "
            f"{seed}: {max_difference}"
        )

    summary = pd.read_csv(
        mechanism_dir
        / "mechanism_summary_v3.csv"
    )

    finite = pd.read_csv(
        mechanism_dir
        / "mechanism_finite_time_v3.csv"
    )

    if len(summary) != 2:
        raise RuntimeError(
            f"seed={seed}: expected 2 summary rows; "
            f"found {len(summary)}"
        )

    if len(finite) != 6:
        raise RuntimeError(
            f"seed={seed}: expected 6 finite rows; "
            f"found {len(finite)}"
        )

    summary["case"] = "surrogate"
    summary["run_dir"] = str(run_dir)

    finite["case"] = "surrogate"
    finite["run_dir"] = str(run_dir)

    summary_chunks.append(summary)
    finite_chunks.append(finite)

    inventory_rows.append({
        "source_seed": seed,
        "run_dir": str(run_dir),
        "qc_pass": qc_pass,
        "max_v3_qc_abs_difference": (
            max_difference
        ),
        "n_summary_rows": len(summary),
        "n_finite_rows": len(finite),
        "numpy_version": metadata.get(
            "numpy_version"
        ),
        "scipy_version": metadata.get(
            "scipy_version"
        ),
    })


inventory = pd.DataFrame(inventory_rows)

surrogate_summary = pd.concat(
    summary_chunks,
    ignore_index=True,
)

surrogate_finite = pd.concat(
    finite_chunks,
    ignore_index=True,
)

if surrogate_summary["source_seed"].nunique() != 32:
    raise RuntimeError(
        "Surrogate summary does not contain "
        "32 unique seeds."
    )

if surrogate_finite["source_seed"].nunique() != 32:
    raise RuntimeError(
        "Surrogate finite table does not "
        "contain 32 unique seeds."
    )


dns_dir = (
    ROOT
    / "phase_ensemble_v1"
    / "runs"
    / "64_reference_dns"
    / "mechanism_v3"
)

dns_metadata_path = (
    dns_dir
    / "mechanism_run_metadata_v3.json"
)

if not dns_metadata_path.exists():
    raise FileNotFoundError(
        "DNS mechanism reference is missing: "
        f"{dns_metadata_path}"
    )

dns_metadata = json.loads(
    dns_metadata_path.read_text(
        encoding="utf-8"
    )
)

if dns_metadata.get("qc_pass") is not True:
    raise RuntimeError(
        "DNS mechanism QC did not pass."
    )

if float(
    dns_metadata[
        "max_v3_qc_abs_difference"
    ]
) > 1.0e-12:
    raise RuntimeError(
        "DNS mechanism QC tolerance exceeded."
    )

dns_summary = pd.read_csv(
    dns_dir
    / "mechanism_summary_v3.csv"
)

dns_finite = pd.read_csv(
    dns_dir
    / "mechanism_finite_time_v3.csv"
)

dns_summary["case"] = "dns"
dns_finite["case"] = "dns"


instantaneous_metrics = [
    "Pi_normalized_mean",
    "Pi_sign_bias",
    "M_normalized_mean",
    "M_sign_bias",
    "T_normalized_mean",
    "T_sign_bias",
    "quarter_W_normalized_mean",
    "quarter_W_sign_bias",
    "corr_Pi_M_pooled",
    "corr_Pi_T_pooled",
    "corr_Pi_quarter_W_pooled",
    "top5_positive_M_z_frame_mean",
    "top5_positive_T_z_frame_mean",
    "top5_positive_quarter_W_z_frame_mean",
    "corr_Pi_Pi_dev_pooled",
    "corr_Pi_Pi_vol_pooled",
]

finite_metrics = [
    "I_Pi_normalized_mean",
    "I_Pi_sign_bias",
    "I_M_normalized_mean",
    "I_M_sign_bias",
    "I_T_normalized_mean",
    "I_T_sign_bias",
    "I_quarter_W_normalized_mean",
    "I_quarter_W_sign_bias",
    "corr_I_Pi_I_M",
    "corr_I_Pi_I_T",
    "corr_I_Pi_I_quarter_W",
]


comparison_instantaneous = (
    compare_dns_to_surrogates(
        surrogate=surrogate_summary,
        dns=dns_summary,
        group_columns=["scale"],
        metrics=instantaneous_metrics,
    )
)

comparison_finite = (
    compare_dns_to_surrogates(
        surrogate=surrogate_finite,
        dns=dns_finite,
        group_columns=[
            "scale",
            "window_frames",
        ],
        metrics=finite_metrics,
    )
)


instantaneous_pairs = [
    (
        "M_normalized_mean",
        "Pi_normalized_mean",
    ),
    (
        "M_sign_bias",
        "Pi_sign_bias",
    ),
    (
        "T_normalized_mean",
        "Pi_normalized_mean",
    ),
    (
        "T_sign_bias",
        "Pi_sign_bias",
    ),
    (
        "quarter_W_normalized_mean",
        "Pi_normalized_mean",
    ),
    (
        "quarter_W_sign_bias",
        "Pi_sign_bias",
    ),
]

finite_pairs = [
    (
        "I_M_normalized_mean",
        "I_Pi_normalized_mean",
    ),
    (
        "I_M_sign_bias",
        "I_Pi_sign_bias",
    ),
    (
        "I_T_normalized_mean",
        "I_Pi_normalized_mean",
    ),
    (
        "I_T_sign_bias",
        "I_Pi_sign_bias",
    ),
    (
        "I_quarter_W_normalized_mean",
        "I_Pi_normalized_mean",
    ),
    (
        "I_quarter_W_sign_bias",
        "I_Pi_sign_bias",
    ),
]

cross_seed_instantaneous = regression_rows(
    data=surrogate_summary,
    group_columns=["scale"],
    metric_pairs=instantaneous_pairs,
    analysis_name="instantaneous_surrogate_seeds",
)

cross_seed_finite = regression_rows(
    data=surrogate_finite,
    group_columns=[
        "scale",
        "window_frames",
    ],
    metric_pairs=finite_pairs,
    analysis_name="finite_time_surrogate_seeds",
)

cross_seed = pd.concat(
    [
        cross_seed_instantaneous,
        cross_seed_finite,
    ],
    ignore_index=True,
)


outputs = {
    "mechanism_qc_inventory_v3.csv": inventory,
    "mechanism_surrogate_summary_v3.csv": (
        surrogate_summary
    ),
    "mechanism_surrogate_finite_time_v3.csv": (
        surrogate_finite
    ),
    "mechanism_dns_summary_v3.csv": (
        dns_summary
    ),
    "mechanism_dns_finite_time_v3.csv": (
        dns_finite
    ),
    "mechanism_dns_vs_surrogate_instantaneous_v3.csv": (
        comparison_instantaneous
    ),
    "mechanism_dns_vs_surrogate_finite_time_v3.csv": (
        comparison_finite
    ),
    "mechanism_cross_seed_regression_v3.csv": (
        cross_seed
    ),
}

for filename, dataframe in outputs.items():
    path = OUT_DIR / filename
    dataframe.to_csv(
        path,
        index=False,
    )
    print("saved:", path)


metadata = {
    "n_surrogate_seeds": 32,
    "seed_start": SEEDS[0],
    "seed_end": SEEDS[-1],
    "scales": sorted(
        surrogate_summary["scale"]
        .unique()
        .tolist()
    ),
    "window_frames": sorted(
        surrogate_finite["window_frames"]
        .unique()
        .tolist()
    ),
    "empirical_p_minimum": 1.0 / 33.0,
    "dns_qc_pass": True,
    "surrogate_qc_pass_count": int(
        inventory["qc_pass"].sum()
    ),
    "output_files": list(outputs.keys()),
}

metadata_path = (
    OUT_DIR
    / "mechanism_ensemble_metadata_v3.json"
)

metadata_path.write_text(
    json.dumps(
        metadata,
        indent=2,
    ),
    encoding="utf-8",
)

print("saved:", metadata_path)


print("\n" + "=" * 88)
print("ENSEMBLE AUDIT")
print("=" * 88)
print("surrogate seeds       =", len(inventory))
print(
    "unique surrogate seeds=",
    surrogate_summary[
        "source_seed"
    ].nunique(),
)
print(
    "QC pass count         =",
    int(inventory["qc_pass"].sum()),
)
print(
    "maximum QC difference =",
    inventory[
        "max_v3_qc_abs_difference"
    ].max(),
)
print(
    "DNS QC difference     =",
    dns_metadata[
        "max_v3_qc_abs_difference"
    ],
)


print("\n" + "=" * 88)
print("KEY INSTANTANEOUS DNS-vs-NULL RESULTS")
print("=" * 88)

key_instantaneous = (
    comparison_instantaneous[
        comparison_instantaneous[
            "metric"
        ].isin([
            "Pi_normalized_mean",
            "Pi_sign_bias",
            "M_normalized_mean",
            "M_sign_bias",
            "corr_Pi_M_pooled",
            "top5_positive_M_z_frame_mean",
        ])
    ]
)

print(
    key_instantaneous[
        [
            "scale",
            "metric",
            "dns_value",
            "surrogate_mean",
            "surrogate_q025",
            "surrogate_q975",
            "surrogate_min",
            "surrogate_max",
            "z_effect",
            "n_surrogate_ge_dns",
            "empirical_p_greater",
            "dns_above_all_surrogates",
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 88)
print("KEY FINITE-TIME RESULTS: N_tau=10")
print("=" * 88)

key_finite = (
    comparison_finite[
        (
            comparison_finite[
                "window_frames"
            ] == 10
        )
        & comparison_finite[
            "metric"
        ].isin([
            "I_Pi_normalized_mean",
            "I_Pi_sign_bias",
            "I_M_normalized_mean",
            "I_M_sign_bias",
            "corr_I_Pi_I_M",
        ])
    ]
)

print(
    key_finite[
        [
            "scale",
            "window_frames",
            "metric",
            "dns_value",
            "surrogate_mean",
            "surrogate_q025",
            "surrogate_q975",
            "surrogate_min",
            "surrogate_max",
            "z_effect",
            "n_surrogate_ge_dns",
            "empirical_p_greater",
            "dns_above_all_surrogates",
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 88)
print("CROSS-SEED Pi-M RELATIONS")
print("=" * 88)

key_cross_seed = cross_seed[
    cross_seed["x_metric"].isin([
        "M_normalized_mean",
        "M_sign_bias",
        "I_M_normalized_mean",
        "I_M_sign_bias",
    ])
]

print(
    key_cross_seed.to_string(index=False)
)

print("\nPASS: 32 surrogate mechanism seeds and DNS aggregated.")
