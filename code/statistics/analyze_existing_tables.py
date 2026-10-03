#!/usr/bin/env python3
"""Conditional mask uncertainty and descriptive crossed-case sensitivity.

No field evolution, no voxel resampling, no independent-DNS inference.
Run: python analyze_existing_tables.py
Dependencies: numpy, pandas, scipy. Input/output paths are package-relative.
"""
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / "data/statistics_inputs"
OUT = ROOT.parents[1] / "outputs/statistical_audit"
OUT.mkdir(parents=True, exist_ok=True)
BOOTSTRAP_REPLICATES = 50000
RANDOM_SEED = 20261003


def summarize(values, reference, indices):
    x = np.asarray(values, dtype=float)
    assert np.isfinite(x).all() and np.isfinite(reference)
    n = len(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    boot_means = x[indices].mean(axis=1)
    boot_low, boot_high = np.quantile(boot_means, [0.025, 0.975])
    half_width = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    n_ge = int(np.sum(x >= reference))
    deleted_means = (x.sum() - x) / (n - 1)
    return dict(
        n_masks=n, reference=reference, mask_mean=mean, mask_sd=sd,
        reference_minus_mask_mean=reference - mean,
        mask_mean_bootstrap95_low=float(boot_low),
        mask_mean_bootstrap95_high=float(boot_high),
        effect_bootstrap95_low=float(reference - boot_high),
        effect_bootstrap95_high=float(reference - boot_low),
        effect_t95_low=reference - mean - half_width,
        effect_t95_high=reference - mean + half_width,
        n_masks_ge_reference=n_ge,
        surrogate_rank_fraction_plus_one=(1 + n_ge) / (n + 1),
        leave_one_mask_effect_min=float((reference - deleted_means).min()),
        leave_one_mask_effect_max=float((reference - deleted_means).max()),
    )


def holm_adjust(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    indices32 = rng.integers(0, 32, (BOOTSTRAP_REPLICATES, 32))
    indices24 = rng.integers(0, 24, (BOOTSTRAP_REPLICATES, 24))
    rows = []
    main_masks = pd.read_csv(DATA / "main32_seed_metrics.csv")
    main_summary = pd.read_csv(DATA / "main32_ensemble_summary.csv")
    keys = ["statistic_type", "scale", "window_frames", "metric"]
    for key, group in main_masks.groupby(keys, sort=True):
        group = group.sort_values("seed")
        assert len(group) == group.seed.nunique() == 32
        match = main_summary
        for column, value in zip(keys, key):
            match = match[match[column] == value]
        assert len(match) == 1
        reference = float(match.iloc[0].dns_value)
        result = summarize(group.value, reference, indices32)
        assert np.isclose(result["mask_mean"], match.iloc[0].surrogate_mean, atol=1e-12)
        rows.append(dict(block="main32", comparator="recorded_DNS", **dict(zip(keys, key)), **result))

    independent = pd.read_csv(DATA / "independent24_case_metrics.csv")
    metrics = ["mean_Pi", "normalized_mean_Pi", "sign_bias_Pi"]
    metrics += [f"{prefix}_w{w}" for prefix in ["mean_I", "normalized_mean_I", "sign_bias_I"] for w in [5, 10, 20]]
    for scale in [4, 8]:
        reference_rows = independent[(independent.case_type == "reference_star32") & (independent.scale == scale)]
        group = independent[(independent.case_type == "surrogate") & (independent.scale == scale)].sort_values("seed")
        assert len(reference_rows) == 1 and len(group) == group.seed.nunique() == 24
        for metric in metrics:
            finite = "_w" in metric
            rows.append(dict(
                block="independent24", comparator="reference_star32", scale=scale,
                metric=metric, statistic_type="finite_time" if finite else "instantaneous",
                window_frames=int(metric.rsplit("_w", 1)[1]) if finite else 0,
                **summarize(group[metric], float(reference_rows.iloc[0][metric]), indices24),
            ))
    intervals = pd.DataFrame(rows)
    intervals.to_csv(OUT / "mask_conditional_intervals.csv", index=False)

    # Audit the existing prespecified family, without changing its rank criterion.
    primary = pd.read_csv(DATA / "independent24_primary_six.csv")
    assert len(primary) == 6
    comparisons = intervals[intervals.block == "independent24"].set_index(["scale", "metric"])
    ranks = np.array([comparisons.loc[(r.scale, r.metric), "surrogate_rank_fraction_plus_one"] for r in primary.itertuples()])
    assert np.allclose(ranks, primary.empirical_p_greater, atol=1e-12)
    primary["holm_adjusted_rank_p_six"] = holm_adjust(ranks)
    primary["holm_reject_at_0_05_under_exchangeable_null"] = primary.holm_adjusted_rank_p_six <= 0.05
    primary.to_csv(OUT / "independent_six_rank_audit.csv", index=False)

    # Dynamic cases are crossed, so report ranges and matched gaps, not iid CIs.
    recovery = pd.read_csv(DATA / "dynamic_heldout_long.csv")
    endpoint = recovery[(recovery.metric == "Pi_normalized_mean") & np.isclose(recovery.relative_time, 1.0)]
    pivot = endpoint.pivot(index=["checkpoint", "phase_seed"], columns="branch", values="recovery")
    frozen = pd.read_csv(DATA / "dynamic_amplitude_matched_long.csv")
    frozen = frozen[(frozen.branch == "amplitude_matched_frozen") & np.isclose(frozen.relative_time, 1.0)]
    frozen = frozen.set_index(["checkpoint", "phase_seed"])["R_Pi_normalized_mean"]
    assert len(pivot) == len(frozen) == 9 and pivot.index.equals(frozen.sort_index().index)
    endpoints = pd.DataFrame({"NL": pivot.nonlinear_recovery, "LIN": pivot.linear_control, "FRZ": frozen})
    endpoints["NL_minus_LIN"] = endpoints.NL - endpoints.LIN
    endpoints["NL_minus_FRZ"] = endpoints.NL - endpoints.FRZ
    assert np.isfinite(endpoints.values).all()
    assert (endpoints[["NL_minus_LIN", "NL_minus_FRZ"]] > 0).all().all()
    endpoints.to_csv(OUT / "dynamic_t1_matched_endpoints.csv")
    sensitivity = []
    reset = endpoints.reset_index()
    for kind, factor in [("checkpoint", "checkpoint"), ("mask", "phase_seed")]:
        for removed in sorted(reset[factor].unique()):
            subset = reset[reset[factor] != removed]
            assert len(subset) == 6
            sensitivity.append(dict(
                omitted_factor=kind, omitted_level=int(removed), retained_cases=len(subset),
                **{f"{c}_mean": float(subset[c].mean()) for c in endpoints.columns},
                NL_minus_LIN_min=float(subset.NL_minus_LIN.min()),
                NL_minus_FRZ_min=float(subset.NL_minus_FRZ.min()),
            ))
    pd.DataFrame(sensitivity).to_csv(OUT / "dynamic_delete_one_factor_sensitivity.csv", index=False)

    input_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.glob("*.csv"))}
    summary = dict(
        bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_rng_seed=RANDOM_SEED,
        bootstrap_method="whole-mask percentile, paired mask indices within each block",
        uncertainty_scope="pointwise conditional Monte Carlo uncertainty for fixed recorded fields and fixed comparator; not flow-population uncertainty",
        rank_test_scope="rank p values are inferential only under an exchangeable phase-Haar reference/null; the algebraic odd-null theorem alone does not give that assumption",
        multiple_comparison_scope="Holm sensitivity audit of the six already recorded independent-block primary metrics, not a new preregistration",
        dynamic_scope="nine crossed checkpoint-by-mask cases; descriptive range and delete-one-factor checks only",
        software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__),
        source_hashes=input_hashes, holm_rejections=int(primary.holm_reject_at_0_05_under_exchangeable_null.sum()),
        dynamic_ranges={c: dict(min=float(endpoints[c].min()), max=float(endpoints[c].max()), mean=float(endpoints[c].mean())) for c in endpoints.columns},
    )
    (OUT / "analysis_metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(intervals[(intervals.block == "independent24") & (intervals.scale == 8) & intervals.metric.isin(["mean_Pi", "normalized_mean_Pi", "sign_bias_Pi"])][["metric", "reference_minus_mask_mean", "effect_bootstrap95_low", "effect_bootstrap95_high", "effect_t95_low", "effect_t95_high", "n_masks_ge_reference"]].to_string(index=False))
    print(primary[["scale", "metric", "empirical_p_greater", "holm_adjusted_rank_p_six"]].to_string(index=False))
    print(json.dumps(summary["dynamic_ranges"], indent=2))


if __name__ == "__main__":
    main()
