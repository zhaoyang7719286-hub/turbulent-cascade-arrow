"""Exact statistic extracted from the upstream compute_pop_amp_isotropic.py.
Only its unrelated field/CLI imports were omitted.
"""
from __future__ import annotations
import numpy as np

def pop_amp_statistics(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]

    if x.size == 0:
        raise RuntimeError("No finite values.")

    positive = x > 0.0
    negative = x < 0.0
    zero = x == 0.0

    n = int(x.size)
    n_pos = int(np.count_nonzero(positive))
    n_neg = int(np.count_nonzero(negative))
    n_zero = int(np.count_nonzero(zero))

    p_pos = n_pos / n
    p_neg = n_neg / n
    p_zero = n_zero / n

    if n_pos == 0 or n_neg == 0:
        raise RuntimeError(
            f"Both signs are required: n_pos={n_pos}, n_neg={n_neg}"
        )

    mu_pos = float(np.mean(x[positive], dtype=np.float64))
    mu_neg = float(np.mean(-x[negative], dtype=np.float64))

    mean_signed = float(np.mean(x, dtype=np.float64))
    mean_absolute = float(np.mean(np.abs(x), dtype=np.float64))

    a_pop = float(p_pos - p_neg)
    a_amp = float(
        (mu_pos - mu_neg)
        / (mu_pos + mu_neg)
    )

    normalized_mean_lhs = float(
        mean_signed / mean_absolute
    )

    normalized_mean_rhs = float(
        (a_pop + a_amp)
        / (1.0 + a_pop * a_amp)
    )

    return {
        "n_samples": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_zero": n_zero,
        "p_positive": p_pos,
        "p_negative": p_neg,
        "p_zero": p_zero,
        "mu_positive": mu_pos,
        "mu_negative_abs": mu_neg,
        "A_pop": a_pop,
        "A_amp": a_amp,
        "mean_signed": mean_signed,
        "mean_absolute": mean_absolute,
        "normalized_mean_lhs": normalized_mean_lhs,
        "normalized_mean_rhs": normalized_mean_rhs,
        "identity_abs_error": float(
            abs(normalized_mean_lhs - normalized_mean_rhs)
        ),
    }
