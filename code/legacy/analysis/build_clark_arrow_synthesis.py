#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
import pandas as pd


ROOT = Path(
    "phase_ensemble_v1/summaries"
)

MECH_DIR = ROOT / "mechanism_main32_v3"

POP_DIR = (
    ROOT
    / "pop_amp_decomposition"
    / "aggregates"
)

MECH_INST = (
    MECH_DIR
    / "mechanism_dns_vs_surrogate_instantaneous_v3.csv"
)

MECH_TIME = (
    MECH_DIR
    / "mechanism_dns_vs_surrogate_finite_time_v3.csv"
)

POP_SUR = (
    POP_DIR
    / "main32"
    / "main32_pop_amp_long.csv"
)

POP_REF = (
    POP_DIR
    / "references"
    / "reference_pop_amp_summary.csv"
)

OUTDIR = (
    ROOT
    / "clark_arrow_synthesis"
)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def unique_row(
    df: pd.DataFrame,
    mask: pd.Series,
    description: str,
) -> pd.Series:
    sub = df.loc[mask]

    if len(sub) != 1:
        raise RuntimeError(
            f"{description}: expected exactly one row, "
            f"found {len(sub)}"
        )

    return sub.iloc[0]


def window_values(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(
        df["window_frames"],
        errors="coerce",
    ).fillna(0).astype(int)


def empirical_summary(
    values: np.ndarray,
    reference: float,
) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size != 32:
        raise RuntimeError(
            f"Expected 32 surrogate values, "
            f"found {values.size}"
        )

    n_ge = int(np.sum(values >= reference))

    return {
        "surrogate_mean": float(np.mean(values)),
        "surrogate_std": float(np.std(values, ddof=1)),
        "surrogate_q025": float(
            np.quantile(values, 0.025)
        ),
        "surrogate_q975": float(
            np.quantile(values, 0.975)
        ),
        "surrogate_min": float(np.min(values)),
        "surrogate_max": float(np.max(values)),
        "n_surrogate_ge_reference": n_ge,
        "empirical_p_greater": (
            n_ge + 1
        ) / (
            values.size + 1
        ),
    }


def get_mechanism_row(
    df: pd.DataFrame,
    scale: int,
    metric: str,
    window=None,
) -> pd.Series:

    mask = (
        pd.to_numeric(
            df["scale"],
            errors="raise",
        ).astype(int).eq(scale)
        &
        df["metric"].astype(str).eq(metric)
    )

    if window is not None:
        mask &= (
            pd.to_numeric(
                df["window_frames"],
                errors="raise",
            ).astype(int).eq(window)
        )

    return unique_row(
        df,
        mask,
        f"mechanism scale={scale}, "
        f"metric={metric}, window={window}",
    )


def get_pop_reference(
    df: pd.DataFrame,
    scale: int,
    variable: str,
    window: int,
) -> pd.Series:

    wf = window_values(df)

    mask = (
        df["reference_case"]
        .astype(str)
        .eq("main_reference")
        &
        df["variable"]
        .astype(str)
        .eq(variable)
        &
        pd.to_numeric(
            df["scale"],
            errors="raise",
        ).astype(int).eq(scale)
        &
        wf.eq(window)
    )

    return unique_row(
        df,
        mask,
        f"population reference "
        f"{variable}, scale={scale}, "
        f"window={window}",
    )


def get_pop_surrogate(
    df: pd.DataFrame,
    scale: int,
    variable: str,
    window: int,
) -> pd.DataFrame:

    wf = window_values(df)

    mask = (
        df["variable"]
        .astype(str)
        .eq(variable)
        &
        pd.to_numeric(
            df["scale"],
            errors="raise",
        ).astype(int).eq(scale)
        &
        wf.eq(window)
    )

    sub = df.loc[mask].copy()

    if len(sub) != 32:
        raise RuntimeError(
            f"population surrogate "
            f"{variable}, scale={scale}, "
            f"window={window}: "
            f"expected 32 rows, found {len(sub)}"
        )

    return sub


def build_instantaneous(
    mech: pd.DataFrame,
    pop_sur: pd.DataFrame,
    pop_ref: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for scale in (4, 8):

        corr = get_mechanism_row(
            mech,
            scale,
            "corr_Pi_M_pooled",
        )

        pi_norm = get_mechanism_row(
            mech,
            scale,
            "Pi_normalized_mean",
        )

        m_norm = get_mechanism_row(
            mech,
            scale,
            "M_normalized_mean",
        )

        pi_sign = get_mechanism_row(
            mech,
            scale,
            "Pi_sign_bias",
        )

        m_sign = get_mechanism_row(
            mech,
            scale,
            "M_sign_bias",
        )

        ref = get_pop_reference(
            pop_ref,
            scale,
            "Pi",
            0,
        )

        sur = get_pop_surrogate(
            pop_sur,
            scale,
            "Pi",
            0,
        )

        pop_stats = empirical_summary(
            sur["A_pop"].to_numpy(),
            float(ref["A_pop"]),
        )

        amp_stats = empirical_summary(
            sur["A_amp"].to_numpy(),
            float(ref["A_amp"]),
        )

        rows.append({
            "scale": scale,

            "corr_Pi_M_reference":
                float(corr["dns_value"]),
            "corr_Pi_M_surrogate_mean":
                float(corr["surrogate_mean"]),
            "corr_Pi_M_surrogate_q025":
                float(corr["surrogate_q025"]),
            "corr_Pi_M_surrogate_q975":
                float(corr["surrogate_q975"]),
            "corr_Pi_M_n_ge_reference":
                int(corr["n_surrogate_ge_dns"]),
            "corr_Pi_M_p_greater":
                float(corr["empirical_p_greater"]),

            "Pi_normalized_reference":
                float(pi_norm["dns_value"]),
            "Pi_normalized_surrogate_mean":
                float(pi_norm["surrogate_mean"]),
            "Pi_normalized_p_greater":
                float(pi_norm["empirical_p_greater"]),

            "M_normalized_reference":
                float(m_norm["dns_value"]),
            "M_normalized_surrogate_mean":
                float(m_norm["surrogate_mean"]),
            "M_normalized_p_greater":
                float(m_norm["empirical_p_greater"]),

            "Pi_sign_bias_reference":
                float(pi_sign["dns_value"]),
            "Pi_sign_bias_surrogate_mean":
                float(pi_sign["surrogate_mean"]),
            "Pi_sign_bias_p_greater":
                float(pi_sign["empirical_p_greater"]),

            "M_sign_bias_reference":
                float(m_sign["dns_value"]),
            "M_sign_bias_surrogate_mean":
                float(m_sign["surrogate_mean"]),
            "M_sign_bias_p_greater":
                float(m_sign["empirical_p_greater"]),

            "A_pop_reference":
                float(ref["A_pop"]),
            "A_pop_surrogate_mean":
                pop_stats["surrogate_mean"],
            "A_pop_surrogate_std":
                pop_stats["surrogate_std"],
            "A_pop_n_ge_reference":
                pop_stats[
                    "n_surrogate_ge_reference"
                ],
            "A_pop_p_greater":
                pop_stats[
                    "empirical_p_greater"
                ],

            "A_amp_reference":
                float(ref["A_amp"]),
            "A_amp_surrogate_mean":
                amp_stats["surrogate_mean"],
            "A_amp_surrogate_std":
                amp_stats["surrogate_std"],
            "A_amp_n_ge_reference":
                amp_stats[
                    "n_surrogate_ge_reference"
                ],
            "A_amp_p_greater":
                amp_stats[
                    "empirical_p_greater"
                ],
        })

    return pd.DataFrame(rows)


def build_finite_time(
    mech: pd.DataFrame,
    pop_sur: pd.DataFrame,
    pop_ref: pd.DataFrame,
    window: int = 10,
) -> pd.DataFrame:

    rows = []

    for scale in (4, 8):

        corr = get_mechanism_row(
            mech,
            scale,
            "corr_I_Pi_I_M",
            window,
        )

        pi_norm = get_mechanism_row(
            mech,
            scale,
            "I_Pi_normalized_mean",
            window,
        )

        m_norm = get_mechanism_row(
            mech,
            scale,
            "I_M_normalized_mean",
            window,
        )

        pi_sign = get_mechanism_row(
            mech,
            scale,
            "I_Pi_sign_bias",
            window,
        )

        m_sign = get_mechanism_row(
            mech,
            scale,
            "I_M_sign_bias",
            window,
        )

        ref = get_pop_reference(
            pop_ref,
            scale,
            "I",
            window,
        )

        sur = get_pop_surrogate(
            pop_sur,
            scale,
            "I",
            window,
        )

        pop_stats = empirical_summary(
            sur["A_pop"].to_numpy(),
            float(ref["A_pop"]),
        )

        amp_stats = empirical_summary(
            sur["A_amp"].to_numpy(),
            float(ref["A_amp"]),
        )

        rows.append({
            "scale": scale,
            "window_frames": window,

            "corr_I_Pi_I_M_reference":
                float(corr["dns_value"]),
            "corr_I_Pi_I_M_surrogate_mean":
                float(corr["surrogate_mean"]),
            "corr_I_Pi_I_M_surrogate_q025":
                float(corr["surrogate_q025"]),
            "corr_I_Pi_I_M_surrogate_q975":
                float(corr["surrogate_q975"]),
            "corr_I_Pi_I_M_n_ge_reference":
                int(corr["n_surrogate_ge_dns"]),
            "corr_I_Pi_I_M_p_greater":
                float(corr["empirical_p_greater"]),

            "I_Pi_normalized_reference":
                float(pi_norm["dns_value"]),
            "I_Pi_normalized_surrogate_mean":
                float(pi_norm["surrogate_mean"]),
            "I_Pi_normalized_p_greater":
                float(pi_norm["empirical_p_greater"]),

            "I_M_normalized_reference":
                float(m_norm["dns_value"]),
            "I_M_normalized_surrogate_mean":
                float(m_norm["surrogate_mean"]),
            "I_M_normalized_p_greater":
                float(m_norm["empirical_p_greater"]),

            "I_Pi_sign_bias_reference":
                float(pi_sign["dns_value"]),
            "I_Pi_sign_bias_surrogate_mean":
                float(pi_sign["surrogate_mean"]),
            "I_Pi_sign_bias_p_greater":
                float(pi_sign["empirical_p_greater"]),

            "I_M_sign_bias_reference":
                float(m_sign["dns_value"]),
            "I_M_sign_bias_surrogate_mean":
                float(m_sign["surrogate_mean"]),
            "I_M_sign_bias_p_greater":
                float(m_sign["empirical_p_greater"]),

            "A_pop_reference":
                float(ref["A_pop"]),
            "A_pop_surrogate_mean":
                pop_stats["surrogate_mean"],
            "A_pop_surrogate_std":
                pop_stats["surrogate_std"],
            "A_pop_n_ge_reference":
                pop_stats[
                    "n_surrogate_ge_reference"
                ],
            "A_pop_p_greater":
                pop_stats[
                    "empirical_p_greater"
                ],

            "A_amp_reference":
                float(ref["A_amp"]),
            "A_amp_surrogate_mean":
                amp_stats["surrogate_mean"],
            "A_amp_surrogate_std":
                amp_stats["surrogate_std"],
            "A_amp_n_ge_reference":
                amp_stats[
                    "n_surrogate_ge_reference"
                ],
            "A_amp_p_greater":
                amp_stats[
                    "empirical_p_greater"
                ],
        })

    return pd.DataFrame(rows)


def main() -> None:

    for path in (
        MECH_INST,
        MECH_TIME,
        POP_SUR,
        POP_REF,
    ):
        require_file(path)

    mech_inst = pd.read_csv(MECH_INST)
    mech_time = pd.read_csv(MECH_TIME)
    pop_sur = pd.read_csv(POP_SUR)
    pop_ref = pd.read_csv(POP_REF)

    instantaneous = build_instantaneous(
        mech_inst,
        pop_sur,
        pop_ref,
    )

    finite_time = build_finite_time(
        mech_time,
        pop_sur,
        pop_ref,
        window=10,
    )

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    inst_path = (
        OUTDIR
        / "clark_arrow_synthesis_instantaneous.csv"
    )

    time_path = (
        OUTDIR
        / "clark_arrow_synthesis_window10.csv"
    )

    audit_path = (
        OUTDIR
        / "clark_arrow_synthesis_audit.json"
    )

    instantaneous.to_csv(
        inst_path,
        index=False,
    )

    finite_time.to_csv(
        time_path,
        index=False,
    )

    audit = {
        "instantaneous_rows":
            int(len(instantaneous)),
        "finite_time_rows":
            int(len(finite_time)),
        "instantaneous_scales":
            sorted(
                instantaneous["scale"]
                .astype(int)
                .tolist()
            ),
        "finite_time_scales":
            sorted(
                finite_time["scale"]
                .astype(int)
                .tolist()
            ),
        "surrogate_count_expected": 32,
        "pass": (
            len(instantaneous) == 2
            and len(finite_time) == 2
            and set(
                instantaneous["scale"]
            ) == {4, 8}
            and set(
                finite_time["scale"]
            ) == {4, 8}
        ),
    }

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
        )
    )

    print("\nINSTANTANEOUS SYNTHESIS")
    print("=" * 120)

    print(
        instantaneous[[
            "scale",
            "corr_Pi_M_reference",
            "corr_Pi_M_surrogate_mean",
            "corr_Pi_M_p_greater",
            "A_pop_reference",
            "A_pop_surrogate_mean",
            "A_pop_p_greater",
            "A_amp_reference",
            "A_amp_surrogate_mean",
            "A_amp_p_greater",
        ]].to_string(index=False)
    )

    print("\nFINITE-TIME SYNTHESIS: N_tau=10")
    print("=" * 120)

    print(
        finite_time[[
            "scale",
            "corr_I_Pi_I_M_reference",
            "corr_I_Pi_I_M_surrogate_mean",
            "corr_I_Pi_I_M_p_greater",
            "A_pop_reference",
            "A_pop_surrogate_mean",
            "A_pop_p_greater",
            "A_amp_reference",
            "A_amp_surrogate_mean",
            "A_amp_p_greater",
        ]].to_string(index=False)
    )

    print("\nAUDIT")
    print(json.dumps(audit, indent=2))

    print("\nsaved:", inst_path)
    print("saved:", time_path)
    print("saved:", audit_path)


if __name__ == "__main__":
    main()
