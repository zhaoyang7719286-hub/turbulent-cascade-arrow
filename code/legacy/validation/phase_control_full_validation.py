#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

SCREEN_SCRIPT = (
    SCRIPT_DIR
    / "phase_control_seed_screen.py"
)

spec = importlib.util.spec_from_file_location(
    "phase_screen",
    SCREEN_SCRIPT,
)
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


ROOT = Path(
    "phase_ensemble_v1/summaries/phase_control"
)

SCREEN_DIR = (
    ROOT
    / "main8_seed_screen"
)

SELECTED_PATH = (
    SCREEN_DIR
    / "phase_control_candidates_selected.csv"
)

OUTDIR = (
    ROOT
    / "full100_validation"
)

OUTPUT_PATH = (
    OUTDIR
    / "phase_control_full100_results.csv"
)

AUDIT_PATH = (
    OUTDIR
    / "phase_control_full100_audit.json"
)


BLOCKS = {
    "main100": {
        "input_dir": Path(
            "datasets/isotropic1024coarse/"
            "velocity_64cube_100frames/"
            "frames_npz"
        ),
        "null_path": Path(
            "phase_ensemble_v1/summaries/"
            "pop_amp_decomposition/aggregates/"
            "main32/main32_pop_amp_long.csv"
        ),
        "expected_null": 32,
    },
    "rep2_100": {
        "input_dir": Path(
            "datasets/isotropic1024coarse/"
            "velocity_64cube_rep2_xyz385_t2501_2600/"
            "frames_npz"
        ),
        "null_path": Path(
            "phase_ensemble_v1/summaries/"
            "pop_amp_decomposition/aggregates/"
            "rep2_24/rep2_24_pop_amp_long.csv"
        ),
        "expected_null": 24,
    },
}


def select_candidates():
    if not SELECTED_PATH.is_file():
        raise FileNotFoundError(SELECTED_PATH)

    selected = pd.read_csv(SELECTED_PATH)

    selected["selection_rank"] = pd.to_numeric(
        selected["selection_rank"],
        errors="raise",
    ).astype(int)

    selected["phase_seed"] = pd.to_numeric(
        selected["phase_seed"],
        errors="raise",
    ).astype(int)

    selected = selected[
        selected["selection_rank"] <= 2
    ].copy()

    selected = selected[[
        "phase_seed",
        "selected_for",
        "selection_rank",
    ]]

    # A seed could theoretically enter both rankings.
    selected = (
        selected
        .sort_values(
            ["selection_rank", "selected_for"]
        )
        .drop_duplicates(
            subset=["phase_seed"],
            keep="first",
        )
        .sort_values(
            ["selected_for", "selection_rank"]
        )
        .reset_index(drop=True)
    )

    if len(selected) != 4:
        raise RuntimeError(
            "Expected four unique candidates, found "
            "{}:\n{}".format(
                len(selected),
                selected,
            )
        )

    return selected


def read_null(block_info, scale=8):
    path = block_info["null_path"]

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    window = pd.to_numeric(
        df["window_frames"],
        errors="coerce",
    ).fillna(0).astype(int)

    mask = (
        df["variable"].astype(str).eq("Pi")
        &
        pd.to_numeric(
            df["scale"],
            errors="raise",
        ).astype(int).eq(scale)
        &
        window.eq(0)
    )

    sub = df.loc[mask].copy()

    expected = int(
        block_info["expected_null"]
    )

    if len(sub) != expected:
        raise RuntimeError(
            "Expected {} null rows in {}, found {}".format(
                expected,
                path,
                len(sub),
            )
        )

    return sub


def null_parameters(df):
    result = {}

    for metric in ("A_pop", "A_amp"):
        values = pd.to_numeric(
            df[metric],
            errors="raise",
        ).to_numpy(dtype=float)

        result[metric] = {
            "mean": float(np.mean(values)),
            "std": float(
                np.std(values, ddof=1)
            ),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return result


def empirical_absolute_p(
    candidate,
    values,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    center = float(np.mean(values))

    candidate_distance = abs(
        float(candidate) - center
    )

    null_distances = np.abs(
        values - center
    )

    n_ge = int(
        np.count_nonzero(
            null_distances
            >= candidate_distance
        )
    )

    return {
        "n_null_abs_ge_candidate": n_ge,
        "empirical_p_absolute": (
            n_ge + 1
        ) / (
            len(values) + 1
        ),
    }


def add_validation_scores(
    row,
    null_df,
    null,
):
    for metric, short in (
        ("A_pop", "pop"),
        ("A_amp", "amp"),
    ):
        value = float(row[metric])

        mean = null[metric]["mean"]
        std = null[metric]["std"]

        if not np.isfinite(std) or std <= 0:
            raise RuntimeError(
                "Invalid null standard deviation "
                "for {}".format(metric)
            )

        row["Z_" + short] = float(
            (value - mean) / std
        )

        empirical = empirical_absolute_p(
            candidate=value,
            values=null_df[
                metric
            ].to_numpy(dtype=float),
        )

        row[
            short
            + "_n_null_abs_ge_candidate"
        ] = empirical[
            "n_null_abs_ge_candidate"
        ]

        row[
            short
            + "_empirical_p_absolute"
        ] = empirical[
            "empirical_p_absolute"
        ]

    row["score_pop_selective"] = float(
        abs(row["Z_pop"])
        - abs(row["Z_amp"])
    )

    row["score_amp_selective"] = float(
        abs(row["Z_amp"])
        - abs(row["Z_pop"])
    )

    row["pop_selective_strict_3sigma"] = bool(
        abs(row["Z_pop"]) >= 3.0
        and abs(row["Z_amp"]) <= 1.0
    )

    row["amp_selective_strict_3sigma"] = bool(
        abs(row["Z_amp"]) >= 3.0
        and abs(row["Z_pop"]) <= 1.0
    )

    row["pop_selective_exploratory_2sigma"] = bool(
        abs(row["Z_pop"]) >= 2.0
        and abs(row["Z_amp"]) <= 1.0
    )

    row["amp_selective_exploratory_2sigma"] = bool(
        abs(row["Z_amp"]) >= 2.0
        and abs(row["Z_pop"]) <= 1.0
    )

    return row


def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected = select_candidates()

    print("=" * 96)
    print("SELECTED FIXED PHASE CANDIDATES")
    print("=" * 96)
    print(selected.to_string(index=False))

    if OUTPUT_PATH.exists():
        results = pd.read_csv(
            OUTPUT_PATH
        )
    else:
        results = pd.DataFrame()

    completed = set()

    if not results.empty:
        for _, row in results.iterrows():
            completed.add(
                (
                    str(row["block"]),
                    int(row["phase_seed"]),
                )
            )

    rows = (
        results.to_dict("records")
        if not results.empty
        else []
    )

    null_metadata = {}

    for block, info in BLOCKS.items():
        input_dir = info["input_dir"]

        files = sorted(
            input_dir.glob("*.npz")
        )

        if len(files) != 100:
            raise RuntimeError(
                "{}: expected 100 frames, found {}"
                .format(
                    block,
                    len(files),
                )
            )

        null_df = read_null(
            info,
            scale=8,
        )

        null = null_parameters(
            null_df
        )

        null_metadata[block] = null

        first_velocity = (
            screen.gate.load_velocity_npz(
                files[0]
            )
        )

        shape = (
            screen.gate.storage_to_phys(
                first_velocity
            ).shape[:3]
        )

        print("\n" + "=" * 96)
        print("BLOCK:", block)
        print("input:", input_dir)
        print(
            "null parameters:",
            json.dumps(null, indent=2),
        )
        print("=" * 96)

        for _, selected_row in (
            selected.iterrows()
        ):
            seed = int(
                selected_row["phase_seed"]
            )

            key = (block, seed)

            if key in completed:
                print(
                    "[cached] block={} seed={}"
                    .format(block, seed),
                    flush=True,
                )
                continue

            print(
                "[evaluate] block={} seed={} "
                "frames=100".format(
                    block,
                    seed,
                ),
                flush=True,
            )

            phase = (
                screen.gate.make_hermitian_phase(
                    shape,
                    seed=seed,
                )
            )

            qc = screen.phase_qc(
                phase
            )

            stats = (
                screen.evaluate_velocity_files(
                    files=files,
                    scale=8,
                    dx=screen.gate.DX_DEFAULT,
                    crop_factor=2,
                    phase=phase,
                )
            )

            row = {
                "block": block,
                "phase_seed": seed,
                "selected_for": str(
                    selected_row[
                        "selected_for"
                    ]
                ),
                "selection_rank": int(
                    selected_row[
                        "selection_rank"
                    ]
                ),
                "scale": 8,
                "n_frames": 100,
                **stats,
                **qc,
            }

            row = add_validation_scores(
                row=row,
                null_df=null_df,
                null=null,
            )

            rows.append(row)

            pd.DataFrame(rows).to_csv(
                OUTPUT_PATH,
                index=False,
            )

    results = pd.DataFrame(rows)

    results = results.sort_values(
        [
            "block",
            "selected_for",
            "selection_rank",
        ]
    ).reset_index(drop=True)

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    expected_rows = (
        len(BLOCKS)
        * len(selected)
    )

    audit = {
        "selected_candidates":
            selected.to_dict("records"),
        "blocks":
            list(BLOCKS.keys()),
        "null_parameters":
            null_metadata,
        "expected_rows":
            int(expected_rows),
        "actual_rows":
            int(len(results)),
        "strict_pop_success_count":
            int(
                results[
                    "pop_selective_strict_3sigma"
                ].astype(bool).sum()
            ),
        "strict_amp_success_count":
            int(
                results[
                    "amp_selective_strict_3sigma"
                ].astype(bool).sum()
            ),
        "exploratory_pop_success_count":
            int(
                results[
                    "pop_selective_exploratory_2sigma"
                ].astype(bool).sum()
            ),
        "exploratory_amp_success_count":
            int(
                results[
                    "amp_selective_exploratory_2sigma"
                ].astype(bool).sum()
            ),
        "max_phase_hermitian_error":
            float(
                results[
                    "phase_hermitian_error_max"
                ].max()
            ),
        "max_phase_unit_modulus_error":
            float(
                results[
                    "phase_unit_modulus_error_max"
                ].max()
            ),
        "max_energy_relative_error":
            float(
                results[
                    "energy_relative_error_max"
                ].max()
            ),
        "pass": bool(
            len(results) == expected_rows
            and results[
                ["A_pop", "A_amp",
                 "Z_pop", "Z_amp"]
            ].apply(
                pd.to_numeric,
                errors="coerce",
            ).notna().all().all()
        ),
    }

    AUDIT_PATH.write_text(
        json.dumps(
            audit,
            indent=2,
        )
    )

    print("\n" + "=" * 120)
    print("FULL-100 VALIDATION RESULTS")
    print("=" * 120)

    print(
        results[[
            "block",
            "phase_seed",
            "selected_for",
            "A_pop",
            "A_amp",
            "Z_pop",
            "Z_amp",
            "score_pop_selective",
            "score_amp_selective",
            "pop_selective_strict_3sigma",
            "amp_selective_strict_3sigma",
            "pop_selective_exploratory_2sigma",
            "amp_selective_exploratory_2sigma",
        ]].to_string(index=False)
    )

    print("\nAUDIT")
    print(
        json.dumps(
            audit,
            indent=2,
        )
    )

    print("\nsaved:", OUTPUT_PATH)
    print("saved:", AUDIT_PATH)


if __name__ == "__main__":
    main()
