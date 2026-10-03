#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

GATE_PATH = (
    SCRIPT_DIR
    / "df_phase_randomization_gate_64cube.py"
)

POP_PATH = (
    SCRIPT_DIR
    / "compute_pop_amp_isotropic.py"
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = load_module("phase_gate", GATE_PATH)
popmod = load_module("pop_amp", POP_PATH)
old = popmod.old


DEFAULT_MAIN_INPUT = Path(
    "datasets/isotropic1024coarse/"
    "velocity_64cube_100frames/frames_npz"
)

DEFAULT_OUTDIR = Path(
    "phase_ensemble_v1/summaries/"
    "phase_control/main8_seed_screen"
)


def selected_indices(n_files, n_frames):
    if n_frames < 2:
        raise ValueError("n_frames must be at least 2")

    indices = np.linspace(
        0,
        n_files - 1,
        n_frames,
        dtype=int,
    )

    indices = np.unique(indices)

    if len(indices) != n_frames:
        raise RuntimeError(
            "Could not construct unique frame indices"
        )

    return indices.tolist()


def phase_qc(phase):
    nx, ny, nz = phase.shape

    ix = (-np.arange(nx)) % nx
    iy = (-np.arange(ny)) % ny
    iz = (-np.arange(nz)) % nz

    partner = np.conj(
        phase[np.ix_(ix, iy, iz)]
    )

    hermitian_error = float(
        np.max(np.abs(phase - partner))
    )

    unit_modulus_error = float(
        np.max(np.abs(np.abs(phase) - 1.0))
    )

    return {
        "phase_hermitian_error_max":
            hermitian_error,
        "phase_unit_modulus_error_max":
            unit_modulus_error,
    }


def evaluate_velocity_files(
    files,
    scale,
    dx,
    crop_factor,
    phase=None,
):
    crop_margin = crop_factor * scale + 4

    pi_chunks = []
    energy_errors = []
    rtheta_values = []

    for frame_number, path in enumerate(
        files,
        start=1,
    ):
        velocity = gate.load_velocity_npz(path)

        if phase is None:
            evaluated_velocity = velocity
        else:
            evaluated_velocity = (
                gate.divergence_free_phase_randomize(
                    velocity,
                    phase=phase,
                    dx=dx,
                    rescale_energy=True,
                )
            )

            e0 = float(
                np.sum(
                    velocity * velocity,
                    dtype=np.float64,
                )
            )

            e1 = float(
                np.sum(
                    evaluated_velocity
                    * evaluated_velocity,
                    dtype=np.float64,
                )
            )

            energy_errors.append(
                abs(e1 - e0) / max(abs(e0), 1e-30)
            )

        pi, _, rtheta = (
            old.compute_flux_and_ksgs(
                evaluated_velocity,
                scale,
                dx,
                crop_margin,
            )
        )

        if not np.isfinite(pi).all():
            raise RuntimeError(
                "Non-finite Pi in {}".format(path)
            )

        pi_chunks.append(
            np.asarray(
                pi,
                dtype=np.float32,
            ).ravel()
        )

        rtheta_values.append(float(rtheta))

    pi_all = np.concatenate(pi_chunks)

    stats = popmod.pop_amp_statistics(pi_all)

    stats.update({
        "n_frames": int(len(files)),
        "rtheta_mean": float(
            np.mean(rtheta_values)
        ),
        "rtheta_max": float(
            np.max(rtheta_values)
        ),
        "energy_relative_error_mean": (
            float(np.mean(energy_errors))
            if energy_errors
            else np.nan
        ),
        "energy_relative_error_max": (
            float(np.max(energy_errors))
            if energy_errors
            else np.nan
        ),
    })

    return stats



def discover_null_seeds():
    roots = [
        Path(
            "phase_ensemble_v1/runs/64_smoke"
        ),
        Path(
            "phase_ensemble_v1/runs/64_main"
        ),
    ]

    seeds = []

    for root in roots:
        if not root.is_dir():
            continue

        for run_dir in sorted(root.glob("seed_*")):
            name = run_dir.name

            if not name.startswith("seed_"):
                continue

            try:
                seed = int(
                    name.replace("seed_", "")
                )
            except ValueError:
                continue

            seeds.append(seed)

    seeds = sorted(set(seeds))

    expected = list(
        range(
            2026071301,
            2026071333,
        )
    )

    if seeds != expected:
        raise RuntimeError(
            "Expected formal main seeds "
            "2026071301--2026071332; found: "
            "{}".format(seeds)
        )

    return seeds


def build_matched_null(
    source_files,
    frame_indices,
    seeds,
    scale,
    dx,
    crop_factor,
):
    files = [
        source_files[i]
        for i in frame_indices
    ]

    first_velocity = gate.load_velocity_npz(
        source_files[0]
    )

    shape = gate.storage_to_phys(
        first_velocity
    ).shape[:3]

    rows = []

    for index, seed in enumerate(
        seeds,
        start=1,
    ):
        print(
            "[matched-null {}/{}] seed={} "
            "[deterministic replay]".format(
                index,
                len(seeds),
                seed,
            ),
            flush=True,
        )

        phase = gate.make_hermitian_phase(
            shape,
            seed=seed,
        )

        qc = phase_qc(phase)

        stats = evaluate_velocity_files(
            files=files,
            scale=scale,
            dx=dx,
            crop_factor=crop_factor,
            phase=phase,
        )

        stats.update({
            "seed": seed,
            "scale": scale,
            "reconstruction_protocol": (
                "reference_frames_plus_recorded_phase_seed"
            ),
            **qc,
        })

        rows.append(stats)

    result = pd.DataFrame(rows)

    if len(result) != 32:
        raise RuntimeError(
            "Expected 32 replayed null cases, "
            "found {}".format(len(result))
        )

    if result["seed"].nunique() != 32:
        raise RuntimeError(
            "Replayed null seeds are not unique"
        )

    return result


def null_parameters(null_df):
    result = {}

    for metric in ("A_pop", "A_amp"):
        values = pd.to_numeric(
            null_df[metric],
            errors="raise",
        ).to_numpy(dtype=float)

        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))

        if not np.isfinite(std) or std <= 0:
            raise RuntimeError(
                "Invalid null standard deviation "
                "for {}".format(metric)
            )

        result[metric] = {
            "mean": mean,
            "std": std,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return result


def add_scores(row, null):
    z_pop = (
        float(row["A_pop"])
        - null["A_pop"]["mean"]
    ) / null["A_pop"]["std"]

    z_amp = (
        float(row["A_amp"])
        - null["A_amp"]["mean"]
    ) / null["A_amp"]["std"]

    row["Z_pop"] = float(z_pop)
    row["Z_amp"] = float(z_amp)

    row["score_pop_selective"] = float(
        abs(z_pop) - abs(z_amp)
    )

    row["score_amp_selective"] = float(
        abs(z_amp) - abs(z_pop)
    )

    row["pop_selective_3sigma"] = bool(
        abs(z_pop) >= 3.0
        and abs(z_amp) <= 1.0
    )

    row["amp_selective_3sigma"] = bool(
        abs(z_amp) >= 3.0
        and abs(z_pop) <= 1.0
    )

    return row


def make_selected_table(
    candidate_df,
    top_k,
):
    selected_rows = []

    for target, score_column in (
        (
            "population_selective",
            "score_pop_selective",
        ),
        (
            "amplitude_selective",
            "score_amp_selective",
        ),
    ):
        ranked = (
            candidate_df
            .sort_values(
                score_column,
                ascending=False,
            )
            .head(top_k)
            .reset_index(drop=True)
        )

        for rank, (_, row) in enumerate(
            ranked.iterrows(),
            start=1,
        ):
            record = row.to_dict()
            record["selected_for"] = target
            record["selection_rank"] = rank
            record["selection_score"] = float(
                row[score_column]
            )
            selected_rows.append(record)

    selected = pd.DataFrame(selected_rows)

    return selected.sort_values(
        [
            "selected_for",
            "selection_rank",
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Screen fixed-spectrum Hermitian phase "
            "candidates for selective A_pop/A_amp "
            "separation on eight main-block frames."
        )
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_MAIN_INPUT,
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
    )

    parser.add_argument(
        "--scale",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--n_screen_frames",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--n_candidates",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--seed_start",
        type=int,
        default=2026073001,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--dx",
        type=float,
        default=gate.DX_DEFAULT,
    )

    parser.add_argument(
        "--crop_factor",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_files = sorted(
        args.input_dir.glob("*.npz")
    )

    if len(source_files) != 100:
        raise RuntimeError(
            "Expected 100 source frames, found "
            "{} in {}".format(
                len(source_files),
                args.input_dir,
            )
        )

    frame_indices = selected_indices(
        len(source_files),
        args.n_screen_frames,
    )

    screen_files = [
        source_files[i]
        for i in frame_indices
    ]

    null_path = (
        args.outdir
        / "matched_null_main8.csv"
    )

    reference_path = (
        args.outdir
        / "reference_main8.csv"
    )

    candidates_path = (
        args.outdir
        / "phase_control_candidates_all.csv"
    )

    selected_path = (
        args.outdir
        / "phase_control_candidates_selected.csv"
    )

    audit_path = (
        args.outdir
        / "phase_control_screen_audit.json"
    )

    print("=" * 88)
    print("PHASE-CONTROL MAIN8 SCREEN")
    print("=" * 88)
    print("input:", args.input_dir)
    print("scale:", args.scale)
    print("frame indices:", frame_indices)
    print(
        "frame files:",
        [p.name for p in screen_files],
    )
    print("candidate count:", args.n_candidates)
    print("seed start:", args.seed_start)

    if null_path.exists() and not args.force:
        null_df = pd.read_csv(null_path)

        if len(null_df) != 32:
            raise RuntimeError(
                "Cached matched null has {} rows, "
                "expected 32".format(len(null_df))
            )

        print(
            "\nReusing matched null:",
            null_path,
        )
    else:
        null_seeds = discover_null_seeds()

        null_df = build_matched_null(
            source_files=source_files,
            frame_indices=frame_indices,
            seeds=null_seeds,
            scale=args.scale,
            dx=args.dx,
            crop_factor=args.crop_factor,
        )

        null_df.to_csv(
            null_path,
            index=False,
        )

        print(
            "\nsaved matched null:",
            null_path,
        )

    null = null_parameters(null_df)

    print("\nMATCHED NULL PARAMETERS")
    print(json.dumps(null, indent=2))

    if reference_path.exists() and not args.force:
        reference_df = pd.read_csv(
            reference_path
        )
    else:
        reference_stats = (
            evaluate_velocity_files(
                files=screen_files,
                scale=args.scale,
                dx=args.dx,
                crop_factor=args.crop_factor,
                phase=None,
            )
        )

        reference_stats.update({
            "case": "natural_reference",
            "scale": args.scale,
        })

        reference_stats = add_scores(
            reference_stats,
            null,
        )

        reference_df = pd.DataFrame(
            [reference_stats]
        )

        reference_df.to_csv(
            reference_path,
            index=False,
        )

    print("\nMAIN8 NATURAL REFERENCE")
    print(
        reference_df[[
            "A_pop",
            "A_amp",
            "Z_pop",
            "Z_amp",
            "normalized_mean_lhs",
        ]].to_string(index=False)
    )

    if candidates_path.exists() and not args.force:
        candidate_df = pd.read_csv(
            candidates_path
        )
        completed = set(
            pd.to_numeric(
                candidate_df["phase_seed"],
                errors="raise",
            ).astype(int)
        )
    else:
        candidate_df = pd.DataFrame()
        completed = set()

    shape = gate.storage_to_phys(
        gate.load_velocity_npz(
            source_files[0]
        )
    ).shape[:3]

    candidate_seeds = [
        args.seed_start + i
        for i in range(args.n_candidates)
    ]

    rows = (
        candidate_df.to_dict("records")
        if not candidate_df.empty
        else []
    )

    for candidate_index, seed in enumerate(
        candidate_seeds,
        start=1,
    ):
        if seed in completed:
            print(
                "[candidate {}/{}] seed={} "
                "[cached]".format(
                    candidate_index,
                    args.n_candidates,
                    seed,
                ),
                flush=True,
            )
            continue

        print(
            "[candidate {}/{}] seed={}".format(
                candidate_index,
                args.n_candidates,
                seed,
            ),
            flush=True,
        )

        phase = gate.make_hermitian_phase(
            shape,
            seed=seed,
        )

        qc = phase_qc(phase)

        stats = evaluate_velocity_files(
            files=screen_files,
            scale=args.scale,
            dx=args.dx,
            crop_factor=args.crop_factor,
            phase=phase,
        )

        row = {
            "phase_seed": seed,
            "scale": args.scale,
            "n_screen_frames":
                args.n_screen_frames,
            **stats,
            **qc,
        }

        row = add_scores(
            row,
            null,
        )

        rows.append(row)

        pd.DataFrame(rows).to_csv(
            candidates_path,
            index=False,
        )

    candidate_df = pd.DataFrame(rows)

    candidate_df = candidate_df.sort_values(
        "phase_seed"
    ).reset_index(drop=True)

    candidate_df.to_csv(
        candidates_path,
        index=False,
    )

    if len(candidate_df) != args.n_candidates:
        raise RuntimeError(
            "Expected {} candidates, found {}".format(
                args.n_candidates,
                len(candidate_df),
            )
        )

    selected_df = make_selected_table(
        candidate_df,
        top_k=args.top_k,
    )

    selected_df.to_csv(
        selected_path,
        index=False,
    )

    audit = {
        "input_dir": str(args.input_dir),
        "scale": int(args.scale),
        "n_source_frames":
            int(len(source_files)),
        "screen_frame_indices":
            frame_indices,
        "screen_frame_names":
            [p.name for p in screen_files],
        "n_matched_null":
            int(len(null_df)),
        "n_candidates":
            int(len(candidate_df)),
        "seed_start":
            int(args.seed_start),
        "seed_end":
            int(
                args.seed_start
                + args.n_candidates
                - 1
            ),
        "matched_null_parameters":
            null,
        "n_pop_selective_3sigma":
            int(
                candidate_df[
                    "pop_selective_3sigma"
                ].astype(bool).sum()
            ),
        "n_amp_selective_3sigma":
            int(
                candidate_df[
                    "amp_selective_3sigma"
                ].astype(bool).sum()
            ),
        "max_phase_hermitian_error":
            float(
                candidate_df[
                    "phase_hermitian_error_max"
                ].max()
            ),
        "max_phase_unit_modulus_error":
            float(
                candidate_df[
                    "phase_unit_modulus_error_max"
                ].max()
            ),
        "max_energy_relative_error":
            float(
                candidate_df[
                    "energy_relative_error_max"
                ].max()
            ),
        "pass": bool(
            len(null_df) == 32
            and len(candidate_df)
                == args.n_candidates
            and np.isfinite(
                candidate_df[[
                    "A_pop",
                    "A_amp",
                    "Z_pop",
                    "Z_amp",
                ]].to_numpy(dtype=float)
            ).all()
        ),
    }

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
        )
    )

    print("\n" + "=" * 88)
    print("TOP POPULATION-SELECTIVE CANDIDATES")
    print("=" * 88)

    print(
        candidate_df
        .sort_values(
            "score_pop_selective",
            ascending=False,
        )
        .head(args.top_k)[[
            "phase_seed",
            "A_pop",
            "A_amp",
            "Z_pop",
            "Z_amp",
            "score_pop_selective",
            "pop_selective_3sigma",
        ]]
        .to_string(index=False)
    )

    print("\n" + "=" * 88)
    print("TOP AMPLITUDE-SELECTIVE CANDIDATES")
    print("=" * 88)

    print(
        candidate_df
        .sort_values(
            "score_amp_selective",
            ascending=False,
        )
        .head(args.top_k)[[
            "phase_seed",
            "A_pop",
            "A_amp",
            "Z_pop",
            "Z_amp",
            "score_amp_selective",
            "amp_selective_3sigma",
        ]]
        .to_string(index=False)
    )

    print("\nAUDIT")
    print(json.dumps(audit, indent=2))

    print("\nsaved:", null_path)
    print("saved:", reference_path)
    print("saved:", candidates_path)
    print("saved:", selected_path)
    print("saved:", audit_path)


if __name__ == "__main__":
    main()
