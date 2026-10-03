from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def finite_number(value) -> bool:
    return bool(np.isfinite(float(value)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--expected_frames", type=int, default=100)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    frame_dir = run_dir / "frames_npz"
    gate_dir = run_dir / "gate"
    finite_dir = run_dir / "finite_time_v3"

    errors: list[str] = []
    checks: dict = {}

    frame_files = sorted(frame_dir.glob("*.npz"))
    checks["n_frame_files"] = len(frame_files)

    if len(frame_files) != args.expected_frames:
        errors.append(
            f"Expected {args.expected_frames} frames, "
            f"found {len(frame_files)}"
        )

    phase_hash_path = run_dir / "phase_hash.json"

    if not phase_hash_path.exists():
        errors.append("Missing phase_hash.json")
    else:
        phase_hash = json.loads(
            phase_hash_path.read_text(encoding="utf-8")
        )

        checks["phase_sha256"] = phase_hash.get("phase_sha256")
        checks["hermitian_error_max"] = phase_hash.get(
            "hermitian_error_max"
        )
        checks["unit_modulus_error_max"] = phase_hash.get(
            "unit_modulus_error_max"
        )
        checks["inverse_imaginary_ratio"] = phase_hash.get(
            "inverse_imaginary_ratio"
        )

        if phase_hash["hermitian_error_max"] > 1e-10:
            errors.append("Hermitian symmetry check failed")

        if phase_hash["unit_modulus_error_max"] > 1e-12:
            errors.append("Unit-modulus phase check failed")

        if phase_hash["inverse_imaginary_ratio"] > 1e-12:
            errors.append("Phase inverse-transform reality check failed")

    gate_summary_path = (
        gate_dir / "df_phase_randomization_gate_summary.csv"
    )

    if not gate_summary_path.exists():
        errors.append("Missing gate summary")
    else:
        gate = pd.read_csv(gate_summary_path)

        surrogate = gate[
            gate["case"] == "phase_randomized_D4_projected"
        ].copy()

        if set(surrogate["scale"]) != {4, 8}:
            errors.append("Gate summary lacks scale 4 or 8")

        checks["surrogate_rtheta_max"] = float(
            surrogate["rtheta_max"].max()
        )

        checks["surrogate_volumetric_ratio_max"] = float(
            surrogate["rms_Pi_vol_over_Pi_max"].max()
        )

        if checks["surrogate_rtheta_max"] > 1e-10:
            errors.append("Surrogate D4-divergence QC failed")

        if checks["surrogate_volumetric_ratio_max"] > 1e-10:
            errors.append("Surrogate volumetric-flux QC failed")

    energy_path = (
        gate_dir / "df_phase_randomization_energy_summary.csv"
    )

    if not energy_path.exists():
        errors.append("Missing energy summary")
    else:
        energy = pd.read_csv(energy_path)

        spectrum_error = float(
            energy["spectrum_rel_l2_no_mean"].max(skipna=True)
        )

        energy_ratio_error = float(
            np.nanmax(
                np.abs(
                    energy["total_energy_ratio"].to_numpy(
                        dtype=float
                    ) - 1.0
                )
            )
        )

        checks["spectrum_error_max"] = spectrum_error
        checks["energy_ratio_error_max"] = energy_ratio_error

        if spectrum_error > 1e-12:
            errors.append("Spectrum-preservation QC failed")

        if energy_ratio_error > 1e-12:
            errors.append("Total-energy QC failed")

    instantaneous_path = (
        finite_dir / "instantaneous_arrow_metrics_v3.csv"
    )

    if not instantaneous_path.exists():
        errors.append("Missing instantaneous metrics")
    else:
        instantaneous = pd.read_csv(instantaneous_path)

        if set(instantaneous["scale"]) != {4, 8}:
            errors.append("Instantaneous metrics lack scale 4 or 8")

        for column in [
            "sign_bias_Pi",
            "mean_Pi",
            "std_Pi",
            "normalized_mean_Pi",
        ]:
            if not np.isfinite(
                instantaneous[column].to_numpy(dtype=float)
            ).all():
                errors.append(
                    f"Nonfinite instantaneous metric: {column}"
                )

    finite_path = (
        finite_dir / "finite_time_arrow_summary_v3.csv"
    )

    if not finite_path.exists():
        errors.append("Missing finite-time metrics")
    else:
        finite = pd.read_csv(finite_path)

        primary = finite[
            finite["mode"] == "raw_zero"
        ].copy()

        expected_pairs = {
            (4, 5), (4, 10), (4, 20),
            (8, 5), (8, 10), (8, 20),
        }

        actual_pairs = set(
            zip(
                primary["scale"].astype(int),
                primary["window_frames"].astype(int),
            )
        )

        if actual_pairs != expected_pairs:
            errors.append(
                "Finite-time metrics do not contain all "
                "scale-window combinations"
            )

        for column in [
            "sign_bias_I",
            "normalized_mean_I",
        ]:
            if not np.isfinite(
                primary[column].to_numpy(dtype=float)
            ).all():
                errors.append(
                    f"Nonfinite finite-time metric: {column}"
                )

    result = {
        "run_dir": str(run_dir),
        "expected_frames": args.expected_frames,
        "qc_pass": len(errors) == 0,
        "checks": checks,
        "errors": errors,
    }

    output = run_dir / "qc_status.json"
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
