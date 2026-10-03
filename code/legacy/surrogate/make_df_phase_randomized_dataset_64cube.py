import argparse
import json
from pathlib import Path

import numpy as np

# Reuse the exact functions from the validated gate script.
import importlib.util

GATE_SCRIPT = Path("scripts/df_phase_randomization_gate_64cube.py")
spec = importlib.util.spec_from_file_location("df_gate", GATE_SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--max_frames", type=int, default=100)
    parser.add_argument("--dx", type=float, default=gate.DX_DEFAULT)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.npz"))[:args.max_frames]
    if not files:
        raise FileNotFoundError(f"No npz files found in {input_dir}")

    # Build exactly the same Hermitian phase used in the gate test.
    U0 = gate.load_velocity_npz(files[0])
    U0phys = gate.storage_to_phys(U0)
    phase = gate.make_hermitian_phase(U0phys.shape[:3], seed=args.seed)

    for idx, fp in enumerate(files):
        outfp = outdir / fp.name
        if outfp.exists() and not args.overwrite:
            print(f"[skip] {outfp}")
            continue

        print(f"[{idx+1}/{len(files)}] {fp.name} -> {outfp.name}", flush=True)

        U = gate.load_velocity_npz(fp)
        Ur = gate.divergence_free_phase_randomize(
            U,
            phase=phase,
            dx=args.dx,
            rescale_energy=True,
        )

        np.savez_compressed(outfp, velocity=Ur.astype(np.float32))

    meta = {
        "source_input_dir": str(input_dir),
        "output_dir": str(outdir),
        "n_files": len(files),
        "seed": args.seed,
        "dx": args.dx,
        "phase_protocol": "same Hermitian scalar Fourier phase applied to all components and all time frames",
        "projection": "D4 modified-wavenumber Helmholtz projection",
        "energy_rescale_per_mode": True,
        "note": "Generated from the validated divergence-free phase-randomization gate protocol.",
    }

    with open(outdir / "df_phase_randomized_dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print("Output directory:", outdir)


if __name__ == "__main__":
    main()
