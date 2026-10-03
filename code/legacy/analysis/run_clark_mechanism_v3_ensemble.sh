#!/usr/bin/env bash
set -euo pipefail

ROOT="legacy_workspace"

DNS_DIR="$ROOT/datasets/isotropic1024coarse/velocity_64cube_100frames/frames_npz"
TMP_ROOT="$ROOT/phase_ensemble_v1/tmp_mechanism_rebuild"
LOG_ROOT="$ROOT/phase_ensemble_v1/logs/mechanism_v3"

mkdir -p "$TMP_ROOT" "$LOG_ROOT"

# The original phase ensemble used the default user-site NumPy environment.
unset PYTHONNOUSERSITE

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$ROOT"

echo "============================================================"
echo "MECHANISM ENSEMBLE PREFLIGHT"
echo "============================================================"

PREFLIGHT="$TMP_ROOT/preflight_phase_hash_2026071305.json"

python scripts/write_phase_hash_64cube.py \
    --input_dir "$DNS_DIR" \
    --seed 2026071305 \
    --output "$PREFLIGHT"

python - \
    "$ROOT/phase_ensemble_v1/runs/64_main/seed_2026071305/phase_hash.json" \
    "$PREFLIGHT" <<'PY'
import json
import sys
from pathlib import Path

official = json.loads(
    Path(sys.argv[1]).read_text()
)

current = json.loads(
    Path(sys.argv[2]).read_text()
)

print(
    "official phase hash =",
    official["phase_sha256"],
)
print(
    "current phase hash  =",
    current["phase_sha256"],
)

if (
    official["phase_sha256"]
    != current["phase_sha256"]
):
    raise SystemExit(
        "FAIL: current Python environment does not "
        "reproduce the official phase ensemble."
    )

print("PREFLIGHT PHASE HASH: PASS")
PY

for SEED in $(seq 2026071301 2026071332); do
    if [[ "$SEED" -le 2026071304 ]]; then
        BASE="$ROOT/phase_ensemble_v1/runs/64_smoke"
    else
        BASE="$ROOT/phase_ensemble_v1/runs/64_main"
    fi

    RUN_DIR="$BASE/seed_${SEED}"
    REFERENCE_DIR="$RUN_DIR/finite_time_v3"
    OUT_DIR="$RUN_DIR/mechanism_v3"
    OFFICIAL_FRAME_DIR="$RUN_DIR/frames_npz"
    META="$OUT_DIR/mechanism_run_metadata_v3.json"
    LOG="$LOG_ROOT/seed_${SEED}.log"

    echo
    echo "============================================================"
    echo "seed = $SEED"
    echo "run  = $RUN_DIR"
    echo "============================================================"

    if [[ -f "$META" ]]; then
        if python - "$META" "$SEED" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(
    Path(sys.argv[1]).read_text()
)

seed = int(sys.argv[2])

ok = (
    data.get("source_seed") == seed
    and data.get("qc_pass") is True
    and float(
        data.get(
            "max_v3_qc_abs_difference",
            float("inf"),
        )
    ) <= 1e-12
)

raise SystemExit(0 if ok else 1)
PY
        then
            echo "[skip] valid mechanism result already exists"
            continue
        fi
    fi

    if [[ ! -d "$REFERENCE_DIR" ]]; then
        echo "Missing reference v3 directory:"
        echo "$REFERENCE_DIR"
        exit 1
    fi

    N_EXISTING="$(
        find "$OFFICIAL_FRAME_DIR" \
            -maxdepth 1 \
            -type f \
            -name 'velocity_t*.npz' \
            | wc -l \
            | tr -d ' '
    )"

    GENERATED_TEMP=0

    if [[ "$N_EXISTING" -eq 100 ]]; then
        FRAME_DIR="$OFFICIAL_FRAME_DIR"
        echo "Using retained official frames."
    else
        TEMP_RUN="$TMP_ROOT/seed_${SEED}"
        FRAME_DIR="$TEMP_RUN/frames_npz"

        rm -rf "$TEMP_RUN"
        mkdir -p "$TEMP_RUN"

        echo "Rebuilding temporary surrogate frames."

        python scripts/write_phase_hash_64cube.py \
            --input_dir "$DNS_DIR" \
            --seed "$SEED" \
            --output "$TEMP_RUN/phase_hash.json"

        python - \
            "$RUN_DIR/phase_hash.json" \
            "$TEMP_RUN/phase_hash.json" <<'PY'
import json
import sys
from pathlib import Path

official = json.loads(
    Path(sys.argv[1]).read_text()
)

replay = json.loads(
    Path(sys.argv[2]).read_text()
)

if (
    official["phase_sha256"]
    != replay["phase_sha256"]
):
    raise SystemExit(
        "FAIL: replay phase hash differs "
        "from official phase hash."
    )

print("phase hash: PASS")
PY

        python scripts/make_df_phase_randomized_dataset_64cube.py \
            --input_dir "$DNS_DIR" \
            --outdir "$FRAME_DIR" \
            --max_frames 100 \
            --seed "$SEED"

        N_REBUILT="$(
            find "$FRAME_DIR" \
                -maxdepth 1 \
                -type f \
                -name 'velocity_t*.npz' \
                | wc -l \
                | tr -d ' '
        )"

        if [[ "$N_REBUILT" -ne 100 ]]; then
            echo "Expected 100 rebuilt frames; found $N_REBUILT"
            exit 1
        fi

        GENERATED_TEMP=1
    fi

    rm -rf "$OUT_DIR"

    python scripts/clark_mechanism_v3_one_seed.py \
        --input_dir "$FRAME_DIR" \
        --reference_v3_dir "$REFERENCE_DIR" \
        --outdir "$OUT_DIR" \
        --seed "$SEED" \
        --label "surrogate_${SEED}" \
        --max_frames 100 \
        --scales 4 8 \
        --windows 5 10 20 \
        --dt 0.002 \
        --crop_factor 2 \
        2>&1 | tee "$LOG"

    python - "$META" "$SEED" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
seed = int(sys.argv[2])

if not path.exists():
    raise SystemExit(
        f"Missing mechanism metadata: {path}"
    )

data = json.loads(path.read_text())

if data.get("source_seed") != seed:
    raise SystemExit(
        "FAIL: source seed mismatch."
    )

if data.get("qc_pass") is not True:
    raise SystemExit(
        "FAIL: mechanism QC did not pass."
    )

difference = float(
    data["max_v3_qc_abs_difference"]
)

if difference > 1e-12:
    raise SystemExit(
        f"FAIL: QC difference={difference}"
    )

print(
    "mechanism QC: PASS; "
    f"max difference={difference:.3e}"
)
PY

    if [[ "$GENERATED_TEMP" -eq 1 ]]; then
        rm -rf "$TMP_ROOT/seed_${SEED}"
        echo "Temporary surrogate frames removed."
    fi

    echo "[complete] seed $SEED"
done

echo
echo "============================================================"
echo "ALL 32 MECHANISM SEEDS COMPLETE"
echo "============================================================"
