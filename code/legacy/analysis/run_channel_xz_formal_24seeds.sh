#!/usr/bin/env bash
set -euo pipefail

ROOT="legacy_workspace"
INPUT="$ROOT/datasets/channel/heldout_x1153_1216_y225_288_z961_1024_t3001_3100/uniform_y/frames_npz"
PROFILE="$ROOT/datasets/channel/heldout_x1153_1216_y225_288_z961_1024_t3001_3100/uniform_y/mean_profile_y_component.npy"
GEOMETRY="$ROOT/datasets/channel/heldout_x1153_1216_y225_288_z961_1024_t3001_3100/uniform_y/metadata/uniform_y_summary.json"
BASE="$ROOT/phase_ensemble_v1/runs/channel_xz_heldout_t3001_3100"
LOG_DIR="$ROOT/phase_ensemble_v1/logs/channel_xz_heldout_t3001_3100"

SEED_START=2026072201
SEED_END=2026072224

mkdir -p "$BASE" "$LOG_DIR"
cd "$ROOT"

unset PYTHONNOUSERSITE

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

N_FRAMES="$(
  find "$INPUT" -maxdepth 1 -type f -name '*.npz' | wc -l
)"
if [[ "$N_FRAMES" -ne 100 ]]; then
  echo "Expected 100 prepared channel frames; found $N_FRAMES"
  exit 1
fi

REFERENCE_DIR="$BASE/reference"
if [[ ! -f "$REFERENCE_DIR/DONE" ]]; then
  rm -rf "$REFERENCE_DIR"
  python scripts/run_channel_xz_case_streaming.py \
    --input_dir "$INPUT" \
    --profile "$PROFILE" \
    --geometry "$GEOMETRY" \
    --outdir "$REFERENCE_DIR" \
    --label "channel_xz_heldout_reference" \
    --expected_frames 100 \
    --dt 0.0065 \
    --windows 5 10 20 \
    2>&1 | tee "$LOG_DIR/reference.log"
else
  echo "[skip] reference already complete"
fi

for SEED in $(seq "$SEED_START" "$SEED_END"); do
  RUN_DIR="$BASE/seed_${SEED}"

  if [[ -f "$RUN_DIR/DONE" ]]; then
    echo "[skip] seed $SEED already complete"
    continue
  fi

  rm -rf "$RUN_DIR"

  echo "======================================================================"
  echo "CHANNEL X-Z HELD-OUT SEED $SEED"
  echo "======================================================================"

  python scripts/run_channel_xz_case_streaming.py \
    --input_dir "$INPUT" \
    --profile "$PROFILE" \
    --geometry "$GEOMETRY" \
    --outdir "$RUN_DIR" \
    --label "channel_xz_heldout_surrogate_${SEED}" \
    --seed "$SEED" \
    --expected_frames 100 \
    --dt 0.0065 \
    --windows 5 10 20 \
    2>&1 | tee "$LOG_DIR/seed_${SEED}.log"
done

touch "$BASE/ENSEMBLE_24_DONE"

echo "======================================================================"
echo "CHANNEL X-Z HELD-OUT 24-SEED ENSEMBLE COMPLETE"
echo "======================================================================"
echo "base = $BASE"
