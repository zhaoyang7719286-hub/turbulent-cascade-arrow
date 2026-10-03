#!/usr/bin/env bash
set -euo pipefail

ROOT="legacy_workspace"
INPUT_DNS="$ROOT/datasets/isotropic1024coarse/velocity_64cube_rep2_xyz385_t2501_2600/frames_npz"
BASE="$ROOT/phase_ensemble_v1/runs/64_rep2_xyz385_t2501_2600"
REFERENCE_ROOT="$BASE/reference_star32"
REFERENCE_FRAMES="$REFERENCE_ROOT/frames_npz"
LOG_DIR="$ROOT/phase_ensemble_v1/logs/rep2_xyz385_t2501_2600"
SEED_START=2026072001
SEED_END=2026072024

mkdir -p "$BASE" "$REFERENCE_ROOT" "$LOG_DIR"

unset PYTHONNOUSERSITE

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$ROOT"

N_DNS="$(
  find "$INPUT_DNS" -maxdepth 1 -type f -name '*.npz' | wc -l
)"
if [[ "$N_DNS" -ne 100 ]]; then
  echo "Expected 100 DNS frames; found $N_DNS"
  exit 1
fi

echo "======================================================================"
echo "INDEPENDENT REPLICATION: DNS + REFERENCE"
echo "======================================================================"

python scripts/make_projected_reference_dataset.py \
  --input_dir "$INPUT_DNS" \
  --outdir "$REFERENCE_FRAMES" \
  --max_frames 100 \
  2>&1 | tee "$LOG_DIR/reference_generation.log"

python scripts/eulerian_finite_time_arrow_axisfixed_64cube_v3.py \
  --input_dir "$INPUT_DNS" \
  --outdir "$BASE/dns_raw/finite_time_v3" \
  --label "dns_rep2_raw" \
  --max_frames 100 \
  --scales 4 8 \
  --windows 5 10 20 \
  --dt 0.002 \
  --crop_factor 2 \
  --nbins 80 \
  --zmax 4.0 \
  --min_count_each 50 \
  2>&1 | tee "$LOG_DIR/dns_v3.log"

python scripts/eulerian_finite_time_arrow_axisfixed_64cube_v3.py \
  --input_dir "$REFERENCE_FRAMES" \
  --outdir "$REFERENCE_ROOT/finite_time_v3" \
  --label "reference_star32_rep2" \
  --max_frames 100 \
  --scales 4 8 \
  --windows 5 10 20 \
  --dt 0.002 \
  --crop_factor 2 \
  --nbins 80 \
  --zmax 4.0 \
  --min_count_each 50 \
  2>&1 | tee "$LOG_DIR/reference_v3.log"

echo "======================================================================"
echo "INDEPENDENT REPLICATION: 24 PHASE SURROGATES"
echo "======================================================================"

for SEED in $(seq "$SEED_START" "$SEED_END"); do
  RUN_DIR="$BASE/seed_${SEED}"
  mkdir -p "$RUN_DIR"

  if [[ -f "$RUN_DIR/DONE" ]]; then
    echo "[skip] seed $SEED already complete"
    continue
  fi

  rm -f "$RUN_DIR/FAILED"
  trap 'touch "'"$RUN_DIR"'/FAILED"' ERR

  echo "------------------------------------------------------------------"
  echo "seed = $SEED"
  echo "------------------------------------------------------------------"

  python scripts/write_phase_hash_64cube.py \
    --input_dir "$INPUT_DNS" \
    --seed "$SEED" \
    --output "$RUN_DIR/phase_hash.json" \
    2>&1 | tee "$LOG_DIR/seed_${SEED}_phase_hash.log"

  python scripts/make_df_phase_randomized_dataset_64cube.py \
    --input_dir "$INPUT_DNS" \
    --outdir "$RUN_DIR/frames_npz" \
    --max_frames 100 \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/seed_${SEED}_generation.log"

  N_FRAMES="$(
    find "$RUN_DIR/frames_npz" -maxdepth 1 -type f -name '*.npz' | wc -l
  )"
  if [[ "$N_FRAMES" -ne 100 ]]; then
    echo "Expected 100 generated frames; found $N_FRAMES"
    exit 1
  fi

  python scripts/df_phase_randomization_gate_64cube.py \
    --input_dir "$INPUT_DNS" \
    --outdir "$RUN_DIR/gate" \
    --max_frames 100 \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/seed_${SEED}_gate.log"

  python scripts/eulerian_finite_time_arrow_axisfixed_64cube_v3.py \
    --input_dir "$RUN_DIR/frames_npz" \
    --outdir "$RUN_DIR/finite_time_v3" \
    --label "rep2_surrogate_${SEED}" \
    --max_frames 100 \
    --scales 4 8 \
    --windows 5 10 20 \
    --dt 0.002 \
    --crop_factor 2 \
    --nbins 80 \
    --zmax 4.0 \
    --min_count_each 50 \
    2>&1 | tee "$LOG_DIR/seed_${SEED}_v3.log"

  python scripts/validate_phase_seed_64cube.py \
    --run_dir "$RUN_DIR" \
    --expected_frames 100 \
    2>&1 | tee "$LOG_DIR/seed_${SEED}_validation.log"

  touch "$RUN_DIR/DONE"
  rm -f "$RUN_DIR/FAILED"

  find "$RUN_DIR/frames_npz" \
    -maxdepth 1 -type f -name '*.npz' -delete
  touch "$RUN_DIR/FRAMES_REMOVED_AFTER_SUMMARY"

  trap - ERR
  echo "[complete] seed $SEED"
done

touch "$BASE/ENSEMBLE_24_DONE"

echo "======================================================================"
echo "INDEPENDENT REPLICATION COMPLETE"
echo "======================================================================"
echo "base = $BASE"
