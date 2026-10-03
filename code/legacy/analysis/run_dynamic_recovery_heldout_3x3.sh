#!/usr/bin/env bash
set -euo pipefail

cd legacy_workspace

BASE="phase_ensemble_v1/summaries/dynamic_recovery/heldout_3x3"
LOGBASE="phase_ensemble_v1/logs/dynamic_recovery_heldout_3x3"

mkdir -p "$BASE" "$LOGBASE"

CHECKPOINTS=(
  1750
  2250
  2750
)

SEEDS=(
  2026074101
  2026074102
  2026074103
)

for STEP in "${CHECKPOINTS[@]}"; do
  CHECKPOINT=$(printf \
    "phase_ensemble_v1/summaries/dynamic_recovery/spinup_N64/checkpoints/checkpoint_step_%07d.npz" \
    "$STEP"
  )

  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: missing checkpoint $CHECKPOINT"
    exit 1
  fi

  for SEED in "${SEEDS[@]}"; do
    OUTDIR="$BASE/checkpoint_${STEP}/seed_${SEED}"
    RUNLOG="$LOGBASE/checkpoint_${STEP}_seed_${SEED}.log"
    AUDIT="$OUTDIR/recovery_smoke_audit.json"

    if [[ -f "$AUDIT" ]] && \
       grep -q '"numerical_pass": true' "$AUDIT"; then
      echo "SKIP completed checkpoint=$STEP seed=$SEED"
      continue
    fi

    rm -rf "$OUTDIR"
    mkdir -p "$OUTDIR"

    echo "============================================================"
    echo "RUN checkpoint=$STEP seed=$SEED"
    echo "============================================================"

    python scripts/run_dynamic_recovery_smoke.py \
      --checkpoint "$CHECKPOINT" \
      --outdir "$OUTDIR" \
      --phase_seed "$SEED" \
      --n_steps 300 \
      --output_every 10 \
      --scale 8 \
      --workers 16 \
      --dt 0.01 \
      --force \
      2>&1 | tee "$RUNLOG"
  done
done

echo "============================================================"
echo "HELDOUT 3x3 COMPLETE"
echo "============================================================"
