#!/usr/bin/env bash
set -uo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
OUTDIR="${OUTDIR:-outputs_vector_mmd_2602sec61_standard}"
PRESET="${PRESET:-standard}"
TRAIN_N="${TRAIN_N:-10000}"
THREADS="${THREADS:-20}"

cd "$(dirname "$0")"

run_all() {
  extra_args=()
  if [ -n "${ALL_TAU_LOW:-}" ]; then
    extra_args+=(--all-tau-low "$ALL_TAU_LOW")
  fi
  if [ -n "${ALL_TAU_HIGH:-}" ]; then
    extra_args+=(--all-tau-high "$ALL_TAU_HIGH")
  fi
  if [ -n "${VAL_METRIC:-}" ]; then
    extra_args+=(--val-metric "$VAL_METRIC")
  fi
  if [ -n "${LOSS_STATISTIC:-}" ]; then
    extra_args+=(--loss-statistic "$LOSS_STATISTIC")
  fi
  if [ -n "${SINGLE_TAUS:-}" ]; then
    extra_args+=(--single-taus "$SINGLE_TAUS")
  fi
  if [ -n "${DATASETS:-}" ]; then
    extra_args+=(--datasets "$DATASETS")
  fi

  "$PYTHON_BIN" test_wbvm_vector_mmd_experiment.py &&
  "$PYTHON_BIN" test_merge_vector_mmd_results.py &&
  "$PYTHON_BIN" wbvm_vector_mmd_experiment.py \
    --preset "$PRESET" \
    --train-n "$TRAIN_N" \
    --outdir "$OUTDIR" \
    --num-threads "$THREADS" \
    "${extra_args[@]}"
}

run_all
status=$?
if [ -n "${EXIT_FILE:-}" ]; then
  echo "$status" > "$EXIT_FILE"
fi
exit "$status"
