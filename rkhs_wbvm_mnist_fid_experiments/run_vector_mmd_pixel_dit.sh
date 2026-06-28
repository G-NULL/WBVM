#!/usr/bin/env bash
set -uo pipefail

cd /root/rkhs_wbvm_mnist_fid_experiments
echo $$ > outputs_mnist_route2_v3_pixel_dit.pid
echo running > outputs_mnist_route2_v3_pixel_dit.exit

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
VECTOR_STATISTIC="${VECTOR_STATISTIC:-u}"

"$PYTHON_BIN" mnist_fid_experiment.py \
  --preset standard \
  --model-space pixel \
  --methods wbvm_vector \
  --direct-backbone dit \
  --fid-backend mnist \
  --selection-metric mnist_fid \
  --vector-loss-statistic "$VECTOR_STATISTIC" \
  --num-threads 20 \
  --num-workers 8 \
  --outdir outputs_mnist_route2_v3_pixel_dit \
  > outputs_mnist_route2_v3_pixel_dit.log 2>&1
status=$?
echo "$status" > outputs_mnist_route2_v3_pixel_dit.exit
exit "$status"
