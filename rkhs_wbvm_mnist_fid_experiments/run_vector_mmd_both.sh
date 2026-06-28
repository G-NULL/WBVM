#!/usr/bin/env bash
set -uo pipefail

cd /root/rkhs_wbvm_mnist_fid_experiments
echo $$ > outputs_mnist_route2_standard.pid
echo running > outputs_mnist_route2_tests.exit
echo running > outputs_mnist_route2_standard_pixel.exit
echo waiting > outputs_mnist_route2_standard_latent.exit

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
VECTOR_STATISTIC="${VECTOR_STATISTIC:-u}"

"$PYTHON_BIN" test_mnist_fid_experiment.py
test_status=$?
if [[ "$test_status" -ne 0 ]]; then
  echo "$test_status" > outputs_mnist_route2_tests.exit
  exit "$test_status"
fi
echo 0 > outputs_mnist_route2_tests.exit

COMMON_ARGS=(
  --preset standard
  --methods wbvm_vector
  --fid-backend inception
  --selection-metric mnist_fid
  --vector-loss-statistic "$VECTOR_STATISTIC"
  --num-threads 20
  --num-workers 8
)

"$PYTHON_BIN" mnist_fid_experiment.py \
  "${COMMON_ARGS[@]}" \
  --model-space pixel \
  --outdir outputs_mnist_route2_standard_pixel \
  > outputs_mnist_route2_standard_pixel.log 2>&1
pixel_status=$?
echo "$pixel_status" > outputs_mnist_route2_standard_pixel.exit
if [[ "$pixel_status" -ne 0 ]]; then
  exit "$pixel_status"
fi

"$PYTHON_BIN" mnist_fid_experiment.py \
  "${COMMON_ARGS[@]}" \
  --model-space latent \
  --outdir outputs_mnist_route2_standard_latent \
  > outputs_mnist_route2_standard_latent.log 2>&1
latent_status=$?
echo "$latent_status" > outputs_mnist_route2_standard_latent.exit
exit "$latent_status"
