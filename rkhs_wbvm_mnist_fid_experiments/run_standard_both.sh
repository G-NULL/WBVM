#!/usr/bin/env bash
set -uo pipefail

cd /root/rkhs_wbvm_mnist_fid_experiments
echo $$ > outputs_mnist_v3_standard.pid

COMMON_ARGS=(
  --preset standard
  --methods wbvm_all,wbvm_single,meanflow,shortcut,drifting
  --fid-backend inception
  --selection-metric mnist_fid
  --tune-baselines
  --num-threads 20
  --num-workers 8
)

/root/miniconda3/bin/python3 mnist_fid_experiment.py \
  "${COMMON_ARGS[@]}" \
  --model-space pixel \
  --outdir outputs_mnist_v3_standard_pixel \
  > outputs_mnist_v3_standard_pixel.log 2>&1
pixel_status=$?
echo "$pixel_status" > outputs_mnist_v3_standard_pixel.exit
if [[ "$pixel_status" -ne 0 ]]; then
  exit "$pixel_status"
fi

/root/miniconda3/bin/python3 mnist_fid_experiment.py \
  "${COMMON_ARGS[@]}" \
  --model-space latent \
  --outdir outputs_mnist_v3_standard_latent \
  > outputs_mnist_v3_standard_latent.log 2>&1
latent_status=$?
echo "$latent_status" > outputs_mnist_v3_standard_latent.exit
exit "$latent_status"
