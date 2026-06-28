#!/usr/bin/env bash
set -uo pipefail

cd /root/rkhs_wbvm_mnist_fid_experiments
echo $$ > outputs_mnist_corrected_cnn_mnistfid.pid
echo running > outputs_mnist_corrected_cnn_mnistfid.exit

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"

"$PYTHON_BIN" mnist_fid_experiment.py \
  --preset standard \
  --model-space pixel \
  --methods wbvm_all,wbvm_single,meanflow,shortcut,drifting \
  --direct-backbone cnn \
  --fid-backend mnist \
  --selection-metric mnist_fid \
  --single-steps 2500 \
  --wbvm-single-taus 0.3,0.5,0.7 \
  --num-threads 20 \
  --num-workers 8 \
  --outdir outputs_mnist_corrected_cnn_mnistfid \
  > outputs_mnist_corrected_cnn_mnistfid.log 2>&1
status=$?
echo "$status" > outputs_mnist_corrected_cnn_mnistfid.exit
exit "$status"
