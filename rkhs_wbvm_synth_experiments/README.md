# RKHS-WBVM Synthetic Experiments

This folder contains a self-contained implementation of the synthetic benchmark
suite requested for `weak_bridge_velocity_matching_spine_talk.pdf`, using the
RKHS derivative-kernel route.

The setup mirrors Section 6.1 of `2602.19600v1`:

- datasets: `rings2d`, `spiral2d`, `moons2d`, `checker2d`, `helix3d`, `torus3d`;
- one-step endpoint generator `h_theta: R^d -> R^D`;
- bridge variables
  `X_tau = (1 - tau) X0 + tau X1`, `V = X1 - X0`;
- RKHS weak flux discrepancy with an RBF derivative kernel;
- qualitative 6-by-4 sample grid;
- `n / K / tau` sensitivity grid.

Here `K` is not an anchor count. For RKHS-WBVM it denotes the kernel batch size,
i.e. the number of bridge samples from each side used to estimate the
closed-form flux discrepancy per generator step.

## Files

- `wbvm_rkhs_experiment.py`: main experiment script.
- `wbvm_vector_mmd_experiment.py`: route-two vector-flux MMD WBVM script. It
  replaces the derivative-kernel Hessian loss with the vector-valued kernel
  objective `k(x,x') v^T v'`.
- `test_wbvm_vector_mmd_experiment.py`: small unit tests for the vector-flux
  MMD loss.
- `merge_vector_mmd_results.py`: merges an existing derivative-kernel RKHS
  output directory with a vector-flux MMD output directory into one combined
  Table-3/Figure-1-style result set.
- `test_merge_vector_mmd_results.py`: unit test for the result-merging labels
  and Flow Matching de-duplication.
- `requirements.txt`: minimal Python package list.
- `remote_paramiko.py`: helper for uploading this folder to the remote server,
  launching a run, and fetching outputs.

## Local smoke test

```powershell
python wbvm_rkhs_experiment.py --preset smoke --outdir outputs_smoke
```

Route-two vector-flux MMD smoke test:

```powershell
python wbvm_vector_mmd_experiment.py --preset smoke --outdir outputs_vector_mmd_smoke
```

Merge route-one and route-two outputs:

```powershell
python merge_vector_mmd_results.py --rkhs-dir outputs_remote_standard_noknt --vector-dir outputs_vector_mmd_standard --outdir outputs_merged_routes
```

## Standard run

```powershell
python wbvm_rkhs_experiment.py --preset standard --outdir outputs_standard --skip-knt
```

Expected output files:

- `rkhs_wbvm_toy_6x4.png`
- `metrics_summary.csv`
- `table1_metrics.csv`
- `table1_metrics.md`
- `table1_metrics.tex`
- `run_config.json`

The standard preset is the paper-faithful setting used for the corrected run:

- generator / FM backbone: 5 hidden layers, width 512, ReLU;
- RKHS-WBVM all-time: 8,000 generator updates, kernel U-statistic batch `K=1024`;
- RKHS training loss: U-statistic theta-dependent terms; full U-statistic flux
  residual is kept as a held-out diagnostic;
- validation-selected RKHS-WBVM single-level: 4,000 updates per candidate `tau`;
- RKHS bandwidth: fixed data-bridge median `sigma0` with multi-scale average
  `sigma0 * [0.5, 1, 2, 4]`;
- RKHS velocities: stop-gradient batch RMS normalization before the flux loss;
- bridge coupling: model and data bridge samples share the same base noise `X0`;
- WBVM-all bridge times: `tau ~ Unif(0.35,0.90)`;
- flow matching: 8,000 updates, 32-dimensional sinusoidal time features;
- FM sampling: midpoint ODE, NFE=20;
- table metrics: 10,000 generated samples per method/dataset.
- runtime parallelism: one H800 GPU, GPU-vectorized pairwise RKHS batches, TF32 enabled,
  and up to 20 CPU threads for NumPy/PyTorch CPU work.

`WBVM-single` now implements validation-selected single-level WBVM:

- candidate levels: `T={0.1,0.2,...,0.9}`;
- one candidate generator is trained for each `tau` with 4000 optimizer steps in
  the standard preset;
- validation uses held-out full RKHS/WBVM flux residual on the validation set by
  default; endpoint metrics such as energy distance, sliced W2, and W2 remain
  available as ablations;
- the selected `tau*` is reported in the table and in
  `wbvm_single_validation_selection.csv`.

`WBVM-all` samples bridge times uniformly from `[0.35,0.90]` by default,
following the stabilization plan in `rkhs_wbvm_improvement_plan.md`. This only
changes the WBVM-all bridge range; flow matching still uses the broader
`[0.05,0.95]` interval unless overridden. Following the original WBVM route,
the generator is not time-conditioned; final samples use `h(U)`.

All WBVM latent variables use the ambient/final dimension `D`.
The optional `--tie-latent-to-base` ablation sets `U=X0` inside the WBVM bridge
training loss, turning the generator into a more tightly paired base-to-data
transport map for comparison.
