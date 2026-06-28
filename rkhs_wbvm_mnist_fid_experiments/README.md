# RKHS-WBVM MNIST FID Experiments

This folder contains a compact MNIST image-generation benchmark for the
RKHS-WBVM route, with FID as the evaluation metric.

The first run is intended as a pilot rather than a claim-level reproduction of
large-scale ImageNet papers:

- `WBVM-all`: original endpoint generator `h(U)`, independent latent `U`, no
  `tied_X0`; RKHS derivative-kernel bridge loss over `tau in [0.35, 0.90]`.
- `MeanFlow`: average-velocity identity from arXiv:2505.13447, with sorted
  uniform `(r,t)` pairs, the JVP target, and one-step sampling
  `x=e-u(e,0,1)`.
- `ShortcutFlow`: arXiv:2410.12557-style step-size-conditioned shortcut model,
  with `d_min=1/128`, flow-matching grounding at the smallest step, discrete
  two-half-step self-consistency, and EMA bootstrap/evaluation.
- `Drifting`: arXiv:2602.04770 Algorithm 1/2-style drifting target regression
  in a frozen MNIST classifier feature space, using positive real samples,
  self-generated negatives, and row/column softmax weights.

Outputs:

- `metrics_summary.csv`
- `mnist_fid_table.png`
- `mnist_samples_grid.png`
- `run_config.json`
- `training_logs/<run>/{manifest.json,steps.jsonl,steps.csv,summary.json}`

Example:

```bash
python mnist_fid_experiment.py --preset quick --outdir outputs_mnist_quick
```

Use standard torch-fidelity Inception-FID with:

```bash
python mnist_fid_experiment.py --preset quick --fid-backend inception --outdir outputs_mnist_quick_inception
```

If the remote server downloads GitHub releases slowly, pre-cache
`weights-inception-2015-12-05-6726825d.pth` at:

```bash
/root/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth
```
