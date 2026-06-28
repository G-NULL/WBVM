# MNIST standard experiments: networks and parameters

Date: 2026-06-15

This note records the exact network and parameter settings used for the two
formal MNIST standard runs:

- Pixel run: `outputs_mnist_v3_standard_pixel`
- Latent run: `outputs_mnist_v3_standard_latent`

Both runs use the same method list:

`WBVM-all, WBVM-single, MeanFlow, ShortcutFlow, Drifting`

## Runtime

Remote machine:

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- CUDA available: yes
- PyTorch: 2.7.0+cu128
- torchvision: 0.22.0+cu128
- Torch CPU threads: 20
- DataLoader workers: 8
- TF32: enabled

Commands:

```bash
python mnist_fid_experiment.py \
  --preset standard \
  --model-space pixel \
  --methods wbvm_all,wbvm_single,meanflow,shortcut,drifting \
  --fid-backend inception \
  --selection-metric mnist_fid \
  --tune-baselines \
  --num-threads 20 \
  --num-workers 8 \
  --outdir outputs_mnist_v3_standard_pixel

python mnist_fid_experiment.py \
  --preset standard \
  --model-space latent \
  --methods wbvm_all,wbvm_single,meanflow,shortcut,drifting \
  --fid-backend inception \
  --selection-metric mnist_fid \
  --tune-baselines \
  --num-threads 20 \
  --num-workers 8 \
  --outdir outputs_mnist_v3_standard_latent
```

## Shared Data And Training Settings

| Setting | Value |
|---|---:|
| Seed | 7 |
| Train samples | 50,000 |
| Validation samples | 5,000 |
| Evaluation samples | 10,000 |
| Main training steps | 5,000 |
| WBVM-single steps per tau | 4,000 |
| Batch size | 256 |
| WBVM kernel batch | 128 |
| Base AdamW learning rate | 2e-4 |
| AdamW weight decay | 0.01 |
| Gradient clipping | 2.0 |
| Evaluation batch size | 256 |
| Inception-FID backend | torchmetrics Inception feature 2048 |
| Extra MNIST metrics | LeNet-FID, normalized KID, raw KID |
| KID subset size / subsets | 1000 / 20 |

Important implementation note: `hidden`, `latent_hidden`, `latent_depth`,
`meanflow_gap_prob`, and `shortcut_empirical_frac` remain in `run_config.json`
as legacy fields, but the current DiT-based training path does not use them for
model capacity or loss construction. The effective network capacity is controlled
by `pixel_dit_hidden`, `pixel_dit_depth`, `pixel_dit_heads`, and patch size.

## Model Spaces

### Pixel Space

- Training variable shape: `(1, 32, 32)`, matching the MNIST resize used by `tyfeld/drifting-model`.
- Data normalization: image pixels `x in [0,1]` are mapped to model space
  `2x - 1 in [-1,1]`
- Direct generators output `tanh(net(u))`, so generated pixels are bounded in
  model space.
- LeNet-based MNIST-FID/KID evaluation resizes generated and real images back
  to 28 x 28 before feature extraction.

### Latent Space

- Latent variable shape: `(16, 4, 4)`, i.e. 256 dimensions.
- Encoder: LeNet-5 convolutional trunk.
- Decoder: learned lightweight transpose-convolution decoder.
- Latents are standardized by feature-wise mean/std estimated from 50,000
  training samples.
- All five methods train and sample in this same normalized latent space; samples
  are decoded back to pixels only for visualization and metrics.

## Shared DriftDiT-Tiny-Style Backbone

All trainable generative methods use the same DriftDiT-Tiny-style backbone
from `tyfeld/drifting-model`'s MNIST setup.

| Setting | Pixel run | Latent run |
|---|---:|---:|
| Hidden size | 256 | 256 |
| Depth | 6 | 6 |
| Attention heads | 4 | 4 |
| Input channels | 1 | 16 |
| Spatial size | 32 x 32 | 4 x 4 |
| Patch size | 4 | 1 |
| Patch token count | 64 | 16 |
| Register tokens | 8 | 8 |

Backbone structure:

- Patch embedding: `Conv2d(in_channels, hidden, kernel=patch, stride=patch)`.
- Positional encoding: RoPE in attention, with register tokens prepended.
- Conditioning: sinusoidal embeddings for time `t` and step/gap variable,
  each passed through a two-layer MLP and added.
- Optional style embedding: 32 random style tokens sampled from a 64-entry
  learnable codebook, following the reference implementation.
- Transformer block: RMSNorm, QK-Norm attention, SwiGLU MLP ratio 4, and
  adaLN-Zero condition-generated shift/scale/gates.
- Final layer: AdaLN final projection, followed by patch unpatchifying.

Single-head DiT is used by:

- `WBVM-all`
- `WBVM-single`
- `ShortcutFlow`
- `Drifting`

Dual-head DiT is used by:

- `MeanFlow`, with separate `u` and `v` output heads.

For dual-head MeanFlow, the full six-block trunk is shared and only the final
projection is split into separate `u` and `v` heads.

## LeNet Feature Net And Latent Codec

The same LeNet feature extractor is used for:

- MNIST-FID / normalized KID / raw KID evaluation.
- Latent-space encoder.
- WBVM-single validation FID selection.

LeNet feature network:

```text
Conv2d(1, 6, kernel=5) -> tanh -> avg_pool2d(2)
Conv2d(6, 16, kernel=5) -> tanh -> avg_pool2d(2)
flatten 16*4*4 = 256
Linear(256, 120) -> tanh
Linear(120, 84) -> tanh
Linear(84, 10)
```

Feature used for MNIST-FID/KID: the 84-dimensional penultimate feature.

Classifier training:

- Epochs: 3
- Optimizer: AdamW
- Learning rate: 1e-3

Latent decoder:

```text
latent (16,4,4)
nearest upsample x2
ConvTranspose2d(16, 6, kernel=5) -> tanh
nearest upsample x2
ConvTranspose2d(6, 1, kernel=5) -> sigmoid
```

Decoder training:

- Epochs: 5
- Optimizer: AdamW
- Learning rate: 1e-3
- Loss: binary cross entropy reconstruction loss.

## WBVM-all

Network:

- Direct single-head DiT generator `h(U)`.
- Pixel: `U ~ N(0, I)` with shape `(1,28,28)`.
- Latent: `U ~ N(0, I)` with shape `(16,4,4)`.

Training:

- Steps: 5,000
- Learning rate: 2e-4
- Weight decay: 1e-4
- Tau interval: uniform in `[0.35, 0.90]`
- Independent bridge base: `X0 ~ N(0, I)`
- No `tied_X0`
- Generator has no time input in WBVM; it is the original endpoint generator
  route `h(U)`.

RKHS flux loss:

```text
X_theta = (1 - tau) X0 + tau h(U)
X_data  = (1 - tau) X0 + tau X1
V_theta = normalize(h(U) - X0)
V_data  = normalize(X1 - X0)

loss = <V_theta, V_theta>_H - 2 <V_theta, V_data>_H
```

The data-data term is omitted because it does not depend on theta.

Kernel bandwidth:

- Precompute a base bandwidth from bridge samples using 2,048 points.
- Base bandwidth uses median distance with clamp `[1, 64]`.
- Multi-scale RBF derivative kernel uses:

```text
sigma = sigma0 * [0.5, 1.0, 2.0, 4.0]
```

Velocity normalization epsilon: `1e-6`.

## WBVM-single

Network and loss:

- Same direct single-head DiT generator and RKHS flux loss as `WBVM-all`.
- Each candidate fixes a single tau.

Tau candidates:

```text
{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
```

Training:

- 4,000 steps per tau candidate.
- Base learning rate: 2e-4.
- Weight decay: 1e-4.

Validation selection:

- Selection metric: held-out MNIST-FID using the shared LeNet feature extractor.
- Validation set size: 5,000.
- Pixel selected tau in this run: `0.1`.
- Latent selected tau in this run: `0.1`.

## MeanFlow

Network:

- Dual-head DiT with `u` and `v` heads.
- Same hidden size/depth/head count as the shared DiT backbone.
- Implements the Pixel MeanFlow-style core in both pixel and latent spaces.

Time sampling:

- `t, r` sampled from logit-normal distribution:

```text
logit-normal mean = -0.4
logit-normal std  = 1.0
```

- 10% of pairs are replaced by uniform samples.
- `data_proportion = 0.5`, so half of the batch uses `r = t`.
- The code sorts the pair so `t >= r`.

Training target:

```text
z = (1 - t) x + t e
v_target = e - x
```

The model predicts `u` and `v`. The JVP route forms:

```text
compound_v = u + (t - r) * d u / d t
```

Loss:

```text
loss = adaptive_loss(compound_v - v_target)
     + adaptive_loss(v_pred - v_target)
```

Adaptive loss:

```text
per_example = ||error||_2^2
weight = stopgrad((per_example + 0.01)^1.0)
adaptive_loss = mean(per_example / weight)
```

Tuning grid:

```text
learning rate in {1e-4, 2e-4, 5e-4}
```

Selection metric: validation MNIST-FID.

Selected in this run:

- Pixel MeanFlow: `lr = 5e-4`
- Latent MeanFlow: `lr = 5e-4`

## ShortcutFlow

Network:

- Single-head time/step-conditioned DiT.
- Same hidden size/depth/head count as the shared DiT backbone.
- EMA model is used for bootstrap targets and returned for evaluation.

Core settings:

| Setting | Value |
|---|---:|
| `shortcut_min_steps` | 128 |
| Bootstrap fraction | `batch_size / shortcut_bootstrap_every` |
| Default `shortcut_bootstrap_every` | 8 |
| EMA decay default | 0.999 |

Training target:

- Bootstrap part: uses EMA model two half-steps to construct the shortcut target.
- Flow-grounding part: uses minimum-step flow matching target.
- `dt_base` values are balanced across powers of two.

Loss:

```text
MSE(predicted_velocity, target_velocity)
```

Tuning grid:

```text
learning rate in {1e-4, 2e-4}
shortcut_bootstrap_every in {4, 8}
EMA decay in {0.99, 0.995}
```

Selection metric: validation MNIST-FID.

Selected in this run:

- Pixel ShortcutFlow: `lr = 2e-4`, `shortcut_bootstrap_every = 4`,
  `EMA decay = 0.99`
- Latent ShortcutFlow: `lr = 1e-4`, `shortcut_bootstrap_every = 4`,
  `EMA decay = 0.995`

## Drifting

Network:

- Direct single-head DiT generator.
- Same backbone as WBVM direct generator.
- Pixel version trains in pixel space.
- Latent version trains in normalized LeNet latent space.

Training:

- Steps: 5,000
- Learning rate: 2e-4
- Weight decay: 1e-4
- Step size: 0.08
- Temperatures: `[0.05, 0.10, 0.20]`
- Positive samples: real batch samples.
- Negative samples: current self-generated samples.
- Distance normalization: enabled.
- Drift vector normalization: enabled.

Loss:

```text
field = sum_t drifting_vector_t(y, real_batch, generated_batch)
target = stopgrad(y + step_size * field)
loss = MSE(y, target)
```

## Evaluation Metrics

Final table reports:

- Inception-FID on 10,000 samples.
- MNIST-FID using LeNet 84-dimensional features.
- Normalized polynomial KID, reported as `KID-z x1000`.
- Raw polynomial KID, reported as `KID-raw x1000`.
- NFE. All methods here use one generator evaluation for sampling, so `NFE = 1`.

For normalized KID, feature normalization is fit on real validation/test features
and then applied to generated features:

```text
feature_z = (feature - mean_real) / std_real
```

Raw KID is also retained to expose any feature-scale artifacts.

## Output Files

Pixel outputs:

- `outputs_mnist_v3_standard_pixel/metrics_summary.csv`
- `outputs_mnist_v3_standard_pixel/baseline_tuning.csv`
- `outputs_mnist_v3_standard_pixel/wbvm_single_selection.csv`
- `outputs_mnist_v3_standard_pixel/mnist_fid_table.png`
- `outputs_mnist_v3_standard_pixel/mnist_samples_grid.png`

Latent outputs:

- `outputs_mnist_v3_standard_latent/metrics_summary.csv`
- `outputs_mnist_v3_standard_latent/baseline_tuning.csv`
- `outputs_mnist_v3_standard_latent/wbvm_single_selection.csv`
- `outputs_mnist_v3_standard_latent/mnist_fid_table.png`
- `outputs_mnist_v3_standard_latent/mnist_samples_grid.png`

Combined outputs:

- `standard_pixel_latent_summary.csv`
- `standard_pixel_latent_summary_table.png`
