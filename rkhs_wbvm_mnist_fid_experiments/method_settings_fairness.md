# MNIST 方法设置与公平性检查

本文档记录当前 corrected MNIST standard 实验中各方法的网络、训练参数、loss/objective 和潜在不公平点，用于检查横向比较是否合理。

## 结果来源说明

当前合并表由两次 standard 运行组成：

- `WBVM-all` 和 `WBVM-single` 来自 `outputs_mnist_inception_standard`。
- corrected `MeanFlow`、`ShortcutFlow`、`Drifting` 来自 `outputs_mnist_corrected_baselines_standard`。

两次运行的公共配置一致，包括 MNIST 数据划分、seed、FID backend、CNN hidden size、batch size 等。WBVM 代码在 corrected baseline 之前没有改动，因此结果可以暂时合并比较；不过严格论文表最好再用同一份最新脚本一次性重跑全方法。

## 公共设置

| 项 | 设置 |
|---|---|
| 数据集 | MNIST |
| 训练集 | `50000` |
| validation | `5000` |
| test / FID samples | `10000` |
| 图像空间 | `[-1, 1]` |
| 图像 shape | `1 x 28 x 28` |
| FID | torch-fidelity / torchmetrics Inception-FID |
| batch size | `256` |
| optimizer | AdamW |
| learning rate | `2e-4` |
| weight decay | `1e-4` |
| base CNN hidden | `64` |
| DataLoader workers | `8` |
| seed | `7` |
| final NFE | all methods use `1` |
| device | CUDA |
| TF32 | enabled |

## 共同 CNN Backbone

除额外说明外，生成器和速度网络都基于同一个 `CondConvNet(hidden=64)`：

```text
input: 1 or 1 + cond_dim channels, 28 x 28
Conv2d(in, 64, 3x3)
Conv2d(64, 128, 4x4, stride=2)
Conv2d(128, 256, 4x4, stride=2)
2 x ResBlock(256)
ConvTranspose2d(256, 128, 4x4, stride=2)
ConvTranspose2d(128, 64, 4x4, stride=2)
ResBlock(64)
GroupNorm + SiLU + Conv2d(64, 1, 3x3)
```

`ResBlock(C)` 为：

```text
GroupNorm + SiLU + Conv2d(C, C, 3x3)
GroupNorm + SiLU + Conv2d(C, C, 3x3)
residual connection
```

## 各方法设置

| Method | 网络 | Steps | 主要参数 | Loss / objective |
|---|---|---:|---|---|
| `WBVM-all` | `DirectGenerator(cond_dim=0, out_tanh=True)` | `5000` | `tau ~ Unif(0.35, 0.90)`，`kernel_batch=128`，independent `U`，common bridge `X0` | RKHS derivative-kernel U-stat theta terms |
| `WBVM-single` | `DirectGenerator(cond_dim=0, out_tanh=True)` | 每候选 `2500` | tau candidates `{0.3, 0.5, 0.7}`，选中 `0.5` | 每个 tau 训练 WBVM，然后用 validation FID 选择 |
| `MeanFlow` | `VelocityNet(cond_dim=2, out_tanh=False)` | `5000` | sorted uniform `(r,t)`，one-step sample `x = e - u(e, 0, 1)` | MeanFlow JVP target |
| `ShortcutFlow` | `VelocityNet(cond_dim=2, out_tanh=False)` | `5000` | `d_min=1/128`，`shortcut_empirical_frac=0.75`，EMA decay `0.999` | 75% flow grounding + 25% two-half-step self-consistency |
| `Drifting` | `DirectGenerator(cond_dim=0, out_tanh=True)` + MNIST feature net | `5000` | `step_size=0.08`，temperatures `[0.05, 0.1, 0.2]`，row/column softmax | arXiv:2602.04770 Algorithm 1/2-style drifting target |

## WBVM 细节

### WBVM-all

- Latent: `U ~ N(0, I)`，shape 与图像一致，即 `1 x 28 x 28`。
- 不使用 `tied_X0`。
- Bridge:

```text
X_model_tau = (1 - tau) X0 + tau h(U)
X_data_tau  = (1 - tau) X0 + tau X1
```

- `X0` 在 model/data bridge 之间共享，用于降低 bridge loss 方差。
- RKHS bandwidth:

```text
sigma0 = median bandwidth on data bridge samples
sigmas = sigma0 * [0.5, 1.0, 2.0, 4.0]
```

- `kernel_bandwidth_points = 2048`
- `kernel_batch = 128`
- velocity RMS normalization: enabled，`velocity_eps=1e-6`
- loss 使用 theta-dependent U-stat terms：

```text
loss = <model, model>_U - 2 <model, data>
```

data-data 项不参与训练梯度。

### WBVM-single

- Candidate tau set:

```text
{0.3, 0.5, 0.7}
```

- 每个 candidate 训练 `2500` steps。
- validation metric: Inception-FID on validation set。
- selected tau:

```text
tau = 0.5
```

validation FID:

| tau | validation FID |
|---:|---:|
| `0.3` | `97.774` |
| `0.5` | `59.894` |
| `0.7` | `65.877` |

## MeanFlow 细节

当前实现对应 arXiv:2505.13447 的核心 average-velocity identity：

- Network: `VelocityNet(cond_dim=2)`。
- Conditioning: concatenated `(r, t)` scalar maps。
- Sampling:

```text
e ~ N(0, I)
x_gen = e - u_theta(e, r=0, t=1)
```

- Training:

```text
x ~ data
e ~ noise
t = max(a, b), r = min(a, b), a,b ~ Uniform(0,1)
z_t = (1 - t) x + t e
v = e - x
u = u_theta(z_t, r, t)
target = v - (t - r) * d/dt u_theta(z_t, r, t)
loss = MSE(u, stopgrad(target))
```

JVP 用 `torch.func.jvp` 计算。

## ShortcutFlow 细节

当前实现对应 arXiv:2410.12557 的 shortcut/self-consistency 思路：

- Network: `VelocityNet(cond_dim=2)`。
- Conditioning: `(t, dt)` scalar maps。
- `shortcut_min_steps = 128`，因此 `d_min = 1/128`。
- EMA bootstrap/eval:

```text
ema_decay = 0.999
```

- Batch split:

```text
75% flow-matching grounding
25% shortcut self-consistency
```

Flow grounding:

```text
x_t = (1 - t) x0 + t x1
target = x1 - x0
loss_fm = MSE(s_theta(x_t, t, d_min), target)
```

Self-consistency:

```text
s1 = ema_model(x_t, t, dt/2)
x_mid = x_t + (dt/2) s1
s2 = ema_model(x_mid, t + dt/2, dt/2)
target = 0.5 * (s1 + s2)
loss_sc = MSE(s_theta(x_t, t, dt), stopgrad(target))
```

Final sampling:

```text
x_gen = x0 + s_ema(x0, t=0, dt=1)
```

## Drifting 细节

当前实现对应 arXiv:2602.04770 Algorithm 1/2 的 drifting target regression 思路，但为了在 MNIST 上稳定计算，drifting field 在一个冻结的 MNIST classifier feature space 中计算。

### 额外 MNIST feature net

```text
Conv2d(1, 32, 3x3) + SiLU
Conv2d(32, 64, 3x3) + SiLU
AvgPool2d
Conv2d(64, 128, 3x3) + SiLU
AvgPool2d
Linear(128 * 7 * 7, 256) + SiLU
Linear(256, 10)
```

- classifier epochs: `3`
- classifier lr: `1e-3`
- final training acc: about `0.9865`
- drifting uses the 256-dimensional hidden feature before the classifier head。

### Drifting field

- Positive samples: real MNIST features。
- Negative samples: self-generated fake features。
- Temperatures:

```text
[0.05, 0.1, 0.2]
```

- Distance normalization: enabled。
- Drift vector RMS normalization: enabled。
- Row/column softmax weighting:

```text
logits = [-dist_pos / temp, -dist_neg / temp]
a_row = softmax(logits, dim=1)
a_col = softmax(logits, dim=0)
a = sqrt(a_row * a_col)
```

- Target regression:

```text
target_feature = fake_feature + step_size * drifting_field
loss = MSE(fake_feature, stopgrad(target_feature))
```

where:

```text
step_size = 0.08
```

## Corrected Standard FID Results

| Method | FID ↓ | NFE |
|---|---:|---:|
| `WBVM-all` | `48.711` | `1` |
| `WBVM-single` | `58.197` | `1` |
| `Drifting` | `80.509` | `1` |
| `MeanFlow` | `148.976` | `1` |
| `ShortcutFlow` | `341.593` | `1` |

## 潜在不公平点

### 1. Drifting 使用额外监督 classifier

这是当前最明显的不公平来源。

`Drifting` 的 loss 不在 pixel space 直接计算，而是在一个额外训练的 MNIST classifier feature space 中计算。这个 feature net 使用标签监督训练，其他方法没有使用标签。

影响：

- Drifting 获得了额外的 supervised representation。
- Drifting objective 更接近 feature matching，而其他方法主要在 pixel/bridge space 中训练。
- 如果目标是严格无监督生成模型比较，这一项不公平。

### 2. WBVM-single 做了 validation tau selection

`WBVM-single` 训练了 3 个候选 tau：

```text
tau = 0.3, 0.5, 0.7
```

最终报告的是 validation FID 最好的 `tau=0.5`。

影响：

- 它消耗了约 3 倍 single-level 训练。
- 表中的 single 结果是 model selection 后结果，不是单个固定 tau 的结果。

### 3. ShortcutFlow 使用 EMA

`ShortcutFlow` 用 EMA model 进行 bootstrap target 和最终 eval。

影响：

- EMA 是论文式设置，但它是额外稳定化机制。
- MeanFlow 当前没有 EMA，可能相对吃亏。

### 4. MeanFlow 没有额外稳定化

MeanFlow 当前没有：

- EMA
- validation model selection
- feature-space loss

因此在当前比较中，MeanFlow 设置更朴素。

### 5. WBVM 使用 `kernel_batch=128`

公共 batch size 是 `256`，但 WBVM RKHS pairwise kernel 用：

```text
kernel_batch = 128
```

这是显存和速度折中。

影响：

- WBVM loss 的 Monte Carlo 方差和其他 MSE objective 不同。
- 但如果设成 256，pairwise kernel 成本会明显增加。

### 6. 合并表不是同一次进程跑出

当前 corrected combined table 中：

- WBVM 来自旧 standard run。
- corrected baselines 来自新 standard run。

公共配置一致，但严格来说不是同一次脚本运行。

## 建议的更公平版本

建议后续分两张表：

### 表 A：paper-faithful 设置

保留每篇论文原始思路中的关键机制：

- WBVM-single 允许 validation selection。
- ShortcutFlow 使用 EMA。
- Drifting 使用 feature-space drifting。
- MeanFlow 使用 JVP target。

这张表回答：

> 每种方法按自己论文思路实现后，在当前 MNIST setup 下表现如何？

### 表 B：strict shared-backbone 设置

尽量强制统一：

- 不使用 supervised classifier feature space。
- 所有方法统一是否使用 EMA。
- 所有方法统一训练预算。
- WBVM-single 固定 tau 或把所有方法也允许 validation selection。
- 所有方法使用同一个 backbone capacity 和相同 sampling NFE。

这张表回答：

> 在尽量相同的资源和 inductive bias 下，方法本身差异如何？

