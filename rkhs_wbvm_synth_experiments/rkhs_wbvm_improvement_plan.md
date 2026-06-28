# RKHS-WBVM Synthetic Experiments: Implementation Improvement Plan

## 1. 当前问题诊断

当前模拟实验中，WBVM 生成结果出现明显的星形射线、中心坍缩和扇形连接结构。这通常说明有限样本 RKHS flux loss 给生成器提供了某种捷径：模型不必真正贴近数据流形，也可以通过局部速度范数、kernel Hessian 或批量噪声结构来降低训练目标。

从当前输出目录和日志看，实验使用的是 V-statistic 版本，并且 `last_loss` 大量为负：

```python
flux = rkhs_flux_v_stat(..., include_data_data=False)
```

这不是完整的 squared discrepancy，而只是与参数 \(\theta\) 有关的部分。因此 loss 下降不一定代表弱桥流量差异真的变小。

## 2. 修改一：用 U-statistic 替代当前 V-statistic 训练版

建议先把训练中的 flux loss 改成去掉同组对角项的 U-statistic：

```python
flux = rkhs_flux_u_stat(x_model, v_model, x_data, v_data)
```

原因是 RBF derivative kernel 的同组对角项包含

\[
v^\top H_{k_\sigma}(x,x)v
=
\frac{\|v\|^2}{\sigma^2}.
\]

如果使用带对角项的 V-statistic，loss 容易被速度范数主导，模型可能通过产生射线状高速结构来降低目标。U-statistic 去掉同组对角项后，可以减轻这种速度范数捷径。

同时建议额外记录 held-out full discrepancy：

\[
\widehat D_{\rm heldout}^2(\theta,\tau),
\]

但只把它作为诊断指标，不一定直接作为训练 loss。这样可以区分“训练 loss 下降”和“真实弱流量差异下降”。

## 3. 修改二：加入 velocity RMS normalization

当前星形伪解很可能与速度范数和 kernel Hessian 的尺度耦合有关。因此建议在进入 RKHS flux loss 前，对模型速度和数据速度做 batch RMS normalization：

\[
\widetilde V^\theta
=
\frac{
V^\theta
}{
\operatorname{sg}\left(
\sqrt{\mathbb E_{\rm batch}\|V^\theta\|^2}
+\delta
\right)
},
\]

\[
\widetilde V^d
=
\frac{
V^d
}{
\operatorname{sg}\left(
\sqrt{\mathbb E_{\rm batch}\|V^d\|^2}
+\delta
\right)
}.
\]

这里 \(\operatorname{sg}(\cdot)\) 表示 stop-gradient。训练时用

\[
\widetilde V^\theta,\qquad \widetilde V^d
\]

代替原始

\[
V^\theta,\qquad V^d
\]

进入 RKHS flux discrepancy。

对应代码思路：

```python
def normalize_velocity(v, eps=1e-6):
    scale = torch.sqrt(v.pow(2).sum(dim=1).mean()).detach()
    return v / (scale + eps)

v_model_n = normalize_velocity(v_model)
v_data_n = normalize_velocity(v_data)
flux = rkhs_flux_u_stat(x_model, v_model_n, x_data, v_data_n)
```

这个修改优先级很高，通常比继续微调 learning rate 更有效。

## 4. 修改三：使用 common base noise 降低有限样本方差

当前实现中，模型桥和数据桥使用两组独立的 base noise：

\[
X_{\theta,\tau}
=
(1-\tau)X_0^\theta+\tau h_\theta(U),
\]

\[
X^d_\tau
=
(1-\tau)X_0^d+\tau X_1.
\]

在 population 层面这没有问题，但在有限 batch 下会显著增加方差。建议改成共享同一批 base noise：

\[
X_{\theta,\tau}
=
(1-\tau)X_0+\tau h_\theta(U),
\]

\[
X^d_\tau
=
(1-\tau)X_0+\tau X_1.
\]

也就是 model bridge 和 data bridge 使用同一个 \(X_0\)。这样 weak flux comparison 更像 paired bridge comparison，梯度会更稳定。

对应代码结构：

```python
x0 = torch.randn(batch, data.D, device=device)
u = torch.randn(batch, latent_dim, device=device)
x1 = take_batch(train_x, batch)

y = model(u)

x_model = (1.0 - tau) * x0 + tau * y
v_model = y - x0

x_data = (1.0 - tau) * x0 + tau * x1
v_data = x1 - x0
```

## 5. 修改四：调整 \(\tau\) 采样范围

当前 all-time 训练直接使用

\[
\tau\sim{\rm Unif}(0.05,0.95).
\]

但从 single-level 选择结果看，较优 \(\tau\) 往往落在 \(0.8\) 或 \(0.9\)。这说明高 \(\tau\) 的 endpoint 信号更强，但过高 \(\tau\) 又容易导致核梯度尖锐、posterior-like collapse 和射线伪解。

建议第一版先改成中等偏高的稳定区间：

\[
\tau\sim{\rm Unif}(0.35,0.85).
\]

这个范围去掉了两个不稳定端点：

- 极小 \(\tau\)：桥几乎是 base noise，endpoint 监督信号很弱；
- 极大 \(\tau\)：桥过度接近 endpoint，kernel Hessian 和速度项容易尖锐。

也可以使用 curriculum：

\[
[0.35,0.65]
\rightarrow
[0.25,0.85]
\rightarrow
[0.15,0.90].
\]

这样先在稳定中间层训练出大致结构，再逐步扩大到更靠近 endpoint 的区域。

## 6. 推荐实验顺序

建议不要一次性全改，而是按以下顺序做 ablation：

1. **U-statistic only**

   把训练 loss 从 `rkhs_flux_v_stat(..., include_data_data=False)` 改成 `rkhs_flux_u_stat(...)`。

2. **U-statistic + velocity normalization**

   在 U-statistic 基础上加入 \(V^\theta,V^d\) 的 RMS normalization。

3. **U-statistic + velocity normalization + common \(X_0\)**

   进一步共享 model/data bridge 的 base noise。

4. **调整 \(\tau\) 区间**

   把 all-time 训练从

   \[
   [0.05,0.95]
   \]

   改成

   \[
   [0.35,0.85].
   \]

5. **curriculum \(\tau\)**

   如果第 4 步有效，再尝试

   \[
   [0.35,0.65]
   \rightarrow
   [0.25,0.85]
   \rightarrow
   [0.15,0.90].
   \]

## 7. 预期观察

如果这些修改有效，应该首先看到：

- 星形射线减少；
- 中心坍缩减弱；
- held-out flux residual 更稳定；
- endpoint sliced \(W_2\) 改善；
- off-manifold rate 降低。

如果这些修改后仍然明显失败，则说明问题可能不只是实现细节，而是 pure weak flux 在有限样本下约束不足，需要进一步加入 endpoint / marginal stabilization 或 mixture latent structure。
