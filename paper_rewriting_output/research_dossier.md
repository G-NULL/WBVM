# Endpoint-Posterior Velocity Matching: Research Dossier

## 核心定位

`endpoint_posterior_velocity_matching_roadmap.pdf` 的独特主张不是再训练一个自由形式的时间速度网络 `v_theta(x,t)`，而是让速度场由同一个 one-step generator `h_theta(U)` 诱导出来：

```text
endpoint posterior mean -> endpoint-induced velocity -> one-step sample h_theta(U)
```

无噪声线性 bridge 下，核心对象是

```text
m_{theta,tau}(x) = E[h_theta(U) | X_{theta,tau}=x],
v_{theta,tau}(x) = (m_{theta,tau}(x)-x)/(1-tau).
```

Gaussian-base noisy bridge 下，`X0` 可被积掉，得到更稳定的低维 latent posterior：

```text
p_theta(u | x_tau) proportional pi(u) N(x_tau; tau h_theta(u), (1-tau^2)I),
v^gamma_{theta,tau}(x_tau) = (m_{theta,tau}(x_tau)-tau x_tau)/(1-tau^2).
```

因此最尖锐的论文定位可以写成：用 endpoint posterior 约束速度场，同时保留 one-step latent-to-data transport。

## 已有基础

本地 MAGT 源文档已经提供了一个强基线：固定 Gaussian smoothing level、posterior anchors、低维到高维的 one-shot transport、single-level score pull-back theorem。E-PVM 的任务不是重复 MAGT，而是把 fixed-level score/denoiser posterior 推进到 velocity/interpolant 语言。

从本地证明笔记看：

- Gaussian-corruption velocity 版本几乎可以继承 MAGT 的 denoiser/score/Fisher/W2 proof chain。
- 真正三项 stochastic interpolant 版本保留了 velocity loss 的正交分解，但不再天然等价 Fisher divergence。
- 因此三项版本需要一个新的 single-level velocity pull-back theorem。
- 如果 `X0` 是全维 Gaussian，intrinsic-dimension rate 可能被 ambient dimension 污染；低维或 shared-latent base endpoint 是保留流形优势的关键设计。

## 外部 SOTA 压力

截至 2026-06-08，外部坐标很明确：

- Flow Matching 学 fixed conditional probability paths 的向量场，重点在 simulation-free CNF 训练和 OT/diffusion paths。
- Stochastic Interpolants 已经给出 flows/diffusions/Schrodinger bridge 的统一桥架。
- Consistency/Shortcut/Flow Map Matching/MeanFlow 都在把多步动态压缩为一/少步 map 或 average velocity。
- Posterior-Augmented Flow Matching (PAFM, 2026) 已经提出用给定中间态下的 approximate posterior target completions 替代单目标 FM supervision，这是最接近的撞题对象。

所以 E-PVM 必须和 PAFM 拉开距离：PAFM 仍是 free velocity-field training 的 posterior-augmented supervision；E-PVM 应主打 generator-induced velocity identity、latent endpoint anchors、one-step transport、流形/低维理论。

## 五个关键选择

1. **Noisy bridge vs no-noise bridge**  
   No-noise 公式直观，但 posterior 权重在 `tau -> 1` 时极尖。Gaussian-base noisy bridge 的分母从 `(1-tau)^2` 变成 `1-tau^2`，更适合有限 anchor。

2. **Fixed tau vs adaptive tau**  
   Fixed tau 保持 one-step 方法的辨识度；adaptive tau 或少量 tau levels 能显著稳定 posterior，但会削弱 single-level 理论叙事。

3. **Full Gaussian base vs low-dimensional/shared base**  
   Full Gaussian base 便于推导；low-dimensional/shared-latent base 更可能保留 intrinsic dimension。

4. **Prior anchors vs proposal/retrieval anchors**  
   Prior anchors 是最小版本；retrieval、amortized、mixture proposal 是算法创新和实验提升的主要空间。

5. **Score proof inheritance vs velocity proof novelty**  
   保守 Gaussian-corruption version 更容易证明；真正 stochastic-interpolant version 更有新意但理论风险更高。

## 初步判断

最值得押注的主线是：

```text
Noisy endpoint-posterior velocity matching
+ ESS-aware/retrieval latent anchors
+ single-level velocity-to-generation analysis
```

这条线既能继承 MAGT 的 anchor/posterior 优势，又能与 Flow Matching、MeanFlow、PAFM 做出清晰区分。
