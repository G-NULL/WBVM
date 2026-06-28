# Endpoint-Only Discrepancy Search for WBVM

## Question

For `weak_bridge_velocity_matching_spine_talk.pdf`, can the weak bridge flux discrepancy be stripped of the stochastic/interpolant process and written only on the endpoint \(X_1\)? Are there existing papers or methods doing this?

## Short Answer

Yes, but it changes the object.

The WBVM flux discrepancy is a path-level continuity-equation discrepancy:

\[
\Delta_\theta(\varphi,\tau)
=
\mathbb E[\nabla\varphi(X_{\theta,\tau})^\top V^\theta]
-
\mathbb E[\nabla\varphi(X_\tau^d)^\top V^d].
\]

It compares signed/vector fluxes along a bridge. If the bridge is removed and only the endpoint is kept, the natural object becomes an endpoint distribution discrepancy:

\[
D_{\mathcal F}(P_\theta,P_{\rm data})
=
\sup_{f\in\mathcal F,\ \|f\|_{\mathcal F}\le 1}
\left|
\mathbb E[f(h_\theta(U))]
-
\mathbb E[f(X_1)]
\right|.
\]

This is no longer a flux discrepancy. It is an IPM / MMD / Wasserstein / energy / adversarial distribution matching objective.

## What Cannot Be Kept

Without a path \(X_t\) and velocity \(V_t=\dot X_t\), there is no canonical flux

\[
\rho_t v_t
\]

and no continuity-equation weak residual

\[
\frac{d}{dt}\mathbb E[\varphi(X_t)]
-
\mathbb E[\nabla\varphi(X_t)^\top V_t].
\]

At \(X_1\) alone, any formula involving

\[
\mathbb E[\nabla\varphi(X_1)^\top V]
\]

requires choosing an extra endpoint vector field \(V\). That vector field is not intrinsic unless one introduces an OT map, a score, a Stein operator, or a generator-parameter tangent.

## Endpoint Limit of the Linear Bridge

There is one important nuance. If we do not completely discard the bridge, but instead take the endpoint limit of the linear bridge used in WBVM, then a purely endpoint-looking formula does appear.

For the linear bridge,

\[
X_{\theta,\tau}=(1-\tau)X_0+\tau h_\theta(U),
\qquad
V^\theta=h_\theta(U)-X_0,
\]

\[
X_\tau^d=(1-\tau)X_0+\tau X_1,
\qquad
V^d=X_1-X_0.
\]

At \(\tau=1\),

\[
X_{\theta,1}=h_\theta(U),
\qquad
X_1^d=X_1,
\]

and the weak flux discrepancy becomes

\[
\Delta_\theta(\varphi,1)
=
\mathbb E
\left[
\nabla\varphi(h_\theta(U))^\top
\{h_\theta(U)-X_0\}
\right]
-
\mathbb E
\left[
\nabla\varphi(X_1)^\top
\{X_1-X_0\}
\right].
\]

If \(X_0\) is independent of both endpoints and \(\mathbb E[X_0]=0\), this reduces to

\[
\Delta_\theta(\varphi,1)
=
\mathbb E
\left[
\nabla\varphi(h_\theta(U))^\top h_\theta(U)
\right]
-
\mathbb E
\left[
\nabla\varphi(X_1)^\top X_1
\right].
\]

Thus an endpoint-only derivative discrepancy can be defined:

\[
D_{\rm end\text{-}grad}^2(P_\theta,P_{\rm data})
=
\sup_{\|\varphi\|_{\mathcal F}\le1}
\left|
\mathbb E_{P_\theta}
\left[
X^\top\nabla\varphi(X)
\right]
-
\mathbb E_{P_{\rm data}}
\left[
X^\top\nabla\varphi(X)
\right]
\right|^2.
\]

Equivalently, with the differential operator

\[
L\varphi(x)=x^\top\nabla\varphi(x),
\]

this is an IPM over the transformed function class \(L\mathcal F\):

\[
D_{\rm end\text{-}grad}(P_\theta,P_{\rm data})
=
\sup_{\|\varphi\|_{\mathcal F}\le1}
\left|
\mathbb E_{P_\theta}[L\varphi(X)]
-
\mathbb E_{P_{\rm data}}[L\varphi(X)]
\right|.
\]

This is not the original bridge flux anymore; it is the endpoint action of the dilation vector field \(x\). In distributional form,

\[
\mathbb E[X^\top\nabla\varphi(X)]
=
-\langle \varphi,\nabla\cdot(xP)\rangle,
\]

so the discrepancy measures

\[
\nabla\cdot\{x(P_\theta-P_{\rm data})\}
\]

in a negative Sobolev / RKHS dual norm. This connects it to Sobolev IPMs and Stein-type operator discrepancies.

If \(\mathcal F\) is an RKHS with kernel \(k\), then

\[
\mathbb E[X^\top\nabla\varphi(X)]
=
\left\langle
\varphi,
\mathbb E[X^\top\nabla_x k(X,\cdot)]
\right\rangle_{\mathcal H},
\]

and the closed-form squared discrepancy is

\[
\begin{aligned}
D_{\rm end\text{-}grad}^2
&=
\mathbb E_{X,X'\sim P_\theta}
\left[
X^\top\nabla_x\nabla_{x'}k(X,X')X'
\right]
\\
&\quad+
\mathbb E_{Y,Y'\sim P_{\rm data}}
\left[
Y^\top\nabla_y\nabla_{y'}k(Y,Y')Y'
\right]
\\
&\quad-
2\mathbb E_{X\sim P_\theta,Y\sim P_{\rm data}}
\left[
X^\top\nabla_x\nabla_y k(X,Y)Y
\right].
\end{aligned}
\]

This is the closest endpoint-only analogue of the derivative-kernel WBVM formula.

However, it is best viewed as an endpoint Sobolev/Stein-like IPM, not as a new flow-matching objective. It inherits a vector field \(x\) from the chosen bridge; a different bridge would produce a different endpoint operator.

## Endpoint-Only Families

### 1. IPM / GAN / WGAN

Endpoint-only objective:

\[
D_{\mathcal F}(P_\theta,P_{\rm data})
=
\sup_{f\in\mathcal F}
\left\{
\mathbb E_{X_1\sim P_{\rm data}}f(X_1)
-
\mathbb E_{U\sim\pi}f(h_\theta(U))
\right\}.
\]

If \(\mathcal F\) is a 1-Lipschitz class, this is Wasserstein-1:

\[
W_1(P_\theta,P_{\rm data})
=
\sup_{\|f\|_{\rm Lip}\le1}
\left[
\mathbb E f(X_1)-\mathbb E f(h_\theta(U))
\right].
\]

Representative methods:

- WGAN.
- Fisher GAN.
- Sobolev GAN.

### 2. MMD / GMMN / MMD-GAN

Endpoint-only kernel discrepancy:

\[
\mathrm{MMD}^2(P_\theta,P_{\rm data})
=
\left\|
\mathbb E k(h_\theta(U),\cdot)
-
\mathbb E k(X_1,\cdot)
\right\|_{\mathcal H}^2.
\]

Closed form:

\[
\begin{aligned}
\mathrm{MMD}^2
&=
\mathbb E k(X_1,X_1')
+
\mathbb E k(h_\theta(U),h_\theta(U'))
\\
&\quad
-2\mathbb E k(X_1,h_\theta(U)).
\end{aligned}
\]

Representative methods:

- Generative Moment Matching Networks.
- MMD-GAN.

This is the cleanest endpoint-only analogue of the RKHS part of WBVM, but it discards velocity/flux.

### 3. Energy Distance / Cramer GAN

Endpoint-only energy distance:

\[
\mathcal E(P_\theta,P_{\rm data})
=
2\mathbb E\|X_1-h_\theta(U)\|
-
\mathbb E\|X_1-X_1'\|
-
\mathbb E\|h_\theta(U)-h_\theta(U')\|.
\]

This is also an endpoint-only two-sample discrepancy.

### 4. Sinkhorn / OT Endpoint Matching

Endpoint-only entropic OT:

\[
\mathrm{OT}_\varepsilon(P_\theta,P_{\rm data})
=
\min_{\pi\in\Pi(P_\theta,P_{\rm data})}
\mathbb E_{\pi}[c(x,y)]
+
\varepsilon\,\mathrm{KL}(\pi\|P_\theta\otimes P_{\rm data}).
\]

The Sinkhorn divergence corrects entropic bias and gives a differentiable endpoint distribution loss. This is closer to transport geometry than MMD, but not to WBVM flux.

### 5. Stein / Score-Based Endpoint Discrepancy

If a score for the target endpoint distribution is available,

\[
s_{\rm data}(x)=\nabla\log p_{\rm data}(x),
\]

then one can define a Stein discrepancy:

\[
\mathcal S(P_\theta,P_{\rm data})
=
\sup_{\|g\|_{\mathcal G}\le1}
\left|
\mathbb E_{X\sim P_\theta}
\left[
\nabla\cdot g(X)+g(X)^\top s_{\rm data}(X)
\right]
\right|.
\]

This has an endpoint vector-field flavor. It does not need data samples inside the expectation if the target score is known or learned. Related methods include Kernel Stein Discrepancy and measure transport with KSD.

### 6. Distribution Matching Distillation

Recent one-step diffusion distillation methods train an endpoint generator by matching distributions through score differences:

\[
\nabla_\theta D(P_\theta\|P_{\rm data})
\quad
\text{is approximated using}
\quad
s_\theta(x)-s_{\rm data}(x).
\]

Representative methods:

- Diff-Instruct.
- Distribution Matching Distillation (DMD).
- DMD2.

This is probably the closest modern endpoint-only analogue if the goal is a one-step generator.

## Answer to the Main Question

If "similar formula" means a weak dual discrepancy with test functions, then yes:

\[
\sup_{\|f\|\le1}
\left|
\mathbb E f(h_\theta(U))-\mathbb E f(X_1)
\right|
\]

is exactly the endpoint-only analogue. It is already a large existing literature: IPM, MMD, WGAN, Sinkhorn, energy distance, Sobolev GAN.

If "similar formula" means a flux/continuity-equation residual, then no: not without adding an extra endpoint vector field, score, or transport map. WBVM's flux term is intrinsically path-level.

## Suggested Positioning

Do not pitch endpoint-only WBVM as new. It will be read as MMD/WGAN/Sinkhorn/Stein/DMD depending on the test class.

The defensible positioning is:

\[
\boxed{
\text{WBVM is not endpoint distribution matching; it is path-level weak continuity-equation matching.}
}
\]

Then use endpoint-only methods as baselines:

1. MMD / MMD-GAN;
2. WGAN / Sobolev GAN;
3. Sinkhorn divergence;
4. energy distance / Cramer;
5. DMD if using pretrained diffusion scores.

## Practical Recommendation

For your project:

- If you want the cleanest endpoint-only baseline, use MMD or sliced Wasserstein.
- If you want the strongest endpoint-only one-step generator competitor, use WGAN/Sinkhorn or DMD-style score-difference matching.
- If you want to keep a "weak PDE" interpretation, do not remove the bridge; instead use all-time WBVM or explicitly add an endpoint validation loss.

## Representative References

Endpoint-only distribution matching:

- Generative Moment Matching Networks: https://arxiv.org/abs/1502.02761
- MMD GAN: https://arxiv.org/abs/1705.08584
- Wasserstein GAN: https://arxiv.org/abs/1701.07875
- Learning Generative Models with Sinkhorn Divergences: https://arxiv.org/abs/1706.00292
- Sobolev GAN: https://arxiv.org/abs/1711.04894
- Fisher GAN: https://arxiv.org/abs/1705.09675
- Cramer distance / Cramer GAN: https://arxiv.org/abs/1705.10743
- Sliced-Wasserstein Autoencoder: https://arxiv.org/abs/1804.01947

Endpoint score / Stein / distillation directions:

- Kernelized Stein Discrepancy: https://arxiv.org/abs/1602.03253
- Diff-Instruct: https://arxiv.org/abs/2305.18455
- One-step Diffusion with Distribution Matching Distillation: https://arxiv.org/abs/2311.18828
- Improved Distribution Matching Distillation / DMD2: https://arxiv.org/abs/2405.14867
