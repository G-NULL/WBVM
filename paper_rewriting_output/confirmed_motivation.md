# Confirmed Motivation

## Confirmation Status

User confirmed the main route:

```text
A + B
```

Updated route:

```text
Endpoint-posterior velocity identity from noisy/general stochastic interpolants
+ population theory for general stochastic interpolants
+ anchors left as an estimator-design choice
```

## Exact Confirmed Motivation

The paper will study a one-step generative transport method whose velocity field is not a free neural vector field, but is induced by a low-dimensional endpoint generator through an endpoint posterior identity.

The main methodological spine is to start from a general noisy stochastic interpolant:

\[
I_{\theta,t}
=
a_tX_0+b_t h_\theta(U)+\gamma_t\varepsilon,
\qquad
\varepsilon\sim\mathcal N(0,I_D),
\qquad
X_1^\theta=h_\theta(U).
\]

The induced velocity field is the conditional expectation of the pathwise velocity:

\[
v_{\theta,t}(x)
=
\mathbb E_\theta
\left[
\dot a_tX_0+\dot b_t h_\theta(U)+\dot\gamma_t\varepsilon
\mid
I_{\theta,t}=x
\right].
\]

When \(\gamma_t>0\), this admits an endpoint-posterior velocity identity. Since

\[
\varepsilon
=
\frac{x-a_tX_0-b_t h_\theta(U)}{\gamma_t},
\]

we have

\[
v_{\theta,t}(x)
=
\lambda_t x
+ A_t m_{\theta,t}^0(x)
+ B_t m_{\theta,t}^1(x),
\]

where

\[
\lambda_t=\frac{\dot\gamma_t}{\gamma_t},
\qquad
A_t=\dot a_t-\lambda_t a_t,
\qquad
B_t=\dot b_t-\lambda_t b_t,
\]

and

\[
m_{\theta,t}^0(x)
=
\mathbb E_\theta[X_0\mid I_{\theta,t}=x],
\qquad
m_{\theta,t}^1(x)
=
\mathbb E_\theta[h_\theta(U)\mid I_{\theta,t}=x].
\]

The Gaussian-base noisy bridge

\[
a_t=1-t,
\qquad
b_t=t,
\qquad
X_0\sim\mathcal N(0,I_D),
\qquad
\gamma_t^2=2t(1-t)
\]

is the clean special case where \(X_0\) can be integrated out and the velocity reduces to a latent endpoint posterior mean:

\[
p_\theta(u\mid x_t)
\propto
\pi(u)
\phi
\left(
x_t;t h_\theta(u),(1-t^2)I_D
\right),
\]

\[
v_{\theta,t}(x_t)
=
\frac{
m_{\theta,t}^1(x_t)-t x_t
}{
1-t^2
}.
\]

This special case is useful as the first tractable model, but the paper's theoretical target is broader: identify what must be proved for general stochastic interpolants.

## Theory Target for General Stochastic Interpolants

For a data interpolant

\[
I_t
=
a_tX_0+b_tX_1+\gamma_t\varepsilon,
\qquad
v_{\mathrm{data},t}(x)
=
\mathbb E
\left[
\dot a_tX_0+\dot b_tX_1+\dot\gamma_t\varepsilon
\mid
I_t=x
\right].
\]

the first population training objective is

\[
\mathcal L_{\mathrm{vel}}(\theta)
=
\mathbb E
\left[
\left\|
v_{\theta,\tau}(I_\tau)-\dot I_\tau
\right\|_2^2
\right].
\]

One must prove the orthogonal decomposition

\[
\mathcal L_{\mathrm{vel}}(\theta)
=
\mathcal L_{\mathrm{irr}}
+
\mathbb E_{I_\tau\sim p_\tau}
\left[
\left\|
v_{\theta,\tau}(I_\tau)-v_{\mathrm{data},\tau}(I_\tau)
\right\|_2^2
\right].
\]

The main theoretical contribution should then be a single-level velocity-to-generation theorem, ideally of the form

\[
W_2
\left(
P_{X_1},
P_{h_\theta(U)}
\right)
\le
C_\tau
\left\|
v_{\theta,\tau}
-
v_{\mathrm{data},\tau}
\right\|_{L^2(p_\tau)}
+
R_\tau,
\]

where \(R_\tau\) collects any necessary intermediate-density, regularity, or finite-estimation terms. A weaker theorem is acceptable if it clearly states the extra assumptions.

## Rejected / Deprioritized Options

| Option | Status | Reason |
|---|---|---|
| C. Retrieval-Residual E-PVM alone | Estimator option | Useful if anchors are chosen as the implementation route, but not a confirmed contribution |
| D. Iso-ESS Adaptive Tau | Estimator/training option | Helpful for robustness, but not part of the core theory statement |
| E. Shared-Latent Bridge | Future/high-risk | More original but adds identifiability and proof complexity |
| F. Endpoint Evidence for OOD | Extension | Interesting application, not the main contribution |
| G. Posterior-Rank Dimension Selection | Extension | Statistical side project, not central to velocity matching |
| H. Direct PAFM confrontation | Required positioning, not main route | Must be discussed, but the main paper should not become only a PAFM comparison |

## Scope Limits

The paper should not overclaim that Gaussian noisy E-PVM is entirely distinct from MAGT. In the Gaussian-corruption regime, the method is closely related to MAGT's fixed-level posterior score/denoiser identity.

The novelty should be framed as:

1. deriving the endpoint-posterior velocity identity for noisy/general stochastic interpolants;
2. showing exactly when the Gaussian-base case collapses to a MAGT-like posterior mean identity;
3. identifying what additional theory is required for general stochastic interpolants beyond MAGT's score pull-back;
4. establishing a single-level velocity-to-generation analysis, at least under a tractable general-SI subcase.

Anchors should be treated as a plug-in estimator of the endpoint posterior expectations, not as the central confirmed contribution.

## Forbidden Overclaims

- Do not claim that E-PVM is a completely new generative paradigm independent of MAGT.
- Do not claim a full stochastic-interpolant pull-back theorem unless it is actually proved.
- Do not claim superiority over Flow Matching, MeanFlow, PAFM, or diffusion models without direct experiments.
- Do not ignore PAFM; it is the closest posterior-supervision competitor and must be positioned explicitly.
- Do not present any particular anchor design as the main contribution until experiments show it is necessary.

## Working Paper Arc

1. Start from the limitation of free velocity fields for one-step generation on low-dimensional data.
2. Introduce generator-induced velocity fields through endpoint posterior means.
3. Derive the endpoint-posterior velocity identity for a general noisy stochastic interpolant.
4. Show the Gaussian-base noisy bridge as the clean MAGT-adjacent special case.
5. Prove the velocity loss decomposition.
6. Prove, or state sharply as the main theorem target, a single-level velocity pull-back / identifiability theorem.
7. Treat anchors as implementation choices satisfying a generic posterior-estimation error condition.
8. Validate on synthetic manifolds first, then structured/image benchmarks.
9. Compare against MAGT, Flow Matching, diffusion/DDIM, MeanFlow or shortcut-style baselines, and PAFM if feasible.
