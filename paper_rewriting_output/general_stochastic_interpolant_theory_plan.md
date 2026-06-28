# General Stochastic-Interpolant Theory Plan

## Main Shift

The main paper should not treat anchors as the defining novelty. The defining novelty should be:

\[
\boxed{
\text{endpoint-posterior velocity identity for generator-induced stochastic interpolants}
}
\]

and the theory should answer:

\[
\boxed{
\text{What does fixed-level velocity matching prove for endpoint generation?}
}
\]

Anchors are then one possible estimator for the posterior expectations appearing in the identity.

## 1. General Interpolant Setup

Let the data bridge be

\[
I_t
=
a_tX_0+b_tX_1+\gamma_t\varepsilon,
\qquad
\varepsilon\sim\mathcal N(0,I_D),
\]

and the model bridge be

\[
I_{\theta,t}
=
a_tX_0+b_t h_\theta(U)+\gamma_t\varepsilon,
\qquad
U\sim\pi.
\]

The pathwise velocities are

\[
\dot I_t
=
\dot a_tX_0+\dot b_tX_1+\dot\gamma_t\varepsilon,
\]

\[
\dot I_{\theta,t}
=
\dot a_tX_0+\dot b_t h_\theta(U)+\dot\gamma_t\varepsilon.
\]

The conditional velocities are

\[
v_{\mathrm{data},t}(x)
=
\mathbb E[\dot I_t\mid I_t=x],
\]

\[
v_{\theta,t}(x)
=
\mathbb E_\theta[\dot I_{\theta,t}\mid I_{\theta,t}=x].
\]

## 2. Endpoint-Posterior Velocity Identity

When \(\gamma_t>0\),

\[
\varepsilon
=
\frac{x-a_tX_0-b_tX_1}{\gamma_t}
\]

on the data bridge, and

\[
\varepsilon
=
\frac{x-a_tX_0-b_t h_\theta(U)}{\gamma_t}
\]

on the model bridge.

Define

\[
\lambda_t
=
\frac{\dot\gamma_t}{\gamma_t},
\qquad
A_t
=
\dot a_t-\lambda_ta_t,
\qquad
B_t
=
\dot b_t-\lambda_tb_t.
\]

Then

\[
v_{\mathrm{data},t}(x)
=
\lambda_t x
+A_t m_{\mathrm{data},t}^0(x)
+B_t m_{\mathrm{data},t}^1(x),
\]

where

\[
m_{\mathrm{data},t}^0(x)
=
\mathbb E[X_0\mid I_t=x],
\qquad
m_{\mathrm{data},t}^1(x)
=
\mathbb E[X_1\mid I_t=x].
\]

Similarly,

\[
v_{\theta,t}(x)
=
\lambda_t x
+A_t m_{\theta,t}^0(x)
+B_t m_{\theta,t}^1(x),
\]

where

\[
m_{\theta,t}^0(x)
=
\mathbb E_\theta[X_0\mid I_{\theta,t}=x],
\qquad
m_{\theta,t}^1(x)
=
\mathbb E_\theta[h_\theta(U)\mid I_{\theta,t}=x].
\]

This is the central identity.

## 3. What Must Be Proved

### 3.1 Conditional-Expectation Calibration

The first theorem is the standard projection identity:

\[
\mathbb E
\left[
\left\|
v_{\theta,\tau}(I_\tau)-\dot I_\tau
\right\|^2
\right]
=
\mathbb E
\left[
\left\|
v_{\mathrm{data},\tau}(I_\tau)-\dot I_\tau
\right\|^2
\right]
+
\mathbb E_{p_\tau}
\left[
\left\|
v_{\theta,\tau}-v_{\mathrm{data},\tau}
\right\|^2
\right].
\]

This proves that the population velocity objective is calibrated to the fixed-level conditional velocity mismatch.

### 3.2 Fixed-Level Identifiability

For general stochastic interpolants, the hard question is:

\[
v_{\theta,\tau}
\approx
v_{\mathrm{data},\tau}
\quad
\Longrightarrow
\quad
P_{h_\theta(U)}
\approx
P_{X_1}.
\]

This is not automatic. A theorem must state conditions under which the map

\[
P_{X_1}
\mapsto
v_{\mathrm{data},\tau}
\]

is stable or locally identifiable.

A possible target theorem is:

\[
W_2(P_{X_1},P_{h_\theta(U)})
\le
C_\tau
\left\|
v_{\theta,\tau}-v_{\mathrm{data},\tau}
\right\|_{L^2(p_\tau)}
+R_\tau.
\]

The residual \(R_\tau\) may include an intermediate-density discrepancy, model-class approximation error, or regularity constants.

### 3.3 Single-Level Pull-Back

The continuity equations are

\[
\partial_t p_t+\nabla\cdot(p_t v_{\mathrm{data},t})=0,
\]

\[
\partial_t p_{\theta,t}+\nabla\cdot(p_{\theta,t} v_{\theta,t})=0.
\]

Multi-time velocity control would give a standard stability result:

\[
W_2(P_{X_1},P_{h_\theta(U)})
\lesssim
W_2(p_\tau,p_{\theta,\tau})
+
\int_\tau^1
\left\|
v_{\mathrm{data},t}-v_{\theta,t}
\right\|_{L^2(p_t)}
\,dt.
\]

But the desired result is single-level. Therefore one needs a growth or propagation lemma:

\[
\int_\tau^1
\left\|
v_{\mathrm{data},t}-v_{\theta,t}
\right\|_{L^2(p_t)}
\,dt
\le
C_\tau
\left\|
v_{\mathrm{data},\tau}-v_{\theta,\tau}
\right\|_{L^2(p_\tau)}
+R_\tau.
\]

This is the velocity analogue of the score-gap growth machinery used by MAGT.

### 3.4 Intermediate Density Issue

Unlike score matching, velocity matching at one level does not automatically yield a Fisher divergence. Thus one must handle

\[
W_2(p_\tau,p_{\theta,\tau})
\]

or avoid it through a direct identifiability argument.

Possible routes:

1. prove that fixed-level velocity equality implies fixed-level density equality under the model bridge;
2. include \(W_2(p_\tau,p_{\theta,\tau})\) as an explicit residual term;
3. add a weak density/evidence matching condition at level \(\tau\);
4. restrict to a Gaussian-base subcase where velocity posterior means determine the smoothed law.

### 3.5 Regularity and Geometry Assumptions

The theorem will likely need assumptions such as:

\[
a_t,b_t,\gamma_t\in C^2,
\qquad
\gamma_t>0
\quad
\text{for }t\in(0,1),
\]

bounded moments of \(X_0,X_1,h_\theta(U)\), Lipschitz or Sobolev regularity of the conditional velocities, and nondegeneracy of the intermediate densities.

For manifold-supported data, additional assumptions may be needed:

- positive reach or tubular neighborhood;
- bounded curvature;
- regular generator \(h_\theta\);
- control of constants as \(\gamma_t\to0\) or \(t\to1\).

## 4. Anchor Estimation as an Undecided Plug-In

Anchors should be framed through a generic estimator condition:

\[
\mathbb E_{p_\tau}
\left[
\left\|
\widehat v_{\theta,K,\tau}(X)
-
v_{\theta,\tau}(X)
\right\|^2
\right]
\le
\varepsilon_K^2.
\]

Any concrete anchor scheme only needs to provide a bound or empirical evidence for \(\varepsilon_K\).

Reference options:

1. **Prior anchors**

\[
u_k\sim\pi(u).
\]

Simple baseline, likely low ESS near endpoints.

2. **QMC anchors**

Low-discrepancy latent anchors for moderate intrinsic dimension.

3. **MAP/Laplace anchors**

Find a local posterior mode and sample around a Gaussian approximation.

4. **Retrieval anchors**

Maintain a codebook

\[
\mathcal C=\{u_j,h_\theta(u_j)\}_{j=1}^M
\]

and retrieve top-\(K\) likely endpoints for each \(x_\tau\).

5. **Amortized proposal**

Train

\[
q_\phi(u\mid x_\tau,\tau)
\]

with importance correction.

6. **Mixture proposal**

\[
q(u\mid x_\tau)
=
\lambda\pi(u)+(1-\lambda)q_\phi(u\mid x_\tau).
\]

Useful for coverage and mode protection.

These should remain options until experiments decide which one is worth making a contribution.

## 5. Suggested Paper Claim

The clean claim is:

\[
\boxed{
\text{We derive generator-induced endpoint-posterior velocity identities for noisy stochastic interpolants}
}
\]

and

\[
\boxed{
\text{we identify and partially solve the single-level velocity pull-back problem needed for one-step generation.}
}
\]

The anchor estimator is presented as an implementation layer, not the central novelty.
