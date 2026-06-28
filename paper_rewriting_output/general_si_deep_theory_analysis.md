# Deep Theory Analysis: General Stochastic-Interpolant E-PVM

## 0. Main Takeaway

For a general stochastic interpolant, the endpoint-posterior velocity identity is clean, but fixed-level velocity matching alone is usually not enough to prove endpoint generation accuracy.

The reason is structural:

\[
v_t(x)
=
\lambda_t x
+A_t m_t^0(x)
+B_t m_t^1(x)
\]

is only one vector equation, while the posterior contains two unknown endpoint posterior means:

\[
m_t^0(x)=\mathbb E[X_0\mid I_t=x],
\qquad
m_t^1(x)=\mathbb E[X_1\mid I_t=x].
\]

Thus the theory must either:

1. choose a special bridge/base where \(m_t^0\) can be integrated out;
2. add an intermediate density/score control that gives a second equation;
3. prove a new single-level velocity identifiability theorem under strong structural assumptions.

This should be the central theoretical framing.

## 1. General Identity

Let

\[
I_t=a_tX_0+b_tX_1+\gamma_t\varepsilon,
\qquad
\varepsilon\sim\mathcal N(0,I_D),
\]

and model

\[
I_{\theta,t}=a_tX_0+b_t h_\theta(U)+\gamma_t\varepsilon.
\]

Define

\[
\lambda_t=\frac{\dot\gamma_t}{\gamma_t},
\qquad
A_t=\dot a_t-\lambda_t a_t,
\qquad
B_t=\dot b_t-\lambda_t b_t.
\]

Then the data velocity is

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

The model velocity is

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

This identity is worth stating as the first theorem.

## 2. Why General SI Is Harder Than MAGT

MAGT's Gaussian smoothing effectively has one posterior mean:

\[
Y_t=\alpha_tX+\sigma_tZ,
\qquad
m_t(x)=\mathbb E[X\mid Y_t=x].
\]

The score satisfies a Tweedie identity:

\[
s_t(x)
=
\nabla\log p_t(x)
=
-\frac{x-\alpha_t m_t(x)}{\sigma_t^2}.
\]

Therefore controlling the posterior mean is equivalent to controlling the score, and the proof chain is:

\[
\text{posterior mean error}
\Longleftrightarrow
\text{score/Fisher error}
\Longrightarrow
W_2(p_t,p_{\theta,t})
\Longrightarrow
W_2(P_X,P_{h_\theta(U)}).
\]

For a general stochastic interpolant, however,

\[
v_t(x)
=
\lambda_t x+A_t m_t^0(x)+B_t m_t^1(x)
\]

does not directly give a Fisher divergence. It mixes the base posterior and endpoint posterior. This is the main theoretical obstruction.

## 3. A Useful Second Equation: Score Closure

The intermediate score is

\[
s_t(x)
=
\nabla\log p_t(x)
=
-\frac{
x-a_t m_t^0(x)-b_t m_t^1(x)
}{
\gamma_t^2
}.
\]

Equivalently,

\[
a_t m_t^0(x)+b_t m_t^1(x)
=
x+\gamma_t^2 s_t(x).
\]

Together with

\[
A_t m_t^0(x)+B_t m_t^1(x)
=
v_t(x)-\lambda_t x,
\]

we get a two-equation system for \((m_t^0,m_t^1)\):

\[
\begin{pmatrix}
a_t & b_t\\
A_t & B_t
\end{pmatrix}
\begin{pmatrix}
m_t^0(x)\\
m_t^1(x)
\end{pmatrix}
=
\begin{pmatrix}
x+\gamma_t^2s_t(x)\\
v_t(x)-\lambda_t x
\end{pmatrix}.
\]

The determinant is

\[
\Delta_t
=
a_tB_t-b_tA_t
=
a_t\dot b_t-b_t\dot a_t.
\]

Thus, if

\[
\Delta_t\ne0,
\]

then the pair \((s_t,v_t)\) identifies both posterior means.

This gives an important theorem route:

\[
\text{velocity mismatch}
+
\text{score/density mismatch}
\Longrightarrow
\text{endpoint posterior mismatch}.
\]

For the endpoint posterior mean, explicitly:

\[
m_t^1(x)
=
\frac{
a_t\{v_t(x)-\lambda_t x\}
-A_t\{x+\gamma_t^2s_t(x)\}
}{
\Delta_t
}.
\]

Therefore,

\[
\|m_{\theta,t}^1-m_{\mathrm{data},t}^1\|
\le
\frac{|a_t|}{|\Delta_t|}
\|v_{\theta,t}-v_{\mathrm{data},t}\|
+
\frac{|A_t|\gamma_t^2}{|\Delta_t|}
\|s_{\theta,t}-s_{\mathrm{data},t}\|.
\]

This is a very useful bridge between general SI velocity theory and MAGT-style score theory.

## 4. Gaussian-Base Reduction: The MAGT-Adjacent Special Case

Suppose

\[
X_0\sim\mathcal N(0,I_D),
\qquad
X_0\perp X_1,
\]

and

\[
I_t=a_tX_0+b_tX_1+\gamma_t\varepsilon.
\]

Let

\[
S_t=a_t^2+\gamma_t^2.
\]

Conditioned on \(X_1=y\),

\[
I_t\mid X_1=y
\sim
\mathcal N(b_ty,S_tI_D).
\]

Thus the bridge is just a Gaussian corruption of the endpoint \(X_1\), with signal coefficient \(b_t\) and noise variance \(S_t\). Moreover,

\[
m_t^0(x)
=
\frac{a_t}{S_t}
\left(
x-b_tm_t^1(x)
\right).
\]

Substitute this into the velocity identity:

\[
v_t(x)
=
D_t x+C_t m_t^1(x),
\]

where

\[
D_t
=
\lambda_t+\frac{A_ta_t}{S_t},
\qquad
C_t
=
B_t-\frac{A_ta_tb_t}{S_t}.
\]

If

\[
C_t\ne0,
\]

then

\[
m_t^1(x)
=
\frac{v_t(x)-D_tx}{C_t}.
\]

In this special case, velocity matching is equivalent to endpoint posterior mean matching. The score is also

\[
s_t(x)
=
-\frac{x-b_tm_t^1(x)}{S_t}.
\]

So

\[
v_{\theta,t}-v_{\mathrm{data},t}
=
C_t
\left(
m_{\theta,t}^1-m_{\mathrm{data},t}^1
\right)
=
-\frac{C_tS_t}{b_t}
\left(
s_{\theta,t}-s_{\mathrm{data},t}
\right),
\]

assuming \(b_t\ne0\).

This is the clean subcase where the theory can inherit MAGT almost directly.

For the roadmap's linear noisy choice,

\[
a_t=1-t,
\qquad
b_t=t,
\qquad
\gamma_t^2=2t(1-t),
\]

we have

\[
S_t=1-t^2,
\]

and the velocity simplifies to

\[
v_t(x)
=
\frac{m_t^1(x)-tx}{1-t^2}.
\]

This special case should be presented as the first tractable theorem, not as the whole novelty.

## 5. What a General SI Theory Must Prove

### Theorem 1: Endpoint-Posterior Velocity Identity

Prove:

\[
v_{\theta,t}(x)
=
\lambda_t x
+A_t m_{\theta,t}^0(x)
+B_t m_{\theta,t}^1(x).
\]

This is mostly algebra, but it establishes the conceptual object.

### Theorem 2: Population Velocity Calibration

For any measurable \(v_{\theta,\tau}\),

\[
\mathbb E
\left[
\|v_{\theta,\tau}(I_\tau)-\dot I_\tau\|^2
\right]
=
\mathbb E
\left[
\|v_{\mathrm{data},\tau}(I_\tau)-\dot I_\tau\|^2
\right]
+
\mathbb E_{p_\tau}
\left[
\|v_{\theta,\tau}-v_{\mathrm{data},\tau}\|^2
\right].
\]

This says the velocity objective is calibrated to fixed-level conditional velocity error.

### Theorem 3: Gaussian-Base Velocity-Score Equivalence

Under Gaussian base and independent endpoint coupling,

\[
\|v_{\theta,\tau}-v_{\mathrm{data},\tau}\|_{L^2(p_\tau)}^2
\asymp
\|s_{\theta,\tau}-s_{\mathrm{data},\tau}\|_{L^2(p_\tau)}^2,
\]

with explicit constants determined by \(a_\tau,b_\tau,\gamma_\tau\).

This theorem explains precisely why the Gaussian noisy case is close to MAGT.

### Theorem 4: General SI Closure With Score/Density Residual

For general base/interpolant, prove something like:

\[
\|m_{\theta,\tau}^1-m_{\mathrm{data},\tau}^1\|_{L^2(p_\tau)}
\le
C_\tau
\left(
\|v_{\theta,\tau}-v_{\mathrm{data},\tau}\|_{L^2(p_\tau)}
+
\gamma_\tau^2
\|s_{\theta,\tau}-s_{\mathrm{data},\tau}\|_{L^2(p_\tau)}
\right),
\]

provided

\[
\Delta_\tau=a_\tau\dot b_\tau-b_\tau\dot a_\tau
\ne0.
\]

This is not yet endpoint \(W_2\), but it is the missing algebraic closure.

### Theorem 5: Single-Level Velocity Pull-Back

The ambitious target is:

\[
W_2(P_{X_1},P_{h_\theta(U)})
\le
C_\tau
\|v_{\theta,\tau}-v_{\mathrm{data},\tau}\|_{L^2(p_\tau)}
+
R_\tau.
\]

The residual \(R_\tau\) should be made explicit. Likely candidates:

\[
R_\tau
=
C_\tau'
W_2(p_{\theta,\tau},p_\tau),
\]

or

\[
R_\tau
=
C_\tau'
\|s_{\theta,\tau}-s_{\mathrm{data},\tau}\|_{L^2(p_\tau)}.
\]

The pure velocity-only version is the hardest and may be false without extra assumptions.

### Theorem 6: Finite Estimator Perturbation

For anchors or any estimator,

\[
\mathbb E_{p_\tau}
\left[
\|\widehat v_{\theta,K,\tau}-v_{\theta,\tau}\|^2
\right]
\le
\varepsilon_K^2.
\]

Then the population theorem receives an additive perturbation:

\[
W_2(P_{X_1},P_{h_\theta(U)})
\le
C_\tau
\left(
\widehat{\mathcal E}_{\mathrm{vel}}^{1/2}
+
\varepsilon_K
\right)
+
R_\tau.
\]

This keeps anchors out of the main theorem until a concrete estimator is selected.

## 6. Recommended Proof Strategy

### Stage A: Safe Theorem

Prove the Gaussian-base reduction:

\[
X_0\sim\mathcal N(0,I_D)
\quad
\Longrightarrow
\quad
v_t(x)=D_tx+C_tm_t^1(x).
\]

Then show velocity matching is equivalent to score/denoiser matching and inherits MAGT-style pull-back.

This theorem is low risk but MAGT-adjacent.

### Stage B: General Identity and Closure

Prove the two-equation closure:

\[
(s_t,v_t)
\Longleftrightarrow
(m_t^0,m_t^1)
\quad
\text{if}
\quad
\Delta_t\ne0.
\]

This is genuinely general stochastic-interpolant theory and clarifies why velocity alone is insufficient.

### Stage C: Partial Pull-Back

Prove a theorem with an explicit residual:

\[
W_2(P_{X_1},P_{h_\theta(U)})
\le
C_\tau
\|v_{\theta,\tau}-v_{\mathrm{data},\tau}\|_{L^2(p_\tau)}
+
C_\tau'
W_2(p_{\theta,\tau},p_\tau).
\]

This is easier and honest: fixed-level velocity needs intermediate distribution alignment unless the Gaussian-base reduction applies.

### Stage D: Pure Velocity Pull-Back, If Possible

Try to remove the residual using a growth lemma:

\[
\|v_{\theta,t}-v_{\mathrm{data},t}\|_{L^2(p_t)}
\le
G(t,\tau)
\|v_{\theta,\tau}-v_{\mathrm{data},\tau}\|_{L^2(p_\tau)}.
\]

This likely requires strong assumptions:

\[
\|\nabla v_t\|_{\infty}\le L_t,
\qquad
\|\nabla^2\log p_t\|_{\infty}\le H_t,
\]

plus nondegenerate density and manifold tube conditions.

## 7. Bridge-Design Insight

The coefficients

\[
A_t=\dot a_t-\lambda_ta_t,
\qquad
B_t=\dot b_t-\lambda_tb_t
\]

determine how much the velocity depends on base posterior versus endpoint posterior.

If one can choose a schedule with small \(|A_\tau|\), then

\[
v_{\theta,\tau}(x)
\approx
\lambda_\tau x+B_\tau m_{\theta,\tau}^1(x),
\]

which makes endpoint recovery easier.

The condition

\[
A_\tau=0
\]

means

\[
\frac{\dot a_\tau}{a_\tau}
=
\frac{\dot\gamma_\tau}{\gamma_\tau}.
\]

This may be impossible globally with endpoint boundary conditions, but it can guide a single-level bridge design. This is a potential theoretical design knob distinct from anchors.

## 8. Suggested Revised Contribution Statement

The paper should claim:

1. We derive endpoint-posterior velocity identities for generator-induced noisy stochastic interpolants.
2. We show that the Gaussian-base noisy bridge reduces to a MAGT-like endpoint posterior mean and admits velocity-score equivalence.
3. For general stochastic interpolants, we prove that velocity alone mixes base and data endpoint posteriors; a score/density closure or special bridge design is required.
4. We formulate and partially prove a single-level velocity pull-back theorem with explicit residual terms.
5. We treat anchors as a generic plug-in estimator satisfying an \(L^2\) velocity approximation condition.

This is a much cleaner and more defensible theory story than making anchor design the primary novelty.
