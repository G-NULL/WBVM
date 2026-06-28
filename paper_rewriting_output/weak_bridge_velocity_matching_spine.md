# Weak Bridge Velocity Matching Spine

## 1. Working Motivation

Current endpoint-posterior velocity matching learns a one-step generator by estimating a pointwise conditional velocity

\[
v_{\theta,\tau}(x)
=
\mathbb E[h_\theta(U)-X_0\mid X_{\theta,\tau}=x].
\]

The weak-form alternative changes the object being matched.  Instead of estimating this conditional expectation, it asks whether the generator-induced bridge and the data bridge satisfy the same continuity equation in weak form.  The central motivation is:

\[
\boxed{
\text{learn } h_\theta \text{ by aligning bridge fluxes, not by estimating pointwise conditional velocities.}
}
\]

This makes the direction genuinely different from anchor-based E-PVM.  Anchors are no longer the bottleneck; the bottleneck becomes the design of a stable test-function or critic class that can detect mismatched bridge flux.

## 2. Basic Bridge Setup

Data bridge:

\[
X_\tau^d=(1-\tau)X_0+\tau X_1,
\qquad
V^d=X_1-X_0.
\]

Generator-induced bridge:

\[
X_{\theta,\tau}=(1-\tau)X_0+\tau h_\theta(U),
\qquad
V^\theta=h_\theta(U)-X_0.
\]

Both are sample-level paths.  Their Markovian projected velocity fields would be

\[
v_{d,\tau}(x)=\mathbb E[V^d\mid X_\tau^d=x],
\qquad
v_{\theta,\tau}(x)=\mathbb E[V^\theta\mid X_{\theta,\tau}=x].
\]

E-PVM tries to approximate \(v_{\theta,\tau}\) or its endpoint posterior mean.  Weak Bridge Velocity Matching avoids that step.

## 3. Weak Continuity Equation

Let \(\rho_\tau^d\) and \(\rho_{\theta,\tau}\) be the laws of \(X_\tau^d\) and \(X_{\theta,\tau}\).  Formally,

\[
\partial_\tau \rho_\tau+\nabla\cdot(\rho_\tau v_\tau)=0.
\]

For a smooth test function \(\varphi:\mathbb R^D\to\mathbb R\),

\[
\frac{d}{d\tau}\mathbb E[\varphi(X_\tau)]
=
\mathbb E[\nabla\varphi(X_\tau)^\top V].
\]

Thus define the weak bridge flux discrepancy

\[
\Delta_\theta(\varphi,\tau)
=
\mathbb E[\nabla\varphi(X_{\theta,\tau})^\top V^\theta]
-
\mathbb E[\nabla\varphi(X_\tau^d)^\top V^d].
\]

If we also need to match the intermediate marginals, define

\[
M_\theta(\varphi,\tau)
=
\mathbb E[\varphi(X_{\theta,\tau})]
-
\mathbb E[\varphi(X_\tau^d)].
\]

A first population objective is

\[
\mathcal L_{\mathrm{WBVM}}(\theta)
=
\int_0^1
\sup_{\varphi\in\mathcal F}
\left\{
|M_\theta(\varphi,\tau)|^2
+
\lambda |\Delta_\theta(\varphi,\tau)|^2
\right\}
d\tau.
\]

For a fixed-level version, replace the integral by one selected \(\tau\):

\[
\mathcal L_{\mathrm{WBVM}}^\tau(\theta)
=
\sup_{\varphi\in\mathcal F}
\left\{
|M_\theta(\varphi,\tau)|^2
+
\lambda |\Delta_\theta(\varphi,\tau)|^2
\right\}.
\]

The all-time version has cleaner identifiability.  The fixed-level version is closer to the current one-step fixed-level story, but needs extra assumptions or a local time-consistency correction.

## 4. Critic / Test Function Design

The test-function class should be strong enough to detect flux mismatch but regular enough to train.  A Sobolev critic is natural:

\[
\|\varphi\|_{H^1(\bar\rho_\tau)}^2
=
\mathbb E_{\bar\rho_\tau}\|\nabla\varphi(X)\|^2
+
\epsilon\mathbb E_{\bar\rho_\tau}|\varphi(X)|^2,
\]

where

\[
\bar\rho_\tau
=
\frac12\rho_{\theta,\tau}+\frac12\rho_\tau^d.
\]

Then a squared flux IPM can be written as

\[
D_{\mathrm{flux}}^2(\theta,\tau)
=
\sup_{\varphi}
\left\{
2\Delta_\theta(\varphi,\tau)
-
\|\varphi\|_{H^1(\bar\rho_\tau)}^2
\right\}.
\]

This is the bridge-flux analogue of a Sobolev IPM.  Sobolev GAN uses a Sobolev-ball critic to compare distributions; here the same idea is applied to the vector-valued flux term
\(\rho_\tau v_\tau\), not merely to the marginal law.

Practical neural objective:

\[
\max_\eta
2\Delta_\theta(\varphi_\eta,\tau)
-
\alpha
\mathbb E_{\bar\rho_\tau}\|\nabla\varphi_\eta(X)\|^2
-
\epsilon
\mathbb E_{\bar\rho_\tau}|\varphi_\eta(X)|^2.
\]

Generator update:

\[
\min_\theta
D_{\mathrm{marg}}^2(\theta,\tau)
+
\lambda D_{\mathrm{flux}}^2(\theta,\tau).
\]

Here \(D_{\mathrm{marg}}\) can be MMD, sliced Wasserstein, a second Sobolev IPM term, or a simple critic on \(X_{\theta,\tau}\) versus \(X_\tau^d\).

## 5. Stronger Space-Time Version

For a space-time test function \(\psi(\tau,x)\), the weak path equation is

\[
\mathbb E\left[
\partial_\tau\psi(\tau,X_\tau)
+
\nabla_x\psi(\tau,X_\tau)^\top V
\right].
\]

This gives a path-level residual:

\[
\Delta_\theta^{\mathrm{st}}(\psi)
=
\int_0^1
\Big(
\mathbb E[
\partial_\tau\psi(\tau,X_{\theta,\tau})
+
\nabla_x\psi(\tau,X_{\theta,\tau})^\top V^\theta
]
-
\mathbb E[
\partial_\tau\psi(\tau,X_\tau^d)
+
\nabla_x\psi(\tau,X_\tau^d)^\top V^d
]
\Big)
d\tau.
\]

The space-time version is more faithful to the continuity equation and naturally supports a theorem:

\[
\Delta_\theta^{\mathrm{st}}(\psi)=0
\quad\forall \psi
\quad\Longrightarrow\quad
\rho_{\theta,\tau}=\rho_\tau^d
\quad\forall \tau
\]

provided the two bridges share the same initial law and the weak solution is unique.

## 6. Relationship to Current E-PVM

Pointwise velocity matching implies weak matching.  If

\[
\rho_{\theta,\tau}=\rho_\tau^d
\quad\text{and}\quad
v_{\theta,\tau}=v_{d,\tau}
\quad\rho_\tau\text{-a.e.},
\]

then

\[
\Delta_\theta(\varphi,\tau)=0
\qquad
\forall \varphi.
\]

But the converse only holds in a projected/divergence sense.  Weak matching controls

\[
\nabla\cdot(\rho_{\theta,\tau}v_{\theta,\tau}
-
\rho_\tau^d v_{d,\tau})
\]

rather than the pointwise difference \(v_{\theta,\tau}-v_{d,\tau}\).  This is precisely why WBVM is not just E-PVM in disguise.

Interpretation:

\[
\boxed{
\text{E-PVM asks: what is the average endpoint direction at }x?
}
\]

\[
\boxed{
\text{WBVM asks: do the two bridges move probability mass through every test surface in the same way?}
}
\]

## 7. Theory Targets

### Proposition 1: Population Identifiability

Assume \(X_{\theta,0}\overset d=X_0\overset d=X_0^d\).  If for all \(\tau\in[0,1]\) and all \(\varphi\in C_c^\infty(\mathbb R^D)\),

\[
\mathbb E[\nabla\varphi(X_{\theta,\tau})^\top V^\theta]
=
\mathbb E[\nabla\varphi(X_\tau^d)^\top V^d],
\]

and the associated continuity equation admits a unique weak solution, then

\[
\rho_{\theta,\tau}=\rho_\tau^d
\qquad
\forall \tau\in[0,1].
\]

In particular,

\[
h_\theta(U)=X_{\theta,1}
\overset d=
X_1.
\]

### Proposition 2: Negative Sobolev Error Bound

For a suitable Sobolev critic class,

\[
D_{\mathrm{flux}}(\theta,\tau)
\approx
\left\|
\nabla\cdot(\rho_{\theta,\tau}v_{\theta,\tau}
-
\rho_\tau^d v_{d,\tau})
\right\|_{H^{-1}(\bar\rho_\tau)}.
\]

An all-time integrated bound should have the form

\[
d_{\mathcal H}(\rho_{\theta,1},\rho_1^d)
\le
C
\int_0^1
\left[
D_{\mathrm{marg}}(\theta,\tau)
+
D_{\mathrm{flux}}(\theta,\tau)
\right]d\tau.
\]

This is the main theorem path.  It replaces the endpoint-posterior pull-back theorem by a weak continuity-equation stability theorem.

### Proposition 3: Fixed-Level Limitation

A single fixed \(\tau\) weak residual generally cannot identify the endpoint distribution without extra structure.  The paper should state this clearly and propose one of:

1. all-time integrated WBVM;
2. a small time-window \([\tau-\delta,\tau+\delta]\);
3. fixed-level marginal matching plus model-bridge linearity assumptions;
4. fixed-level WBVM used only as a regularizer for E-PVM.

This limitation is useful rhetorically because it explains why the original fixed-level endpoint-posterior theory is nontrivial.

## 8. Method Variants

### Variant A: All-Time WBVM

\[
\min_\theta
\int_0^1
\left[
D_{\mathrm{marg}}^2(\theta,\tau)
+
\lambda D_{\mathrm{flux}}^2(\theta,\tau)
\right]d\tau.
\]

Best for theory and clean positioning.

### Variant B: Fixed-Level WBVM

\[
\min_\theta
D_{\mathrm{marg}}^2(\theta,\tau_0)
+
\lambda D_{\mathrm{flux}}^2(\theta,\tau_0).
\]

Best for compatibility with the current MAGT/E-PVM fixed-level paper.  Theory weaker.

### Variant C: Local-Time WBVM

\[
\min_\theta
\mathbb E_{\tau\sim \mathrm{Unif}[\tau_0-\delta,\tau_0+\delta]}
\left[
D_{\mathrm{marg}}^2(\theta,\tau)
+
\lambda D_{\mathrm{flux}}^2(\theta,\tau)
\right].
\]

Best compromise: mostly single-level, but gives a local continuity signal.

### Variant D: Conditional / Guided WBVM

For conditional generation:

\[
X_{\theta,\tau}^c=(1-\tau)X_0+\tau h_\theta(U,c),
\qquad
V^{\theta,c}=h_\theta(U,c)-X_0.
\]

Match class/text-conditional bridge fluxes:

\[
\Delta_\theta(\varphi,\tau,c)
=
\mathbb E[\nabla\varphi(X_{\theta,\tau}^c)^\top V^{\theta,c}]
-
\mathbb E[\nabla\varphi(X_\tau^{d,c})^\top V^{d,c}].
\]

If a pretrained conditional teacher is available, its guided samples can define \(X_\tau^{d,c}\) or a teacher bridge, but the WBVM objective itself does not require a guided score formula.

## 9. Experimental Plan

### Synthetic First

Datasets:

- 2D moons / spiral / checkerboard.
- 3D helix embedded in \(D=50\) or \(D=100\).
- Torus or Swiss roll.

Baselines:

- free Flow Matching;
- Rectified Flow;
- MAGT / fixed-level denoising;
- E-PVM with prior anchors;
- WBVM variants A/B/C.

Metrics:

- endpoint \(W_2\) or sliced \(W_2\);
- MMD / energy distance;
- weak residual on held-out critics;
- path phase-space classifier accuracy for \((\tau,X_\tau,V)\);
- one-step sample quality.

### Structured Data

Use the same structured/scientific benchmark already planned for E-PVM if available.  WBVM should be tested in settings where anchor posterior estimation is hard; this is exactly where weak matching may help.

### Image-Scale Sanity

Use a small class subset first, not full text-to-image.  WBVM has an adversarial critic and may be unstable in high-dimensional images unless applied in latent space.

## 10. Closest Literature and Positioning

| Literature | What It Does | WBVM Difference |
|---|---|---|
| Flow Matching | Regresses a free neural vector field against a target conditional path velocity | WBVM trains a static generator by weakly matching bridge fluxes |
| Stochastic Interpolants | Derives velocity/score objectives for interpolating paths | WBVM uses the weak continuity equation as the learning object |
| Sobolev GAN / Sobolev IPM | Compares distributions using Sobolev-constrained critics | WBVM compares bridge fluxes, not only marginals |
| Lagrangian Flow Matching | Designs probability paths via least-action principles for free velocity fields | WBVM does not design the path; it matches the flux of a generator-induced bridge |
| PAFM | Uses posterior target completions to reduce FM supervision variance | WBVM does not posteriorize target completions; it avoids conditional posterior estimation |
| Augmented Bridge Matching | Preserves coupling information by augmenting velocity with source point | WBVM can incorporate augmentation, but its core object is weak bridge flux |
| Generator Matching | Matches infinitesimal Markov generators | WBVM can be seen as matching the deterministic transport generator in weak form |

## 11. Suggested Paper Arc

1. Start from the limitation of pointwise generator-induced velocity matching: it requires estimating endpoint posterior expectations.
2. Observe that bridges are probability flows governed by a continuity equation.
3. Replace pointwise velocity matching by weak flux matching over test functions.
4. Define WBVM objectives: all-time, fixed-level, and local-time variants.
5. Prove population identifiability for the all-time weak formulation.
6. Derive a negative-Sobolev stability bound.
7. Show empirically that WBVM avoids anchor collapse and gives robust one-step generation on low-dimensional manifolds.
8. Position E-PVM and WBVM as complementary:
   - E-PVM gives sharper pointwise supervision when conditional posterior estimation is accurate.
   - WBVM gives a posterior-free alternative when anchors are unreliable.

## 12. Recommended Claim Boundary

Safe claim:

\[
\boxed{
\text{WBVM is a posterior-free weak formulation for learning generator-induced bridges.}
}
\]

Avoid claiming:

- that weak matching always beats endpoint-posterior matching;
- that fixed-level WBVM alone identifies endpoints without assumptions;
- that the method scales to text-to-image without latent-space experiments;
- that WBVM is unrelated to Flow Matching or Sobolev IPMs.

Best short title:

\[
\textbf{Weak Bridge Velocity Matching for One-step Generative Transport}
\]

## 13. Three Concrete Critic Realizations

The weak discrepancy

\[
\Delta_\theta(\varphi,\tau)
=
\mathbb E[\nabla\varphi(X_{\theta,\tau})^\top V^\theta]
-
\mathbb E[\nabla\varphi(X_\tau^d)^\top V^d]
\]

can be realized in at least three different ways.  These should be treated as method variants rather than mere implementation details.

### 13.1 RKHS Derivative-Kernel WBVM

Let \(\mathcal H_k\) be a scalar RKHS with kernel \(k(x,x')\).  By the derivative reproducing property,

\[
\partial_i \varphi(x)
=
\langle \varphi,\partial_{x_i}k(x,\cdot)\rangle_{\mathcal H_k}.
\]

Therefore

\[
\nabla\varphi(x)^\top v
=
\left\langle
\varphi,\sum_{i=1}^D v_i\partial_{x_i}k(x,\cdot)
\right\rangle_{\mathcal H_k}.
\]

Define the flux mean embedding

\[
\mu_{\theta,\tau}^{\mathrm{flux}}
=
\mathbb E
\left[
\sum_{i=1}^D V_i^\theta
\partial_{x_i}k(X_{\theta,\tau},\cdot)
\right],
\]

\[
\mu_{d,\tau}^{\mathrm{flux}}
=
\mathbb E
\left[
\sum_{i=1}^D V_i^d
\partial_{x_i}k(X_\tau^d,\cdot)
\right].
\]

Then

\[
\Delta_\theta(\varphi,\tau)
=
\langle
\varphi,
\mu_{\theta,\tau}^{\mathrm{flux}}
-
\mu_{d,\tau}^{\mathrm{flux}}
\rangle_{\mathcal H_k}.
\]

Taking the supremum over the unit RKHS ball gives a closed-form discrepancy:

\[
D_{\mathrm{RKHS\text{-}flux}}^2(\theta,\tau)
=
\left\|
\mu_{\theta,\tau}^{\mathrm{flux}}
-
\mu_{d,\tau}^{\mathrm{flux}}
\right\|_{\mathcal H_k}^2.
\]

Equivalently,

\[
\begin{aligned}
D_{\mathrm{RKHS\text{-}flux}}^2
&=
\mathbb E[
(V^\theta)^\top
\nabla_x\nabla_{x'}k(X_{\theta,\tau},X_{\theta,\tau}')
V^{\theta\prime}]
\\
&\quad+
\mathbb E[
(V^d)^\top
\nabla_x\nabla_{x'}k(X_\tau^d,X_\tau^{d\prime})
V^{d\prime}]
\\
&\quad-
2\mathbb E[
(V^\theta)^\top
\nabla_x\nabla_{x'}k(X_{\theta,\tau},X_\tau^{d})
V^{d}].
\end{aligned}
\]

For the RBF kernel

\[
k_\sigma(x,x')
=
\exp\left(-\frac{\|x-x'\|^2}{2\sigma^2}\right),
\]

\[
\nabla_x\nabla_{x'}k_\sigma(x,x')
=
k_\sigma(x,x')
\left[
\frac{I_D}{\sigma^2}
-
\frac{(x-x')(x-x')^\top}{\sigma^4}
\right].
\]

For empirical estimation, the model bridge batch and the data bridge batch should be written separately.  Let

\[
\{(x_i^\theta,v_i^\theta)\}_{i=1}^{n_\theta},
\qquad
\{(x_j^d,v_j^d)\}_{j=1}^{n_d},
\qquad
H_k(x,x')=\nabla_x\nabla_{x'}k(x,x').
\]

The natural mini-batch V-statistic is

\[
\begin{aligned}
\widehat D_{\mathrm{RKHS\text{-}flux,V}}^2
&=
\frac{1}{n_\theta^2}
\sum_{i,i'=1}^{n_\theta}
(v_i^\theta)^\top
H_k(x_i^\theta,x_{i'}^\theta)
v_{i'}^\theta
\\
&\quad+
\frac{1}{n_d^2}
\sum_{j,j'=1}^{n_d}
(v_j^d)^\top
H_k(x_j^d,x_{j'}^d)
v_{j'}^d
\\
&\quad-
\frac{2}{n_\theta n_d}
\sum_{i=1}^{n_\theta}\sum_{j=1}^{n_d}
(v_i^\theta)^\top
H_k(x_i^\theta,x_j^d)
v_j^d .
\end{aligned}
\]

Equivalently, one may use the compact notation

\[
\widehat D_{\mathrm{RKHS\text{-}flux,V}}^2
=
\sum_{a,b}\tilde s_a\tilde s_b
v_a^\top H_k(x_a,x_b)v_b,
\]

but then the signs must absorb the sample-size weights:

\[
\tilde s_a=
\begin{cases}
1/n_\theta, & a\text{ is a model bridge sample},\\
-1/n_d, & a\text{ is a data bridge sample}.
\end{cases}
\]

Writing only \(s_a\in\{+1,-1\}\) has the wrong normalization unless the weights are implicitly included.

For \(x=x'\),

\[
H_{k_\sigma}(x,x)=\frac{I_D}{\sigma^2},
\qquad
v^\top H_{k_\sigma}(x,x)v
=
\frac{\|v\|^2}{\sigma^2}.
\]

Thus the diagonal terms of the V-statistic contain a velocity-norm contribution:

\[
\frac{1}{n_\theta^2}
\sum_i
\frac{\|v_i^\theta\|^2}{\sigma^2}
+
\frac{1}{n_d^2}
\sum_j
\frac{\|v_j^d\|^2}{\sigma^2}.
\]

A practical recommendation is to start with the mini-batch V-statistic because it is simple and usually lower variance.  If the diagonal term makes the loss dominated by velocity norms, switch to the U-statistic that removes the within-group diagonal terms:

\[
\begin{aligned}
\widehat D_{\mathrm{RKHS\text{-}flux,U}}^2
&=
\frac{1}{n_\theta(n_\theta-1)}
\sum_{i\ne i'}
(v_i^\theta)^\top
H_k(x_i^\theta,x_{i'}^\theta)
v_{i'}^\theta
\\
&\quad+
\frac{1}{n_d(n_d-1)}
\sum_{j\ne j'}
(v_j^d)^\top
H_k(x_j^d,x_{j'}^d)
v_{j'}^d
\\
&\quad-
\frac{2}{n_\theta n_d}
\sum_{i=1}^{n_\theta}\sum_{j=1}^{n_d}
(v_i^\theta)^\top
H_k(x_i^\theta,x_j^d)
v_j^d .
\end{aligned}
\]

Only the model-model and data-data diagonal terms are removed; the model-data cross term remains unchanged.

Recommended objective:

\[
\mathcal L_{\mathrm{RKHS\text{-}WBVM}}
=
\mathrm{MMD}_k^2(X_{\theta,\tau},X_\tau^d)
+
\lambda
D_{\mathrm{RKHS\text{-}flux}}^2(\theta,\tau).
\]

Why it is useful:

- no neural critic;
- closed-form weak flux discrepancy;
- direct theoretical connection to RKHS IPMs and kernel two-sample testing.

Main risks:

- \(O(B^2)\) minibatch cost;
- bandwidth sensitivity;
- Hessian kernel terms can be numerically noisy in high dimension.

Best first experiment:

Low-dimensional manifolds embedded in \(D=20\) or \(D=100\), using RBF multi-bandwidth kernels.

### 13.2 MMD / Vector-Measure WBVM

The derivative-kernel RKHS discrepancy matches weak divergence of flux.  A simpler MMD variant is to match the vector-valued flux measure directly:

\[
F_{\theta,\tau}
=
\rho_{\theta,\tau}(x)v_{\theta,\tau}(x),
\qquad
F_{d,\tau}
=
\rho_\tau^d(x)v_{d,\tau}(x).
\]

Use a vector-valued RKHS with matrix kernel

\[
K(x,x')=k(x,x')I_D.
\]

The vector MMD is

\[
D_{\mathrm{vMMD}}^2(\theta,\tau)
=
\left\|
\mathbb E[k(X_{\theta,\tau},\cdot)V^\theta]
-
\mathbb E[k(X_\tau^d,\cdot)V^d]
\right\|_{\mathcal H_K}^2.
\]

In closed form,

\[
\begin{aligned}
D_{\mathrm{vMMD}}^2
&=
\mathbb E[
(V^\theta)^\top V^{\theta\prime}
k(X_{\theta,\tau},X_{\theta,\tau}')]
\\
&\quad+
\mathbb E[
(V^d)^\top V^{d\prime}
k(X_\tau^d,X_\tau^{d\prime})]
\\
&\quad-
2\mathbb E[
(V^\theta)^\top V^d
k(X_{\theta,\tau},X_\tau^d)].
\end{aligned}
\]

This is not identical to weak divergence matching.  It is stronger in one sense because it compares the vector flux itself, but it may be too strict because different vector fields can induce the same continuity equation if their difference is divergence-free under the density.

Recommended objective:

\[
\mathcal L_{\mathrm{vMMD\text{-}WBVM}}
=
\mathrm{MMD}_k^2(X_{\theta,\tau},X_\tau^d)
+
\lambda D_{\mathrm{vMMD}}^2(\theta,\tau).
\]

A second MMD option is phase-space MMD.  Define

\[
Z_{\theta,\tau}=(X_{\theta,\tau},V^\theta),
\qquad
Z_\tau^d=(X_\tau^d,V^d).
\]

Then

\[
\mathcal L_{\mathrm{phase\text{-}MMD}}
=
\mathrm{MMD}_{\kappa}^2
\left(
\mathcal L(Z_{\theta,\tau}),
\mathcal L(Z_\tau^d)
\right),
\]

with a product kernel

\[
\kappa((x,v),(x',v'))
=
k_x(x,x')k_v(v,v').
\]

Why it is useful:

- no Hessian kernel terms;
- easy implementation;
- good diagnostic baseline for whether path phase-space statistics match.

Main risks:

- vector MMD may over-penalize harmless divergence-free velocity differences;
- phase-space MMD may become close to adversarial path matching rather than weak PDE matching;
- kernel choice over velocities can dominate the loss.

Best first experiment:

Compare three variants on the same toy manifolds:

\[
\mathrm{MMD}(X)
\quad\text{vs.}\quad
\mathrm{MMD}(X)+D_{\mathrm{vMMD}}
\quad\text{vs.}\quad
\mathrm{MMD}(X,V).
\]

The key metric is whether adding velocity/flux improves endpoint \(W_2\) beyond pure marginal MMD.

### 13.3 Sobolev Neural-Critic WBVM

The Sobolev version keeps the original weak formulation but learns the test function:

\[
D_{\mathrm{Sob\text{-}flux}}^2(\theta,\tau)
=
\sup_{\varphi}
\left\{
2\Delta_\theta(\varphi,\tau)
-
\mathbb E_{\bar\rho_\tau}\|\nabla\varphi(X)\|^2
-
\epsilon\mathbb E_{\bar\rho_\tau}|\varphi(X)|^2
\right\}.
\]

Here

\[
\bar\rho_\tau
=
\frac12\rho_{\theta,\tau}
+
\frac12\rho_\tau^d.
\]

With a neural critic \(\varphi_\eta\):

\[
\max_\eta
\left[
2\Delta_\theta(\varphi_\eta,\tau)
-
\mathbb E_{\bar\rho_\tau}\|\nabla\varphi_\eta(X)\|^2
-
\epsilon\mathbb E_{\bar\rho_\tau}|\varphi_\eta(X)|^2
\right].
\]

Then update the generator by minimizing

\[
\mathcal L_{\mathrm{Sob\text{-}WBVM}}
=
D_{\mathrm{marg}}^2(\theta,\tau)
+
\lambda D_{\mathrm{Sob\text{-}flux}}^2(\theta,\tau).
\]

This is the most flexible variant.  It approximates an \(H^{-1}\)-type norm of the continuity-equation residual:

\[
D_{\mathrm{Sob\text{-}flux}}(\theta,\tau)
\approx
\left\|
\nabla\cdot
(F_{\theta,\tau}-F_{d,\tau})
\right\|_{H^{-1}(\bar\rho_\tau)}.
\]

The optimal critic formally solves the elliptic weak problem

\[
-
\nabla\cdot(\bar\rho_\tau\nabla\varphi^\star)
+
\epsilon\bar\rho_\tau\varphi^\star
=
-
\nabla\cdot(F_{\theta,\tau}-F_{d,\tau}).
\]

Why it is useful:

- adaptive critic, less kernel bandwidth tuning;
- better high-dimensional potential, especially in latent image space;
- strongest conceptual match to weak continuity-equation residual.

Main risks:

- adversarial training instability;
- requires second-order autodiff through \(\nabla\varphi_\eta(X_{\theta,\tau})\) when updating \(\theta\);
- critic may exploit velocity scale artifacts unless normalized.

Stabilizers:

\[
V^\theta \leftarrow \frac{V^\theta}{\mathrm{stopgrad}(\sqrt{\mathbb E\|V^\theta\|^2}+\delta)},
\qquad
V^d \leftarrow \frac{V^d}{\mathrm{stopgrad}(\sqrt{\mathbb E\|V^d\|^2}+\delta)}.
\]

Use spectral normalization or gradient penalty on \(\varphi_\eta\), and train the critic for only a small number of inner steps.

Best first experiment:

Run Sobolev WBVM after RKHS/MMD baselines.  It should be treated as the scalable version, not the first proof-of-concept.

## 14. Which Variant Should Lead?

| Variant | Best Role | Strength | Weakness |
|---|---|---|---|
| RKHS derivative-kernel WBVM | Theory-first and clean synthetic experiments | exact closed-form weak flux discrepancy | \(O(B^2)\), bandwidth sensitivity |
| Vector MMD WBVM | Simple baseline | no Hessian, easy code | not exactly weak continuity residual |
| Phase-space MMD | Diagnostic baseline | checks \((x,v)\) path statistics | may be too strong / less PDE-grounded |
| Sobolev neural critic | Scalable method | adaptive and closest to \(H^{-1}\) residual | adversarial training and autodiff cost |

Recommended progression:

1. Start with RKHS derivative-kernel WBVM to establish that weak flux matching works.
2. Add vector/phase-space MMD as cheap baselines.
3. Use Sobolev critic only after the kernel version validates the idea.

If the paper needs one main method, lead with:

\[
\boxed{
\mathrm{MMD}(X_{\theta,\tau},X_\tau^d)
+
\lambda D_{\mathrm{RKHS\text{-}flux}}^2(\theta,\tau)
}
\]

and present Sobolev critic as the scalable extension.
