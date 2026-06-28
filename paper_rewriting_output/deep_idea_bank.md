# Deep Idea Bank for Endpoint-Posterior Velocity Matching

## 1. ESS-Controlled Endpoint Proposal

Train `q_phi(u | x_tau, tau)` to sample near the endpoint posterior. Keep importance correction, but add ESS or weight-entropy as an auxiliary objective.

Why new: MAGT uses MC/QMC/MAP-Laplace anchors; E-PVM can make posterior efficiency a first-class training signal.

Minimal experiment: helix/torus/CIFAR10-0 with `K in {32,64,128}` comparing prior, MAP-Laplace, amortized ESS proposal.

Risk: proposal collapse and high-dimensional density estimation for `log q_phi`.

## 2. Retrieval-Residual Anchor Bank

Maintain a large codebook `{u_j, h_theta(u_j)}`. Given `x_tau`, retrieve top-K anchors by residual likelihood, then add local latent perturbations around retrieved anchors.

Why new: makes endpoint posterior estimation a memory/retrieval problem rather than pure Monte Carlo.

Minimal experiment: codebook size `M`, local perturbation radius, K, ESS, W2/FID, wall-clock.

Risk: stale codebook; rare modes may be under-retrieved.

## 3. Iso-ESS Adaptive Tau

Choose `tau(x)` or `tau(batch)` so posterior ESS stays near `rho K`. Generation remains one-step because sampling still uses `h_theta(U)`.

Why new: tau is selected by posterior estimability, not by arbitrary validation grid or diffusion schedule.

Minimal experiment: compare fixed tau, validation tau, and iso-ESS tau on synthetic manifolds and CIFAR airplane.

Risk: breaks the clean single-level proof and may over-smooth hard samples.

## 4. Local Tau-Consistency for Endpoint Posterior

Keep the main objective at one tau, but add a small-neighborhood consistency constraint between `m_{theta,tau}` and `m_{theta,tau+delta}` after analytic bridge correction.

Why new: consistency is applied to endpoint posterior means, not to denoising maps or flow maps.

Minimal experiment: low-K regime, measure gradient variance, ESS, W2/FID.

Risk: can average multi-modal posteriors too aggressively.

## 5. Shared-Latent Base Endpoint Bridge

Replace `X0 ~ N(0,I_D)` with `X0=g_psi(U)` or correlated `X0=g_psi(V)`. Both bridge endpoints are low-dimensional generated objects.

Why new: keeps the interpolant close to low-dimensional geometry, potentially preserving intrinsic-dimension rates.

Minimal experiment: Gaussian base vs independent learned base vs shared-latent base on helix/torus/Superconduct.

Risk: `g_psi` may become an unidentifiable second generator.

## 6. Curvature-Flattened Noisy Bridge

Generalize `gamma_tau^2=2 tau(1-tau)` to an ESS/curvature-aware `gamma_tau`, controlling posterior sharpness through `S_tau=(1-tau)^2+gamma_tau^2`.

Why new: bridge noise is designed around anchor posterior stability, not only stochastic-interpolant aesthetics.

Minimal experiment: no-noise, fixed noisy, and ESS-calibrated noisy bridge.

Risk: state-dependent gamma complicates the velocity target and proof.

## 7. Single-Level Velocity Pull-Back Bound

Prove a theorem of the form:

```text
W2(P_data, P_{h_theta(U)})
<= A(tau) * intermediate mismatch
 + B(tau) * ||v_data,tau - v_theta,tau||_{L2(p_tau)}.
```

Why new: MAGT has score pull-back; E-PVM needs velocity pull-back.

Minimal experiment: verify monotonic relation between fixed-tau velocity loss and endpoint W2 on known manifolds.

Risk: single-level velocity may not identify endpoint distribution without extra assumptions.

## 8. Posterior-Rank Intrinsic Dimension Selection

Use high-weight anchor covariance rank, local Jacobian spectrum, or latent gates to estimate/regularize effective dimension.

Why new: anchor posterior gives local dimension evidence that ordinary FM/consistency methods do not expose.

Minimal experiment: embed known `d=1,2,3` manifolds in `D=100`, train with oversized latent, test whether gates recover true dimension.

Risk: bad proposal can look artificially low-rank.

## 9. Endpoint Evidence for OOD and Likelihood

Estimate smoothed bridge evidence, intrinsic density, posterior ESS, and residual norms for test samples. Combine them into an OOD score.

Why new: one-step generator gets a tractable posterior/evidence diagnostic through anchors.

Minimal experiment: MNIST one digit vs others, CIFAR airplane vs non-airplane, Superconduct source split.

Risk: smoothed likelihood can suffer the typicality trap in high dimensions.

## 10. Scientific Constraint Anchor Proposals

Inject domain constraints into `q(u | x_tau)` or an anchor filter: allele frequency, composition constraints, conservation laws, valid molecular rules.

Why new: constraints enter posterior endpoint search rather than only the neural architecture or loss.

Minimal experiment: Genomes or Superconduct with generic vs constraint-aware anchors.

Risk: constraints may remove rare but valid samples.

## 11. PAFM-to-E-PVM Transfer Test

Construct a controlled benchmark where PAFM's posterior target mixture and E-PVM's latent endpoint posterior are both available.

Why new: directly clarifies whether generator-induced posterior velocities add value beyond posterior-augmented free FM supervision.

Minimal experiment: class-conditional MNIST/CIFAR subset, same candidate endpoint pool, compare free velocity PAFM vs one-step E-PVM.

Risk: if PAFM wins strongly on images, E-PVM must reposition toward manifold/scientific data.

## 12. Anchor Uncertainty as Diversity Control

At sampling/training time, use posterior entropy/ESS to modulate latent noise or choose between deterministic `h_theta(u)` and small local refinements.

Why new: diversity control is tied to posterior ambiguity, not classifier-free guidance or temperature.

Minimal experiment: measure precision/recall or diversity metrics under entropy-conditioned sampling.

Risk: adding refinement may dilute the one-step claim.
