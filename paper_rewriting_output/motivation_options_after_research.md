# Motivation Options After Research

## Shortlist

| Option | One-Sentence Motivation | Core Innovation | Why It Is Not Overbroad | Required Evidence | Best-Fit Paper Arc |
|---|---|---|---|---|---|
| A. Noisy/General E-PVM Identity | Derive endpoint-posterior velocity identities for generator-induced noisy stochastic interpolants. | General SI identity plus Gaussian-base tractable special case | Focuses on the population identity, not a fixed anchor design | identity check, velocity loss calibration, special-case reduction to MAGT-like posterior mean | Method/theory paper |
| B. Single-Level Velocity Pull-Back | Establish when fixed-tau velocity mismatch controls endpoint generation error. | New velocity-to-generation theorem | Pure theory target, can use simplified Gaussian-base setting | Theorem + synthetic monotonicity experiments | Theory-first paper |
| C. Retrieval-Residual E-PVM | Turn endpoint posterior approximation into a large-memory retrieval problem over generated endpoints. | Codebook top-K plus residual latent perturbation | Narrow algorithmic improvement to anchors | K/M ablations, ESS, speed, rare-mode recall | Practical algorithm paper |
| D. Iso-ESS Tau Control | Choose tau by posterior estimability rather than a fixed hand-tuned smoothing level. | Tau controller targeting ESS band | Single control variable, still one-step sampling | Tau distribution, stability, W2/FID | Robust training paper |
| E. Shared-Latent Bridge | Replace full-dimensional base endpoints by low-dimensional or shared-latent endpoints to preserve manifold geometry. | `X0=g_psi(U)` or correlated `X0=g_psi(V)` | Specific bridge design, not general FM | Intrinsic dimension, off-manifold rate, proof sketch | Geometry/manifold paper |
| F. Endpoint Evidence for OOD | Use anchor posterior evidence, ESS, and residuals as likelihood/OOD diagnostics for one-step generators. | Repurpose posterior normalization as evidence | Evaluation-focused extension | AUROC/AUPRC, calibration, typicality tests | Applied diagnostics paper |
| G. Posterior-Rank Dimension Selection | Infer effective latent dimension from high-weight anchor posterior covariance and Jacobian spectrum. | Dimension selection from endpoint posterior | Targets one practical pain point | Synthetic true-d recovery, Genomes/Superconduct | Statistical learning paper |
| H. PAFM-Differentiated One-Step E-PVM | Compete directly with PAFM by showing posterior supervision can train a generator-induced one-step transport, not just a free FM vector field. | The posterior is over latent endpoints of `h_theta`, not external target completions | Directly addresses the closest collision | PAFM-style ablations plus one-step NFE=1 | Positioning-heavy ML paper |

## Updated Recommended Route

The strongest first paper route is now a theory/method route:

```text
Endpoint-posterior velocity identity for noisy/general stochastic interpolants
+ general-SI single-level velocity theory
+ anchors as undecided plug-in estimators
```

This keeps the novelty away from "MAGT plus a new anchor trick." The main technical question becomes what fixed-level velocity matching proves for endpoint generation under a general stochastic interpolant.

## Anchor Options Only

Anchors should remain optional until experiments decide which estimator is worth elevating:

- Prior anchors: simplest baseline.
- QMC anchors: variance reduction for moderate latent dimension.
- MAP/Laplace anchors: local posterior approximation.
- Retrieval anchors: codebook top-\(K\) plus optional local perturbation.
- Amortized proposal: learned \(q_\phi(u\mid x_\tau,\tau)\) with importance correction.
- Mixture proposal: prior plus learned proposal for mode coverage.

## Highest-Risk Highest-Reward Route

```text
Shared-latent stochastic-interpolant E-PVM with intrinsic-dimension velocity pull-back.
```

This would be the most original, but it needs careful identifiability control and a new proof route.
