# SOTA Gap Map

| Candidate Contribution | What SOTA Already Does | User Evidence | Real Gap | Claim Strength | Risk |
|---|---|---|---|---|---|
| Endpoint-posterior velocity matching | FM regresses free vector fields along paths; SI gives conditional velocity objectives | Roadmap derives `v_theta,tau` from `E[h(U)|X_tau=x]` | Velocity is restricted to a one-step generator-induced family | Strong if framed against FM/PAFM | Needs clear empirical win beyond rephrasing |
| Gaussian-base noisy E-PVM | MAGT handles Gaussian corruption score; SI handles noisy bridges | Roadmap derives `p(u|x_tau)` with denominator `1-tau^2` | Low-dimensional anchor estimator for noisy endpoint velocity | Strong as first method variant | May look too close to MAGT unless velocity story is central |
| ESS-aware proposal anchors | MAGT uses MC/QMC/MAP-Laplace; PAFM uses importance mixtures over targets | Roadmap highlights ESS collapse near `tau -> 1` | Proposal design is treated as a controlled posterior-estimation problem | Very strong algorithm contribution | Learned proposal can collapse or bias finite-K gradients |
| Retrieval-residual anchor bank | Existing anchors are often sampled or optimized online | Roadmap suggests retrieval anchors | Treat posterior estimation as memory/retrieval over generated endpoints | Strong, easy to test | Codebook staleness and mode imbalance |
| Iso-ESS adaptive tau | MAGT validates fixed smoothing; FM/MeanFlow model time continuously | Roadmap mentions adaptive tau but prefers single tau | Tau chosen by posterior estimability rather than arbitrary schedule | Medium-high | Weakens single-level proof and one-step narrative |
| Shared-latent base endpoint | Standard bridges start from full Gaussian noise; geometric FM uses manifolds but not endpoint posterior anchors | Roadmap proposes `X0=g_psi(U)` or correlated `V,U` | Preserve low-dimensional structure on both bridge endpoints | High novelty | Identifiability and extra generator complexity |
| Single-level velocity pull-back theorem | MAGT has score pull-back; FM theory often uses all-time velocity error | Local proof note says velocity pull-back is missing | Fixed tau velocity error controlling endpoint W2 | Highest theory value | Hard; constants may explode |
| Posterior-rank intrinsic dimension selection | MAGT assumes/chooses latent dimension | Anchor posterior covariance gives local dimension clues | Learn/diagnose intrinsic dimension from posterior anchors | Medium-high | Bad proposals mimic low rank |
| Endpoint evidence for OOD/likelihood | MAGT mentions intrinsic/smoothed density; flow/consistency often lack tractable likelihood | E-PVM anchor weights estimate partition/evidence | Use posterior evidence + ESS + residual as OOD score | Medium | High-dimensional likelihood typicality trap |
| Scientific constraint anchor proposals | Generic generative models learn constraints implicitly | Roadmap already targets manifolds and structured data | Put physical/biological validity into posterior proposal/filter | Strong for applied paper | Constraints may suppress rare valid samples |

## Gap Summary

Most promising paper gap:

```text
PAFM posteriorizes free flow-matching supervision.
E-PVM posteriorizes endpoint velocity under a one-step generator-induced bridge.
```

Most promising theory gap:

```text
MAGT has fixed-level score pull-back.
E-PVM needs fixed-level velocity pull-back, preferably with intrinsic-dimension dependence.
```

Most promising algorithm gap:

```text
Anchor ESS is not just a diagnostic; it can control proposal, tau, bridge noise, and retrieval.
```
