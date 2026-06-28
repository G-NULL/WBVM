# Exemplar Learning Dossier

## Exemplar Inventory

| Family | Representative Source | Why Selected | Pattern Learned |
|---|---|---|---|
| Flow Matching | Lipman et al., 2022/2023 | Canonical free vector-field objective for CNFs | Regress conditional vector fields from prescribed paths |
| Stochastic Interpolants | Albergo, Boffi, Vanden-Eijnden, 2023/2025 | General bridge language covering flows, diffusions, Schrodinger bridge | Conditional velocity is a projection of pathwise velocity |
| MAGT | Local `2602.19600v1` | Closest ancestor: one-shot low-dimensional transport with posterior anchors | Fixed-level posterior identity plus anchor approximation |
| Consistency / FMM / Shortcut | Song et al.; Boffi et al.; Frans et al. | One-step/few-step generation through map consistency or shortcuts | Compress dynamics into maps or step-conditioned updates |
| MeanFlow | Geng et al., 2025 | Strong recent one-step baseline | Average velocity instead of instantaneous velocity |
| PAFM | Stoica et al., 2026 | Closest external collision: posterior completions for FM | Replace single-target supervision with posterior-weighted target mixture |

## Structural Patterns

Strong recent papers in this area tend to follow one of three arcs:

1. **Change the supervised object.**  
   Flow Matching changes score learning into vector-field regression; MeanFlow changes instantaneous velocity into average velocity; PAFM changes single-target velocity supervision into posterior-augmented supervision. E-PVM should follow this pattern by changing the supervised object to an endpoint posterior mean induced by `h_theta`.

2. **Preserve fast sampling while moving complexity into training.**  
   Consistency, Shortcut, FMM, MeanFlow, and MAGT all pay extra training/theory complexity to reduce inference steps. E-PVM fits if sampling remains `x = h_theta(u)` and anchor work is training-time only.

3. **Use a bridge identity as the method's spine.**  
   The best positioning is not "we add anchors"; it is "the endpoint-induced velocity admits a posterior identity, and anchors are the tractable estimator of that identity."

## Rhetorical Patterns

Useful opening move:

```text
Existing flow/diffusion methods learn a time-dependent ambient velocity or score field.
But for low-dimensional data manifolds, a one-step generator already defines a restricted family of admissible velocity fields.
The question is how to train that generator without integrating the field.
```

Useful contrast:

```text
PAFM posteriorizes supervision for a free velocity model.
E-PVM posteriorizes the endpoint itself under a generator-induced bridge, so the learned velocity remains tied to a one-step transport map.
```

## Language Patterns

Use precise terms:

- `generator-induced velocity field`
- `endpoint posterior mean`
- `single-level velocity error`
- `finite-anchor posterior approximation`
- `ESS-aware proposal`
- `intrinsic-dimension-preserving base endpoint`

Avoid overclaiming:

- Do not claim general superiority over diffusion or MeanFlow before image-scale evidence.
- Do not claim full stochastic-interpolant theory unless the velocity pull-back theorem is proved.
- Do not present PAFM as unrelated; it is the nearest competitor and should be acknowledged directly.
