# Phase 3 — Five-seed fixed-state PathMNIST gradient diagnostics

Original full models at initialization and round 50; identical classifier/decoder/private encoder states for the original x and fused x+D(E_i(x)) branches. Five fixed training batches of 128 per client. Float32 raw gradients, CPU Float64 means/Gram reductions, no optimization or test access. Primary mode: native eval BN. Sensitivity mode: batch-stateless BN, buffers restored before/between/after branches. Neither changes stored BN statistics.

gᵒᵢ=∂θ CE(C(x),y), gᶠᵢ=∂θ CE(C(x+D(E_i(x))),y), bᵢ=gᶠᵢ−gᵒᵢ. Γ=(1/10)Σ||gᵢ−mean(g)||²; B=(1/10)Σ||bᵢ−mean(b)||²; Φ=(2/10)Σ〈gᵒᵢ−mean(gᵒ),bᵢ−mean(b)〉. Exact identity Γf=Γo+B+Φ is audited. The five batch gradients are averaged **before** client dispersion; per-batch dispersion is separately retained. The population client divisor is 10; seed SD uses ddof=1.

## initialization — eval

| Seed | Γ original | Γ fused | B | Φ | Γf/Γo | Γ decoder |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 5.662241600e-01 | 6.001539554e-01 | 8.897064674e-02 | -5.504085134e-02 | 1.059923 | 1.172533117e-03 |
| 43 | 6.469239076e-01 | 7.141771269e-01 | 4.805496188e-02 | 1.919825741e-02 | 1.103958 | 6.857435643e-04 |
| 44 | 9.836691163e-01 | 1.037422559e+00 | 2.446990963e-02 | 2.928353278e-02 | 1.054646 | 1.322235671e-03 |
| 45 | 9.309800539e-01 | 1.032010201e+00 | 4.218801423e-02 | 5.884213274e-02 | 1.108520 | 1.129223840e-03 |
| 46 | 1.466749144e+00 | 1.663262839e+00 | 4.139908770e-02 | 1.551146077e-01 | 1.133979 | 2.003482621e-03 |

Γf/Γo 1.092205 ± 0.033920; reduction in 0/5 seeds. Normalized dispersion original→fused 0.989786→0.989559; mean pairwise cosine -0.100951→-0.101489.

## initialization — batch-stateless

| Seed | Γ original | Γ fused | B | Φ | Γf/Γo | Γ decoder |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 7.218934370e+00 | 7.359893176e+00 | 1.162446754e+00 | -1.021487948e+00 | 1.019526 | 2.981622779e-03 |
| 43 | 6.130611208e+00 | 6.192391964e+00 | 6.272167594e-01 | -5.654360031e-01 | 1.010077 | 1.398225536e-03 |
| 44 | 6.643824311e+00 | 6.547469598e+00 | 2.918217940e-01 | -3.881765079e-01 | 0.985497 | 1.851972378e-03 |
| 45 | 5.813235852e+00 | 5.809427953e+00 | 4.403941051e-01 | -4.442020035e-01 | 0.999345 | 2.985792792e-03 |
| 46 | 6.502990974e+00 | 6.469731990e+00 | 1.013948571e+00 | -1.047207555e+00 | 0.994886 | 6.376035863e-03 |

Γf/Γo 1.001866 ± 0.013251; reduction in 3/5 seeds. Normalized dispersion original→fused 0.917749→0.917668; mean pairwise cosine -0.028905→-0.029099.

## final-round50 — eval

| Seed | Γ original | Γ fused | B | Φ | Γf/Γo | Γ decoder |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 3.206712991e+03 | 8.651488482e+03 | 1.041398182e+04 | -4.969206331e+03 | 2.697930 | 6.639030577e+03 |
| 43 | 1.151540518e+04 | 2.282897241e+03 | 1.250673766e+04 | -2.173924560e+04 | 0.198247 | 1.663482435e+02 |
| 44 | 2.674920257e+03 | 1.959428623e+03 | 3.147247274e+03 | -3.862738908e+03 | 0.732519 | 9.899409918e+02 |
| 45 | 4.185418790e+03 | 2.761099627e+03 | 4.786175729e+03 | -6.210494893e+03 | 0.659695 | 1.131963912e+02 |
| 46 | 4.914834227e+03 | 2.038599318e+03 | 4.944537764e+03 | -7.820772674e+03 | 0.414785 | 9.125941232e+01 |

Γf/Γo 0.940635 ± 1.004737; reduction in 4/5 seeds. Normalized dispersion original→fused 0.700936→0.822183; mean pairwise cosine 0.221775→0.093693.

## final-round50 — batch-stateless

| Seed | Γ original | Γ fused | B | Φ | Γf/Γo | Γ decoder |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 9.749477654e+02 | 5.707793011e+02 | 4.215673580e+02 | -8.257358222e+02 | 0.585446 | 4.722384341e-01 |
| 43 | 7.513013812e+02 | 7.356121684e+02 | 8.531202509e+01 | -1.010012380e+02 | 0.979117 | 1.123656180e+02 |
| 44 | 8.539784139e+02 | 7.080224673e+02 | 1.155677845e+02 | -2.615237311e+02 | 0.829087 | 3.543186547e+02 |
| 45 | 8.849318428e+02 | 7.328770715e+02 | 1.101494854e+02 | -2.622042567e+02 | 0.828173 | 1.494294228e+00 |
| 46 | 9.102319907e+02 | 8.907183612e+02 | 1.474199773e+02 | -1.669336068e+02 | 0.978562 | 2.100219051e+00 |

Γf/Γo 0.840077 ± 0.160942; reduction in 5/5 seeds. Normalized dispersion original→fused 0.888618→0.894538; mean pairwise cosine -0.017369→-0.028022.

## Interpretation, controls and comparison with Digits

All parameters and BN buffers are unchanged by paired measurement; RNG unchanged. Raw gradient norms/decoder dispersion, per-client/per-batch records and Gram matrices are archived. A raw Γ decrease can follow smaller gradients rather than better angular alignment: inspect normalized dispersion and cosines together. Batch-stateless BN couples examples, so it is a distinct batch-conditioned gradient objective; native eval remains the inference-aligned measurement.

Digits final models: native-eval Γf/Γo 0.791656 ± 0.253408, batch-stateless 0.148367 ± 0.140864. Native normalized dispersion 0.794160→0.799588 and mean pairwise cosine 0.090559→0.043966 already showed why raw scale reduction alone does not establish alignment. PathMNIST uses a different classifier, label-skewed clients, persistent optimizers and uncalibrated original settings. These are descriptive cross-benchmark observations, not a controlled causal comparison.

The final classifier was trained on fused inputs. Its original-input branch is a counterfactual at the same weights, not an independently trained raw-input classifier. Native BN statistics also reflect fused training inputs. Consequently a smaller fused gradient can reflect loss calibration or an input-distribution shift, beyond any alignment effect. Batch-stateless BN is a distinct objective and cannot fully remove that interpretation limit.

Two anchors and finite training probes do not describe every round or the population objective. Five seeds do not justify a universal reduction claim. No tuning or additional generalization evaluation was performed; the manuscript is unchanged.

Phase calendar 23.185 s; sum process time 93.378 s. Per-seed costs/memory in artifacts/summary.json. Configuration, exact probe IDs and anchor hashes: ../plan.json, ../probe.json, ../anchors.json. Executed commands and exit codes: artifacts/campaign.json.
