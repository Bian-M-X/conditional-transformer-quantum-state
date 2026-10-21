# Round 2 descriptive results

Five paired seeds; mean +/- sample SD. No confidence intervals or significance claims.
Grid values within one seed are correlated. Stage A is development data.

## A: direct low-field training fidelity

| Architecture | Mode | Budget | F at h=0.2 | Seed range |
|---|---|---:|---:|---:|
| mlp | even | 600 | 0.999793 +/- 0.000108 | 0.999675 to 0.999892 |
| mlp | even | 1200 | 0.999879 +/- 5.27e-05 | 0.999826 to 0.999956 |
| mlp | even | 2400 | 0.999879 +/- 5.27e-05 | 0.999826 to 0.999956 |
| mlp | raw | 600 | 0.528694 +/- 0.00559 | 0.521398 to 0.534996 |
| mlp | raw | 1200 | 0.528694 +/- 0.00559 | 0.521398 to 0.534996 |
| mlp | raw | 2400 | 0.528694 +/- 0.00559 | 0.521398 to 0.534996 |
| transformer | even | 600 | 0.999552 +/- 0.000624 | 0.998466 to 0.999981 |
| transformer | even | 1200 | 0.99964 +/- 0.000659 | 0.998466 to 0.999996 |
| transformer | even | 2400 | 0.999686 +/- 0.000682 | 0.998466 to 0.999999 |
| transformer | raw | 600 | 0.655996 +/- 0.22 | 0.50045 to 0.999808 |
| transformer | raw | 1200 | 0.663477 +/- 0.225 | 0.500129 to 0.999892 |
| transformer | raw | 2400 | 0.659149 +/- 0.222 | 0.500102 to 0.999892 |

## B: held-out mean infidelity

| Architecture | Policy | Predictor | Mean infidelity | Mean worst infidelity |
|---|---|---|---:|---:|
| mlp | continue | covariant_average | 0.244545 +/- 0.000407 | 0.4759 +/- 0.000299 |
| mlp | continue | direct | 0.500262 +/- 0.000359 | 0.999377 +/- 1.91e-05 |
| mlp | continue | lookup | 0.500262 +/- 0.000357 | 0.99938 +/- 1.67e-05 |
| mlp | continue | sign_map | 0.306899 +/- 0.288 | 0.585961 +/- 0.231 |
| mlp | continue | train_span_ritz | 0.500258 +/- 0.000359 | 0.999377 +/- 1.86e-05 |
| mlp | log | covariant_average | 0.244532 +/- 0.000416 | 0.475883 +/- 0.000301 |
| mlp | log | direct | 0.50026 +/- 0.000359 | 0.999377 +/- 1.86e-05 |
| mlp | log | lookup | 0.50026 +/- 0.000359 | 0.999377 +/- 1.86e-05 |
| mlp | log | sign_map | 0.306893 +/- 0.288 | 0.585958 +/- 0.231 |
| mlp | log | train_span_ritz | 0.500214 +/- 0.000392 | 0.999379 +/- 1.56e-05 |
| mlp | random | covariant_average | 0.244543 +/- 0.000412 | 0.475896 +/- 0.000303 |
| mlp | random | direct | 0.500261 +/- 0.000359 | 0.999377 +/- 1.9e-05 |
| mlp | random | lookup | 0.500261 +/- 0.000358 | 0.999379 +/- 1.72e-05 |
| mlp | random | sign_map | 0.306897 +/- 0.288 | 0.58596 +/- 0.231 |
| mlp | random | train_span_ritz | 0.500221 +/- 0.00038 | 0.999379 +/- 1.52e-05 |
| mlp | ritz_gain | covariant_average | 0.244595 +/- 0.000384 | 0.475932 +/- 0.000282 |
| mlp | ritz_gain | direct | 0.500306 +/- 0.00033 | 0.999379 +/- 1.71e-05 |
| mlp | ritz_gain | lookup | 0.500306 +/- 0.00033 | 0.999379 +/- 1.71e-05 |
| mlp | ritz_gain | sign_map | 0.306959 +/- 0.288 | 0.58602 +/- 0.231 |
| mlp | ritz_gain | train_span_ritz | 0.500294 +/- 0.000358 | 0.999383 +/- 1.51e-05 |
| mlp | uniform | covariant_average | 0.244532 +/- 0.000414 | 0.475883 +/- 0.0003 |
| mlp | uniform | direct | 0.50026 +/- 0.000357 | 0.999377 +/- 1.84e-05 |
| mlp | uniform | lookup | 0.50026 +/- 0.000357 | 0.999377 +/- 1.84e-05 |
| mlp | uniform | sign_map | 0.306893 +/- 0.288 | 0.585959 +/- 0.231 |
| mlp | uniform | train_span_ritz | 0.500224 +/- 0.00043 | 0.999379 +/- 2.23e-05 |
| mlp | variance | covariant_average | 0.244552 +/- 0.0004 | 0.475899 +/- 0.000291 |
| mlp | variance | direct | 0.500277 +/- 0.000343 | 0.999378 +/- 1.74e-05 |
| mlp | variance | lookup | 0.500277 +/- 0.000343 | 0.999378 +/- 1.74e-05 |
| mlp | variance | sign_map | 0.306916 +/- 0.288 | 0.585984 +/- 0.231 |
| mlp | variance | train_span_ritz | 0.500221 +/- 0.000445 | 0.999378 +/- 2.45e-05 |
| transformer | continue | covariant_average | 0.24377 +/- 2.24e-05 | 0.475456 +/- 5.12e-05 |
| transformer | continue | direct | 0.498972 +/- 0.000436 | 0.99929 +/- 4.39e-05 |
| transformer | continue | lookup | 0.498972 +/- 0.000436 | 0.99929 +/- 4.41e-05 |
| transformer | continue | sign_map | 0.43447 +/- 0.353 | 0.688028 +/- 0.284 |
| transformer | continue | train_span_ritz | 0.499006 +/- 0.000398 | 0.99929 +/- 4.37e-05 |
| transformer | log | covariant_average | 0.243764 +/- 2.37e-05 | 0.475437 +/- 5.13e-05 |
| transformer | log | direct | 0.499004 +/- 0.000409 | 0.999293 +/- 4.12e-05 |
| transformer | log | lookup | 0.499004 +/- 0.000409 | 0.999293 +/- 4.12e-05 |
| transformer | log | sign_map | 0.434507 +/- 0.353 | 0.688063 +/- 0.284 |
| transformer | log | train_span_ritz | 0.498943 +/- 0.00046 | 0.999284 +/- 4.77e-05 |
| transformer | random | covariant_average | 0.243769 +/- 1.97e-05 | 0.475459 +/- 4.75e-05 |
| transformer | random | direct | 0.49898 +/- 0.000435 | 0.999291 +/- 4.36e-05 |
| transformer | random | lookup | 0.498981 +/- 0.000434 | 0.999291 +/- 4.37e-05 |
| transformer | random | sign_map | 0.434481 +/- 0.353 | 0.688046 +/- 0.284 |
| transformer | random | train_span_ritz | 0.498981 +/- 0.000249 | 0.999279 +/- 4.55e-05 |
| transformer | ritz_gain | covariant_average | 0.243765 +/- 2.45e-05 | 0.475447 +/- 5.55e-05 |
| transformer | ritz_gain | direct | 0.498965 +/- 0.000445 | 0.999288 +/- 4.41e-05 |
| transformer | ritz_gain | lookup | 0.498965 +/- 0.000445 | 0.999288 +/- 4.41e-05 |
| transformer | ritz_gain | sign_map | 0.434472 +/- 0.353 | 0.687987 +/- 0.284 |
| transformer | ritz_gain | train_span_ritz | 0.498885 +/- 0.000345 | 0.999255 +/- 9.28e-05 |
| transformer | uniform | covariant_average | 0.243763 +/- 2.38e-05 | 0.475432 +/- 5.03e-05 |
| transformer | uniform | direct | 0.499007 +/- 0.000402 | 0.999293 +/- 4.05e-05 |
| transformer | uniform | lookup | 0.499007 +/- 0.000402 | 0.999293 +/- 4.05e-05 |
| transformer | uniform | sign_map | 0.434511 +/- 0.353 | 0.68806 +/- 0.284 |
| transformer | uniform | train_span_ritz | 0.498939 +/- 0.000413 | 0.999285 +/- 4.26e-05 |
| transformer | variance | covariant_average | 0.243765 +/- 2.92e-05 | 0.475433 +/- 4.85e-05 |
| transformer | variance | direct | 0.498959 +/- 0.000446 | 0.999288 +/- 4.57e-05 |
| transformer | variance | lookup | 0.498959 +/- 0.000446 | 0.999288 +/- 4.57e-05 |
| transformer | variance | sign_map | 0.434465 +/- 0.353 | 0.68797 +/- 0.284 |
| transformer | variance | train_span_ritz | 0.498929 +/- 0.000438 | 0.999282 +/- 4.69e-05 |

## Paired comparisons

Negative delta is better for the candidate policy. Wins count seeds; no p-values.

| Architecture | Predictor | Policy | Reference | Mean delta | Wins / 5 |
|---|---|---|---|---:|---:|
| mlp | direct | random | continue | -6.87446e-07 | 1 |
| mlp | direct | random | uniform | 5.95433e-07 | 3 |
| mlp | direct | random | log | 9.10373e-07 | 2 |
| mlp | direct | variance | continue | 1.58181e-05 | 1 |
| mlp | direct | variance | uniform | 1.7101e-05 | 0 |
| mlp | direct | variance | log | 1.7416e-05 | 1 |
| mlp | direct | ritz_gain | continue | 4.46472e-05 | 0 |
| mlp | direct | ritz_gain | uniform | 4.593e-05 | 0 |
| mlp | direct | ritz_gain | log | 4.6245e-05 | 0 |
| mlp | lookup | random | continue | -1.11088e-06 | 1 |
| mlp | lookup | random | uniform | 6.22411e-07 | 3 |
| mlp | lookup | random | log | 9.97029e-07 | 2 |
| mlp | lookup | variance | continue | 1.52941e-05 | 1 |
| mlp | lookup | variance | uniform | 1.70274e-05 | 0 |
| mlp | lookup | variance | log | 1.7402e-05 | 1 |
| mlp | lookup | ritz_gain | continue | 4.39904e-05 | 0 |
| mlp | lookup | ritz_gain | uniform | 4.57237e-05 | 0 |
| mlp | lookup | ritz_gain | log | 4.60983e-05 | 0 |
| mlp | sign_map | random | continue | -2.06192e-06 | 1 |
| mlp | sign_map | random | uniform | 3.3866e-06 | 2 |
| mlp | sign_map | random | log | 3.7333e-06 | 2 |
| mlp | sign_map | variance | continue | 1.76887e-05 | 1 |
| mlp | sign_map | variance | uniform | 2.31373e-05 | 0 |
| mlp | sign_map | variance | log | 2.3484e-05 | 0 |
| mlp | sign_map | ritz_gain | continue | 5.99086e-05 | 0 |
| mlp | sign_map | ritz_gain | uniform | 6.53571e-05 | 0 |
| mlp | sign_map | ritz_gain | log | 6.57038e-05 | 0 |
| mlp | covariant_average | random | continue | -2.35157e-06 | 1 |
| mlp | covariant_average | random | uniform | 1.06961e-05 | 1 |
| mlp | covariant_average | random | log | 1.02668e-05 | 1 |
| mlp | covariant_average | variance | continue | 7.23505e-06 | 2 |
| mlp | covariant_average | variance | uniform | 2.02827e-05 | 1 |
| mlp | covariant_average | variance | log | 1.98534e-05 | 1 |
| mlp | covariant_average | ritz_gain | continue | 5.00636e-05 | 0 |
| mlp | covariant_average | ritz_gain | uniform | 6.31113e-05 | 0 |
| mlp | covariant_average | ritz_gain | log | 6.2682e-05 | 0 |
| mlp | train_span_ritz | random | continue | -3.766e-05 | 3 |
| mlp | train_span_ritz | random | uniform | -3.79985e-06 | 3 |
| mlp | train_span_ritz | random | log | 6.45658e-06 | 2 |
| mlp | train_span_ritz | variance | continue | -3.75585e-05 | 2 |
| mlp | train_span_ritz | variance | uniform | -3.69838e-06 | 2 |
| mlp | train_span_ritz | variance | log | 6.55806e-06 | 2 |
| mlp | train_span_ritz | ritz_gain | continue | 3.5488e-05 | 1 |
| mlp | train_span_ritz | ritz_gain | uniform | 6.93482e-05 | 0 |
| mlp | train_span_ritz | ritz_gain | log | 7.96046e-05 | 0 |
| transformer | direct | random | continue | 8.2049e-06 | 3 |
| transformer | direct | random | uniform | -2.7001e-05 | 3 |
| transformer | direct | random | log | -2.34426e-05 | 4 |
| transformer | direct | variance | continue | -1.37402e-05 | 2 |
| transformer | direct | variance | uniform | -4.89462e-05 | 4 |
| transformer | direct | variance | log | -4.53877e-05 | 4 |
| transformer | direct | ritz_gain | continue | -7.67959e-06 | 2 |
| transformer | direct | ritz_gain | uniform | -4.28855e-05 | 3 |
| transformer | direct | ritz_gain | log | -3.93271e-05 | 3 |
| transformer | lookup | random | continue | 8.17224e-06 | 3 |
| transformer | lookup | random | uniform | -2.6887e-05 | 3 |
| transformer | lookup | random | log | -2.33275e-05 | 4 |
| transformer | lookup | variance | continue | -1.38829e-05 | 2 |
| transformer | lookup | variance | uniform | -4.89421e-05 | 4 |
| transformer | lookup | variance | log | -4.53827e-05 | 4 |
| transformer | lookup | ritz_gain | continue | -7.80446e-06 | 2 |
| transformer | lookup | ritz_gain | uniform | -4.28637e-05 | 3 |
| transformer | lookup | ritz_gain | log | -3.93042e-05 | 3 |
| transformer | sign_map | random | continue | 1.13514e-05 | 2 |
| transformer | sign_map | random | uniform | -3.01613e-05 | 4 |
| transformer | sign_map | random | log | -2.57672e-05 | 4 |
| transformer | sign_map | variance | continue | -4.90434e-06 | 2 |
| transformer | sign_map | variance | uniform | -4.64171e-05 | 4 |
| transformer | sign_map | variance | log | -4.2023e-05 | 4 |
| transformer | sign_map | ritz_gain | continue | 2.36509e-06 | 2 |
| transformer | sign_map | ritz_gain | uniform | -3.91477e-05 | 3 |
| transformer | sign_map | ritz_gain | log | -3.47536e-05 | 3 |
| transformer | covariant_average | random | continue | -1.08997e-06 | 3 |
| transformer | covariant_average | random | uniform | 6.38368e-06 | 1 |
| transformer | covariant_average | random | log | 4.56135e-06 | 3 |
| transformer | covariant_average | variance | continue | -4.70386e-06 | 3 |
| transformer | covariant_average | variance | uniform | 2.76979e-06 | 2 |
| transformer | covariant_average | variance | log | 9.47458e-07 | 2 |
| transformer | covariant_average | ritz_gain | continue | -4.69338e-06 | 4 |
| transformer | covariant_average | ritz_gain | uniform | 2.78028e-06 | 1 |
| transformer | covariant_average | ritz_gain | log | 9.57943e-07 | 2 |
| transformer | train_span_ritz | random | continue | -2.46393e-05 | 4 |
| transformer | train_span_ritz | random | uniform | 4.15164e-05 | 3 |
| transformer | train_span_ritz | random | log | 3.78049e-05 | 3 |
| transformer | train_span_ritz | variance | continue | -7.62948e-05 | 4 |
| transformer | train_span_ritz | variance | uniform | -1.01391e-05 | 3 |
| transformer | train_span_ritz | variance | log | -1.38506e-05 | 3 |
| transformer | train_span_ritz | ritz_gain | continue | -0.000120966 | 5 |
| transformer | train_span_ritz | ritz_gain | uniform | -5.48107e-05 | 3 |
| transformer | train_span_ritz | ritz_gain | log | -5.85222e-05 | 3 |

## Measured costs

Training plus acquisition = 2400 FORWARD H-vector actions in every stage-B run. Backward H work, gradient steps and wall time differ (ADR 0003).
Costs for each seed, selected points, and model calls are in stage_b_costs.csv.

| Architecture | Policy | Updates (range) | Query H (range) | Wall seconds (mean) |
|---|---|---:|---:|---:|
| mlp | continue | 1200-1200 | 0-0 | 1.349 |
| mlp | uniform | 1200-1200 | 0-0 | 1.344 |
| mlp | log | 1200-1200 | 0-0 | 1.374 |
| mlp | random | 1200-1200 | 0-0 | 1.336 |
| mlp | variance | 1175-1175 | 51-51 | 1.324 |
| mlp | ritz_gain | 1124-1124 | 153-153 | 1.264 |
| transformer | continue | 1200-1200 | 0-0 | 3.983 |
| transformer | uniform | 1200-1200 | 0-0 | 4.167 |
| transformer | log | 1200-1200 | 0-0 | 4.162 |
| transformer | random | 1200-1200 | 0-0 | 4.041 |
| transformer | variance | 1175-1175 | 51-51 | 3.964 |
| transformer | ritz_gain | 1124-1124 | 153-153 | 3.733 |
