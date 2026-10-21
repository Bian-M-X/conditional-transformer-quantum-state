# Round 3 descriptive results

Five paired seeds. Mean +/- sample SD, not confidence intervals. Negative paired deltas favor the first condition.

## A: low-field training fidelity

| Architecture | Training | Checkpoint | Mean F at h=0.2 | Seed range |
|---|---|---|---:|---:|
| mlp | raw | selected | 0.5286937 +/- 0.00559 | 0.5213979 to 0.5349958 |
| mlp | raw | final | 0.5034926 +/- 0.00134 | 0.501247 to 0.5045707 |
| mlp | even | selected | 0.9998786 +/- 5.27e-05 | 0.9998261 to 0.9999561 |
| mlp | even | final | 0.9999864 +/- 8.17e-06 | 0.9999728 to 0.9999921 |
| transformer | raw | selected | 0.6591491 +/- 0.222 | 0.500102 to 0.9998922 |
| transformer | raw | final | 0.6514864 +/- 0.224 | 0.5001553 to 0.9996006 |
| transformer | even | selected | 0.999686 +/- 0.000682 | 0.998466 to 0.999999 |
| transformer | even | final | 0.999997 +/- 2.81e-06 | 0.9999928 to 0.9999996 |

## B: validation-selected test infidelity

| N | Architecture | Scale | Training | Predictor | Mean infidelity | Mean worst infidelity |
|---:|---|---:|---|---|---:|---:|
| 4 | mlp | 0.04 | covariant | direct | 0.01538024 +/- 0.00859 | 0.05379684 +/- 0.0301 |
| 4 | mlp | 0.04 | covariant | lookup | 0.01538848 +/- 0.00855 | 0.05382698 +/- 0.0299 |
| 4 | mlp | 0.04 | covariant | post_covariant | 0.01538024 +/- 0.00859 | 0.05379684 +/- 0.0301 |
| 4 | mlp | 0.04 | covariant | train_span_ritz | 0.01527106 +/- 0.00854 | 0.05343489 +/- 0.0299 |
| 4 | mlp | 0.04 | raw | direct | 3.7761e-05 +/- 1.64e-05 | 9.964917e-05 +/- 4.61e-05 |
| 4 | mlp | 0.04 | raw | lookup | 0.0001313266 +/- 9.75e-06 | 0.0003666305 +/- 2.2e-05 |
| 4 | mlp | 0.04 | raw | post_covariant | 3.236702e-05 +/- 1.54e-05 | 7.257008e-05 +/- 3.45e-05 |
| 4 | mlp | 0.04 | raw | train_span_ritz | 1.580097e-06 +/- 5.87e-07 | 2.901737e-06 +/- 1.7e-06 |
| 4 | mlp | 1.0 | covariant | direct | 0.01548508 +/- 0.00864 | 0.05418347 +/- 0.0303 |
| 4 | mlp | 1.0 | covariant | lookup | 0.01550322 +/- 0.00859 | 0.05423334 +/- 0.0301 |
| 4 | mlp | 1.0 | covariant | post_covariant | 0.01548508 +/- 0.00864 | 0.05418347 +/- 0.0303 |
| 4 | mlp | 1.0 | covariant | train_span_ritz | 0.01544923 +/- 0.00864 | 0.05406914 +/- 0.0302 |
| 4 | mlp | 1.0 | raw | direct | 7.211033e-05 +/- 2.12e-05 | 0.0001768667 +/- 5.2e-05 |
| 4 | mlp | 1.0 | raw | lookup | 0.000154993 +/- 1.92e-05 | 0.0004049998 +/- 4.1e-05 |
| 4 | mlp | 1.0 | raw | post_covariant | 6.589177e-05 +/- 1.74e-05 | 0.0001418336 +/- 3.2e-05 |
| 4 | mlp | 1.0 | raw | train_span_ritz | 4.142789e-06 +/- 2.81e-06 | 1.235398e-05 +/- 1.12e-05 |
| 4 | transformer | 0.04 | covariant | direct | 0.0002897385 +/- 0.000541 | 0.00103443 +/- 0.00201 |
| 4 | transformer | 0.04 | covariant | lookup | 0.0005548439 +/- 0.000951 | 0.001685037 +/- 0.003 |
| 4 | transformer | 0.04 | covariant | post_covariant | 0.0002897385 +/- 0.000541 | 0.00103443 +/- 0.00201 |
| 4 | transformer | 0.04 | covariant | train_span_ritz | 3.347791e-05 +/- 6.61e-05 | 9.564053e-05 +/- 0.000198 |
| 4 | transformer | 0.04 | raw | direct | 0.0001465059 +/- 0.000235 | 0.0003937475 +/- 0.000666 |
| 4 | transformer | 0.04 | raw | lookup | 0.0002454173 +/- 0.000237 | 0.0007009234 +/- 0.000726 |
| 4 | transformer | 0.04 | raw | post_covariant | 0.0001080452 +/- 0.000213 | 0.0003062098 +/- 0.000616 |
| 4 | transformer | 0.04 | raw | train_span_ritz | 7.021643e-05 +/- 0.000152 | 0.0002375896 +/- 0.000521 |
| 4 | transformer | 1.0 | covariant | direct | 0.0001881712 +/- 0.000323 | 0.0005047951 +/- 0.000897 |
| 4 | transformer | 1.0 | covariant | lookup | 0.000289784 +/- 0.000342 | 0.000783293 +/- 0.000954 |
| 4 | transformer | 1.0 | covariant | post_covariant | 0.0001881712 +/- 0.000323 | 0.0005047951 +/- 0.000897 |
| 4 | transformer | 1.0 | covariant | train_span_ritz | 4.552687e-05 +/- 9.55e-05 | 0.000141207 +/- 0.000306 |
| 4 | transformer | 1.0 | raw | direct | 6.809149e-05 +/- 4.51e-05 | 0.0001341418 +/- 6.28e-05 |
| 4 | transformer | 1.0 | raw | lookup | 0.0001764178 +/- 4.87e-05 | 0.0004392261 +/- 8.8e-05 |
| 4 | transformer | 1.0 | raw | post_covariant | 3.591063e-05 +/- 2.06e-05 | 7.66924e-05 +/- 5.59e-05 |
| 4 | transformer | 1.0 | raw | train_span_ritz | 8.432183e-06 +/- 5.96e-06 | 2.618219e-05 +/- 2.27e-05 |
| 6 | mlp | 0.04 | covariant | direct | 0.07159358 +/- 0.0529 | 0.1922276 +/- 0.144 |
| 6 | mlp | 0.04 | covariant | lookup | 0.0734455 +/- 0.05 | 0.197143 +/- 0.137 |
| 6 | mlp | 0.04 | covariant | post_covariant | 0.07159358 +/- 0.0529 | 0.1922276 +/- 0.144 |
| 6 | mlp | 0.04 | covariant | train_span_ritz | 0.06587862 +/- 0.0593 | 0.1775548 +/- 0.16 |
| 6 | mlp | 0.04 | raw | direct | 0.4556181 +/- 0.0572 | 0.9259324 +/- 0.0522 |
| 6 | mlp | 0.04 | raw | lookup | 0.4555214 +/- 0.0574 | 0.9257709 +/- 0.0525 |
| 6 | mlp | 0.04 | raw | post_covariant | 0.1105841 +/- 0.00368 | 0.2984878 +/- 0.0114 |
| 6 | mlp | 0.04 | raw | train_span_ritz | 0.4545021 +/- 0.0594 | 0.9237733 +/- 0.0569 |
| 6 | mlp | 1.0 | covariant | direct | 0.09030339 +/- 0.0436 | 0.2434219 +/- 0.119 |
| 6 | mlp | 1.0 | covariant | lookup | 0.09157984 +/- 0.0407 | 0.2468777 +/- 0.111 |
| 6 | mlp | 1.0 | covariant | post_covariant | 0.09030339 +/- 0.0436 | 0.2434219 +/- 0.119 |
| 6 | mlp | 1.0 | covariant | train_span_ritz | 0.08769242 +/- 0.0487 | 0.2366902 +/- 0.132 |
| 6 | mlp | 1.0 | raw | direct | 0.4543877 +/- 0.0592 | 0.9258851 +/- 0.0519 |
| 6 | mlp | 1.0 | raw | lookup | 0.4543499 +/- 0.0592 | 0.9258265 +/- 0.052 |
| 6 | mlp | 1.0 | raw | post_covariant | 0.1109545 +/- 0.00293 | 0.2996148 +/- 0.00903 |
| 6 | mlp | 1.0 | raw | train_span_ritz | 0.4536954 +/- 0.0605 | 0.9243289 +/- 0.0554 |
| 6 | transformer | 0.04 | covariant | direct | 0.0811122 +/- 0.0465 | 0.2161013 +/- 0.128 |
| 6 | transformer | 0.04 | covariant | lookup | 0.08270134 +/- 0.0438 | 0.2207264 +/- 0.12 |
| 6 | transformer | 0.04 | covariant | post_covariant | 0.0811122 +/- 0.0465 | 0.2161013 +/- 0.128 |
| 6 | transformer | 0.04 | covariant | train_span_ritz | 0.07703418 +/- 0.0499 | 0.2046521 +/- 0.137 |
| 6 | transformer | 0.04 | raw | direct | 0.004908971 +/- 0.00145 | 0.01444315 +/- 0.00401 |
| 6 | transformer | 0.04 | raw | lookup | 0.01356566 +/- 0.00129 | 0.04012513 +/- 0.00484 |
| 6 | transformer | 0.04 | raw | post_covariant | 0.004120648 +/- 0.00129 | 0.01074992 +/- 0.00357 |
| 6 | transformer | 0.04 | raw | train_span_ritz | 0.0001391806 +/- 9.22e-05 | 0.0002614354 +/- 0.000195 |
| 6 | transformer | 1.0 | covariant | direct | 0.05096187 +/- 0.0559 | 0.1363096 +/- 0.153 |
| 6 | transformer | 1.0 | covariant | lookup | 0.05528122 +/- 0.0518 | 0.1483489 +/- 0.141 |
| 6 | transformer | 1.0 | covariant | post_covariant | 0.05096187 +/- 0.0559 | 0.1363096 +/- 0.153 |
| 6 | transformer | 1.0 | covariant | train_span_ritz | 0.04587924 +/- 0.06 | 0.1236196 +/- 0.163 |
| 6 | transformer | 1.0 | raw | direct | 0.01869014 +/- 0.0265 | 0.050036 +/- 0.0697 |
| 6 | transformer | 1.0 | raw | lookup | 0.02569919 +/- 0.0235 | 0.07042289 +/- 0.0613 |
| 6 | transformer | 1.0 | raw | post_covariant | 0.0171269 +/- 0.027 | 0.04236094 +/- 0.066 |
| 6 | transformer | 1.0 | raw | train_span_ritz | 0.009033121 +/- 0.0196 | 0.02047722 +/- 0.0446 |

## Paired changes in mean test infidelity

| Comparison | N | Architecture | Checkpoint | Scale/mode | Mean delta | Wins/5 |
|---|---:|---|---|---|---:|---:|
| scaled minus physical | 4 | mlp | selected | /raw | -3.434933e-05 | 5 |
| scaled minus physical | 4 | mlp | selected | /covariant | -0.0001048418 | 4 |
| covariant training minus raw direct | 4 | mlp | selected | 1.0/ | 0.01541297 | 1 |
| covariant training minus raw post_covariant | 4 | mlp | selected | 1.0/ | 0.01541919 | 1 |
| covariant training minus raw direct | 4 | mlp | selected | 0.04/ | 0.01534248 | 1 |
| covariant training minus raw post_covariant | 4 | mlp | selected | 0.04/ | 0.01534787 | 1 |
| scaled minus physical | 4 | mlp | final | /raw | -3.434933e-05 | 5 |
| scaled minus physical | 4 | mlp | final | /covariant | -0.002817531 | 5 |
| covariant training minus raw direct | 4 | mlp | final | 1.0/ | 0.01540732 | 1 |
| covariant training minus raw post_covariant | 4 | mlp | final | 1.0/ | 0.01541354 | 1 |
| covariant training minus raw direct | 4 | mlp | final | 0.04/ | 0.01262414 | 1 |
| covariant training minus raw post_covariant | 4 | mlp | final | 0.04/ | 0.01262953 | 1 |
| final minus selected | 4 | mlp | paired | 1.0/raw | 0 | 0 |
| final minus selected | 4 | mlp | paired | 1.0/covariant | -5.64728e-06 | 1 |
| final minus selected | 4 | mlp | paired | 0.04/raw | 0 | 0 |
| final minus selected | 4 | mlp | paired | 0.04/covariant | -0.002718336 | 3 |
| scaled minus physical | 4 | transformer | selected | /raw | 7.841436e-05 | 2 |
| scaled minus physical | 4 | transformer | selected | /covariant | 0.0001015673 | 2 |
| covariant training minus raw direct | 4 | transformer | selected | 1.0/ | 0.0001200797 | 3 |
| covariant training minus raw post_covariant | 4 | transformer | selected | 1.0/ | 0.0001522606 | 2 |
| covariant training minus raw direct | 4 | transformer | selected | 0.04/ | 0.0001432327 | 3 |
| covariant training minus raw post_covariant | 4 | transformer | selected | 0.04/ | 0.0001816933 | 2 |
| scaled minus physical | 4 | transformer | final | /raw | 7.373093e-05 | 2 |
| scaled minus physical | 4 | transformer | final | /covariant | -7.218097e-05 | 3 |
| covariant training minus raw direct | 4 | transformer | final | 1.0/ | 0.0001118438 | 3 |
| covariant training minus raw post_covariant | 4 | transformer | final | 1.0/ | 0.0001505511 | 2 |
| covariant training minus raw direct | 4 | transformer | final | 0.04/ | -3.406815e-05 | 2 |
| covariant training minus raw post_covariant | 4 | transformer | final | 0.04/ | 4.005609e-06 | 2 |
| final minus selected | 4 | transformer | paired | 1.0/raw | 9.781997e-06 | 0 |
| final minus selected | 4 | transformer | paired | 1.0/covariant | 1.546029e-06 | 1 |
| final minus selected | 4 | transformer | paired | 0.04/raw | 5.098565e-06 | 0 |
| final minus selected | 4 | transformer | paired | 0.04/covariant | -0.0001722022 | 1 |
| scaled minus physical | 6 | mlp | selected | /raw | 0.001230393 | 1 |
| scaled minus physical | 6 | mlp | selected | /covariant | -0.01870981 | 5 |
| covariant training minus raw direct | 6 | mlp | selected | 1.0/ | -0.3640843 | 5 |
| covariant training minus raw post_covariant | 6 | mlp | selected | 1.0/ | -0.02065108 | 4 |
| covariant training minus raw direct | 6 | mlp | selected | 0.04/ | -0.3840245 | 5 |
| covariant training minus raw post_covariant | 6 | mlp | selected | 0.04/ | -0.03899049 | 3 |
| scaled minus physical | 6 | mlp | final | /raw | 0.001230393 | 1 |
| scaled minus physical | 6 | mlp | final | /covariant | -0.03414415 | 5 |
| covariant training minus raw direct | 6 | mlp | final | 1.0/ | -0.3650358 | 5 |
| covariant training minus raw post_covariant | 6 | mlp | final | 1.0/ | -0.02160252 | 4 |
| covariant training minus raw direct | 6 | mlp | final | 0.04/ | -0.4004103 | 5 |
| covariant training minus raw post_covariant | 6 | mlp | final | 0.04/ | -0.05537627 | 4 |
| final minus selected | 6 | mlp | paired | 1.0/raw | 0 | 0 |
| final minus selected | 6 | mlp | paired | 1.0/covariant | -0.0009514362 | 1 |
| final minus selected | 6 | mlp | paired | 0.04/raw | 0 | 0 |
| final minus selected | 6 | mlp | paired | 0.04/covariant | -0.01638578 | 1 |
| scaled minus physical | 6 | transformer | selected | /raw | -0.01378117 | 4 |
| scaled minus physical | 6 | transformer | selected | /covariant | 0.03015033 | 2 |
| covariant training minus raw direct | 6 | transformer | selected | 1.0/ | 0.03227173 | 1 |
| covariant training minus raw post_covariant | 6 | transformer | selected | 1.0/ | 0.03383497 | 1 |
| covariant training minus raw direct | 6 | transformer | selected | 0.04/ | 0.07620323 | 0 |
| covariant training minus raw post_covariant | 6 | transformer | selected | 0.04/ | 0.07699156 | 0 |
| scaled minus physical | 6 | transformer | final | /raw | -0.004731213 | 5 |
| scaled minus physical | 6 | transformer | final | /covariant | -0.02543587 | 3 |
| covariant training minus raw direct | 6 | transformer | final | 1.0/ | 0.03996496 | 1 |
| covariant training minus raw post_covariant | 6 | transformer | final | 1.0/ | 0.04239532 | 1 |
| covariant training minus raw direct | 6 | transformer | final | 0.04/ | 0.0192603 | 1 |
| covariant training minus raw post_covariant | 6 | transformer | final | 0.04/ | 0.01964516 | 1 |
| final minus selected | 6 | transformer | paired | 1.0/raw | -0.01023864 | 1 |
| final minus selected | 6 | transformer | paired | 1.0/covariant | -0.002545421 | 4 |
| final minus selected | 6 | transformer | paired | 0.04/raw | -0.001188689 | 2 |
| final minus selected | 6 | transformer | paired | 0.04/covariant | -0.05813162 | 5 |
