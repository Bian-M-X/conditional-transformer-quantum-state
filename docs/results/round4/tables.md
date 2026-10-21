# Round 4 complete descriptive results

Mean ± sample SD across five seeds. Lower infidelity is better; no confidence or significance claim.

P=physical; U=scaled unchanged weights; I=scaled matched initial function; C=matched function + physical-coordinate Adam.

| h / architecture / checkpoint / split / method | P | U | I | C |
|---|---|---|---|---|
| 0.4 / mlp / selected / train / direct | 0.4999328 ± 0.000496 | 0.4325323 ± 0.151 | 0.499861 ± 0.000671 | 0.4999328 ± 0.000496 |
| 0.4 / mlp / selected / train / lookup | 0.4999328 ± 0.000496 | 0.4325323 ± 0.151 | 0.499861 ± 0.000671 | 0.4999328 ± 0.000496 |
| 0.4 / mlp / selected / train / train_span_ritz | 0.4998906 ± 0.000546 | 0.4002666 ± 0.223 | 0.4998453 ± 0.000671 | 0.4998906 ± 0.000546 |
| 0.4 / mlp / selected / train / post_covariant | 0.3138075 ± 0.000486 | 0.2514023 ± 0.14 | 0.3137798 ± 0.000568 | 0.3138075 ± 0.000486 |
| 0.4 / mlp / selected / validation / direct | 0.4998741 ± 0.00057 | 0.4939573 ± 0.0138 | 0.4997946 ± 0.00077 | 0.4998741 ± 0.00057 |
| 0.4 / mlp / selected / validation / lookup | 0.4998712 ± 0.000571 | 0.4834574 ± 0.0373 | 0.4997908 ± 0.000771 | 0.4998712 ± 0.000571 |
| 0.4 / mlp / selected / validation / train_span_ritz | 0.4998254 ± 0.000627 | 0.4003365 ± 0.223 | 0.4997739 ± 0.00077 | 0.4998254 ± 0.000627 |
| 0.4 / mlp / selected / validation / post_covariant | 0.2835436 ± 0.0004 | 0.2715891 ± 0.0271 | 0.2835362 ± 0.000457 | 0.2835436 ± 0.0004 |
| 0.4 / mlp / selected / test / direct | 0.4997748 ± 0.000645 | 0.4742717 ± 0.0577 | 0.4996806 ± 0.000877 | 0.4997748 ± 0.000645 |
| 0.4 / mlp / selected / test / lookup | 0.499773 ± 0.000646 | 0.4683254 ± 0.071 | 0.4996783 ± 0.000878 | 0.499773 ± 0.000646 |
| 0.4 / mlp / selected / test / train_span_ritz | 0.4997183 ± 0.000713 | 0.4002515 ± 0.223 | 0.4996576 ± 0.000878 | 0.4997183 ± 0.000713 |
| 0.4 / mlp / selected / test / post_covariant | 0.2447468 ± 0.000439 | 0.2174907 ± 0.0613 | 0.2447353 ± 0.000503 | 0.2447468 ± 0.000439 |
| 0.4 / mlp / final / train / direct | 0.4999328 ± 0.000496 | 0.4325323 ± 0.151 | 0.499861 ± 0.000671 | 0.4999328 ± 0.000496 |
| 0.4 / mlp / final / train / lookup | 0.4999328 ± 0.000496 | 0.4325323 ± 0.151 | 0.499861 ± 0.000671 | 0.4999328 ± 0.000496 |
| 0.4 / mlp / final / train / train_span_ritz | 0.4998906 ± 0.000546 | 0.4002666 ± 0.223 | 0.4998453 ± 0.000671 | 0.4998906 ± 0.000546 |
| 0.4 / mlp / final / train / post_covariant | 0.3138075 ± 0.000486 | 0.2514023 ± 0.14 | 0.3137798 ± 0.000568 | 0.3138075 ± 0.000486 |
| 0.4 / mlp / final / validation / direct | 0.4998741 ± 0.00057 | 0.4939573 ± 0.0138 | 0.4997946 ± 0.00077 | 0.4998741 ± 0.00057 |
| 0.4 / mlp / final / validation / lookup | 0.4998712 ± 0.000571 | 0.4834574 ± 0.0373 | 0.4997908 ± 0.000771 | 0.4998712 ± 0.000571 |
| 0.4 / mlp / final / validation / train_span_ritz | 0.4998254 ± 0.000627 | 0.4003365 ± 0.223 | 0.4997739 ± 0.00077 | 0.4998254 ± 0.000627 |
| 0.4 / mlp / final / validation / post_covariant | 0.2835436 ± 0.0004 | 0.2715891 ± 0.0271 | 0.2835362 ± 0.000457 | 0.2835436 ± 0.0004 |
| 0.4 / mlp / final / test / direct | 0.4997748 ± 0.000645 | 0.4742717 ± 0.0577 | 0.4996806 ± 0.000877 | 0.4997748 ± 0.000645 |
| 0.4 / mlp / final / test / lookup | 0.499773 ± 0.000646 | 0.4683254 ± 0.071 | 0.4996783 ± 0.000878 | 0.499773 ± 0.000646 |
| 0.4 / mlp / final / test / train_span_ritz | 0.4997183 ± 0.000713 | 0.4002515 ± 0.223 | 0.4996576 ± 0.000878 | 0.4997183 ± 0.000713 |
| 0.4 / mlp / final / test / post_covariant | 0.2447468 ± 0.000439 | 0.2174907 ± 0.0613 | 0.2447353 ± 0.000503 | 0.2447468 ± 0.000439 |
| 0.4 / transformer / selected / train / direct | 0.4970975 ± 0.00144 | 0.497348 ± 0.00108 | 0.4969328 ± 0.00164 | 0.4970975 ± 0.00144 |
| 0.4 / transformer / selected / train / lookup | 0.4970975 ± 0.00144 | 0.497348 ± 0.00108 | 0.4969328 ± 0.00164 | 0.4970975 ± 0.00144 |
| 0.4 / transformer / selected / train / train_span_ritz | 0.497089 ± 0.00144 | 0.4973343 ± 0.00109 | 0.4969413 ± 0.00162 | 0.497089 ± 0.00144 |
| 0.4 / transformer / selected / train / post_covariant | 0.3128613 ± 0.000462 | 0.3128233 ± 0.000383 | 0.3123528 ± 0.000937 | 0.3128613 ± 0.000462 |
| 0.4 / transformer / selected / validation / direct | 0.4966201 ± 0.00166 | 0.496911 ± 0.00124 | 0.4964541 ± 0.00188 | 0.4966201 ± 0.00166 |
| 0.4 / transformer / selected / validation / lookup | 0.4966176 ± 0.00166 | 0.4969084 ± 0.00124 | 0.4964488 ± 0.00188 | 0.4966176 ± 0.00166 |
| 0.4 / transformer / selected / validation / train_span_ritz | 0.4966195 ± 0.00166 | 0.4968946 ± 0.00126 | 0.4964628 ± 0.00186 | 0.4966195 ± 0.00166 |
| 0.4 / transformer / selected / validation / post_covariant | 0.2826698 ± 0.000125 | 0.2826579 ± 0.000103 | 0.2825319 ± 0.000258 | 0.2826698 ± 0.000125 |
| 0.4 / transformer / selected / test / direct | 0.4960158 ± 0.00196 | 0.4963588 ± 0.00146 | 0.4958167 ± 0.00221 | 0.4960158 ± 0.00196 |
| 0.4 / transformer / selected / test / lookup | 0.4960143 ± 0.00196 | 0.4963573 ± 0.00146 | 0.4958136 ± 0.00222 | 0.4960143 ± 0.00196 |
| 0.4 / transformer / selected / test / train_span_ritz | 0.4960136 ± 0.00196 | 0.4963395 ± 0.00148 | 0.4958271 ± 0.00219 | 0.4960136 ± 0.00196 |
| 0.4 / transformer / selected / test / post_covariant | 0.2438068 ± 0.000192 | 0.2437895 ± 0.000158 | 0.2435953 ± 0.000393 | 0.2438068 ± 0.000192 |
| 0.4 / transformer / final / train / direct | 0.4970975 ± 0.00144 | 0.497348 ± 0.00108 | 0.4969328 ± 0.00164 | 0.4970975 ± 0.00144 |
| 0.4 / transformer / final / train / lookup | 0.4970975 ± 0.00144 | 0.497348 ± 0.00108 | 0.4969328 ± 0.00164 | 0.4970975 ± 0.00144 |
| 0.4 / transformer / final / train / train_span_ritz | 0.497089 ± 0.00144 | 0.4973343 ± 0.00109 | 0.4969413 ± 0.00162 | 0.497089 ± 0.00144 |
| 0.4 / transformer / final / train / post_covariant | 0.3128613 ± 0.000462 | 0.3128233 ± 0.000383 | 0.3123528 ± 0.000937 | 0.3128613 ± 0.000462 |
| 0.4 / transformer / final / validation / direct | 0.4966201 ± 0.00166 | 0.496911 ± 0.00124 | 0.4964541 ± 0.00188 | 0.4966201 ± 0.00166 |
| 0.4 / transformer / final / validation / lookup | 0.4966176 ± 0.00166 | 0.4969084 ± 0.00124 | 0.4964488 ± 0.00188 | 0.4966176 ± 0.00166 |
| 0.4 / transformer / final / validation / train_span_ritz | 0.4966195 ± 0.00166 | 0.4968946 ± 0.00126 | 0.4964628 ± 0.00186 | 0.4966195 ± 0.00166 |
| 0.4 / transformer / final / validation / post_covariant | 0.2826698 ± 0.000125 | 0.2826579 ± 0.000103 | 0.2825319 ± 0.000258 | 0.2826698 ± 0.000125 |
| 0.4 / transformer / final / test / direct | 0.4960158 ± 0.00196 | 0.4963588 ± 0.00146 | 0.4958167 ± 0.00221 | 0.4960158 ± 0.00196 |
| 0.4 / transformer / final / test / lookup | 0.4960143 ± 0.00196 | 0.4963573 ± 0.00146 | 0.4958136 ± 0.00222 | 0.4960143 ± 0.00196 |
| 0.4 / transformer / final / test / train_span_ritz | 0.4960136 ± 0.00196 | 0.4963395 ± 0.00148 | 0.4958271 ± 0.00219 | 0.4960136 ± 0.00196 |
| 0.4 / transformer / final / test / post_covariant | 0.2438068 ± 0.000192 | 0.2437895 ± 0.000158 | 0.2435953 ± 0.000393 | 0.2438068 ± 0.000192 |
| 0.6 / mlp / selected / train / direct | 0.0005903384 ± 0.000157 | 0.0001565293 ± 4.63e-05 | 0.0001762112 ± 8.13e-05 | 0.0005903384 ± 0.000157 |
| 0.6 / mlp / selected / train / lookup | 0.0005903384 ± 0.000157 | 0.0001565293 ± 4.63e-05 | 0.0001762112 ± 8.13e-05 | 0.0005903384 ± 0.000157 |
| 0.6 / mlp / selected / train / train_span_ritz | 0.0001326581 ± 7.54e-05 | 5.552244e-05 ± 3.23e-05 | 7.568366e-05 ± 5.08e-05 | 0.0001326581 ± 7.54e-05 |
| 0.6 / mlp / selected / train / post_covariant | 0.0005538362 ± 0.000143 | 0.0001177502 ± 5.29e-05 | 0.0001492776 ± 6.94e-05 | 0.0005538362 ± 0.000143 |
| 0.6 / mlp / selected / validation / direct | 0.002596025 ± 0.000266 | 0.001722901 ± 0.000208 | 0.001667917 ± 0.000287 | 0.002596025 ± 0.000266 |
| 0.6 / mlp / selected / validation / lookup | 0.004448197 ± 0.000141 | 0.00393138 ± 7.63e-05 | 0.003958727 ± 0.000122 | 0.004448197 ± 0.000141 |
| 0.6 / mlp / selected / validation / train_span_ritz | 9.00134e-05 ± 5.5e-05 | 4.469825e-05 ± 2.57e-05 | 6.130899e-05 ± 4.18e-05 | 9.00134e-05 ± 5.5e-05 |
| 0.6 / mlp / selected / validation / post_covariant | 0.002557576 ± 0.000248 | 0.001666658 ± 0.000214 | 0.001626199 ± 0.000281 | 0.002557576 ± 0.000248 |
| 0.6 / mlp / selected / test / direct | 0.001984196 ± 0.000258 | 0.001155045 ± 0.00016 | 0.001131213 ± 0.00023 | 0.001984196 ± 0.000258 |
| 0.6 / mlp / selected / test / lookup | 0.0033921 ± 0.000189 | 0.002722564 ± 9.91e-05 | 0.002754988 ± 0.000147 | 0.0033921 ± 0.000189 |
| 0.6 / mlp / selected / test / train_span_ritz | 9.896377e-05 ± 5.97e-05 | 4.59163e-05 ± 2.69e-05 | 6.33692e-05 ± 4.38e-05 | 9.896377e-05 ± 5.97e-05 |
| 0.6 / mlp / selected / test / post_covariant | 0.001947117 ± 0.00024 | 0.001105947 ± 0.000166 | 0.001094558 ± 0.000223 | 0.001947117 ± 0.00024 |
| 0.6 / mlp / final / train / direct | 0.0005903384 ± 0.000157 | 0.0001565293 ± 4.63e-05 | 0.0001762112 ± 8.13e-05 | 0.0005903384 ± 0.000157 |
| 0.6 / mlp / final / train / lookup | 0.0005903384 ± 0.000157 | 0.0001565293 ± 4.63e-05 | 0.0001762112 ± 8.13e-05 | 0.0005903384 ± 0.000157 |
| 0.6 / mlp / final / train / train_span_ritz | 0.0001326581 ± 7.54e-05 | 5.552244e-05 ± 3.23e-05 | 7.568366e-05 ± 5.08e-05 | 0.0001326581 ± 7.54e-05 |
| 0.6 / mlp / final / train / post_covariant | 0.0005538362 ± 0.000143 | 0.0001177502 ± 5.29e-05 | 0.0001492776 ± 6.94e-05 | 0.0005538362 ± 0.000143 |
| 0.6 / mlp / final / validation / direct | 0.002596025 ± 0.000266 | 0.001722901 ± 0.000208 | 0.001667917 ± 0.000287 | 0.002596025 ± 0.000266 |
| 0.6 / mlp / final / validation / lookup | 0.004448197 ± 0.000141 | 0.00393138 ± 7.63e-05 | 0.003958727 ± 0.000122 | 0.004448197 ± 0.000141 |
| 0.6 / mlp / final / validation / train_span_ritz | 9.00134e-05 ± 5.5e-05 | 4.469825e-05 ± 2.57e-05 | 6.130899e-05 ± 4.18e-05 | 9.00134e-05 ± 5.5e-05 |
| 0.6 / mlp / final / validation / post_covariant | 0.002557576 ± 0.000248 | 0.001666658 ± 0.000214 | 0.001626199 ± 0.000281 | 0.002557576 ± 0.000248 |
| 0.6 / mlp / final / test / direct | 0.001984196 ± 0.000258 | 0.001155045 ± 0.00016 | 0.001131213 ± 0.00023 | 0.001984196 ± 0.000258 |
| 0.6 / mlp / final / test / lookup | 0.0033921 ± 0.000189 | 0.002722564 ± 9.91e-05 | 0.002754988 ± 0.000147 | 0.0033921 ± 0.000189 |
| 0.6 / mlp / final / test / train_span_ritz | 9.896377e-05 ± 5.97e-05 | 4.59163e-05 ± 2.69e-05 | 6.33692e-05 ± 4.38e-05 | 9.896377e-05 ± 5.97e-05 |
| 0.6 / mlp / final / test / post_covariant | 0.001947117 ± 0.00024 | 0.001105947 ± 0.000166 | 0.001094558 ± 0.000223 | 0.001947117 ± 0.00024 |
| 0.6 / transformer / selected / train / direct | 0.0001703189 ± 4.41e-05 | 0.0001357377 ± 0.000164 | 7.042773e-05 ± 3.25e-05 | 0.0001703189 ± 4.41e-05 |
| 0.6 / transformer / selected / train / lookup | 0.0001703189 ± 4.41e-05 | 0.0001357377 ± 0.000164 | 7.042773e-05 ± 3.25e-05 | 0.0001703189 ± 4.41e-05 |
| 0.6 / transformer / selected / train / train_span_ritz | 5.335642e-05 ± 3.13e-05 | 3.795915e-05 ± 1.73e-05 | 4.095156e-05 ± 2.03e-05 | 5.335642e-05 ± 3.13e-05 |
| 0.6 / transformer / selected / train / post_covariant | 9.680971e-05 ± 3.45e-05 | 5.803043e-05 ± 2.29e-05 | 5.42288e-05 ± 2.37e-05 | 9.680971e-05 ± 3.45e-05 |
| 0.6 / transformer / selected / validation / direct | 0.0009431939 ± 0.000414 | 0.001019484 ± 0.000383 | 0.001112219 ± 0.00055 | 0.0009431939 ± 0.000414 |
| 0.6 / transformer / selected / validation / lookup | 0.003922075 ± 5.54e-05 | 0.003851582 ± 0.000182 | 0.003774122 ± 4.89e-05 | 0.003922075 ± 5.54e-05 |
| 0.6 / transformer / selected / validation / train_span_ritz | 4.334545e-05 ± 1.97e-05 | 3.406531e-05 ± 1.34e-05 | 3.731011e-05 ± 1.38e-05 | 4.334545e-05 ± 1.97e-05 |
| 0.6 / transformer / selected / validation / post_covariant | 0.000839348 ± 0.000458 | 0.0008937915 ± 0.000393 | 0.001040816 ± 0.000518 | 0.000839348 ± 0.000458 |
| 0.6 / transformer / selected / test / direct | 0.000664016 ± 0.000256 | 0.0007017305 ± 0.000238 | 0.0007212237 ± 0.000326 | 0.000664016 ± 0.000256 |
| 0.6 / transformer / selected / test / lookup | 0.002705974 ± 6.12e-05 | 0.002619792 ± 0.000192 | 0.002536779 ± 5.75e-05 | 0.002705974 ± 6.12e-05 |
| 0.6 / transformer / selected / test / train_span_ritz | 4.440486e-05 ± 2.23e-05 | 3.378273e-05 ± 1.41e-05 | 3.704719e-05 ± 1.52e-05 | 4.440486e-05 ± 2.23e-05 |
| 0.6 / transformer / selected / test / post_covariant | 0.0005705907 ± 0.000292 | 0.0005934086 ± 0.000231 | 0.0006599107 ± 0.000306 | 0.0005705907 ± 0.000292 |
| 0.6 / transformer / final / train / direct | 0.0001433468 ± 6.01e-05 | 0.0001321539 ± 0.000166 | 5.547482e-05 ± 1.6e-05 | 0.0001433468 ± 6.01e-05 |
| 0.6 / transformer / final / train / lookup | 0.0001433468 ± 6.01e-05 | 0.0001321539 ± 0.000166 | 5.547482e-05 ± 1.6e-05 | 0.0001433468 ± 6.01e-05 |
| 0.6 / transformer / final / train / train_span_ritz | 4.378047e-05 ± 1.51e-05 | 3.645417e-05 ± 1.74e-05 | 3.22337e-05 ± 7.37e-06 | 4.378047e-05 ± 1.51e-05 |
| 0.6 / transformer / final / train / post_covariant | 8.406206e-05 ± 2.38e-05 | 5.523118e-05 ± 2.14e-05 | 4.362559e-05 ± 4.68e-06 | 8.406206e-05 ± 2.38e-05 |
| 0.6 / transformer / final / validation / direct | 0.0008888708 ± 0.000436 | 0.001005493 ± 0.000405 | 0.001066729 ± 0.000601 | 0.0008888708 ± 0.000436 |
| 0.6 / transformer / final / validation / lookup | 0.003890861 ± 7.94e-05 | 0.003845126 ± 0.000184 | 0.003748181 ± 2.55e-05 | 0.003890861 ± 7.94e-05 |
| 0.6 / transformer / final / validation / train_span_ritz | 3.816588e-05 ± 1.11e-05 | 3.328442e-05 ± 1.34e-05 | 3.138458e-05 ± 6.99e-06 | 3.816588e-05 ± 1.11e-05 |
| 0.6 / transformer / final / validation / post_covariant | 0.0007963247 ± 0.000469 | 0.0008788021 ± 0.000411 | 0.001000854 ± 0.000562 | 0.0007963247 ± 0.000469 |
| 0.6 / transformer / final / test / direct | 0.0006217821 ± 0.000278 | 0.0006918915 ± 0.000252 | 0.0006828659 ± 0.000366 | 0.0006217821 ± 0.000278 |
| 0.6 / transformer / final / test / lookup | 0.002669264 ± 8.63e-05 | 0.002611945 ± 0.000194 | 0.002506113 ± 2.85e-05 | 0.002669264 ± 8.63e-05 |
| 0.6 / transformer / final / test / train_span_ritz | 3.820491e-05 ± 1.19e-05 | 3.283994e-05 ± 1.42e-05 | 3.047273e-05 ± 7.08e-06 | 3.820491e-05 ± 1.19e-05 |
| 0.6 / transformer / final / test / post_covariant | 0.0005401976 ± 0.000304 | 0.00058275 ± 0.000244 | 0.0006265703 ± 0.000338 | 0.0005401976 ± 0.000304 |

## Paired test/direct differences

Alternative minus base; negative is improvement. All five paired deltas are available in CSV/JSON.

| h / model / checkpoint | Alternative - base | Mean ± SD | Improved / 5 | Worsened / 5 |
|---|---|---|---|---|
| 0.4 / mlp / selected | scaled - physical | -0.02550305 ± 0.0571 | 3 | 2 |
| 0.4 / mlp / selected | matched - physical | -9.415641e-05 ± 0.000235 | 4 | 1 |
| 0.4 / mlp / selected | scaled - matched | -0.02540889 ± 0.0568 | 3 | 2 |
| 0.4 / mlp / selected | canonical - physical | 0 ± 0 | 0 | 0 |
| 0.4 / mlp / final | scaled - physical | -0.02550305 ± 0.0571 | 3 | 2 |
| 0.4 / mlp / final | matched - physical | -9.415641e-05 ± 0.000235 | 4 | 1 |
| 0.4 / mlp / final | scaled - matched | -0.02540889 ± 0.0568 | 3 | 2 |
| 0.4 / mlp / final | canonical - physical | 0 ± 0 | 0 | 0 |
| 0.4 / transformer / selected | scaled - physical | 0.0003430344 ± 0.000563 | 3 | 2 |
| 0.4 / transformer / selected | matched - physical | -0.0001991149 ± 0.000293 | 4 | 1 |
| 0.4 / transformer / selected | scaled - matched | 0.0005421492 ± 0.000854 | 2 | 3 |
| 0.4 / transformer / selected | canonical - physical | 0 ± 0 | 0 | 0 |
| 0.4 / transformer / final | scaled - physical | 0.0003430344 ± 0.000563 | 3 | 2 |
| 0.4 / transformer / final | matched - physical | -0.0001991149 ± 0.000293 | 4 | 1 |
| 0.4 / transformer / final | scaled - matched | 0.0005421492 ± 0.000854 | 2 | 3 |
| 0.4 / transformer / final | canonical - physical | 0 ± 0 | 0 | 0 |
| 0.6 / mlp / selected | scaled - physical | -0.0008291508 ± 0.000218 | 5 | 0 |
| 0.6 / mlp / selected | matched - physical | -0.0008529828 ± 0.000451 | 5 | 0 |
| 0.6 / mlp / selected | scaled - matched | 2.383202e-05 ± 0.000363 | 3 | 2 |
| 0.6 / mlp / selected | canonical - physical | 0 ± 0 | 0 | 0 |
| 0.6 / mlp / final | scaled - physical | -0.0008291508 ± 0.000218 | 5 | 0 |
| 0.6 / mlp / final | matched - physical | -0.0008529828 ± 0.000451 | 5 | 0 |
| 0.6 / mlp / final | scaled - matched | 2.383202e-05 ± 0.000363 | 3 | 2 |
| 0.6 / mlp / final | canonical - physical | 0 ± 0 | 0 | 0 |
| 0.6 / transformer / selected | scaled - physical | 3.771451e-05 ± 0.000261 | 3 | 2 |
| 0.6 / transformer / selected | matched - physical | 5.720768e-05 ± 0.000223 | 1 | 4 |
| 0.6 / transformer / selected | scaled - matched | -1.949317e-05 ± 0.000178 | 3 | 2 |
| 0.6 / transformer / selected | canonical - physical | 0 ± 0 | 0 | 0 |
| 0.6 / transformer / final | scaled - physical | 7.010942e-05 ± 0.000319 | 3 | 2 |
| 0.6 / transformer / final | matched - physical | 6.108381e-05 ± 0.000297 | 2 | 3 |
| 0.6 / transformer / final | scaled - matched | 9.02561e-06 ± 0.000188 | 2 | 3 |
| 0.6 / transformer / final | canonical - physical | 0 ± 0 | 0 | 0 |
