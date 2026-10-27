# 第七轮完整统计

统计单位为种子，均值与样本SD为描述量。详见CSV与summary.json。

|模型|优化器|训练|覆盖|训练达标数|最差锚点1-F均值|
|---|---|---|---|---|---|
|mlp|lbfgs|even|narrow|5/5|2.2366475e-06|
|mlp|lbfgs|even|wide|5/5|2.0529886e-05|
|mlp|lbfgs|raw|narrow|4/5|0.0066424|
|mlp|lbfgs|raw|wide|0/5|0.44553298|
|transformer|adam|even|narrow|5/5|7.4070116e-05|
|transformer|adam|even|wide|5/5|0.00027011137|
|transformer|adam|raw|narrow|1/5|0.22810056|
|transformer|adam|raw|wide|1/5|0.38220186|
|transformer|lbfgs|even|narrow|5/5|5.53027e-06|
|transformer|lbfgs|even|wide|5/5|0.00021194026|
|transformer|lbfgs|raw|narrow|3/5|0.087832585|
|transformer|lbfgs|raw|wide|0/5|0.44650725|

## 各区域与方法

|模型/优化器/训练/覆盖|区域|方法|平均1-F（均值 ± SD）|
|---|---|---|---|
|mlp/lbfgs/even/narrow|all_test|direct|0.00056850427 ± 0.0002838|
|mlp/lbfgs/even/narrow|all_test|lookup|0.016529882 ± 1.415e-05|
|mlp/lbfgs/even/narrow|all_test|train_span_ritz|0.001584336 ± 1.771e-05|
|mlp/lbfgs/even/narrow|common_extrapolation|direct|0.0016985376 ± 0.0008664|
|mlp/lbfgs/even/narrow|common_extrapolation|lookup|0.044249463 ± 2.656e-05|
|mlp/lbfgs/even/narrow|common_extrapolation|train_span_ritz|0.0045825878 ± 4.813e-05|
|mlp/lbfgs/even/narrow|common_interpolation|direct|3.1166621e-05 ± 2.16e-05|
|mlp/lbfgs/even/narrow|common_interpolation|lookup|0.00054821329 ± 3.06e-06|
|mlp/lbfgs/even/narrow|common_interpolation|train_span_ritz|5.6824465e-05 ± 1.568e-06|
|mlp/lbfgs/even/narrow|coverage_extension|direct|0.0001239825 ± 7.192e-05|
|mlp/lbfgs/even/narrow|coverage_extension|lookup|0.007726448 ± 1.349e-05|
|mlp/lbfgs/even/narrow|coverage_extension|train_span_ritz|0.00048128078 ± 7.623e-06|
|mlp/lbfgs/even/wide|all_test|direct|0.0029210033 ± 0.004199|
|mlp/lbfgs/even/wide|all_test|lookup|0.0032070834 ± 4.822e-05|
|mlp/lbfgs/even/wide|all_test|train_span_ritz|0.00048116732 ± 9.747e-06|
|mlp/lbfgs/even/wide|common_extrapolation|direct|0.0069276874 ± 0.01061|
|mlp/lbfgs/even/wide|common_extrapolation|lookup|0.0054176539 ± 3.093e-05|
|mlp/lbfgs/even/wide|common_extrapolation|train_span_ritz|0.00056002083 ± 1.99e-05|
|mlp/lbfgs/even/wide|common_interpolation|direct|0.0016994599 ± 0.00207|
|mlp/lbfgs/even/wide|common_interpolation|lookup|0.0030622441 ± 4.425e-05|
|mlp/lbfgs/even/wide|common_interpolation|train_span_ritz|0.0005208313 ± 1.301e-05|
|mlp/lbfgs/even/wide|coverage_extension|direct|0.0008321477 ± 0.001|
|mlp/lbfgs/even/wide|coverage_extension|lookup|0.0016577849 ± 6.444e-05|
|mlp/lbfgs/even/wide|coverage_extension|train_span_ritz|0.0003922792 ± 2.839e-05|
|mlp/lbfgs/raw/narrow|all_test|direct|0.015572343 ± 0.02873|
|mlp/lbfgs/raw/narrow|all_test|lookup|0.019292961 ± 0.00588|
|mlp/lbfgs/raw/narrow|all_test|train_span_ritz|0.007005817 ± 0.01167|
|mlp/lbfgs/raw/narrow|common_extrapolation|direct|0.0364318 ± 0.06596|
|mlp/lbfgs/raw/narrow|common_extrapolation|lookup|0.047700189 ± 0.007323|
|mlp/lbfgs/raw/narrow|common_extrapolation|train_span_ritz|0.014605587 ± 0.02161|
|mlp/lbfgs/raw/narrow|common_interpolation|direct|0.00064363279 ± 0.001179|
|mlp/lbfgs/raw/narrow|common_interpolation|lookup|0.0016637141 ± 0.002379|
|mlp/lbfgs/raw/narrow|common_interpolation|train_span_ritz|0.00078650885 ± 0.001565|
|mlp/lbfgs/raw/narrow|coverage_extension|direct|0.011124282 ± 0.0215|
|mlp/lbfgs/raw/narrow|coverage_extension|lookup|0.011209476 ± 0.007423|
|mlp/lbfgs/raw/narrow|coverage_extension|train_span_ritz|0.0059704708 ± 0.01179|
|mlp/lbfgs/raw/wide|all_test|direct|0.15756516 ± 0.016|
|mlp/lbfgs/raw/wide|all_test|lookup|0.15369656 ± 0.01258|
|mlp/lbfgs/raw/wide|all_test|train_span_ritz|0.14309499 ± 0.009691|
|mlp/lbfgs/raw/wide|common_extrapolation|direct|0.24100069 ± 0.01479|
|mlp/lbfgs/raw/wide|common_extrapolation|lookup|0.22765455 ± 0.01893|
|mlp/lbfgs/raw/wide|common_extrapolation|train_span_ritz|0.25926489 ± 0.01978|
|mlp/lbfgs/raw/wide|common_interpolation|direct|0.039465879 ± 0.007879|
|mlp/lbfgs/raw/wide|common_interpolation|lookup|0.046264696 ± 0.002672|
|mlp/lbfgs/raw/wide|common_interpolation|train_span_ritz|0.018048461 ± 0.002057|
|mlp/lbfgs/raw/wide|coverage_extension|direct|0.18356297 ± 0.02364|
|mlp/lbfgs/raw/wide|coverage_extension|lookup|0.17880197 ± 0.01534|
|mlp/lbfgs/raw/wide|coverage_extension|train_span_ritz|0.14975245 ± 0.01082|
|transformer/adam/even/narrow|all_test|direct|0.0069640291 ± 0.001902|
|transformer/adam/even/narrow|all_test|lookup|0.016616202 ± 0.0001601|
|transformer/adam/even/narrow|all_test|train_span_ritz|0.001655279 ± 0.0001362|
|transformer/adam/even/narrow|common_extrapolation|direct|0.018902935 ± 0.004815|
|transformer/adam/even/narrow|common_extrapolation|lookup|0.044343522 ± 0.0002827|
|transformer/adam/even/narrow|common_extrapolation|train_span_ritz|0.0047029099 ± 0.0002961|
|transformer/adam/even/narrow|common_interpolation|direct|0.00065792852 ± 0.0003886|
|transformer/adam/even/narrow|common_interpolation|lookup|0.0005909599 ± 7.021e-05|
|transformer/adam/even/narrow|common_interpolation|train_span_ritz|9.7778772e-05 ± 3.435e-05|
|transformer/adam/even/narrow|coverage_extension|direct|0.0027394252 ± 0.000897|
|transformer/adam/even/narrow|coverage_extension|lookup|0.0078396425 ± 0.0001534|
|transformer/adam/even/narrow|coverage_extension|train_span_ritz|0.00053768106 ± 0.0001017|
|transformer/adam/even/wide|all_test|direct|0.0025391441 ± 0.001174|
|transformer/adam/even/wide|all_test|lookup|0.0031834567 ± 0.00013|
|transformer/adam/even/wide|all_test|train_span_ritz|0.00050980697 ± 2.963e-05|
|transformer/adam/even/wide|common_extrapolation|direct|0.0019992192 ± 0.0003563|
|transformer/adam/even/wide|common_extrapolation|lookup|0.0053388305 ± 0.0002013|
|transformer/adam/even/wide|common_extrapolation|train_span_ritz|0.00058303856 ± 4.969e-05|
|transformer/adam/even/wide|common_interpolation|direct|0.0037586926 ± 0.002902|
|transformer/adam/even/wide|common_interpolation|lookup|0.0030644653 ± 0.0001315|
|transformer/adam/even/wide|common_interpolation|train_span_ritz|0.00056235401 ± 3.482e-05|
|transformer/adam/even/wide|coverage_extension|direct|0.0020294264 ± 0.0008417|
|transformer/adam/even/wide|coverage_extension|lookup|0.00165617 ± 0.0001429|
|transformer/adam/even/wide|coverage_extension|train_span_ritz|0.000415473 ± 4.961e-05|
|transformer/adam/raw/narrow|all_test|direct|0.13236287 ± 0.08681|
|transformer/adam/raw/narrow|all_test|lookup|0.11263652 ± 0.09102|
|transformer/adam/raw/narrow|all_test|train_span_ritz|0.12847277 ± 0.1061|
|transformer/adam/raw/narrow|common_extrapolation|direct|0.19331858 ± 0.08813|
|transformer/adam/raw/narrow|common_extrapolation|lookup|0.15360068 ± 0.1009|
|transformer/adam/raw/narrow|common_extrapolation|train_span_ritz|0.18203807 ± 0.1361|
|transformer/adam/raw/narrow|common_interpolation|direct|0.059688671 ± 0.07024|
|transformer/adam/raw/narrow|common_interpolation|lookup|0.059507749 ± 0.06422|
|transformer/adam/raw/narrow|common_interpolation|train_span_ritz|0.055395539 ± 0.06625|
|transformer/adam/raw/narrow|coverage_extension|direct|0.14115174 ± 0.1041|
|transformer/adam/raw/narrow|coverage_extension|lookup|0.12175998 ± 0.1051|
|transformer/adam/raw/narrow|coverage_extension|train_span_ritz|0.14310672 ± 0.1194|
|transformer/adam/raw/wide|all_test|direct|0.14519317 ± 0.08335|
|transformer/adam/raw/wide|all_test|lookup|0.14424711 ± 0.08012|
|transformer/adam/raw/wide|all_test|train_span_ritz|0.14341323 ± 0.08215|
|transformer/adam/raw/wide|common_extrapolation|direct|0.19767236 ± 0.1096|
|transformer/adam/raw/wide|common_extrapolation|lookup|0.196191 ± 0.1066|
|transformer/adam/raw/wide|common_extrapolation|train_span_ritz|0.21387224 ± 0.1193|
|transformer/adam/raw/wide|common_interpolation|direct|0.06328539 ± 0.05627|
|transformer/adam/raw/wide|common_interpolation|lookup|0.068447962 ± 0.05069|
|transformer/adam/raw/wide|common_interpolation|train_span_ritz|0.055006209 ± 0.05407|
|transformer/adam/raw/wide|coverage_extension|direct|0.16726461 ± 0.0955|
|transformer/adam/raw/wide|coverage_extension|lookup|0.16213854 ± 0.09012|
|transformer/adam/raw/wide|coverage_extension|train_span_ritz|0.15687423 ± 0.08964|
|transformer/lbfgs/even/narrow|all_test|direct|0.0031325839 ± 0.003186|
|transformer/lbfgs/even/narrow|all_test|lookup|0.016532331 ± 5.076e-06|
|transformer/lbfgs/even/narrow|all_test|train_span_ritz|0.0016025397 ± 5.833e-05|
|transformer/lbfgs/even/narrow|common_extrapolation|direct|0.008672953 ± 0.00867|
|transformer/lbfgs/even/narrow|common_extrapolation|lookup|0.044253113 ± 6.696e-06|
|transformer/lbfgs/even/narrow|common_extrapolation|train_span_ritz|0.004626917 ± 0.0001531|
|transformer/lbfgs/even/narrow|common_interpolation|direct|0.00038084429 ± 0.0005191|
|transformer/lbfgs/even/narrow|common_interpolation|lookup|0.00054691153 ± 4.711e-06|
|transformer/lbfgs/even/narrow|common_interpolation|train_span_ritz|5.9874391e-05 ± 3.005e-06|
|transformer/lbfgs/even/narrow|coverage_extension|direct|0.0010411119 ± 0.001095|
|transformer/lbfgs/even/narrow|coverage_extension|lookup|0.00773081 ± 6.937e-06|
|transformer/lbfgs/even/narrow|coverage_extension|train_span_ritz|0.00049125573 ± 2.912e-05|
|transformer/lbfgs/even/wide|all_test|direct|0.0056513361 ± 0.004441|
|transformer/lbfgs/even/wide|all_test|lookup|0.003247418 ± 0.0001499|
|transformer/lbfgs/even/wide|all_test|train_span_ritz|0.00053302045 ± 0.0001098|
|transformer/lbfgs/even/wide|common_extrapolation|direct|0.0067346828 ± 0.01245|
|transformer/lbfgs/even/wide|common_extrapolation|lookup|0.0053754574 ± 5.293e-05|
|transformer/lbfgs/even/wide|common_extrapolation|train_span_ritz|0.00060305628 ± 6.964e-05|
|transformer/lbfgs/even/wide|common_interpolation|direct|0.0085384752 ± 0.007586|
|transformer/lbfgs/even/wide|common_interpolation|lookup|0.0032215457 ± 0.0003985|
|transformer/lbfgs/even/wide|common_interpolation|train_span_ritz|0.00063692865 ± 0.0002437|
|transformer/lbfgs/even/wide|coverage_extension|direct|0.0026734719 ± 0.001478|
|transformer/lbfgs/even/wide|coverage_extension|lookup|0.0016707926 ± 0.0001128|
|transformer/lbfgs/even/wide|coverage_extension|train_span_ritz|0.00040256242 ± 4.349e-05|
|transformer/lbfgs/raw/narrow|all_test|direct|0.061495175 ± 0.07497|
|transformer/lbfgs/raw/narrow|all_test|lookup|0.051553957 ± 0.06728|
|transformer/lbfgs/raw/narrow|all_test|train_span_ritz|0.053469259 ± 0.09283|
|transformer/lbfgs/raw/narrow|common_extrapolation|direct|0.10848664 ± 0.09988|
|transformer/lbfgs/raw/narrow|common_extrapolation|lookup|0.086058479 ± 0.08002|
|transformer/lbfgs/raw/narrow|common_extrapolation|train_span_ritz|0.086016899 ± 0.1406|
|transformer/lbfgs/raw/narrow|common_interpolation|direct|0.011142755 ± 0.0217|
|transformer/lbfgs/raw/narrow|common_interpolation|lookup|0.017055483 ± 0.03248|
|transformer/lbfgs/raw/narrow|common_interpolation|train_span_ritz|0.011850323 ± 0.02353|
|transformer/lbfgs/raw/narrow|coverage_extension|direct|0.064015894 ± 0.1009|
|transformer/lbfgs/raw/narrow|coverage_extension|lookup|0.051549422 ± 0.08382|
|transformer/lbfgs/raw/narrow|coverage_extension|train_span_ritz|0.060272732 ± 0.1092|
|transformer/lbfgs/raw/wide|all_test|direct|0.1593687 ± 0.03193|
|transformer/lbfgs/raw/wide|all_test|lookup|0.15952616 ± 0.02831|
|transformer/lbfgs/raw/wide|all_test|train_span_ritz|0.152728 ± 0.03248|
|transformer/lbfgs/raw/wide|common_extrapolation|direct|0.23576512 ± 0.02255|
|transformer/lbfgs/raw/wide|common_extrapolation|lookup|0.22788003 ± 0.03745|
|transformer/lbfgs/raw/wide|common_extrapolation|train_span_ritz|0.24852001 ± 0.03981|
|transformer/lbfgs/raw/wide|common_interpolation|direct|0.05243317 ± 0.05001|
|transformer/lbfgs/raw/wide|common_interpolation|lookup|0.05995619 ± 0.03341|
|transformer/lbfgs/raw/wide|common_interpolation|train_span_ritz|0.038549516 ± 0.04722|
|transformer/lbfgs/raw/wide|coverage_extension|direct|0.18227304 ± 0.04954|
|transformer/lbfgs/raw/wide|coverage_extension|lookup|0.18293823 ± 0.03096|
|transformer/lbfgs/raw/wide|coverage_extension|train_span_ritz|0.16651785 ± 0.03607|
