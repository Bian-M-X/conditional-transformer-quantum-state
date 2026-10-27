"""Write the course report and learning/process records from frozen evidence."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'docs/results/round7'


def main():
    s = json.loads((RESULT / 'summary.json').read_text(encoding='utf-8'))
    assert s['completed_jobs'] == 60 and not s['failed_jobs']

    def cell(mode, coverage, region='all_test', method='direct', architecture='transformer', endpoint='selected'):
        return next(c for c in s['cells'] if all(c[k] == v for k, v in dict(
            architecture=architecture, optimizer='lbfgs', training_mode=mode, coverage=coverage,
            region=region, method=method, endpoint=endpoint).items()))

    def fmt(c, key='mean_infidelity'):
        q = c[key]
        return f"{q['mean']:.6g} ± {q['sample_sd']:.3g}"

    def table(headers, rows):
        return '\n'.join(['|'+'|'.join(headers)+'|', '|'+'|'.join(['---']*len(headers))+'|'] +
                         ['|'+'|'.join(map(str, r))+'|' for r in rows])

    gates = table(['优化器/训练', 'narrow达标数', 'wide达标数', 'narrow最坏锚点1-F均值', 'wide最坏锚点1-F均值'], [
        [f'{opt}/{mode}', *[str(next(g['pass_count'] for g in s['anchor_gates'] if
          (g['architecture'],g['optimizer'],g['training_mode'],g['coverage']) ==
          ('transformer',opt,mode,c)))+'/5' for c in ['narrow','wide']],
          *[f"{next(g['worst_anchor_infidelity']['mean'] for g in s['anchor_gates'] if (g['architecture'],g['optimizer'],g['training_mode'],g['coverage']) == ('transformer',opt,mode,c)):.6g}" for c in ['narrow','wide']]]
        for opt in ['adam','lbfgs'] for mode in ['raw','even']])
    region_rows = table(['训练/覆盖', '共同插值1-F', '覆盖延伸1-F', '共同外推1-F'], [
        [mode+'/'+cov, *[fmt(cell(mode,cov,r)) for r in ['common_interpolation','coverage_extension','common_extrapolation']]]
        for mode in ['raw','even'] for cov in ['narrow','wide']])
    paired = []
    for i, seed in enumerate([101,211,307,401,503]):
        vals = [next(p['wide_minus_narrow']['values'][i] for p in s['coverage_pairs'] if
                     (p['architecture'],p['optimizer'],p['training_mode'],p['region']) ==
                     ('transformer','lbfgs',mode,'common_extrapolation')) for mode in ['raw','even']]
        paired.append([seed,*[f'{v:+.6g}' for v in vals]])
    paired_table = table(['种子', 'raw：wide减narrow', 'even：wide减narrow'],paired)
    baselines = table(['训练/覆盖', 'direct', '同源lookup', '同源train-span Ritz'],[
        [m+'/'+c,*[fmt(cell(m,c,method=b)) for b in ['direct','lookup','train_span_ritz']]]
        for m in ['raw','even'] for c in ['narrow','wide']])
    physical = table(['训练/覆盖', '每自旋能量误差/J', '绝对磁化误差'],[
        [m+'/'+c,fmt(cell(m,c),'mean_energy_error_per_spin'),fmt(cell(m,c),'mean_abs_mz_error')]
        for m in ['raw','even'] for c in ['narrow','wide']])
    pages = [r'''# 条件Transformer对未见横场的泛化
## 第七轮：训练可靠性与参数覆盖范围

量子机器学习结课研究 · 课题12 · 2026年10月27日

### 摘要

条件网络在训练点尚未学准时，扩大训练区间能否改善外推便难以单独判断。本轮固定六自旋开放Ising链，使用同样的三个训练点预算，将横场覆盖从[0.4,1.2]扩大为[0.2,1.4]，以五个新种子比较未见横场上的能量、平方保真度与磁化。60组事前冻结实验全部完成。主比较的原始自回归Transformer在宽区间没有种子通过训练精度门槛，共同外推平均不保真度的配对差为+0.1273，未支持覆盖改善。作为预定物理对照，正宇称训练在两种覆盖下均达到训练门槛；扩大区间仍使五个种子的共同插值误差全部增大。窄区间条件预测可优于同源查表，但优势随覆盖和强基准改变。这些结果把训练点精度、布点干预和预测规则区分开来，给出了本题有限系统中泛化成立的具体条件。

### 本轮承接的问题

原题问“一套权重能否在未见h上给出正确量子态”。前六轮已显示，低能量误差、共享网络、更多预算和更强优化之间存在混杂。本轮回到课程指定的h>0、固定链长任务，只研究一个主问题：在固定点数与训练预算下，扩大横场覆盖范围能否改善外推？

|已有工作|它怎样约束本轮|
|---|---|
|第1轮建立多h能量训练、查表及对称性对照|沿用可信模型和物理实现，先跨环境复跑一个原配置|
|第2至4轮检查选模、缩放与坐标等价性|共同验证能量选模；保留最终接受参数；不重做缩放叙事|
|第5至6轮检查优化、独立化与容量|同时保留Adam、L-BFGS与MLP；不把失败唯一归因于参数共享|

主假设是“扩大覆盖能降低共同外推误差”。竞争解释是低场训练本身更困难，或相同点数扩大范围后点间距变大。两种覆盖同时改变这些因素，因此本实验检验明确的布点方案，而不是孤立估计距离的普适因果效应。

本轮是新的数值研究记录。它以本地冻结协议和可运行证据支持结论；团队成员的独立复核与个人答辩理解另行记录。
''', r'''## 1. 从原文到本题的模型

采用开放边界、J=1、h>0，计算基按最左自旋为最高位排列。Pauli矩阵本征值为±1，能量以J计。

$$H(h)=-\sum_{i=1}^{5}Z_iZ_{i+1}-h\sum_{i=1}^{6}X_i.$$

$$p_\theta(\mathbf{s}\mid h)=\prod_{i=1}^{6}p_\theta(s_i\mid\mathbf{s}_{<i},h),\qquad\psi_\theta(\mathbf{s};h)=\sqrt{p_\theta(\mathbf{s}\mid h)}.$$

图1对应必读论文图1及式(1)的条件输入、右移与自回归分解[1]。h>0使计算基上的非对角元非正且连通，有限链基态可选为严格正实振幅，因此本题保留振幅输出即可。相位输出和跨链长任务属于原文的更广设置。

![图1　本题网络与原文方法的对应。条件输入实际为[h,0]；第二分量沿用已有接口并固定为零。示意图中的Bernoulli概率以一个logit实现，与两类softmax的概率参数化等价。](results/round7/model_correspondence.png)

预测s_i的输出位于输入序列的第i个自旋位置：它对应BOS或s_(i-1)，可读h、BOS与s_1到s_(i-1)，不能读s_i及其后缀。h token本身只能读自身。这里的自回归对象是自旋构型，序列位置没有物理时间含义。

链式概率的归一化来自对最后一个自旋逐项求和，再依次消去前面的条件概率。它不要求自旋相互独立；历史前缀正是表示关联的条件。训练和评估全程dropout=0，使用同一前缀时条件概率确定。

|关系|本题实际选择|
|---|---|
|沿用原文|条件token、因果注意力、右移自旋、自回归概率、变分能量|
|本组简化|固定L=6、64构型全枚举、实正振幅、小型单层网络、三个h等权能量平均|
|本轮改动|新种子及保留点、窄/宽覆盖配对、接受步上的验证选模、预算与振幅复算|

Transformer为16维、2头、FFN宽32、单层，共2449参数。MLP为两层32宽tanh，共1569参数。二者容量不等，比较用于检查现象是否依赖注意力结构。
''', r'''## 2. 冻结划分、目标与对照

配置为experiments/configs/round7.json，训练源码冻结于3f1aaa9c9faf。以下划分在执行前写入本地Git；它不是外部注册平台的预注册。

![图2　训练、验证与20个保留点。色带是训练区间，三角是训练点；共同插值和共同外推对两种覆盖含义相同。extension在窄覆盖是外推，在宽覆盖是插值，故单列。横场以h/J表示。](results/round7/parameter_split.png)

|用途|横场h/J|
|---|---|
|窄覆盖narrow训练|0.4, 0.8, 1.2|
|宽覆盖wide训练|0.2, 0.8, 1.4|
|共同验证|0.55, 1.05|
|共同插值，6点|0.475, 0.625, 0.725, 0.875, 0.975, 1.125|
|覆盖延伸，8点|0.225, 0.275, 0.325, 0.375, 1.225, 1.275, 1.325, 1.375|
|共同外推，6点|0.08, 0.125, 0.175, 1.475, 1.625, 1.825|

$$\mathcal{L}(\theta)=\frac{1}{3}\sum_{h\in\mathcal{H}_{\rm tr}}\frac{\langle\psi_\theta(h)|H(h)|\psi_\theta(h)\rangle}{6}.$$

Transformer采用Adam/L-BFGS × raw/even × 两种覆盖 × 五种子，共40组；MLP采用L-BFGS × raw/even × 两种覆盖 × 同五种子，共20组。种子为101、211、307、401、503；同架构同种子的起始参数哈希相同。

raw直接使用链式振幅；even使用归一化的psi+Ppsi，P为全自旋翻转。本题P与H对易，唯一正基态属于正宇称。even是显式加入已知物理信息的全枚举变分对照；其最终态的逐自旋直接采样性质需另行分析。

每组1200次完整目标/梯度closure。Adam学习率0.005；L-BFGS学习率1、history=20、max_iter=1、strong-Wolfe[3,4]，所有试探调用计费。每约100次closure后，只在已接受参数上计算共同验证变分能量并选择最低者。selected为主终点，final保留最后接受参数；两者冻结后才调用ED进行评估。
''', r'''## 3. 先检验训练点是否学准

事前门槛为：每个种子三个训练锚点中最大的1-F不超过0.01；每种覆盖至少4/5种子达标。主比较固定为raw Transformer、L-BFGS、selected，五种子的共同外推平均1-F。覆盖改善还要求配对均值为负且至少4/5种子改善。

![图3　Transformer最坏训练锚点不保真度。每个点对应一个种子，黑短线为五种子算术均值，虚线为0.01门槛。纵轴为对数。全部种子保留；达标数与均值回答不同问题。](results/round7/anchor_reliability.png)

@@GATES@@

主配置在narrow有3/5、wide有0/5种子达标，两侧训练门槛未通过。宽覆盖共同外推的平均1-F为0.235765，窄覆盖为0.108487；其差包含训练不充分的影响，不能单独归为覆盖效应。

这种限制有直接物理依据。对非简并基态，谱分解给出总能量误差epsilon与能隙Delta的标准关系：

$$\epsilon=E_\theta-E_0\geq\Delta(1-F),\qquad F=|\langle\psi_0|\psi_\theta\rangle|^2.$$

h=0.2处能隙为1.2288×10^-4 J。以主配置seed401为例，每自旋能量误差仅4.5559×10^-5 J，F却为0.502331，P期望约0.004717。小能隙使这个能量误差不足以约束高保真度；低能的近简并分支可产生这种现象。它与优化未收敛、有限参数化等解释仍有关联，不能凭投影的收益把原因唯一归为某种优化机制。

even对照的两种覆盖均5/5达标，支持继续比较其未见横场误差；它保持控制身份，不取代未通过的主配置。
''', r'''## 4. 未见横场上的三种评价

能量误差为(E-E0)/6；保真度F采用平方定义。非能量物理量选A=|sum Z_i/6|的期望，比较其与ED基态的绝对差。另保存Z1磁化、端到端连通关联、宇称及方差，用于检查只看能量遗漏的态差异。

![图4　L-BFGS选中模型在20个保留h上的结果。上行为raw、下行为even；依次为不保真度、每自旋能量误差和绝对磁化误差。实线带标记为direct五种子均值，阴影为种子最小到最大值；虚线为同源lookup均值。淡色竖线标记各自三个训练h，连接线只引导阅读，不代表连续h保证。误差均无量纲或以J计，阴影不是置信区间。](results/round7/transformer_generalization.png)

@@REGIONS@@

表中先对每个种子的区域网格求均值，再报五种子的均值±样本SD。h点不是独立重复。even中共同插值误差由0.000380844升至0.00853848，覆盖延伸也变差；共同外推的均值下降幅度远小于种子间差异。扩大训练区间同时稀疏了内部覆盖，并未形成整体更准的条件态族。

raw中能量和保真度的趋势明显不同。图中较小的能量误差不能替代对训练点及保留点的态精度检查。
''', r'''## 5. 配对覆盖效应与同源查表

lookup完全取自同一个冻结网络在三个训练h上生成的正振幅：区间内线性插值后归一化，区间外保持最近边界态。它不使用测试精确态，也不另行训练，因此检验的是向网络输入新h是否比保持已学训练态更有价值。

![图5　每个种子的平均不保真度差。左：wide减narrow；右：wide中direct减lookup。负值表示前者更好。点为全部五个配对种子，黑短线为均值，横线为零，纵轴用symlog保留正负号。区域含义与图2一致。](results/round7/paired_comparisons.png)

@@PAIRS@@

主配置共同外推配对差均值+0.127278±0.0889，仅1/5种子改善，且训练门槛失败，预定改善判据未通过。even控制的配对差为-0.00193827±0.0182，4/5改善；seed401反向变化+0.0283321。该控制提供一个有条件的均值改善观察，不能替换主比较，也不能支持稳定或普遍的改善。

even窄覆盖在共同外推上，direct减lookup为-0.0355802±0.00867，五种子均改善；even宽覆盖则为+0.00135923±0.0125，四种子改善却均值变差。共同插值中，even的wide减narrow五种子全部为正。胜出次数、平均幅度和训练可靠性必须一起读。

这些对照排除了“只要同一网络在多个h上降能，就必然学到可靠外推规则”的推断。它们也显示，一部分有用的条件关系确实能被学到：even窄覆盖在本轮共同外推网格上超过了同源边界保持。
''', r'''## 6. 强基准与物理量交叉检查

train-span Ritz以相同冻结训练态张成最多三维子空间，用目标H(h)的投影矩阵求最低Ritz态[5]。它不以精确基态建基，另有后处理矩阵作用成本。它是额外的已知强基准，不能与不做目标h求解的lookup混为一谈。

![图6　20个保留点的每种子平均不保真度，L-BFGS/selected Transformer。三种方法共享各自冻结的训练态，散点为五种子，黑短线为均值。图按130 mm原始宽度插入，以保留字体尺寸。](results/round7/strong_baselines.png)

@@BASELINES@@

表中even宽覆盖Ritz均值0.000533020，低于direct的0.00565134；even窄覆盖也有相同均值次序。优于简单查表不足以建立对强基准的优势，Ritz的能量最优性质也不自动保证保真度最优。

@@PHYSICAL@@

绝对磁化对全局翻转不变；两种相反磁化分支及其对称组合可能给出接近的该物理量，因而即使它准确，量子态仍可能不准。这解释了同时报告F与非能量观测量的必要性。

MLP的even窄覆盖direct平均1-F为0.000568504±0.000284，Transformer为0.00313258±0.00319。该比较未匹配参数量及架构成本，结果只表明此网格上的条件预测收益并不专属于注意力网络。
''', r'''## 7. 可复现性与实际计算预算

60组主运行无失败；共9000条指标，包括25个h × 3方法 × 2终点 × 60组。25个h包含3训练、2验证及20保留点。所有种子、区域和最终参数均保留，未按测试误差更换设置。

|计算或核验|实际记录|
|---|---|
|每组训练工作|1200 closure；3600次前向H作用及3600次反向H作用|
|60组训练总工作|432000次前向加反向H作用|
|验证与ED|每组24次验证h评价；每组冻结后25次ED；合计1440及1500次|
|训练计时之和|334.495秒；含循环与验证选模|
|各组wall time之和|337.795秒；不含环境安装、作图、写作与额外重跑|
|实际环境|Windows 11，Python 3.12.14，torch 2.14.1+cpu，NumPy 2.3.5，SciPy 1.16.3|
|执行方式|CPU、单线程、float64；完整依赖另存requirements-round7-lock.txt|

even在已枚举的64维数组上翻转并归一化，每次仍只需一次原网络全构型调用，但增加投影运算。相同closure和H作用不等于所有计算成本相同；本轮计时没有回答大系统或训练加多次查询的摊销优势。

33项测试通过，包含原29项和4项新增的分区、选模无ED、已接受参数回调及本地manifest检查。全部运行的原始概率归一化最大误差为2.22×10^-16，枚举后缀变化造成的因果性logit最大差为0。训练/评估模式的浮点差异亦在设定容差内。

复算使用保存的振幅、NumPy eigh及物理实现，检查9000条记录；数值最大差1.084×10^-8，能量容差1e-9、保真度容差1e-6，全部通过。该复算覆盖保存状态的评价，不等于重跑所有训练。

原第1轮even Transformer seed17在本轮环境重新训练，1311个数值指标/训练值与原记录最大差0。新固定配置raw/L-BFGS/wide seed101另目录重跑，10136个指标、训练值与振幅数值逐值相同。这两次都是本Agent运行的复现，独立人类成员复核状态仍为待完成。

selected与final均保存。raw宽覆盖全部保留点平均1-F分别为0.159369及0.183999；final未消除该配置的训练问题。所用源码、配置哈希、初始化与终点参数哈希、命令及预算可从每份manifest追溯。
''', r'''## 8. 本轮结论与交接

在固定六自旋、三个训练h与1200 closure的条件下，原始自回归Transformer的宽覆盖训练没有达到预定门槛，主假设未获支持。even物理对照使训练点达到门槛后，覆盖扩大仍造成共同插值退化；共同外推呈现带有大幅反例的条件性均值改善。窄覆盖的条件网络可超过同源查表，强Ritz基准则进一步收紧了这种精度收益的含义。

因此，本题的回答是有条件的：一个条件网络在本轮部分设置中能泛化到未见h，但训练能量下降和更广覆盖本身不足以保证态准确。这个判断来自训练锚点、配对布点和同源预测规则三个相互补充的证据。

最有价值的下一步是由另一成员按固定run和环境复核训练与图表，再在独立开发h上限定地改善raw宽覆盖训练。如果需要比较新的训练规则，应在新的保留点上重新冻结验证；当前20个h已被本轮分析使用。固定L=6的结果用于有限量子态数值逼近，不能从中提取热力学临界指数。

### 课程材料与实际完成状态

|课程要求|本轮位置与状态|
|---|---|
|研究报告PDF|本报告及同名Markdown、6幅图的PNG/PDF与绘图源码|
|文献学习记录|literature_round7.md：原文定位、概念推导、公式与本题差异；成员需另补本人理解|
|可复现材料|完整源码、冻结配置、60组原始数据及2次重跑、依赖锁定、复现脚本与核验结果|
|过程与分工|process_round7.md及handoff：决策、负结果、AI用途和真实待办；不编造成员贡献|

本轮交付是一份可继续审核的阶段研究文件。课程最终成员署名、个人分工、教师另行通知的篇幅/时长、汇报PPT及现场答辩应在小组整合时完成，不能由运行记录代替。

### 参考文献与一手资料

[1] Y.-H. Zhang and M. Di Ventra, Transformer Quantum State: A Multi-Purpose Model for Quantum Many-Body Problems, Physical Review B 107, 075147 (2023). [arXiv v4](https://arxiv.org/abs/2208.01758v4)，重点图1、式(1)、图2/4及附录G.2。

[2] 作者实现：[transformer_quantum_state](https://github.com/yuanhangzhang98/transformer_quantum_state)。用于定位条件编码、右移与因果结构；本题沿用小型实现，不运行其全部任务。

[3] PyTorch 2.14 [LBFGS官方文档](https://docs.pytorch.org/docs/2.14/generated/torch.optim.LBFGS.html)：closure、history_size和strong_wolfe参数。软件版本以本轮锁定环境为准。

[4] PyTorch 2.14 [Adam官方文档](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html)：优化参数定义。本轮显式设置学习率，未通过测试h选参。

[5] A. Sarkar and D. Lee, Self-learning emulators and eigenvector continuation, Physical Review Research 4, 023214 (2022), [arXiv:2107.13449v2](https://arxiv.org/abs/2107.13449v2)。本轮只核验元数据及摘要，用于已知子空间方法的背景；不据此宣称完成其全文方法复现。

项目来源：团队仓库main 65c367c7115d；科学代码冻结3f1aaa9c9faf；新研究在本地分支归档。写作采用APS Quantum Writing 1.2.0，以课程要求决定结构与篇幅。
''']
    replacements = {'@@GATES@@':gates,'@@REGIONS@@':region_rows,'@@PAIRS@@':paired_table,
                    '@@BASELINES@@':baselines,'@@PHYSICAL@@':physical}
    report='\n<!-- pagebreak -->\n'.join(pages)
    for k,v in replacements.items():report=report.replace(k,v)
    (ROOT/'docs/round7_report.md').write_text(report,encoding='utf-8')

    literature=r'''# 第七轮文献与概念学习记录

日期：2026-10-27。记录者：本轮研究Agent。以下用本项目语言解释实际读取的一手资料，不代表任何成员已完成个人阅读。小组应由各成员分别复述和核查关键推导，并留下自己的改写或疑问。

## 必读文献的阅读范围与定位

Y.-H. Zhang and M. Di Ventra, *Transformer Quantum State: A Multi-Purpose Model for Quantum Many-Body Problems*, PRB 107,075147(2023)，用户提供arXiv2208.01758v4的12页全文已读；重点核对第II节、图1、式(1)、TFIM的图2/4，以及附录G.2。[论文版本入口](https://arxiv.org/abs/2208.01758v4)。作者[代码入口](https://github.com/yuanhangzhang98/transformer_quantum_state)已核对，结构性定位不等于完整运行原库。

原文的对象是一套可接收哈密顿量参数及系统信息的Transformer量子态。图1显示参数/系统信息置于序列前方，自旋序列通过自回归结构表示概率，振幅与相位构成波函数。式(1)的关键是将构型联合概率分解为逐位条件概率。本文固定L、h>0，只保留正实振幅，因此是原机制的短链改编。

图2和图4服务于跨横场/参数预测及泛化表现。附录G.2给出一个特别有用的边界：训练区外能量看起来合理，仍可能得到h约1.24而非1的相变位置和错误标度。这个原文反例提醒我们同时评价态与物理量。六自旋有限链没有热力学非解析相变，不能照搬该有限尺寸标度程序来报临界指数。

## 1. 为什么链式分解既归一化又能表达关联？

对两个自旋，p(s1,s2|h)=p(s1|h)p(s2|s1,h)。对s2求和得到p(s1|h)，再对s1求和得到1。L自旋从最后一位依次消去即可证明归一化。第二位的分布随第一位改变，所以这个分解不要求独立。归一化成立依赖每个前缀对应同一个确定条件分布；本题因此让所有dropout在训练及评价时都为0。

## 2. 右移与因果mask分别防止什么？

输入是[h,BOS,s1,...,s5]。预测si用第i个自旋位置的输出，那里实际输入的是BOS或s(i-1)。右移使当前位置不直接含si；上三角mask使它不能从后方读取si及之后的自旋。只有mask而没有右移时，若保留对角自注意力，当前位置会看到自己的标签。

|预测|允许读取的token|必须隐藏|
|---|---|---|
|s1|h,BOS|s1到s5|
|s2|h,BOS,s1|s2到s5|
|s3|h,BOS,s1,s2|s3到s5|
|s4|h,BOS,s1,s2,s3|s4,s5|
|s5|h,BOS,s1,s2,s3,s4|s5|
|s6|h,BOS,s1到s5|输入中没有s6|

该序列是空间自旋标签顺序，不能把Transformer序列位置理解成动力学时间。检查前缀依赖应直接改变后缀而比较相应logit；只看训练loss不能检验泄漏。

## 3. 为什么这里可只用正实振幅？

h>0时，翻转单自旋连接任意两个计算基构型，H非对角矩阵元为-h。对充分大常数c，cI-H是不可约非负矩阵，其最大本征向量可取严格正，由此对应H的唯一基态。归一化实正模型与本题基态符号结构一致，但不意味着任意小网络都容易优化。h=0不在本题训练或保留点内。

## 4. 能量训练怎样区别于精确态监督？

训练loss直接用已知H和网络振幅计算Rayleigh商，在三个训练h上取每自旋平均。全枚举是将随机期望改为精确求和，不是把ED振幅作为标签。本轮共同验证h也只计算变分能量；两个终点冻结后，ED才提供E0、精确态及能隙用于评价。

每个h的能量都有自己的下界，平均能量下降并不保证每个锚点的态同样准确。评估时报告最坏训练锚点与全部种子，可以看到平均loss掩盖的局部失败。

## 5. 为什么低能量误差不能认证基态？

将归一化态展开为sum_n c_n|n>，有epsilon=sum_(n>0)|c_n|^2(E_n-E0) >= Delta(1-|c0|^2)。因此1-F <= epsilon/Delta。Delta很小时，这个上界可以大于1，没有有效的高保真度保证。它是标准谱分解结论，不是本项目新定理。

本轮h=.2的seed401正好展示这一点：能量看起来已很低，F仍约.502。正确宇称扇区内可改用扇区能隙来分析，但这个谱信息仅用于事后解释，不能回流为号称无真值的训练/选模指标。

## 6. 对称性与观测量的作用

P=prod Xi将s映为逐位翻转构型，P与本题H对易，唯一正基态在P=+1扇区。even=normalize(psi+Ppsi)是已知物理约束；它同时改变变分集合和优化几何，改善不能唯一归因于后者。对于额外纵场lambda不为零时，该对照一般失去根据，本轮固定lambda=0。

A=|sum Zi/L|在翻转下不变，因此分支态与对称组合可能有近似同一A。F、P及有符号Z1能补充不同信息；不能因单个物理量准确而宣布完整量子态正确。端到端连通关联也在原始CSV中保存。

## 7. 查表与Ritz实际使用哪些信息？

lookup用同一网络冻结后的三个训练态，在线只按h插值或取最近边界态。Ritz用相同三个态张成子空间，再求目标h的投影哈密顿量最低态。这是不同的预测规则和查询成本。Ritz未读取ED态，但它查询目标H，降能保证也不保证唯一基态保真度提高。

Sarkar和Lee的[论文元数据及摘要](https://arxiv.org/abs/2107.13449v2)确认了eigenvector continuation与主动训练点选择的已有背景。本轮没有重新阅读其全部正文，也没有运行其自学习误差估计器；本项目的有限维Ritz公式和成本由本地实现直接核查。

## 软件资料与落实

已读取PyTorch官方[LBFGS](https://docs.pytorch.org/docs/2.14/generated/torch.optim.LBFGS.html)中closure、history_size、line_search_fn参数和[Adam](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html)参数定义。L-BFGS一次接受更新可能多次调用closure，所以不能拿更新步数直接与Adam相等；本轮统一计全部1200次closure，并只在接受参数上选模。

## 个人学习与答辩核查入口

成员应各自给出：两自旋归一化推导；预测s4的可见token；能量-保真度界的推导；为什么even可用；为何extension区不是共同外推；为何4/5改善仍可能均值变差。应记录实际姓名、日期、手算或运行证据、发现的错误及修改，而不是仅勾选“已读”。当前没有这些成员个人完成证据。
'''
    (ROOT/'docs/literature_round7.md').write_text(literature,encoding='utf-8')
    process=r'''# 第七轮过程、分工与AI辅助记录

## 决策链与证据身份

1. 用户要求在前六轮基础上正式研究，并指定APS Quantum Writing；随后将远程上传改为D盘新目录本地交付。任务登记local-round7-course-generalization，publication=local-only，未虚构GitHub Issue或独立审阅。
2. 在main 65c367c7115d静态审阅基础上，选择课程原题的固定L=6、lambda=0、多h问题。第4—6轮额外纵场任务不能直接充当同一网络跨h的证据；本轮未新增跨尺寸、相位或动力学任务。
3. 使用独立venv，先通过原29项测试并完整重跑第1轮even Transformer seed17。1311个数值逐值一致，确认环境和沿用实现可重现。
4. 冻结60组全因子设计、5个新种子、20保留h、训练/验证分离、主指标和训练门槛。科学协议提交2a3b81b；两个文件统一LF后的执行冻结提交3f1aaa9。换行修正发生在新训练前，未改变数值算法。
5. 完成60组，无失败、无替换种子、无追加训练预算。预定raw宽覆盖训练门槛失败，主改善判据失败；even按控制身份报告。seed401反例全部保留，不以其不利而删除。
6. 保存selected和final振幅及全过程closure/验证记录。独立目录重跑预定固定seed101配置，与10136个数值逐值一致；它不计入60组。
7. 使用NumPy特征分解重算全部9000条指标，最大差1.084e-8；这是评价复算，尚无另一人独立训练与解释的记录。新旧全部测试及归档校验结果见results/round7。
8. 从同一批CSV生成6幅实际数据图/结构图，写报告与学习记录。按APS Quantum Writing 1.2.0组织中心问题、证据和条件，按课程要求保留流程与交付材料。

## 本轮已观察、推导与待验证

observed：训练门槛通过数、所有表格与配对差、因果/归一化、复跑及实际计算预算。derived：标准能量-保真度界、正实基态与正确宇称的理由。hypothesized：扩大布点的收益及竞争解释，结果未支持主配置改善。planned：成员独立复核、独立开发配置上的训练改进与新的冻结评估，均未实施。

## 实际责任记录

|参与者|本轮可证实的工作|尚需本人完成的工作|
|---|---|---|
|用户|提供课程要求、论文和团队仓库；授权研究、写作skill及本地交付|确认终稿署名与真实分工、参加个人学习核查与答辩|
|原团队成员|第1—6轮贡献由原仓库历史记录，不在本轮重新归名|按实际情况补个人阅读/复现/修订与答辩记录|
|本轮Codex研究Agent|环境重建、方案冻结、代码及实验、数值与视觉核验、报告/记录/材料打包|已交付范围内的运行证据随包保存|

当前不填写未知成员姓名、贡献比例或虚构人工审核。小组应基于真实阅读、设计判断、运行复现和结果解释填写分工，不能按生成代码数量评价贡献。

## AI用途、人工修改与校验方式

主要用途：阅读和整理文献与原仓库、编写回调/运行和分析脚本、执行数值计算、生成图表与报告文本。原始数据来自本机实际程序运行；未用生成图像冒充数据。人类提供了任务要求与范围修正；本轮尚无可核验的人工逐句改稿或独立科学复核证据。

校验方式：源码/配置的Git与SHA256、原29+新增4项测试、已接受步和ED隔离检查、全振幅指标复算、原配置跨环境复跑、新配置精确同机复跑、PDF逐页渲染检查。自动化核验的覆盖范围在报告中分别说明，不将其替代人类理解。

## 异常和流程修正

GitHub连接/插件安装存在问题，用户将交付改为D盘。本轮仅本地分支提交，不发布Issue、PR或main。验证器新增受限的本地任务manifest格式，并测试仅接受已登记local-*任务。

首次科学提交的两个文件有CRLF空白告警；训练前统一LF并再次提交，所有主运行均使用后一个冻结提交。此修正及原提交均保留，未重写历史。

## 小组接力时应留存的真实记录

每名成员记录本人阅读了哪一页/公式、复述了哪个方法、运行了哪个命令、发现并修正了什么问题。下一成员可先执行reproduce_round7.ps1 -Mode audit，再按文档在新目录重跑一个完整固定配置。未通过训练门槛的配置不得在后续汇报中改称“纯外推失败”；所有新调参须使用独立开发数据。
'''
    (ROOT/'docs/process_round7.md').write_text(process,encoding='utf-8')
    assessment='''# 第七轮写作与图表自检

采用APS Quantum Writing 1.2.0的PLANNING/WRITING/FIGURES分支；遵照课程要求而非期刊模板。

统一性评估：A。中心问题始终是“固定点数扩大h覆盖能否改善未见h预测”；训练门槛、正宇称控制与同源查表/Ritz分别消除不同替代解释。六幅图按模型→布点→训练→三指标→配对→强基准支持同一判断。

实质判断：主配置没有支持覆盖改善；训练达标的控制也未表现整体单调收益。负结果、seed401反例与强基准保留。正文不把有限预算失败写成不可能性或新算法。

文稿包含完整适用条件、训练/验证/测试、原文/简化/改动、公式、指标定义、五种子不确定性、成本及复现证据。待成员独立复核与个人答辩材料是真实的协作待办；不以自动核验或文稿评分代替科学确认。最终PDF与图表的视觉检查状态见figure_plan及artifact_checks。
'''
    (ROOT/'docs/round7_editorial_assessment.md').write_text(assessment,encoding='utf-8')
    print('Wrote report, literature, process, and editorial assessment.')


if __name__ == '__main__':
    main()
