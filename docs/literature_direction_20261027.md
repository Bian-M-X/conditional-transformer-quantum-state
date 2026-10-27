# 选题决策的来源核验与先例排查

2026-10-27，Issue17。定向检索，不是穷尽式系统综述或文献首创证明。主代理与两个只读文献/物理子代理核验；这是AI辅助团队核查，不是独立人类阅读认证。摘要、元数据、正文相关章节三个范围分开记录；没有把未读全文的性能陈述当作本项目复现。

## 核心来源与其否决作用

| 来源 | 本次实际阅读范围 | 对方向的约束 |
|---|---|---|
| Zhang–Di Ventra，TQS，[2208.01758v4](https://arxiv.org/html/2208.01758v4) | 子代理正文Fig2–5、附录G.2；沿用课程阅读记录 | 参数泛化、微调与虚构临界行为已有；raw简化实现不能代表完整TQS普遍失败 |
| Rende等，FNQS，[2502.09488v3](https://arxiv.org/html/2502.09488v3) | 子代理摘要、主结果与Methods中的coupling-space QGT | 多哈密顿量共享、参数导数/相图用途已有 |
| Zhang–Jiang–Luo，FNEH，[2608.14208v1](https://arxiv.org/html/2608.14208v1) | 主代理与子代理正文方法、DSS、Fig4e及Discussion | 算符缓存及累计查询成本已覆盖；不能把小系统交叉曲线当首创。该文是预印本 |
| Sarkar–Lee，EC主动学习，[2107.13449v2](https://arxiv.org/html/2107.13449v2) | 子代理约束方程、误差估计与补充误差几何 | 方差式代理及加点已有，需检查准确快照/局部误差展开的假设 |
| Steih–Urban，[2014作者原稿](https://www.uni-ulm.de/fileadmin/website_uni_ulm/mawi2/dokumente/preprint-server/2014/1402_su-adap-20140505-submitted.pdf)；Ali–Steih–Urban，[arXiv1407.1708](https://arxiv.org/abs/1407.1708)、[修订稿](https://www.uni-ulm.de/fileadmin/website_uni_ulm/mawi.inst.070/urban/papers/ASU-Revised.pdf) | 主代理原稿结构/元数据，两个子代理§3.3–3.4、Eq3.15、§4.2及修订稳定性假设 | 重选旧参数时精修快照与总成本讨论已有；原稿与修订稿的作者、具体策略不可混写 |
| Li–Yang–Zhao，FeynmanNet–EC，[2606.12998v3](https://arxiv.org/html/2606.12998v3) | 主代理官方元数据/摘要；子代理§II.1、§III与讨论 | 已有VMC训练态和EC；Gram采样噪声限制基底数并采用截断。PRC114,024302(2026) |
| Zhang–Wang，多态变分原理，[2504.21459v1](https://arxiv.org/html/2504.21459v1) | 主代理摘要、目标与附录讨论；未重现算法 | Tr(S^-1 H)整体低能子空间优化已有；不以多头或子空间输出本身立题 |
| Heightman等，纯态拓扑障碍，[2609.07591v1](https://arxiv.org/html/2609.07591v1) | 子代理摘要、纯态障碍与operator-valued Theorem3 | 当前有限正实TFIM族不满足相应非平凡丛前提；operator-valued出路亦已有。预印本 |
| Campostrini–Pelissetto–Vicari，边界场Ising，[1507.08391](https://arxiv.org/html/1507.08391) | 主代理官方元数据/摘要，物理子代理§3.1 Eq9–19 | 辅助边界自旋/JW强参考需保留；单边极限须核对扇区，不能直接套等强双边最终公式 |
| Epperly–Lin–Nakatsukasa，QSD理论，[2110.07492v2](https://arxiv.org/abs/2110.07492v2) | 主代理官方摘要/期刊信息，未通读证明 | 含噪广义本征问题与截断稳定性已系统研究，不能仅重写简单扰动界 |
| Lee–Lee–Huh，QKSD采样误差，[2307.16279v3](https://arxiv.org/abs/2307.16279v3) | 主代理官方摘要/期刊信息，未通读证明 | 条件数、采样误差和坏基底剔除已有；Quantum8,1477(2024) |
| Nakamura–Yano–Yoshioka，自适应测量，[2311.07893](https://arxiv.org/abs/2311.07893) | 主代理原始论文检索结果摘要；出版社全文被robots限制 | 测量预算自适应已有；本次未读全文，下一轮需精读并移植可比基线，不直接照搬其硬件性能 |
| Szołdra，NQS overlap误差，[2311.14820](https://arxiv.org/abs/2311.14820) | 主代理原始论文检索摘要 | 振幅比可能无界，不能只看样本方差就宣布普适置信界；需继续读其自回归特例 |
| Rende等，NQS微调，[2403.07795v3](https://arxiv.org/abs/2403.07795v3) | 子代理元数据与摘要；本文未重新读全部正文 | warm start及廉价微调应成为强基线，不与从头训练一种方式比较 |
| Cook等，Parametric matrix models，[期刊原文](https://www.nature.com/articles/s41467-025-61362-4) | 主代理摘要和比较EC的结果段，非通读 | 以可学习小矩阵替代大系统也是已知邻近方向；不是本次默认转题方案 |

## 检索问题与范围

检索日期为2026-10-27。主要问题/检索串：conditional NQS与foundation/subspace/near-degeneracy；inexact snapshots adaptive accuracy greedy；neural quantum states active learning optimization budget；quantum subspace adaptive measurement/Gram noise；Born sampling of linear combinations；boundary-field TFIM free-fermion mapping。只把作者原稿、arXiv、出版社和官方代码作为技术证据，搜索聚合站仅用于定位。

本次排除了几项显而易见的首创叙事：第一次共享多个Hamiltonian、第一次NQS+Ritz、第一次比较总查询成本、第一次精修旧点、第一次学低能子空间、第一次给含噪Gram做截断。尚未证实的是这些因素在非凸NQS中共同存在时是否形成可泛化的预算决策机制。

## 阅读优先级和应产出的东西

1. 精读2014自适应快照原稿与Sarkar–Lee，把输入信息、内层精度假设与具体规则改写成可运行对照；注明NQS缺少哪些证书。
2. 精读FNEH与FeynmanNet–EC，复核缓存、DSS/混合采样、Gram截断与总成本口径；不能实现同等功能时标为简化基线。
3. 精读QSD与自适应测量、NQS overlap误差，再设计M动作；需要先证明或验证估计器假设，不能先许诺高概率保真度保证。
4. 仅在主方向通过先例排查后，开展新的N=8先导；其后再冻结N=10与第二家族验证。所有上述新性能实验本次均未执行。

## 对历史文献地图的勘误

- “单个破缺分支与猫态正交，F约0.5”应改为“平方重叠约1/2”；正交对应平方重叠0。
- even投影收益同时改变可表示集合和优化几何，不能单独否定容量解释。
- FNEH Fig4e已有累计wall成本与交叉点，因此旧路线C只能作强基线或进一步限定的新任务，不能称为尚未被回答的一般问题。
- 修旧锚点/加新锚点的抽象动作及自适应精度已有降阶先例，必须比较而非只在NQS文献内宣称空白。
- 原地图为历史写作快照，其PDF不在本次改写；最新选题判断以本核验表及research_direction_20261027.md为准。
