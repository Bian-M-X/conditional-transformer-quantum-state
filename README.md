# Conditional Transformer Quantum State (CTQS)

**Research status: exploratory / early-stage validation.** 研究目标是考察**近简并区域中条件神经网络量子态的可靠训练与泛化**。第八轮精修/加点先导未过投资门槛，暂缓当前动作菜单上的复杂调度器开发，先独立复核与重新立题。近期文献已覆盖一般摊销交叉点及自适应快照精度，不能将这些想法本身当作新贡献。科学结论仍待独立成员复核。

**当前候选题目（未定稿）**：近简并系统中条件神经网络量子态的训练可靠性与摊销价值

**English**: Training Reliability and Amortization Value of Conditional Neural Quantum States near Quasi-Degeneracies

## 第八轮：精修/加点选择空间的止损判别（Issue #19）

- [报告](docs/round8_report.md) / [PDF](docs/round8_report.pdf) / [冻结协议](docs/round8_preregistration.md) / [交接](docs/handoffs/issue-19.md)
- 40组N8链/梯、MLP/Transformer、5配对种子及一次同机重跑完成；8280行主指标、600完整本地检查点、50项测试通过。独立Kronecker/NumPy复算和重跑回执见[完整结果](docs/results/round8/tables.md)。
- 主端点8条件均未通过投资门槛；7/8中固定中心加点等于逐种子事后最优动作，唯一非零相对空间8.32%，绝对改善2.05e-6。停止当前菜单上的复杂调度器开发，先独立复核与重新立题。
- 新矩阵自由H、缓存仿射Ritz、可恢复优化器/RNG已验证；没有学得调度器、重采样收益、部署成本优势或ICLR成熟度结论。本轮不能替代原题多h共享模型证据。
- 复核：`python scripts/verify_round8.py`；本地保留检查点可加`--local-checkpoints`。源码冻结d2595d7；独立成员复核仍待完成。

## 选题决策的前期记录（2026-10-27，Issue #17）

- [决策正文](docs/research_direction_20261027.md) / [PDF](docs/research_direction_20261027.pdf) / [来源与勘误](docs/literature_direction_20261027.md) / [交接](docs/handoffs/issue-17.md)
- 已同步协作者[第七轮](docs/round7_report.md)和原文献地图；最新main来源44e03c6，保留各分支历史。
- 建议先做“下一单位预算应精修旧态、增加新锚点还是重采样”的可学性判别，再决定是否开发方法。新颖性、强对照、计费探测与跨家族门槛详见正文，尚无新方法结果。
- 本机有16GiB GPU硬件，但现环境仍为CPU版PyTorch；该决策时尚未启动新训练，后续N8先导及止损判断见上方第八轮。采样/GPU扩展仍须另行验证。

## 第六轮研究入口（Issue #11，SR/容量控制与 main 同步）

- [第六轮报告](docs/round6_report.md) / [PDF](docs/round6_report.pdf) / [交接记录](docs/handoffs/issue-11.md)
- [冻结协议](docs/round6_preregistration.md)、[SR 推导与数值检查](docs/theory_round6.md)、[完整统计](docs/results/round6/tables.md)、[main 整合授权记录](docs/decisions/0007-round6-main-integration.md)

60组开发对照及一次同机重跑完成。低场加宽共享 MLP/Transformer 达标5/5、4/5；固定设置的 SR 未稳定解决低场问题，独立化也无一致优势。
加宽 Transformer 虽通过种子数门槛，均值仍被失败种子拉高，不能宣称通用改善。撤回把共享特有失效提升为论文主张的倾向，继续比较强优化、容量与总查询成本。
复核：`.venv\Scripts\python.exe scripts/verify_round6.py`。29项测试、2260行本轮指标与40个初始化对照；N=8/10与完整成本验证尚未执行。
用户明确要求本轮上传同步 main；保留远程 README 改动和全部研究历史。main 的整合不代表独立成员科学复核，结论仍为 provisional。

## 课题 12 文献答案地图与六轮研究史（2026-10-26）

- [正文](docs/topic12_literature_and_history.md) / [PDF](docs/topic12_literature_and_history.pdf)
- 回答“哪些问题已被 TQS / FNQS / FNEH / Heightman 等论文回答、哪些还没有”，并把六轮的动机、结果、被排除的解释和 ICLR 差距串成一条线。
- 这是文献与项目史复核，不是新的实验结论；引用核验范围逐篇标注，独立成员复核仍未完成。
- 此为历史快照；成本交叉点、快照精修先例及猫态表述的更新见[2026-10-27核验与勘误](docs/literature_direction_20261027.md)。

## 第五轮研究入口（Issue #9，路线决策，待独立复核）

- [前四轮路线评估](docs/strategy_after_round4.md) / [第五轮报告](docs/round5_report.md) / [PDF](docs/round5_report.pdf) / [交接记录](docs/handoffs/issue-9.md)
- [冻结协议](docs/round5_preregistration.md)、[完整统计](docs/results/round5/tables.md)、[事后物理见证及边界](docs/theory_round5.md)

完成80组共享/独立 × Adam/L-BFGS 对照及一次同机重跑。低场 Transformer 的可靠拟合种子数依次为0/5、3/5、0/5、5/5；
独立 Adam 仍失败，故共享并非失败必要条件；独立 L-BFGS 的收益仍混有容量与更新分配差异。
强 Ritz 基线进一步要求后续研究证明共享网络的查询总成本价值。暂缓主动选点和局部缩放扫描，下一门槛为 SR、容量/更新控制与新尺寸验证。
全部2840行指标和振幅公开，复核：`.venv\Scripts\python.exe scripts/verify_round5.py`。课程原始多 h、内插/外推证据保持独立入口。

## 第四轮研究入口（Issue #7，待独立复核）

- [第四轮报告](docs/round4_report.md) / [PDF](docs/round4_report.pdf) / [交接记录](docs/handoffs/issue-7.md)
- [冻结协议](docs/round4_preregistration.md)、[坐标等价性证明](docs/theory_round4.md)、[完整统计](docs/results/round4/tables.md)

80 组初始化/优化坐标对照及一次同机重跑已完成。20 对完整补偿与物理输入轨迹逐值一致；
匹配初始函数后的普通 Adam 缩放收益依赖模型与 h，低场训练失败仍在。
全部 10880 行指标与振幅已保存，复核：`.venv\Scripts\python.exe scripts/verify_round4.py`。
这是优化机制对照，不是新优化器或投稿级算法；独立成员复核仍待完成。

## 第三轮研究入口（Issue #5，待独立复核）

- [第三轮报告 PDF](docs/round3_report.pdf) / [正文](docs/round3_report.md) / [交接记录](docs/handoffs/issue-5.md)
- [冻结协议](docs/round3_preregistration.md)、[完整统计](docs/results/round3/tables.md)、[运行索引](experiments/round3_index.json)
- [缩放的函数等价性与 Adam 参数化分析](docs/theory_round3.md)

新完成100组配置：20组旧轨迹补评最终模型、80组尺度/协变/尺寸对照，另有一次同机重跑。
原任务的普通 MLP 最终步仍未解决低场错误；缩放与协变训练的效果依赖模型和尺寸，不能宣称通用收益。
H预算已计入反向传播，网络调用和时间差异另列。复核：`.venv\Scripts\python.exe scripts/verify_round3.py`。
完整保留负面结果；正式结论仍等待其他成员独立复核。

## 第二轮研究入口（Issue #3，待独立复核）

- [第二轮中文报告 PDF](docs/round2_report.pdf) / [正文](docs/round2_report.md) / [交接记录](docs/handoffs/issue-3.md)
- [冻结协议](docs/round2_preregistration.md)、[完整统计](docs/results/round2/tables.md)、[运行索引](experiments/round2_index.json)
- [协变反例与敏感度界](docs/theory_round2.md)、[前向/反向成本修正](docs/decisions/0003-forward-budget-accounting.md)

完成 60 组训练预算配置与 60 组选点配置，每项五个配对种子，其中 12 组复用首轮。
当前选模流程下的低场错误未被增加最大训练步数稳定解决；带符号纵场任务中，主动选点没有通过预设优势标准。
匹配的是 2400 次前向训练加选点 H 作用，**并非总计算成本**；反向成本、失败与重跑均公开。
复核：`.venv\Scripts\python.exe scripts/verify_round2.py`；重建表图：`.venv\Scripts\python.exe scripts/summarize_round2.py`。
独立复跑命令及独立审核要求见交接记录，不要把同机重跑当作跨成员复现。

## 第一轮研究入口（Issue #1，待独立复核）

- [中文研究报告 PDF](docs/round1_report.pdf) / [可编辑正文](docs/round1_report.md)
- [下一位协作者的接力记录](docs/handoffs/issue-1.md)、[考核要求与学习记录](docs/course_requirements.md)
- [冻结实验协议](docs/round1_preregistration.md)、[完整表格](docs/results/round1/tables.md)、[运行索引](experiments/round1_index.json)
- [文献与新颖性边界](docs/literature_round1.md)、[谱推导与反例](docs/theory_round1.md)、[历史材料审计](docs/audit/round1_archive_audit.md)

本轮实跑12组 N=6 基准与一个同机重跑，支持先区分训练点失败与参数泛化。
没有证明新主动采样方法或 ICLR 投稿成熟度；`docs/STATE.md` 的正式结论等待独立成员复核。

研究环境建议使用隔离的 Python 3.13 venv（不继承 Anaconda 系统 site-packages）：

```bash
python -m venv .venv
# Windows: .venv\Scripts\python.exe；Linux/macOS: .venv/bin/python
.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements-research.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts/run_round1.py
.venv\Scripts\python.exe scripts/verify_round1.py
.venv\Scripts\python.exe scripts/summarize_round1.py
```

训练要求先提交 tracked 变更，以保证 manifest 对应实际代码。新复跑产生独立目录；汇总默认仍使用已提交的首轮索引，替换/追加分析索引必须显式记录。
PDF 重建另需 `reportlab` 与 CJK 字体：`python scripts/build_round1_pdf.py --font PATH_TO_CJK_FONT`。
Windows sandbox 若系统临时目录不可写，可将当前进程 `TEMP`/`TMP` 指向仓库 `tmp/`；勿通过允许重复 OpenMP 运行库来绕过依赖冲突。

## 开始协作（3 名研究者 + 各自 Agent；任务轮换）

不按成员永久划分研究主线；按 **GitHub Issue → 独立分支 → 实验记录 → PR → 合并** 轮换接力。每个 Issue 有一个当前负责人，但后续可交给任何成员及其 Agent。Agent 的对话历史不视为项目记录。

1. **进入工作前**依次阅读 [`AGENTS.md`](AGENTS.md)、[`docs/STATE.md`](docs/STATE.md)、[`docs/PROTOCOL.md`](docs/PROTOCOL.md)，然后阅读当前 Issue、对应交接记录及近期提交。
2. 从 GitHub Issues 认领一项任务；没有 Issue 先建 Issue。每位成员同时推进不超过一个需要合并的主要任务，避免竞争修改。
3. 从 `main` 新建 `issue/<编号>-<简述>` 分支，**禁止 Agent 直接推送 `main`**。
4. 每次正式实验创建一个**独立的** `experiments/runs/<run_id>/manifest.json`，记录配置、随机种子、commit、运行命令与预算。大文件及敏感文件不提交 Git。
5. 完成阶段工作时，把状态写到 `docs/handoffs/issue-<编号>.md`；在 PR 中说明做了什么、没做什么、怎么验证、下一步怎么接续。
6. 重要研究结论先经过独立复现，再写入 `docs/STATE.md`；设计决策写入新的 `docs/decisions/NNNN-*.md`。PR 经另一位成员审阅后合并。

**接力最短路径：** `AGENTS.md` → `docs/STATE.md` → 任务 Issue → `docs/handoffs/issue-N.md` → 指定实验 manifest → `git status` / `git log`。不要仅靠聊天记录交接。

## 仓库目录

```text
AGENTS.md                         # 所有 Agent 的共同工作约束与接力流程
README.md                         # 总入口
CONTRIBUTING.md                   # 人员协作、分支与评审规则
docs/
  STATE.md                        # 项目当前共识；只维护已核实信息
  PROTOCOL.md                     # 物理模型、指标、基准和公平预算
  RESEARCH_LOG.md                 # 重要阶段摘要（不记流水账）
  decisions/0001-collaboration.md # 不可变的决策记录，后续新增文件
  handoffs/_TEMPLATE.md          # Issue 对应的接力记录模板
  研究主题与预实验报告.docx
experiments/
  README.md                       # 实验记录标准
  runs/<run_id>/manifest.json     # 每次运行的独立记录
  预实验1.zip
  预实验2.zip
src/                              # 后续模型、训练、评测模块
scripts/new_run.py                # 为一次实验创建记录
scripts/validate_workspace.py     # 校验仓库与实验记录结构
tests/                             # 不依赖 ML 框架的基础检查
.github/ISSUE_TEMPLATE/           # 任务模板
.github/PULL_REQUEST_TEMPLATE.md  # 评审模板
.github/workflows/                 # 轻量自动化检查
```

## 已有材料

此前上传的报告位于 `docs/研究主题与预试验报告.docx`，两份 ZIP 位于 `experiments/预实验1.zip`、`experiments/预实验2.zip`，**不会自动宣称其内容可复现**。本轮已清点哈希、源码结构和部分原始指标，但没有完整重跑历史二维实验。未经独立复核不得把 ZIP 内结果登记为 confirmed。

## 本地快速检查

需要 Python 3.10+。manifest检查不需要PyTorch；研究测试与实验需先安装相应依赖，第七轮使用Python 3.12与requirements-round7-lock.txt：

```bash
python scripts/validate_workspace.py
python -m unittest discover -s tests -v
python scripts/new_run.py --slug tfim-baseline --owner YOUR_GITHUB_ID --issue 1 --seed 0
```

最后一条只**创建实验记录，不会运行模型**。把 `--issue 1` 替换为真实 Issue 编号。

论文进度与待解决问题见 [`docs/STATE.md`](docs/STATE.md)。外部数据、权重、大型中间文件请使用受控存储，提交校验和、获取步骤与必要的聚合统计即可。

## 第七轮本地阶段交付（2026-10-27）

本地任务`local-round7-course-generalization`：固定六自旋、多h能量训练，60组新种子/布点对照；预定主配置训练门槛失败，保留全部负面结果与物理对照。用户将本轮远程发布改为D盘保存，本分支不代表远程main或独立成员确认。

**后续GitHub发布**：用户在本地交付后恢复上传授权。已完成的冻结成果由[Issue #15](https://github.com/Bian-M-X/conditional-transformer-quantum-state/issues/15)关联，使用943410298abc-ctrl的fork分支`issue/15-round7-course-generalization`向团队提交PR，交接见[issue-15](docs/handoffs/issue-15.md)。原始本地manifest及研究数据保持原样；本地交付阶段的说明是历史记录，正式STATE仍待独立成员复核。

从[阅读与复现入口](docs/round7_reproduction.md)、[阶段报告](docs/round7_report.pdf)和[交接](docs/handoffs/local-round7.md)进入。完整源码、依赖、原始数据及审核记录随包保存；团队正式STATE共识保持原状态。新环境需安装PyTorch等依赖，见requirements-round7-lock.txt；33项测试与60组证据核验可用`python scripts/reproduce_round7.py --mode audit`运行。
