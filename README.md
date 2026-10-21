# Conditional Transformer Quantum State (CTQS)

**Research status: exploratory / early-stage validation.** 研究目标是考察**近简并区域中条件神经网络量子态的可靠泛化**，重点评估有限训练预算下的参数采样策略。当前结果是预研究，不代表已有新的投稿级算法。

**拟定题目**：近简并系统中条件神经网络量子态的可靠泛化与计算预算感知的自适应参数采样  
**English**: Reliable Generalization and Budget-Aware Adaptive Parameter Sampling for Conditional Neural Quantum States near Quasi-Degeneracies

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
experiments/
  README.md                       # 实验记录标准
  runs/<run_id>/manifest.json     # 每次运行的独立记录
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

需要 Python 3.10+，不需要先安装 PyTorch：

```bash
python scripts/validate_workspace.py
python -m unittest discover -s tests -v
python scripts/new_run.py --slug tfim-baseline --owner YOUR_GITHUB_ID --issue 1 --seed 0
```

最后一条只**创建实验记录，不会运行模型**。把 `--issue 1` 替换为真实 Issue 编号。

论文进度与待解决问题见 [`docs/STATE.md`](docs/STATE.md)。外部数据、权重、大型中间文件请使用受控存储，提交校验和、获取步骤与必要的聚合统计即可。
