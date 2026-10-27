# 第七轮阅读、复现与协作入口

本轮执行日期2026-10-27。用户已将GitHub上传改为D盘本地交付；所有新工作在local/round7-course-generalization分支，远程main未修改。最终归档哈希见交付目录delivery_manifest.json；全部主运行的科学源码冻结于3f1aaa9c9faf0ad7045b28b4e095dc2b2a572a6c，来源main为65c367c7115dba578462038fd6cf6c1e5fe64478。

后续发布：D盘快照完成后，用户恢复GitHub上传授权；通过[Issue #15](https://github.com/Bian-M-X/conditional-transformer-quantum-state/issues/15)及fork分支`issue/15-round7-course-generalization`归档并提交审阅。上段保留首次交付时的状态，科学配置与结果未改变。当前协作入口见[发布交接](handoffs/issue-15.md)与[ADR0009](decisions/0009-round7-github-publication.md)。

## 先读什么

- docs/round7_report.pdf：9页阶段研究报告，按课程要求整理；可编辑源为docs/round7_report.md。
- docs/literature_round7.md：必读论文的定位、关键概念与本题关系。
- docs/process_round7.md：决策、负结果、真实分工与AI辅助记录。
- docs/round7_preregistration.md和experiments/configs/round7.json：看结果前的冻结设计。
- docs/results/round7/summary.json、tables.md、seed_endpoints.csv：全部单元及种子的统计；all_metrics.csv包含9000条记录。
- experiments/round7_index.json：60组主运行唯一索引。每个run有manifest、配置、训练/验证、振幅、检查与指标；2次复跑另存，不进入60组统计。

科学结论仍按observed/derived/provisional记录，尚待独立成员复核。docs/STATE.md的团队正式共识未由本轮改写。

## 环境

原执行环境为Windows 11、Python 3.12.14、CPU float64单线程；torch 2.14.1+cpu，numpy 2.3.5，scipy 1.16.3，matplotlib 3.10.6。全部包见根目录requirements-round7-lock.txt，操作系统与版本也保存在manifest。交付不携带venv、缓存或任何凭据。

在完整项目根目录操作，先安装Python 3.12与Git。示例PowerShell命令：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-round7-lock.txt
.\reproduce_round7.ps1 -Mode audit -Python .\.venv\Scripts\python.exe
```

原venv位于交付外的work/round7_venv，不应在新电脑使用manifest内旧绝对路径。上面的入口会为当前项目创建tmp目录。核心科学计算使用torch/numpy/scipy；生成报告另需Windows宋体，PDF渲染检查使用Poppler。完整项目ZIP包括本地.git供git show核对冻结内容，解压时保留它。

## 三种复现层次

audit默认只运行33项测试、验证全部manifest、从保存振幅复算全部第七轮指标，并重新比较已保存的两次复跑。它不会重新训练，也不自动证明有人独立理解或复现了研究。

```powershell
.\reproduce_round7.ps1 -Mode replay -Python .\.venv\Scripts\python.exe
.\reproduce_round7.ps1 -Mode full -Python .\.venv\Scripts\python.exe
```

replay重新训练固定的raw Transformer/L-BFGS/wide/seed101，约十秒的原机量级。full按原设计重新训练60组，原机训练与评估约5.6分钟；不同机器实际计时不同。两种模式创建带时间戳的新run目录，保留原始索引，按架构/优化器/训练方式/覆盖/种子逐个比较指标、训练轨迹和振幅，输出tmp/*_comparison.json。

新机器比较的事前容差为atol=1e-8、rtol=1e-6。浮点库与硬件差异可能改变优化轨迹；失败输出应保留并检查，不放宽容差后宣布复现。本轮原机重跑实测最大差为0，见docs/results/round7/round7_replay.json。原跨环境基准见baseline_replay.json。

## 重新生成表、图与报告

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/summarize_round7.py
.\.venv\Scripts\python.exe -X utf8 scripts/build_round7_materials.py
.\.venv\Scripts\python.exe -X utf8 scripts/build_round7_pdf.py
```

这些命令使用原始60组索引。新复跑不进入原统计。PNG为600 dpi，图PDF为矢量；公式在报告中以600 dpi栅格排版。字体默认C:/Windows/Fonts/simsun.ttc，可用--font指定兼容CJK的TrueType字体。生成后的PDF须重新渲染检查，不能只凭脚本退出码判断排版。

## 结果及下一步

预定主配置的训练门槛失败，扩大覆盖没有通过改善判据。even控制训练达标但未形成整体单调收益；窄覆盖可超过同源查表，Ritz强基准进一步限制优势叙述。所有失败种子和反向结果已保留。

下一位成员先独立执行audit和一个完整replay，再解释训练门槛、能量-保真度界和seed401反例。后续调参用新的开发h，新的规则用新的冻结保留点验证。汇报PPT、成员个人学习与真实贡献在团队整合时另补；本轮未虚构这些人工工作。
