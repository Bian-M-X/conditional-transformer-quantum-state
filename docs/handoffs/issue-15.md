# Issue #15：第七轮成果发布与审阅交接

## 目标与版本

将已完成的第七轮课程横场泛化研究发布到GitHub，供团队检查、复现和整合。本任务不追加科学实验。[Issue #15](https://github.com/Bian-M-X/conditional-transformer-quantum-state/issues/15)创建在本地研究后；发布路径为943410298abc-ctrl的fork分支issue/15-round7-course-generalization及发往团队main的PR。

来源main：65c367c7115dba578462038fd6cf6c1e5fe64478。科学执行冻结：3f1aaa9c9faf0ad7045b28b4e095dc2b2a572a6c。结果/报告快照：3fa53418558ca02cce08b3c0ce87ce44772efe04。发布说明见ADR0009；原始本地任务与ADR0008保留。

## 已完成的研究与材料

- 固定N=6、lambda=0、h>0，三训练h的窄/宽覆盖，五个新种子，20个事前冻结保留h；60/60主运行，无失败或换种子。
- 预定raw Transformer/L-BFGS/selected训练门槛为窄3/5、宽0/5；共同外推wide-minus-narrow为+0.127278±0.0889，改善判据失败。even两侧5/5训练达标，但共同插值扩大覆盖后五种子全变差，共同外推保留seed401反例。
- 9页PDF、可编辑报告、6幅图、全统计、同源lookup和Ritz、文献记录及真实过程/AI分工。报告为docs/round7_report.pdf，入口为docs/round7_reproduction.md。
- experiments/round7_index.json列出60组唯一主运行。每组目录保存manifest/config/training/amplitudes/checks/metrics；两个额外复跑不进入主统计。

## 实际核验与边界

本地已通过33项测试、510份manifest检查和第七轮9000条指标的NumPy振幅复算；原基准1311值、新固定配置10136值复跑最大差均为0。证据见docs/results/round7/final_checks.txt、evidence_audit.json及两个replay.json。

本次发布准备补齐原仓库完整Git历史后，实际运行verify_round1.py至verify_round6.py，六个保存证据核验均通过。它们检查历史记录、预算、物理关系和保存的复跑，不是重新训练前六轮。现有GitHub研究工作流亦包含这六项检查；第七轮自动化入口可由reproduce_round7.py运行。

全部主运行的源码、配置和数值记录保持原样；发布不升级团队STATE中的正式科学共识。五种子及离散h的结果具有明确适用范围，even为已知物理对照，参数量/H工作与总成本不是同一个比较。

## 未完成与下一步

尚缺另一成员独立执行与个人理解/分工记录、课程终稿署名及最终汇报PPT。本轮没有新的数值失败需要补跑。fork的PR须由团队维护者审核合并；是否允许外部贡献者CI由仓库设置和维护者决定。

下一成员安装锁定环境后，在完整项目根目录执行：

```powershell
.\reproduce_round7.ps1 -Mode audit -Python .\.venv\Scripts\python.exe
.\reproduce_round7.ps1 -Mode replay -Python .\.venv\Scripts\python.exe
```

验收时应解释训练门槛、光谱界和seed401反例，并核对图/表来自同一索引。若要调整raw训练规则，使用独立开发h和新的冻结保留点；当前20个h已进入本轮结果分析。
