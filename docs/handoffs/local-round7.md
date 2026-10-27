# 第七轮本地研究交接

- 目标：固定6自旋、lambda=0、三个训练h，在确认训练可靠性的条件下比较窄/宽覆盖对未见h的预测。
- 来源：main 65c367c7115d；科学源码/配置冻结3f1aaa9c9faf；本地任务local-round7-course-generalization。用户将发布改为D盘交付，未创建远程Issue/PR。
- 已完成：60/60主运行、5新种子、20新保留点、9000条指标；报告、6图、文献、过程、预算与原始证据齐备。索引experiments/round7_index.json。
- 核验：33项测试；全部manifest；全振幅NumPy复算；原第1轮基准重跑1311值、新固定配置重跑10136值均最大差0；最终检查见results/round7。
- 观察：主配置raw/L-BFGS宽覆盖训练0/5达标、窄覆盖3/5；共同外推wide减narrow为+0.127278±0.0889，预定改善判据失败。
- 控制：even两侧5/5训练达标，共同插值wide减narrow五种子全正；共同外推均值小幅改善但有seed401大反例。窄覆盖direct胜同源lookup，强Ritz均值还能更好。
- 未完成：另一位成员独立复现与个人理解/分工记录、课程终稿署名和汇报PPT。原始主结果没有数值运行失败，也没有替换种子或测试后调参。
- 下一步命令：在已安装锁定环境的完整项目根目录执行`.\reproduce_round7.ps1 -Mode audit -Python .\.venv\Scripts\python.exe`，再执行replay；复述训练门槛和光谱界而不是只报PASS。
- 限制：五种子与离散h的描述性比较；even为已知物理控制，full enumeration不建立可扩展采样或总成本优势；尚无独立人类复核。
- 审核入口：docs/round7_report.pdf、round7_reproduction.md、process_round7.md与各run manifest；正式STATE共识保持原团队状态。
