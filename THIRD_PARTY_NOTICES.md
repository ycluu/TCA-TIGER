# Third-Party Notices

本仓库由上游 [GenRec](https://github.com/phonism/genrec)（MIT）扩展而来，保留根目录 `LICENSE`。

`genrec/models/sasrec_teacher.py` 是为读取本项目冻结教师 checkpoint 而维护的 SASRec 适配实现，其来源可追溯至 [pmixer/SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch)，按 Apache License 2.0 提供。许可证副本见 `licenses/SASRec-Apache-2.0.txt`。本仓库不分发 SASRec checkpoint。

Full-Vocabulary TCA 是对 [TCA4Rec](https://github.com/critical88/TCA4Rec) 所述 token-level collaborative alignment 思想的 TIGER 适配，不是官方实现。

Amazon Beauty 数据、Sentence-T5 权重、生成的 embedding、Semantic ID、教师 cache 和训练 checkpoint 不属于本仓库发布内容。使用者应分别遵守原始数据、模型与软件的许可条款。
