# TCA-TIGER Phase 1 最终报告

## 1. 结论

本阶段冻结的主线为：Amazon Beauty → Sentence-T5 768D → RQ-KMeans（3×256）→ TIGER → Full-Vocabulary Token Collaborative Alignment。所有模型均按验证集 NDCG@10 选择 checkpoint，测试集不参与选择。Full-Vocabulary TCA 在固定 seed 42 的测试集上提升全部四项 SID-level 指标。

## 2. 问题与动机

TIGER 通过商品文本语义构造层次 Semantic ID，并自回归生成下一商品 SID。语义表示本身不直接编码用户—商品协同行为，因此本项目引入冻结 SASRec 教师，将 item 空间的序列偏好转换为 SID 空间的分层软监督。

## 3. 数据与 Semantic ID

- 数据集：Amazon Beauty 5-core。
- 序列：timestamp 稳定排序；leave-two-out。
- 商品表示：Sentence-T5，768 维。
- 量化：RQ-KMeans，3 层，每层 256 code。
- TIGER token：0 为 padding/decoder start；三层区间依次为 1–256、257–512、513–768。
- 最大用户历史长度：50。

冻结的 `semantic_ids.pt` 未在最终发布整理中重新生成或修改。

## 4. SASRec 教师与离线 cache

冻结的 SASRec 接收每个 TIGER 训练样本的因果 item 历史，输出全商品 item score。对目标 SID 的每一层，按真实前缀筛选商品并将 item 概率 scatter-sum 到 256 个 code，得到三个 prefix-conditioned 分布。

cache 共有 131,413 个训练样本，张量形状 `[131413, 3, 256]`，类型 FP16，约 202 MiB。每条数据保存确定性 sample key；训练加载时验证数据集、SID artifact 与样本顺序。教师仅参与 cache 生成，TIGER 训练和推理不在线加载教师。

## 5. Full-Vocabulary TCA

目标 SID 为 $y=[c_0,c_1,c_2]$，第 $t$ 层教师分布为 $Q_{CF}^{(t)}$：

$$Q_t=(1-\alpha)Q_{hard}+\alpha Q_{CF}^{(t)}, \qquad \alpha=0.1,\;\tau=1.$$

教师概率只放入当前层对应的合法 token 区间，但模型概率始终在完整 769-token 词表上归一化：

$$\mathcal{L}_{TCA}=-\sum_t\sum_{v=0}^{768}Q_t(v)\log P_\theta(v\mid H,y_{<t}).$$

这项设计保持 TIGER 原始自回归输出空间，同时把 item-level collaborative knowledge 注入层次 SID token 预测。

## 6. 冻结配置

- T5 encoder/decoder：4/4 层。
- `d_model=128`，`d_ff=1024`，`d_kv=64`，6 heads。
- dropout 0.1，ReLU。
- Adam，学习率 `1e-4`，weight decay 0，无 scheduler。
- batch 256，inference batch 96，beam 10，seed 42。
- checkpoint selection：validation NDCG@10；patience 10。
- TCA：conventional soft CE，alpha 0.1，temperature 1。

## 7. 最终结果

### Validation

| 模型 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Original TIGER | 0.0645904078 | 0.0951570336 | 0.0425696823 | 0.0524112724 |
| Full-Vocabulary TCA | 0.0668901022 | 0.0976159031 | 0.0442361943 | 0.0541202280 |

### Test

| 模型 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Original TIGER | 0.0438740000 | 0.0676304649 | 0.0278123773 | 0.0354940226 |
| Full-Vocabulary TCA | 0.0467750242 | 0.0693764838 | 0.0305066295 | 0.0377905855 |
| 相对提升 | +6.61217156% | +2.58170474% | +9.68724147% | +6.47028052% |

Original TIGER checkpoint 为 epoch 86；Full-Vocabulary TCA checkpoint 为 epoch 111，SHA-256 为 `17d6ae7b8914ee43176757669e0a592b229a9ad2c41d9225b9f607e73a950870`。最终 replay strict load 成功，validation/test 共八项指标与存储结果的绝对差均为 0。

## 8. 评估语义

指标比较生成的三-token SID 与目标 SID。当前评估不做 SID collision resolution、item-level reranking 或历史商品过滤。最终 TCA 的 validation beam-1 known-SID rate 为 100%，Top-10 known-SID rate 为 99.9499%；test 分别为 100% 与 99.9606%。

## 9. 可复现性

公开复现入口、依赖与隔离输出目录命令见根目录 README。小型机器可读结果见 `results/main_results.json`。原始数据、embedding、Semantic ID、teacher cache 和 checkpoint 不随普通 Git 仓库分发。

## 10. 局限

- Amazon Beauty 单数据集、单随机种子，不能推出统计显著性或普适收益。
- 结果是 SID-level；碰撞情况下多个 item 仍不可区分。
- 教师监督质量依赖冻结的 SASRec checkpoint。
- evaluator 保留项目既有的 batch 聚合与无约束 Beam Search 语义。

## 11. 冻结声明

Phase 1 的方法、配置、checkpoint 与指标已经冻结。公开主线只包含 Original TIGER 与 Full-Vocabulary TCA；探索性实验保留在内部历史区域，不属于公开项目叙事。
