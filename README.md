# TIGER 语义 ID 量化消融实验

基于 [phonism/genrec](https://github.com/phonism/genrec) 的 TIGER 管线，比较三种 item 语义 ID（Semantic ID, SID）生成方式：端到端 **RQ-VAE**、无监督 **RQ-KMeans**，以及保留最终残差的 **RQ-OPQ**。

项目关注一个实用问题：**更低的量化重构误差或更低的 SID 碰撞率，是否一定带来更好的生成式推荐效果？**

## 核心结论

- 在 3-token 主实验中，**RQ-KMeans** 的 Tiger 下游 Recall@10 最高（`0.06763`），尽管它的 SID 碰撞率和重构误差都比 RQ-VAE 更差。
- **RQ-OPQ** 显著降低碰撞率并改善重构误差，但直接生成完整 5-token SID 后，Recall@10 为 `0.06144`，没有超过 3-token RQ-KMeans。
- 将 RQ-OPQ 的两个 OPQ token 仅作为历史上下文、只预测前 3 个 RQ token，可将 Recall@10 提升至 `0.06557`，说明“残差信息辅助上下文”比“把残差 token 也作为生成目标”更有效。
- 结论：**量化质量与下游生成难度存在权衡；不能仅用 MSE 或 collision rate 选择 SID 方案。**

## 方法

所有实验使用 Amazon 2014 Beauty 5-core 数据，固定随机种子 `42`，共 `12,101` 个 item。每个 codebook 的大小均为 `256`。

```text
RQ-VAE:      item embedding → encoder → residual VQ → [c0, c1, ...]
RQ-KMeans:   item embedding → residual K-Means → [c0, c1, ...]
RQ-OPQ:      item embedding → 3-layer RQ → residual → OPQ/PQ → [c0, c1, c2, o0, o1]
```

RQ-OPQ 的前 3 个 token 表示粗粒度层次语义；后 2 个 token 用正交旋转后的子空间量化编码最终残差，以减少被丢弃的信息。

## 实验设置

- Item 表示：`sentence-t5-base` 生成的 768 维文本 embedding。
- 序列切分：leave-one-out；最后一次交互测试、倒数第二次验证。
- 历史窗口：最多 50 个 item。
- 排序：TIGER beam search，`beam_size=10`。
- 指标：Recall@5 / Recall@10 / NDCG@5 / NDCG@10。
- 量化指标：完整 SID collision rate、最大碰撞组、重构 MSE、平均重构 L2。

> 下游指标衡量生成序列与目标 SID 的 exact match。碰撞率须与 Recall/NDCG 联合解读：不同 item 共用一个 SID 时，SID 命中不等同于唯一 item 命中。

## 结果一：3-token 主实验

RQ-VAE 和 RQ-KMeans 均生成 3 个 token；RQ-OPQ 生成完整 5 个 token（3 RQ + 2 OPQ），因此本表适合观察系统层面的量化—生成权衡，但不是严格控制 token budget 的比较。

| 方法 | SID | Collision ↓ | Recon MSE ↓ | Recall@5 ↑ | Recall@10 ↑ | NDCG@5 ↑ | NDCG@10 ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| RQ-VAE | `[c0,c1,c2]` | 2.4874% | 6.3519e-05 | 0.03824 | 0.05888 | 0.02511 | 0.03177 |
| RQ-KMeans | `[c0,c1,c2]` | 7.5531% | 6.5769e-05 | **0.04387** | **0.06763** | 0.02781 | **0.03549** |
| RQ-OPQ | `[c0,c1,c2,o0,o1]` | **0.8016%** | **5.3378e-05** | 0.03839 | 0.06144 | 0.02543 | 0.03285 |

RQ-KMeans 的每层 256 个 code 都被使用，但它的完整 SID 碰撞率最高；这说明单层 usage 不能代表组合 SID 的区分能力。相反，RQ-OPQ 的量化指标最好，却增加了解码长度与生成难度。

## 结果二：5-token 长度匹配实验

为排除 SID 长度影响，RQ-VAE 与 RQ-KMeans 扩展为 5 层残差量化。三种方法均使用 5-token target、`1281 = 5 × 256 + 1` 的 Tiger vocabulary，以及相同的 50-item 历史窗口。

| 方法 | SID | Collision ↓ | Recon MSE ↓ | Recall@5 ↑ | Recall@10 ↑ | NDCG@5 ↑ | NDCG@10 ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| RQ-VAE-5 | `[c0,c1,c2,c3,c4]` | **0.0413%** | 5.6370e-05 | 0.03775 | 0.05848 | 0.02466 | 0.03131 |
| RQ-KMeans-5 | `[c0,c1,c2,c3,c4]` | 0.7851% | **5.2538e-05** | **0.03945** | 0.05995 | **0.02570** | 0.03229 |
| RQ-OPQ | `[c0,c1,c2,o0,o1]` | 0.8016% | 5.3378e-05 | 0.03839 | **0.06144** | 0.02543 | **0.03285** |

在 token budget 对齐后，RQ-OPQ 的 Recall@10 与 NDCG@10 最好。RQ-VAE-5 几乎消除了 SID 碰撞，但其下游指标没有随之提升，再次验证了“唯一性不是充分条件”。

## 结果三：RQ-OPQ 的 target 消融

该实验固定同一份 RQ-OPQ SID：历史输入始终是 `[c0,c1,c2,o0,o1]`，只改变 decoder 的预测目标。

| 输入历史 | 预测目标 | Recall@5 ↑ | Recall@10 ↑ | NDCG@5 ↑ | NDCG@10 ↑ |
|---|---|---:|---:|---:|---:|
| 5 token | `[c0,c1,c2,o0,o1]` | 0.03839 | 0.06144 | 0.02543 | 0.03285 |
| 5 token | `[c0,c1,c2]` | **0.04316** | **0.06557** | **0.02823** | **0.03549** |

完整生成 OPQ token 增强了 item 区分能力，但也显著扩大了序列生成空间。仅预测粗粒度 RQ token 时，OPQ 残差仍能作为历史上下文被 encoder 利用，获得更好的下游结果。

## 复现

### 环境

```bash
conda create -n genrec python=3.10 -y
conda activate genrec
pip install -r requirements.txt
```

所有命令在仓库根目录执行；将 `config/base.gin` 中的 `MODEL_HUB_SENTENCE_T5_BASE` 修改为本地 `sentence-t5-base` 路径。

### 训练与评估 RQ-KMeans（3 token）

```bash
python genrec/trainers/rqkmeans_trainer.py config/tiger/amazon/sid_rqkmeans.gin --split beauty
python genrec/trainers/tiger_trainer.py config/tiger/amazon/tiger_rqkmeans.gin --split beauty
```

### 训练与评估 RQ-OPQ（3 RQ + 2 OPQ）

```bash
python genrec/trainers/rqkmeans_trainer.py config/tiger/amazon/sid_rqopq.gin --split beauty
python genrec/trainers/tiger_trainer.py config/tiger/amazon/tiger_rqopq.gin --split beauty
```

### RQ-OPQ：完整输入、仅预测前三个 RQ token

复用上一步生成的 `semantic_ids.pt`：

```bash
python genrec/trainers/tiger_trainer.py config/tiger/amazon/tiger_rqopq_target3.gin --split beauty
```

### 5-token 长度匹配实验

```bash
# RQ-VAE-5：训练、导出统一 SID artifact、训练 TIGER
python genrec/trainers/rqvae_trainer.py config/tiger/amazon/rqvae_5token.gin --split beauty
python genrec/trainers/sid_export_trainer.py config/tiger/amazon/sid_rqvae_5token_export.gin --split beauty
python genrec/trainers/tiger_trainer.py config/tiger/amazon/tiger_rqvae_5token.gin --split beauty

# RQ-KMeans-5：量化阶段会同时生成 SID artifact
python genrec/trainers/rqkmeans_trainer.py config/tiger/amazon/sid_rqkmeans_5token.gin --split beauty
python genrec/trainers/tiger_trainer.py config/tiger/amazon/tiger_rqkmeans_5token.gin --split beauty
```

结果保存在 `out/tiger/amazon/beauty/<method>/results.json`；量化器输出的 `semantic_ids.pt` 是 Tiger 使用的统一 item-to-SID 映射。

## 工程实现

```text
genrec/models/
├── rqvae.py              # 端到端残差量化 VAE
├── rqkmeans.py           # 多层残差 K-Means
├── rqopq.py              # RQ 后对最终残差执行 OPQ/PQ
└── semantic_id.py        # 统一 SID artifact 与量化指标

genrec/trainers/
├── rqvae_trainer.py      # RQ-VAE 训练
├── rqkmeans_trainer.py   # RQ-KMeans / RQ-OPQ 训练入口
├── sid_export_trainer.py # RQ-VAE checkpoint → SID artifact
└── tiger_trainer.py      # TIGER 训练与评估
```

统一 SID artifact 将量化器与下游生成模型解耦：TIGER 只读取 `item_id → SID` 映射，因此 RQ-VAE、RQ-KMeans、RQ-OPQ 可以使用一致的训练与评估接口。

## 局限与下一步

- 当前结果仅报告单个随机种子；应补充 3 个 seed 的均值与标准差。
- RQ-OPQ 的 target=3 实验以粗粒度 SID 评估，不能直接与 5-token exact-match 的数值横向比较；它回答的是“OPQ 是否有助于上下文建模”。
- 可进一步加入 constrained decoding / SID trie，以减少无效 codebook-slot token 的生成空间。
- 可在 Sports、Toys 等品类复验，检验结论的跨域稳定性。

## 致谢

本项目基于 [GenRec](https://github.com/phonism/genrec) 实现，TIGER 模型参考 [TIGER: Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)。
