# TIGER 语义 ID 量化消融实验

基于 [phonism/genrec](https://github.com/phonism/genrec) 的 TIGER 管线，比较三种 item 语义 ID（Semantic ID, SID）生成方式：端到端 **RQ-VAE**、无监督 **RQ-KMeans**，以及保留最终残差的 **RQ-OPQ**。

项目关注一个实用问题：**更低的量化重构误差或更低的 SID 碰撞率，是否一定带来更好的生成式推荐效果？** 在此基础上，进一步研究 **Semantic ID 的表示能力与自回归生成难度之间的权衡**。

## 核心结论

- 在 3-token 主实验中，**RQ-KMeans** 的 TIGER 下游 Recall@10 最高（`0.06763`），尽管它的 SID 碰撞率和重构误差都比 RQ-VAE 更高。这说明 **embedding reconstruction quality 与下游生成式推荐效果并不严格单调对应**。
- 将 RQ 深度从 3 层增加到 5 层并没有带来稳定的下游收益。尤其对于 RQ-KMeans，5-token 版本的 Recall@10 低于 3-token 版本，说明增加 Semantic ID 长度虽然提高了表示能力，但同时也增加了 TIGER 的自回归生成难度。
- **RQ-OPQ** 在量化层面取得了更低的重构 MSE 和更低的 SID collision rate，说明对 RQ 最终残差进行 OPQ/PQ 编码能够保留更多细粒度信息。但当 TIGER 被要求完整生成 `[c0,c1,c2,o0,o1]` 时，额外的 OPQ token 也会增加生成长度和预测空间，因此量化指标的改善并没有直接转化为更高的 Recall。
- 在相同的 RQ-OPQ Semantic ID 下，将 `[o0,o1]` 保留在历史输入中、但只让 decoder 预测 `[c0,c1,c2]`，Recall@10 从 `0.06144` 提升到 `0.06557`。这一结果表明，在当前实验设置下，**细粒度 residual information 更适合作为 encoder context，而不一定适合作为 autoregressive generation target**。
- 总体而言，实验支持一个重要观察：**Semantic ID 的选择不能只根据 reconstruction MSE、codebook usage 或 collision rate 判断；对于 TIGER，还需要考虑 Semantic ID 的离散结构、生成长度以及 decoder 的可预测性。**

## 方法

所有实验使用 Amazon 2014 Beauty 5-core 数据，固定随机种子 `42`，共 `12,101` 个 item。每个 codebook 的大小均为 `256`。

```text
RQ-VAE:      item embedding → encoder → residual VQ → [c0, c1, ...]
RQ-KMeans:   item embedding → residual K-Means → [c0, c1, ...]
RQ-OPQ:      item embedding → 3-layer RQ → residual → OPQ/PQ → [c0, c1, c2, o0, o1]
```

RQ-OPQ 的前 3 个 token 表示粗粒度层次语义；后 2 个 token 用正交旋转后的子空间量化编码 RQ 最后一层产生的 residual，以减少传统 RQ 在最后量化阶段丢弃的信息。

## 实验设置

- Item 表示：`sentence-t5-base` 生成的 768 维文本 embedding。
- 序列切分：leave-two-out；最后一次交互测试、倒数第二次验证。
- 历史窗口：最多 50 个 item。
- 排序：TIGER beam search，`beam_size=10`。
- 指标：Recall@5 / Recall@10 / NDCG@5 / NDCG@10。
- 量化指标：完整 SID collision rate、最大碰撞组、重构 MSE、平均重构 L2。

> 下游指标衡量生成序列与目标 SID 的 exact match。碰撞率须与 Recall/NDCG 联合解读：不同 item 共用一个 SID 时，SID 命中不等同于唯一 item 命中。

## 结果一：3-token 主实验

RQ-VAE 和 RQ-KMeans 均生成 3 个 token；RQ-OPQ 生成完整 5 个 token（3 RQ + 2 OPQ），因此本表适合观察系统层面的量化—生成权衡，但不是严格控制 token budget 的比较。

| 方法        | SID                | Collision ↓ |    Recon MSE ↓ | Recall@5 ↑ | Recall@10 ↑ | NDCG@5 ↑ |  NDCG@10 ↑ |
| --------- | ------------------ | ----------: | -------------: | ----------: | -----------: | --------: | ----------: |
| RQ-VAE    | `[c0,c1,c2]`       |     2.4874% |     6.3519e-05 |     0.03824 |      0.05888 |   0.02511 |     0.03177 |
| RQ-KMeans | `[c0,c1,c2]`       |     7.5531% |     6.5769e-05 | **0.04387** |  **0.06763** | **0.02781** | **0.03549** |
| RQ-OPQ    | `[c0,c1,c2,o0,o1]` | **0.8016%** | **5.3378e-05** |     0.03839 |      0.06144 |   0.02543 |     0.03285 |

### 分析

RQ-KMeans-3 获得了本组实验中最高的 Recall@10 和 Recall@5，分别为 `0.06763` 和 `0.04387`。值得注意的是，它的 reconstruction MSE（`6.5769e-05`）高于 RQ-VAE-3（`6.3519e-05`），collision rate（`7.5531%`）也明显更高。

因此，本实验并不支持“量化重构越准确，TIGER 推荐效果越好”这一简单关系。一个可能的解释是，RQ-VAE 的训练目标主要关注连续 embedding 的重构，而 TIGER 的目标是学习用户历史到下一 Semantic ID 的条件概率：

```text
RQ-VAE:
        embedding reconstruction
                ↓
        continuous representation

TIGER:
        user history
                ↓
        Semantic ID generation
```

二者优化目标并不完全一致。因此，Semantic ID 是否形成了适合序列建模和自回归生成的离散结构，可能比单纯的 reconstruction fidelity 更重要。

同时，RQ-KMeans 的较高 collision rate 也没有阻止其取得更好的下游 Recall。这进一步说明 **collision rate 是重要的 tokenizer 指标，但不是 downstream recommendation performance 的充分代理指标**。

RQ-OPQ 在量化层面取得了最好的 reconstruction MSE（`5.3378e-05`）和最低的 collision rate（`0.8016%`），说明对 RQ 最终 residual 进行 OPQ/PQ 编码确实能够保留更多信息。但它同时将 SID 长度从 3 增加到 5，因此其 TIGER 结果不能直接证明“更好的量化一定带来更好的生成效果”。

## 结果二：5-token 长度匹配实验

为排除 SID 长度影响，RQ-VAE 与 RQ-KMeans 扩展为 5 层残差量化。三种方法均使用 5-token target、`1281 = 5 × 256 + 1` 的 Tiger vocabulary，以及相同的 50-item 历史窗口。

| 方法          | SID                | Collision ↓ |    Recon MSE ↓ | Recall@5 ↑ | Recall@10 ↑ |   NDCG@5 ↑ |  NDCG@10 ↑ |
| ----------- | ------------------ | ----------: | -------------: | ----------: | -----------: | ----------: | ----------: |
| RQ-VAE-5    | `[c0,c1,c2,c3,c4]` | **0.0413%** |     5.6370e-05 |     0.03775 |      0.05848 |     0.02466 |     0.03131 |
| RQ-KMeans-5 | `[c0,c1,c2,c3,c4]` |     0.7851% | **5.2538e-05** | **0.03945** |      0.05995 | **0.02570** |     0.03229 |
| RQ-OPQ      | `[c0,c1,c2,o0,o1]` |     0.8016% |     5.3378e-05 |     0.03839 |  **0.06144** |     0.02543 | **0.03285** |

### 分析

在严格控制为 5-token target 后，RQ-OPQ 的 Recall@10（`0.06144`）和 NDCG@10（`0.03285`）在本组实验中最高；RQ-KMeans-5 的 Recall@5 和 NDCG@5 则分别达到 `0.03945` 和 `0.02570`。

因此，RQ-OPQ 在 5-token setting 下表现出一定的 downstream 优势，但这种优势并不意味着“OPQ 的 reconstruction 更好，所以推荐效果必然更好”。更准确的解释是：

> **在相同 5-token 生成预算下，使用 3 层 RQ 表达层次语义，再使用 OPQ 编码最终 residual，是一种具有竞争力的 Semantic ID 构造方式。**

同时，RQ-KMeans-5 的 reconstruction MSE（`5.2538e-05`）实际上低于 RQ-OPQ（`5.3378e-05`），但其 Recall@10 仍略低于 RQ-OPQ。这再次表明 reconstruction MSE 与 TIGER downstream performance 并不是一一对应的。

RQ-VAE-5 则展示了另一个值得关注的现象：它的 collision rate 只有 `0.0413%`，显著低于 RQ-VAE-3，但 Recall@10 仅为 `0.05848`，没有因为 SID 唯一性提高而获得收益。

因此，本实验更适合支持如下结论：

> **增加 Semantic ID 的表达能力或唯一性，并不保证生成式推荐性能同步提升；当 token 数量增加时，还需要考虑 TIGER 对更长自回归目标的建模能力。**

## 结果三：RQ-OPQ 的 target 消融

该实验固定同一份 RQ-OPQ SID：历史输入始终是 `[c0,c1,c2,o0,o1]`，只改变 decoder 的预测目标。

| 输入历史    | 预测目标               | Recall@5 ↑ | Recall@10 ↑ |   NDCG@5 ↑ |  NDCG@10 ↑ |
| ------- | ------------------ | ----------: | -----------: | ----------: | ----------: |
| 5 token | `[c0,c1,c2,o0,o1]` |     0.03839 |      0.06144 |     0.02543 |     0.03285 |
| 5 token | `[c0,c1,c2]`       | **0.04316** |  **0.06557** | **0.02823** | **0.03549** |

### 分析

该消融实验与前两个实验不同：这里 **Semantic ID 本身没有发生变化**，历史输入始终包含完整的：

```text
[c0,c1,c2,o0,o1]
```

唯一改变的是 decoder 是否需要生成 OPQ token。

当 decoder 完整生成 5-token SID 时：

```text
[c0,c1,c2,o0,o1]
```

Recall@10 为 `0.06144`。

当 decoder 只预测前 3 个 RQ token 时：

```text
[c0,c1,c2]
```

Recall@10 提升到 `0.06557`。

相对提升约为：

\[
\frac{0.06557-0.06144}{0.06144}\approx6.7\%
\]

这一结果说明，在当前实验设置下，OPQ token 中包含的 residual information **可能对表示用户历史有帮助，但不一定值得作为 decoder 的自回归生成目标**。

可以将两种方案理解为：

```text
Full generation:

history
  ↓
[c0,c1,c2,o0,o1]
  ↓
TIGER
  ↓
generate [c0,c1,c2,o0,o1]

Representation + coarse generation:

history
  ↓
[c0,c1,c2,o0,o1]
  ↓
TIGER Encoder
  ↓
rich contextual representation
  ↓
Decoder
  ↓
generate [c0,c1,c2]
```

后一种方案把 OPQ token 的作用从 **generation target** 转变成 **contextual information**。在本实验中，这种设计取得了更好的 Recall/NDCG。

需要注意的是，这一结果不能直接证明 OPQ token 在所有生成式推荐场景中都应该只用于 encoder context。它更准确地说明：

> **在当前 Amazon Beauty、TIGER、beam size=10 的实验设置下，完整生成细粒度 residual token 的额外建模成本可能超过其带来的收益。**

因此，RQ-OPQ-target3 更适合作为一个 **generation strategy ablation**，而不是新的 tokenizer baseline。

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

- 当前结果仅报告单个随机种子；应补充 3 个 seed 的均值与标准差，判断 RQ-KMeans 与 RQ-OPQ 的差异是否具有统计稳定性。
- 当前实验仅使用 Amazon Beauty；需要在 Sports、Toys 等其他品类上复验，判断“3-token 优于更深 RQ”以及“OPQ residual 更适合作为 context”的现象是否具有跨域稳定性。
- RQ-OPQ 的 target=3 实验与完整 5-token target 并不是相同的 evaluation target，因此不应简单把 `0.06557` 与 `0.06144` 理解为两个 tokenizer 的直接性能排名；该实验主要回答 **生成目标设计** 的问题。
- 可进一步加入 constrained decoding / SID trie，在每个 decoder position 限制合法 codebook token，减少无效 token 生成空间。
- 可进一步研究 coarse-to-fine decoding：先生成 `[c0,c1,c2]`，再仅针对候选 coarse SID 预测 `[o0,o1]`，从而同时利用 residual 信息和分层 Semantic ID 结构。
- 可以增加 tokenizer 层面的 Codebook Perplexity、prefix sharing / semantic cohesion 等指标，进一步解释为什么 reconstruction MSE、collision rate 与 TIGER Recall 并不完全一致。
- 如果希望证明 RQ-OPQ 的优势来自“residual encoding”而不是单纯增加 token 数量，还需要设计更严格的 matched-budget ablation，例如在相同 token 数和相同 codebook capacity 下比较不同 residual quantization strategy。

