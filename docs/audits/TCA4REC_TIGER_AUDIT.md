# TCA4Rec → TIGER Token-Level Collaborative Alignment Audit

Audit date: 2026-09-05. Scope: source and artifact audit only; no model, trainer, tokenizer, checkpoint, or evaluation code was changed. Official-source anchors are the [TCA4Rec repository](https://github.com/critical88/TCA4Rec), [`models/LLM4Rec.py`](https://github.com/critical88/TCA4Rec/blob/main/models/LLM4Rec.py), [`models/backbone/SASRec.py`](https://github.com/critical88/TCA4Rec/blob/main/models/backbone/SASRec.py), [`data.py`](https://github.com/critical88/TCA4Rec/blob/main/data.py), and [`main.py`](https://github.com/critical88/TCA4Rec/blob/main/main.py).

## 1. Executive Summary

Decision: **GO WITH CHANGES**.

The method is technically compatible with this TIGER, but it cannot be driven by `sasrec_beauty_item_embeddings.pt` alone. TCA needs a history-conditioned user vector, so the complete trained SASRec++ state plus its architecture/config and deterministic preprocessing contract must be available. The smallest safe integration is an isolated frozen-teacher adapter, not a dependency on the SASRec++ repository.

Recommended first experiment: offline, leakage-safe precomputation of the final three prefix-conditioned token distributions (exact full-catalog aggregation, FP16 storage), with `alpha=0.1`. This is more faithful than Top-M and much cheaper during TIGER training than online SASRec inference. Keep Top-M as an optional fallback, explicitly labelled an approximation.

Three decisive reasons:

1. Full checkpoint inference is required: item embeddings provide candidate vectors but not the user vector computed from ordered history.
2. ID alignment is sound, but the boundary conventions differ: SASRec uses canonical items `1..12101`, while TIGER uses item rows `0..12100` and offset SID vocabulary tokens.
3. Official code is title-token/causal-LM oriented; TIGER is a fixed three-token T5 encoder-decoder. Trie, label positions, EOS, and loss insertion therefore need a deliberate adapter rather than a literal copy.

## 2. Official TCA4Rec Data Flow

| Step | Repository location | Function/class | Input | Output | Indexing assumption |
|---|---|---|---|---|---|
| Data module selection | `data.py` | `create_data_module` | args, tokenizer | `LLM4RecDataModule` | delegates to `utils/LLM4Rec_data.py` |
| Batch entry | `models/LLM4Rec.py:486-505` | `training_step` | `user_idx`, `title_idx`, `input_ids`, `hist_id`, mask | scalar loss | item padding is 0 |
| CF user state | `LLM4Rec.forward:244` | `SASRec.get_user_embs` | user IDs (unused), left-padded history `[B,10]` | `[B,64]` | final sequence position is the user state |
| CF item state | `forward:245` | `get_item_embs(arange(n_item+1))` | IDs `0..n_item` | `[n_item+1,64]` | row 0 is padding |
| Item logits | `forward:246` | matrix multiply | `[B,64] @ [64,n_item+1]` | `[B,n_item+1]` | padding is included and not masked |
| Prefix lookup | `forward:251-304` | `Trie.get_node_for_sequence` | ground-truth generated-token prefix | node | empty prefix means root |
| Item→token aggregation | `forward:278-304` | `scatter_sum` | prefix candidates' scores/items/tokens | full vocabulary vector | softmax over candidate-item occurrences, then sum by next token |
| Soft target | `forward:314-342` | manual loss | shifted LLM logits, one-hot label, CF token distribution | scalar mean loss | autoregressive one-token shift |

End-to-end: ordered history → frozen SASRec final hidden state → dot products with all item embeddings → ground-truth-prefix Trie node → softmax over matching item scores → scatter-sum by next token → mixture with one-hot target → soft-target cross entropy.

Important official-code qualification: `title_idx` is passed but the active aggregation does not use it; commented code once contemplated target retention/filtering. Historical items are not masked. `beta=1` is hard-coded. `tau` (default 4.5 when `use_msl`) applies to generator logits, not CF softmax.

## 3. Official SASRec Teacher Interface

`LLM4Rec.init_cf_model` constructs `SASRec(n_user, n_item, device="cpu", maxlen=10, hidden_units=64)`, loads `cf_model/sasrec/{dataset}.pt` with `load_state_dict`, calls `eval()`, and sets every parameter's `requires_grad=False`.

The official backbone left-pads histories with item 0. `log2feats` embeds items, adds positional embeddings, applies two one-head causal self-attention blocks and returns `[B,T,64]`. `get_user_embs` returns `log_feats[:, -1, :]`; `get_item_embs` indexes the learned table. The verified score is

`S_cf(u,i) = h_u^T e_i`, hence `cf_logits = user_embs @ item_embs.T` with shape `[B, n_item+1]`.

**Answer B: full SASRec checkpoint + model inference is required.** The exported `[12102,50]` item table supplies only `e_i`; it cannot reconstruct `h_u=f_theta(history)`, which depends on item/position embeddings, layer norms, attention and feed-forward weights.

Required self-contained teacher package:

- `best_val_ndcg10.pth` model state (local file exists, about 7.70 MB);
- frozen SASRec++ architecture or a stable inference-only equivalent;
- config: 12101 items, hidden 50, maxlen 50, 2 blocks, 1 head, dropout 0.2, `norm_first=true`, `use_time_gap=false`, padding 0;
- canonical sequence/left-padding/truncation contract and checkpoint format/version;
- a manifest/hash tying checkpoint, config and canonical Beauty metadata together.

## 4. SASRec++ Compatibility

| Requirement from TCA | Official SASRec | Local SASRec++ | Compatible? | Smallest adaptation |
|---|---|---|---|---|
| History input | left-padded `[B,10]` | left-padded `[B,50]`, canonical IDs | Yes | retain last 50; never use official maxlen 10 |
| Padding ID | 0 | 0 | Yes | assert row/history pad is 0 |
| Max length | 10 | 50 | Adapted | load local config exactly |
| User representation | `get_user_embs`; last hidden | `log2feats(... )[:, -1, :]` | Yes | expose inference wrapper |
| Item representation | `item_emb(ids)` | `item_emb(ids)` `[12102,50]` | Yes | exclude row 0 from probability universe |
| All-item scoring | `[B,64]@[64,N+1]` | `final_feat@[1:,50]^T` | Yes | return `[B,12101]` for IDs 1..12101 |
| Checkpoint | raw state dict | dict containing `model_state_dict` plus optimizer/epoch metadata | Adapted | unwrap `model_state_dict`, strict load |
| Time feature | none | disabled for Beauty | Yes | pass no time gaps |

Local `predict` already implements the same dot product. For bulk scoring, use `log2feats(history)[:, -1] @ item_emb.weight[1:].T` under `eval()` and `torch.no_grad()`; do not call `predict` 12101 times.

## 5. Current TIGER Vocabulary / ID Spaces

`genrec/trainers/tiger_trainer.py:t5_collate` implements `token = raw_code + level*256 + 1`. `train` sets vocabulary size `256*3+1=769`, pad 0 and EOS 0. `genrec/models/tiger.py:Tiger.__init__` also sets decoder start to the pad ID.

| Meaning | Raw value | TIGER token-ID range |
|---|---:|---:|
| RQ level 0 | 0..255 | 1..256 |
| RQ level 1 | 0..255 | 257..512 |
| RQ level 2 | 0..255 | 513..768 |
| Padding | — | 0 |
| EOS | — | 0 (shared config ID) |
| Decoder start | — | 0 (shared with pad/EOS) |

There are no extra BOS or semantic special tokens. Labels produced by `t5_collate` have shape `[B,3]` and contain only the three offset SID tokens.

Exact item conversions:

```text
canonical_item_id (1..12101) == sasrec_item_id
sid_artifact_row = canonical_item_id - 1
tiger_sequence_item_index = sid_artifact_row
canonical_item_id = sid_artifact_row + 1 = tiger_sequence_item_index + 1
```

Evidence: `AmazonSeqDataset._load_sequences` enumerates first-seen ASINs from 0, then `__getitem__` directly indexes `sem_ids_list[item_id]`. The local SASRec export uses canonical 1-based IDs with padding 0. Future code must round-trip all 12101 IDs and verify SID row count/checksum before training.

Risk: TIGER's `user_id = hash(reviewerID) % 10000` is unstable/colliding and is not the canonical SASRec user ID. It is irrelevant if the teacher is driven solely by each sample's canonical item history; it must not be used to look up user embeddings.

## 6. Trie and Collaborative Tokenizer

Each official Trie node represents a token prefix. Its `token_item_pairs` is populated on the parent during insertion with one `(next_token,item_id)` occurrence per item passing through that edge. Thus:

```text
A [12,31,7], B [12,31,91], C [12,83,22], D [47,9,16]
root:       (12,A),(12,B),(12,C),(47,D)
prefix 12:  (31,A),(31,B),(83,C)
prefix 12,31: (7,A),(91,B)
```

Official item title token sequences append EOS before insertion. For TIGER, construct from every `semantic_ids.pt` row and canonical ID `row+1`, using **final TIGER vocabulary IDs** `[c0+1,c1+257,c2+513]` on edges. This removes repeated offset conversions from the hot path and makes a prefix directly comparable to decoder labels. Do not insert EOS: TIGER has fixed depth 3 and EOS shares ID 0 with padding/start.

At position `t`, for node prefix `p=y_{<t}`, let `C_p` be candidate items under that prefix and `g_p(i)` their next token. Official code computes

`q_p(i)=softmax_{j in C_p}(S_cf(u,j)/1)` and `Q_p(v)=sum_{i in C_p:g_p(i)=v} q_p(i)`.

This is **sum**, implemented with `scatter_sum`, not mean/max/logsumexp. Duplicate full SIDs remain separate item occurrences until their probabilities are summed into identical token branches.

## 7. Soft Label Alignment

Official code makes one-hot `e_y`, then

`soft_target = (1-alpha)e_y + alpha Q_p`.

Its manual loss is algebraically soft-label cross entropy:

`L = -log(sum_v exp(z_v/tau)*soft_target_v) + log(sum_v exp(z_v/tau))`.

This differs from conventional `-sum_v soft_target_v log_softmax(z/tau)_v`: the official expression is a log of an expectation, not an expectation of log probabilities. A faithful reproduction should preserve this exact objective for the first comparison, while naming it accurately. `main.py` defaults alpha to 0.0; the README TCA command explicitly uses `--use_msl --alpha 0.1`. Recommendation: one controlled run at `alpha=0.1`, no sweep.

Official `use_msl` masks generator logits to Trie-valid tokens; this is separable from soft-label alignment. For a clean TIGER experiment, keep existing generation/evaluation unchanged and enable only the training soft-target objective unless constrained training logits are explicitly added as a second ablation.

## 8. EOS / Padding / Teacher Forcing

Official TCA queries the Trie with slices of the **ground-truth tokens already present in `input_ids`**, not generated samples. It is teacher forcing.

For TIGER, Hugging Face T5 internally forms decoder inputs by right-shifting `[y0,y1,y2]` and prepending decoder-start 0:

| Supervised token | Decoder input at that position | Trie prefix |
|---|---|---|
| `c0` token | 0 | `[]` |
| `c1` token | `c0` | `[c0]` |
| `c2` token | `c1` (with prior context available causally) | `[c0,c1]` |

Current TIGER labels contain no EOS; therefore current CE loss covers exactly c0/c1/c2. Apply TCA to those three positions only. Do not add an EOS target and do not treat ID 0 as a Trie edge. Padding is absent from `[B,3]` targets; encoder padding is governed by `attention_mask`.

## 9. Computational Cost

Current TIGER batch is 256, catalog 12101, SID length 3, vocabulary 769, SASRec hidden 50.

- User states `[256,50]`: 0.05 MB FP32.
- Non-padding item table `[12101,50]`: 2.42 MB FP32.
- Full CF logits `[256,12101]`: 3,097,856 values = 12.39 MB FP32 or 6.20 MB FP16.
- Dense token distributions `[256,3,769]`: 2.36 MB FP32 or 1.18 MB FP16 (only three 256-token level slices are smaller).
- Teacher activations for `[256,50]` histories add attention/FFN workspace; no backward graph is needed.

The raw tensors fit in 8 GB, so online computation is possible, but TIGER already consumes most of the budget and official Python per-example/per-position Trie loops add synchronization and fragmentation. On an RTX 3070 Ti, online same-GPU inference is operationally risky and slows every epoch. CPU teacher avoids VRAM but likely bottlenecks training.

## 10. Offline vs Online Teacher Recommendation

Beauty has 198,502 retained interactions and 22,363 users. TIGER sliding-window training examples are `sum(L-3)=131,413`.

| Option | Approximate disk | Training GPU cost | Speed | Faithfulness |
|---|---:|---|---|---|
| A. Online full teacher | checkpoint only | SASRec + `[B,12101]` each batch | slowest/variable | exact |
| B. Offline all-item logits | 6.36 GB FP32 / 3.18 GB FP16 | aggregation only | fast, high I/O | exact modulo storage precision |
| C. Offline Top-M IDs+scores | at M=256: ~202 MB (int32+FP32), ~135 MB (int32+FP16) | sparse aggregation | fastest | approximate; discarded mass changes prefix distributions |

Preferred fourth representation derived exactly from B during precompute: store the **three final prefix-conditioned 256-way distributions** per training example. Size is about 404 MB FP32 or 202 MB FP16, training requires no teacher/Trie, and it remains full-catalog exact modulo precision. Persist sample identity/history hash, target and prefix so stale or reordered caches fail loudly. If constrained to A/B/C, choose B; if disk pressure dominates, choose C with M=256 and log retained probability mass.

## 11. Leakage Audit

`AmazonSeqDataset._generate_samples` excludes validation/test (`full_seq[:-2]`) and creates targets for `i=1..len(train_seq)-1`, history `train_seq[:i]`, target `train_seq[i]`. Therefore the teacher input for every example must be **that sample's `history` only**, truncated to the last 50 canonical IDs and left-padded with 0. Never score using the user's entire train sequence for early sliding-window targets; that leaks later train interactions. Validation and test interactions must never enter cache construction or teacher training.

The SASRec++ teacher was trained on train interactions, but offline generation still needs per-example causal histories. Cache keys should include ordered canonical history, target, split and teacher checkpoint hash. Assert target and all later items are absent by construction (a repeated historical occurrence of the same item may legitimately equal the target ID and should be handled as a documented data property, not silently removed).

Official TCA does not mask historical items and includes padding in logits. The Beauty adaptation should exclude padding row 0, preserve historical items to match the active official behavior, and keep the target available. Any history masking would be a separate deviation/ablation.

## 12. Collision Behavior

If items A and B share the same complete SID, both occur in every matching node's `(token,item)` pairs. Their separately normalized CF masses are summed at c0, c1 and c2. At the final branch this strengthens the shared token probability but cannot identify which collided item was intended. Ambiguity remains after c2 and existing SID-level evaluation behavior remains unchanged. No collision-aware expansion is proposed.

## 13. TIGER Loss Insertion Point

Current path:

- `genrec/trainers/tiger_trainer.py:291-300`: batch tensors are moved to device and `Tiger.forward(..., labels)` returns loss/logits.
- `genrec/models/tiger.py:45-51`: `T5ForConditionalGeneration` computes its built-in hard CE and exposes `outputs.logits`.
- available batch tensors: encoder `input_ids [B,150]`, `attention_mask [B,150]`, offset `labels [B,3]`, and currently unused `raw_targets [B,3]`; decoder inputs are generated internally and are not returned by the collator.

Smallest future change: keep `Tiger.forward` untouched or add an opt-in return path; under `use_tca`, take returned logits `[B,3,769]`, labels and cached teacher distributions and calculate the chosen soft loss in the trainer. Under `use_tca=false`, use the existing `outputs.loss` byte-for-byte path. If online teacher history is needed, extend the dataset/collator to return canonical history and stable sample key; do not reverse semantic IDs back into items.

## 14. Proposed Config

```text
use_tca = false
tca_alpha = 0.1
tca_objective = official_log_mixture
cf_teacher = sasrecpp_beauty
cf_checkpoint = null
cf_config = null
cf_cache = null
cf_mode = offline_token_distribution
cf_topk = 0
cf_temperature = 1.0
tca_supervise_eos = false
tca_mask_history_items = false
```

Defaults preserve the baseline without changing existing gin files. Require checkpoint/cache paths only when `use_tca=true`.

## 15. Required Tests

1. All 12101 IDs round-trip SASRec ID ↔ SID row ↔ TIGER index and match artifact hashes.
2. Toy SID mapping constructs the expected root and prefix nodes.
3. Prefix lookup returns only items under that prefix.
4. Analytic item probabilities scatter-sum into the exact next-token distribution.
5. Every supervised token distribution and mixed target sums to 1 within tolerance.
6. `use_tca=true, alpha=0` equals baseline hard CE numerically. Because the official log-mixture loss reduces to hard CE at alpha 0, this must pass.
7. Teacher is eval/no-grad; no parameter receives a gradient and repeated inference is deterministic.
8. Every cache history equals the TIGER sample history and excludes future/validation/test events.
9. Only three SID positions are supervised; ID 0/EOS/padding receives no TCA mass.
10. Full-catalog aggregation equals Top-M when M=12101; collision masses sum correctly.
11. Cache key detects sample reorder, checkpoint/config mismatch and maxlen mismatch.

## 16. Official TCA vs TIGER Adaptation

| Component | Official TCA4Rec | TIGER adaptation | Exact / Adapted |
|---|---|---|---|
| CF teacher | frozen reference SASRec 64D/maxlen10 | frozen local SASRec++ 50D/maxlen50 | Adapted architecture, same interface |
| Teacher dataset | selected Amazon category | canonical Beauty train split | Adapted |
| Generative model | causal LLM + LoRA, title tokens | T5 encoder-decoder TIGER | Adapted |
| Item representation | variable title-token sequence + EOS | fixed RQ-KMeans 3-token SID | Adapted |
| Collaborative tokenizer | Trie + candidate softmax + scatter-sum | same formula over offset SID tokens | Exact algorithm, adapted vocabulary |
| Soft labels | `(1-a)onehot+aQ` | same | Exact |
| Alpha | README 0.1 | 0.1 | Exact experiment setting |
| Training objective | official log-mixture expression, optional MSL mask | retain expression; no new decoding mask | Soft loss exact, MSL constraint omitted |

## 17. Interview Defensibility

### We can claim

“We adapted TCA4Rec's token-level collaborative alignment to a Semantic-ID TIGER. A frozen SASRec++ teacher produces history-conditioned item scores, which are aggregated through the RQ-KMeans SID prefix tree into token-level soft targets.”

“The item-to-token aggregation and alpha mixture follow the released TCA4Rec code, while the teacher architecture and token vocabulary are explicit adaptations.”

### We cannot claim

- “We reproduced original TCA4Rec exactly.”
- “Item embeddings alone provide collaborative supervision.”
- “TCA resolves SID collisions.”
- “The teacher uses validation/test interactions.”
- “Top-M is mathematically identical to full-catalog TCA,” unless M is the full catalog.
- “Official TCA uses conventional soft-label CE”; its released loss is the log-mixture form described above.

## 18. Phase-2 Insertion Point

The future teacher adapter naturally yields per-example all-item/Top-M scores and per-prefix token distributions. The item-score output can later rank TIGER-generated candidate items into preferred/rejected pairs after generation and canonical item recovery. The likely insertion point is between candidate generation and a future preference-loss data builder, not inside the current TCA token-loss loop. Collided full SIDs still need an explicit item recovery policy in that later phase; none is designed here.

## 19. GO / NO-GO Recommendation

**GO WITH CHANGES**

1. Export/package the full SASRec++ checkpoint, architecture config and sequence contract; the item table alone is insufficient.
2. Add an explicit canonical-ID adapter and build the Trie in final offset TIGER token IDs, supervising only c0/c1/c2.
3. Precompute leakage-safe full-catalog prefix token distributions for the 131,413 causal training examples; avoid online same-GPU teacher inference on the 8 GB card.

