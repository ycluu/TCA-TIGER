# GitHub Release Audit

## 1. Executive Summary

**Status: READY** for a code/configuration/results-metadata GitHub release. The public story is limited to Original TIGER and the Full-Vocabulary TCA extension. No training, inference replay, checkpoint mutation, commit, push or history rewrite was performed during release cleanup.

## 2. Repository Snapshot

- Branch: `main`
- Audited HEAD before cleanup: `1a709cce7ba2019cca940f835fa6d5aac0f89d57`
- The worktree was already dirty; all pre-existing edits were preserved.
- A pre-change inventory and frozen-artifact hashes are retained locally under `docs/internal/release/` (ignored from publication).

## 3. Frozen Public Scope

Public mainline: Amazon Beauty → Sentence-T5 768D → RQ-KMeans 3×256 → TIGER → frozen SASRec teacher → SID-prefix aggregation → Full-Vocabulary TCA. Evaluation is SID-level. Item-level serving, collision resolution, hybrid retrieval, GenPAS, preference/DPO alignment and other exploratory directions are outside the release narrative.

Position-CE experiments are deliberately retained only as internal/historical material and excluded from the public mainline. Their code/config support was not removed because doing so would create unnecessary regression risk.

## 4. README Review

The root README was rewritten in Chinese for GitHub and interview readers. It presents the motivation, architecture, prefix-conditioned teacher construction, full-vocabulary objective, frozen results, protocol, executable entrypoints, limitations, licensing and references without requiring historical ablations.

## 5. Frozen Results and Provenance

The public result file was built from existing artifacts, not recomputed:

| Model | Epoch | R@5 | R@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original TIGER | 86 | 0.0438740000 | 0.0676304649 | 0.0278123773 | 0.0354940226 |
| Full-Vocabulary TCA | 111 | 0.0467750242 | 0.0693764838 | 0.0305066295 | 0.0377905855 |

All models are described publicly as selected by validation NDCG@10. The final TCA replay strict-loaded checkpoint SHA-256 `17d6ae7b8914ee43176757669e0a592b229a9ad2c41d9225b9f607e73a950870`; all eight stored/replayed validation and test differences are zero. `results/main_results.json` was programmatically compared with the raw results/replay and passed.

## 6. Frozen Architecture and Protocol

Verified config: encoder/decoder 4/4 layers, `d_model=128`, `d_ff=1024`, `d_kv=64`, 6 heads, dropout 0.1, history 50, SID length 3, codebook 256, vocabulary 769, batch 256, Adam LR `1e-4`, beam 10, seed 42, early-stop patience 10. Final TCA uses `alpha=0.1`, temperature 1 and conventional full-vocabulary soft CE.

## 7. Public Result Metadata

Added `results/main_results.json`, containing only the frozen dataset/task identifiers, protocol, exact results, relative gains, epochs, checkpoint hashes and local provenance paths. Historical result JSON files were not overwritten.

## 8. Documentation Classification

- PUBLIC CORE: `README.md`, `docs/TCA_PHASE1_FINAL_REPORT.md`, `results/main_results.json`.
- PUBLIC SUPPLEMENTARY: teacher-cache and TCA4Rec design audits under `docs/audits/`.
- INTERNAL / EXPLORATORY: GenPAS, GRID, LETTER, OneRec, Phase 2 and hybrid audits under `docs/internal/exploratory/`.
- OBSOLETE / SUPERSEDED: Position-CE/generation diagnostics, early TCA objectives, integration and selection-history reports under `docs/internal/superseded/`.

The internal tree is ignored, so historical evidence remains local without becoming public-facing. The pre-release README is preserved there as `README_before_release.md`.

## 9. Exploratory Code

Option B was used: functioning exploratory modules/tests/configs remain at their existing paths but are absent from the main README. Three standalone audit scripts with machine-specific inputs and the release-inventory helper are excluded through `.gitignore`; model/trainer code was not refactored.

## 10. Large-File Audit

Twenty-two local files exceed 10 MiB; none is tracked. The inventory includes 13 model/checkpoint files around 17–18 MiB, a 20.43 MiB preference cache, three W&B files (19.26–26.46 MiB), a 35.45 MiB collaborative SID artifact, a 42.74 MiB review archive, a 54.76 MiB embedding parquet, a 94.56 MiB metadata archive and the 201.78 MiB teacher cache.

- `>10 MiB`: 22
- `>50 MiB`: 3
- `>100 MiB`: 1
- tracked large files: 0

`out/`, `dataset/`, `wandb/`, model weight patterns, external worktrees and temporary analysis directories are ignored. No local artifact was deleted.

## 11. Secrets, Privacy and Paths

Public candidates were scanned heuristically for common API/token/private-key patterns, credential assignments, email addresses and Windows/Linux machine paths. No credential-like material was found. The upstream author email was removed from `pyproject.toml`; public config defaults now use Hugging Face model identifiers; teacher-cache CLI defaults use a repository-relative path. Public Markdown contains no local absolute path. This is a working-tree scan, not a guarantee about Git history; history was not rewritten.

## 12. Data and Artifact Distribution

Amazon data is obtained from the UCSD Amazon Review Data source and expected locally under `dataset/amazon/`. Raw data is not distributed. Sentence-T5 weights/embeddings, `semantic_ids.pt`, TIGER checkpoints, SASRec checkpoint and teacher cache are not published or uploaded. README documents required placement/generation without inventing download URLs.

## 13. Dependencies and Configuration Portability

Added explicit `sentencepiece` and `scikit-learn` dependencies used by the documented Sentence-T5/RQ-KMeans pipeline. Machine-specific model locations in `config/base.gin` were replaced with model identifiers, still overridable through gin. The teacher-cache output report now defaults under ignored `out/`. Mainline experiment configs keep W&B disabled.

Audited local environment: Python 3.10.20, PyTorch 2.7.1+cu118, CUDA runtime 11.8, Transformers 4.57.6, scikit-learn 1.7.2; hardware was an NVIDIA GeForce RTX 3070 Ti Laptop GPU. This hardware is not a requirement.

## 14. Reproduction Command Validation

Both documented module entrypoints returned valid `--help` usage with positional config, `--split` and repeated `--gin` overrides. Modified Python files compile. Public commands use isolated `out/reproduce/...` save directories to avoid overwriting frozen runs. Full training and test inference were intentionally not launched.

## 15. Test and Smoke Results

- Final full suite: **60 passed, 0 failed, 0 skipped in 18.33 s**.
- Coverage includes TCA loss, teacher cache/alignment, preference integration and validation selection.
- Result provenance comparison: PASS.
- Synthetic Full-Vocabulary TCA forward/loss/backward: finite, PASS.
- Final TCA checkpoint strict load and exact replay: PASS from the existing frozen replay artifact; no new replay was run.

## 16. License and Attribution

The project retains its upstream MIT `LICENSE`. `THIRD_PARTY_NOTICES.md` records GenRec, SASRec and TCA4Rec provenance. A copy of the SASRec.pytorch Apache-2.0 license is included under `licenses/`. Dataset and model artifacts are explicitly excluded from the software distribution. The existing MIT copyright line was preserved rather than inventing ownership metadata.

## 17. Release Decision

**READY**. Safety gates passed: the post-change inventory verified every frozen artifact hash unchanged; tests and smoke checks pass; no large file is tracked; public narrative and reports exclude Position-CE; public files contain no detected credential or machine path; attribution and limitations are explicit. Exact end-to-end metric reproduction still requires user-provided data and a compatible frozen SASRec checkpoint, which is documented as an artifact prerequisite rather than concealed.
