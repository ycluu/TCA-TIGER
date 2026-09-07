# TCA Teacher Cache Report

## Teacher

- Checkpoint: `<user-provided>/best_val_ndcg10.pth`
- SHA-256: `e285d915ad5032282116eb6382edf7eb7dd4da1887c3729da4e2df30ef8ba8ae`
- Strict load: PASS
- Architecture: SASRec++ hidden=50, maxlen=50, blocks=2, heads=1, norm_first=true, time-aware=false, padding=0, Full CE, seed=42

## Dataset

- Training samples: 131,413
- Items: 12,101

## ID Alignment

- Full 12,101-item round-trip mismatches: 0

## Cache

- Path: `out/tiger/amazon/beauty/tca/tca_teacher_cache.pt`
- Shape: `[131413, 3, 256]`
- Dtype: `torch.float16`
- File size: 201.78 MiB
- Sample-key count: 131,413 (unique: 130,218)
- Stored FP16 maximum sum error: 3.883e-04
- NaN/Inf: 0

## Probability

| Level | FP32 max sum error | Target support | Mean entropy | Median entropy | Mean max P | Median max P | Mean target P | Median target P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c0 | 5.960e-07 | 100.00000000% | 4.267219 | 4.610382 | 0.134453 | 0.089078 | 0.058186 | 0.014678 |
| c1 | 3.576e-07 | 100.00000000% | 1.791823 | 1.908666 | 0.460018 | 0.399962 | 0.345536 | 0.255501 |
| c2 | 3.576e-07 | 100.00000000% | 0.253389 | 0.021305 | 0.899682 | 0.996963 | 0.867596 | 0.996956 |

## Leakage

- Sample mismatch: 0
- Future-item positional violations: 0
- Validation/test positional violations: 0
- Teacher histories were produced from each TIGER sample's causal history, truncated to 50 and left-padded; hashed TIGER user IDs were not used.

## Collision Diagnostic

- Duplicate full-SID groups: 715
- Collided items remain separate item-probability occurrences and are scatter-summed into their common branches.

## Performance

- Device: `cuda`
- Generation time: 8.31 seconds
- Peak allocated GPU memory: 57.16 MiB

## Tests

- PASS — A canonical ID round trip
- PASS — B strict checkpoint load
- PASS — C deterministic inference
- PASS — D toy Trie construction
- PASS — E toy prefix filtering
- PASS — F toy scatter-sum
- PASS — G distribution normalization
- PASS — H target support
- PASS — I no padding probability/edge
- PASS — J deterministic sample key
- PASS — K sample-level no-future leakage
- PASS — L collision mass summation

## Decision

GO
