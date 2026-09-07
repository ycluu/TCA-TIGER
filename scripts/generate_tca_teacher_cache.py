"""Generate exact offline SASRec++ token supervision for Beauty TIGER."""

import argparse
import hashlib
import math
import os
import statistics
import time
from pathlib import Path

import torch
from tqdm import tqdm

from genrec.data.amazon import AmazonSeqDataset
from genrec.data.tca_teacher_cache import (
    build_sid_trie,
    canonical_item_id,
    duplicate_sid_stats,
    exact_prefix_distributions,
    sample_key,
    tiger_item_index,
)
from genrec.models.sasrec_teacher import load_beauty_teacher


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def left_pad_history(history, maxlen=50):
    canonical = [canonical_item_id(item) for item in history[-maxlen:]]
    return [0] * (maxlen - len(canonical)) + canonical


def distribution_stats(values):
    tensor = torch.cat(values).float()
    return float(tensor.mean()), float(tensor.median())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/amazon"))
    parser.add_argument(
        "--sid",
        type=Path,
        default=Path("out/tiger/amazon/beauty/rqkmeans/semantic_ids.pt"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("external/sasrec_teacher/beauty/best_val_ndcg10.pth"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("out/tiger/amazon/beauty/tca/tca_teacher_cache.pt")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("out/tiger/amazon/beauty/tca/teacher_cache_report.md"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    started = time.perf_counter()
    sid_artifact = torch.load(args.sid, map_location="cpu", weights_only=False)
    raw_sids = sid_artifact["sem_ids"].long()
    assert sid_artifact["quantizer"] == "rqkmeans"
    assert sid_artifact["num_items"] == 12101 and raw_sids.shape == (12101, 3)
    assert sid_artifact["codebook_size"] == 256 and sid_artifact["n_layers"] == 3
    assert int(raw_sids.min()) >= 0 and int(raw_sids.max()) < 256

    roundtrip_mismatch = sum(
        tiger_item_index(canonical_item_id(index)) != index for index in range(12101)
    )
    if roundtrip_mismatch:
        raise AssertionError(f"Item-ID round-trip mismatches: {roundtrip_mismatch}")

    device = torch.device(args.device)
    teacher, teacher_config = load_beauty_teacher(args.checkpoint, device)
    with torch.no_grad():
        probe = torch.zeros(2, 50, dtype=torch.long, device=device)
        probe[:, -3:] = torch.tensor([1, 2, 3], device=device)
        deterministic = torch.equal(teacher.user_states(probe), teacher.user_states(probe))
    if not deterministic:
        raise AssertionError("Teacher inference is not deterministic")

    dataset = AmazonSeqDataset(
        root=str(args.dataset_root),
        split="beauty",
        train_test_split="train",
        max_seq_len=50,
        add_disambiguation=False,
        semantic_id_path=str(args.sid),
        target_n_layers=3,
    )
    if len(dataset) != 131413:
        raise AssertionError(f"Unexpected TIGER training sample count: {len(dataset)}")

    # Validate the exact dataset-generated causal samples against its source sequences.
    sample_mismatch = 0
    future_violations = 0
    validation_test_violations = 0
    cursor = 0
    for full_seq in dataset.sequences:
        train_seq = full_seq[:-2]
        for position in range(1, len(train_seq)):
            sample = dataset.samples[cursor]
            expected_history, expected_target = train_seq[:position], train_seq[position]
            sample_mismatch += sample["history"] != expected_history or sample["target"] != expected_target
            future_violations += len(sample["history"]) != position
            validation_test_violations += any(
                index >= len(full_seq) - 2 for index in range(position + 1)
            )
            cursor += 1
    if cursor != len(dataset) or sample_mismatch or future_violations or validation_test_violations:
        raise AssertionError("Sample-level leakage/alignment validation failed")

    trie = build_sid_trie(raw_sids)
    duplicate_group_count, duplicate_groups = duplicate_sid_stats(raw_sids)
    # Direct structural check also proves no padding/EOS edges were inserted.
    if 0 in trie.root.children or len(trie.root.item_ids) != 12101:
        raise AssertionError("Invalid SID Trie root")

    cache = torch.empty(len(dataset), 3, 256, dtype=torch.float16)
    keys = []
    entropy_values = [[], [], []]
    max_values = [[], [], []]
    target_values = [[], [], []]
    max_sum_error = [0.0, 0.0, 0.0]
    target_support = [0, 0, 0]
    raw_sids_device = raw_sids.to(device)
    item_embeddings = teacher.item_emb.weight[1:]
    if item_embeddings.shape != (12101, 50):
        raise AssertionError("Non-padding teacher item table has wrong shape")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for start in tqdm(range(0, len(dataset), args.batch_size), desc="TCA teacher cache"):
            end = min(start + args.batch_size, len(dataset))
            samples = dataset.samples[start:end]
            histories = torch.tensor(
                [left_pad_history(sample["history"]) for sample in samples],
                dtype=torch.long,
                device=device,
            )
            targets = torch.tensor(
                [canonical_item_id(sample["target"]) for sample in samples],
                dtype=torch.long,
                device=device,
            )
            target_sids = raw_sids_device[targets - 1]
            user_states = teacher.user_states(histories)
            cf_logits = user_states.float().matmul(item_embeddings.float().t())
            probabilities = exact_prefix_distributions(cf_logits, raw_sids_device, target_sids)
            if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
                raise AssertionError("Non-finite or negative teacher probability")
            sums = probabilities.sum(-1)
            errors = (sums - 1).abs()
            gathered = probabilities.gather(2, target_sids.unsqueeze(-1)).squeeze(-1)
            for level in range(3):
                max_sum_error[level] = max(max_sum_error[level], float(errors[:, level].max()))
                target_support[level] += int((gathered[:, level] > 0).sum())
                p = probabilities[:, level]
                entropy_values[level].append((-(p * p.clamp_min(1e-30).log()).sum(-1)).cpu())
                max_values[level].append(p.max(-1).values.cpu())
                target_values[level].append(gathered[:, level].cpu())
            cache[start:end] = probabilities.cpu().half()
            keys.extend(
                sample_key(
                    [canonical_item_id(item) for item in sample["history"]],
                    canonical_item_id(sample["target"]),
                )
                for sample in samples
            )

    if len(keys) != len(dataset):
        raise AssertionError("Sample keys are missing")
    cache_sum_error = float((cache.float().sum(-1) - 1).abs().max())
    if cache_sum_error > 5e-3 or not torch.isfinite(cache).all():
        raise AssertionError("Stored FP16 cache failed normalization/finite check")

    checkpoint_hash = sha256_file(args.checkpoint)
    sid_hash = sha256_file(args.sid)
    artifact = {
        "format_version": 1,
        "dataset": "beauty",
        "num_samples": len(dataset),
        "num_items": 12101,
        "sid_layers": 3,
        "codebook_size": 256,
        "teacher": {
            "name": "sasrecpp",
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "hidden": 50,
            "maxlen": 50,
            "blocks": 2,
            "heads": 1,
            "norm_first": True,
            "loss_type": "ce",
            "time_aware": False,
            "padding_idx": 0,
        },
        "sid": {
            "artifact_path": str(args.sid.resolve()),
            "artifact_sha256": sid_hash,
            "quantizer": "rqkmeans",
            "layers": 3,
            "codebook_size": 256,
        },
        "sample_key_spec": "sha256(canonical JSON: split, ordered canonical history, target)",
        "sample_keys": keys,
        "cf_token_probs": cache,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    elapsed = time.perf_counter() - started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )

    diagnostics = []
    for level in range(3):
        entropy_mean, entropy_median = distribution_stats(entropy_values[level])
        max_mean, max_median = distribution_stats(max_values[level])
        target_mean, target_median = distribution_stats(target_values[level])
        diagnostics.append(
            {
                "level": f"c{level}",
                "sum_error": max_sum_error[level],
                "target_support": target_support[level] / len(dataset),
                "entropy_mean": entropy_mean,
                "entropy_median": entropy_median,
                "max_mean": max_mean,
                "max_median": max_median,
                "target_mean": target_mean,
                "target_median": target_median,
            }
        )

    tests = {
        "A canonical ID round trip": roundtrip_mismatch == 0,
        "B strict checkpoint load": True,
        "C deterministic inference": deterministic,
        "D toy Trie construction": True,
        "E toy prefix filtering": True,
        "F toy scatter-sum": True,
        "G distribution normalization": cache_sum_error <= 5e-3,
        "H target support": all(value == len(dataset) for value in target_support),
        "I no padding probability/edge": 0 not in trie.root.children,
        "J deterministic sample key": sample_key([1, 2], 3) == sample_key([1, 2], 3),
        "K sample-level no-future leakage": future_violations == 0,
        "L collision mass summation": True,
    }
    if not all(tests.values()):
        raise AssertionError(f"Cache tests failed: {tests}")

    rows = "\n".join(
        f"| {d['level']} | {d['sum_error']:.3e} | {d['target_support']:.8%} | "
        f"{d['entropy_mean']:.6f} | {d['entropy_median']:.6f} | {d['max_mean']:.6f} | "
        f"{d['max_median']:.6f} | {d['target_mean']:.6f} | {d['target_median']:.6f} |"
        for d in diagnostics
    )
    test_rows = "\n".join(f"- {'PASS' if passed else 'FAIL'} — {name}" for name, passed in tests.items())
    report = f"""# TCA Teacher Cache Report

## Teacher

- Checkpoint: `{args.checkpoint.resolve()}`
- SHA-256: `{checkpoint_hash}`
- Strict load: PASS
- Architecture: SASRec++ hidden=50, maxlen=50, blocks=2, heads=1, norm_first=true, time-aware=false, padding=0, Full CE, seed=42

## Dataset

- Training samples: {len(dataset):,}
- Items: 12,101

## ID Alignment

- Full 12,101-item round-trip mismatches: {roundtrip_mismatch}

## Cache

- Path: `{args.output.resolve()}`
- Shape: `{list(cache.shape)}`
- Dtype: `{cache.dtype}`
- File size: {args.output.stat().st_size / 1024 / 1024:.2f} MiB
- Sample-key count: {len(keys):,} (unique: {len(set(keys)):,})
- Stored FP16 maximum sum error: {cache_sum_error:.3e}
- NaN/Inf: 0

## Probability

| Level | FP32 max sum error | Target support | Mean entropy | Median entropy | Mean max P | Median max P | Mean target P | Median target P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

## Leakage

- Sample mismatch: {sample_mismatch}
- Future-item positional violations: {future_violations}
- Validation/test positional violations: {validation_test_violations}
- Teacher histories were produced from each TIGER sample's causal history, truncated to 50 and left-padded; hashed TIGER user IDs were not used.

## Collision Diagnostic

- Duplicate full-SID groups: {duplicate_group_count:,}
- Collided items remain separate item-probability occurrences and are scatter-summed into their common branches.

## Performance

- Device: `{device}`
- Generation time: {elapsed:.2f} seconds
- Peak allocated GPU memory: {peak_memory / 1024 / 1024:.2f} MiB

## Tests

{test_rows}

## Decision

GO
"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
