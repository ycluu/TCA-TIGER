"""Read-only checkpoint diagnostics for three-token Beauty TIGER generation."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from genrec.data.amazon import AmazonSeqDataset
from genrec.models.tiger import Tiger
from genrec.trainers.tiger_trainer import CollateFn


OFFSETS = (1, 257, 513)
MODEL_CONFIG = {
    "num_layers": 4, "num_decoder_layers": 4, "d_model": 128, "d_ff": 1024,
    "num_heads": 6, "d_kv": 64, "dropout_rate": 0.1, "vocab_size": 769,
    "pad_token_id": 0, "eos_token_id": 0, "feed_forward_proj": "relu", "sem_id_dim": 3,
}


def quantiles(values):
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "median": float(np.median(array)),
            "p10": float(np.percentile(array, 10)), "p90": float(np.percentile(array, 90))}


def load_model(path, device):
    model = Tiger(MODEL_CONFIG).to(device)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()


def diagnose(name, checkpoint, dataset, known_sids, device, batch_size):
    model = load_model(checkpoint, device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=CollateFn(256, 3, 3, 150))
    count = 0
    mass = torch.zeros(3, 4, dtype=torch.float64)
    full_argmax_correct = torch.zeros(3, dtype=torch.long)
    full_argmax_wrong = torch.zeros(3, dtype=torch.long)
    target_prob_full = torch.zeros(3, dtype=torch.float64)
    target_prob_local = torch.zeros(3, dtype=torch.float64)
    nll_full = torch.zeros(3, dtype=torch.float64)
    nll_local = torch.zeros(3, dtype=torch.float64)
    entropy_full = torch.zeros(3, dtype=torch.float64)
    entropy_local = torch.zeros(3, dtype=torch.float64)
    local_ranks = [[], [], []]
    per_position_correct = torch.zeros(3, dtype=torch.long)
    prefix1 = prefix2 = prefix3 = 0
    beam_level_valid = torch.zeros(10, 3, dtype=torch.long)
    beam_full_valid = torch.zeros(10, dtype=torch.long)
    beam_known = torch.zeros(10, dtype=torch.long)
    unique_per_user = []
    duplicate_per_user = []

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
        for batch in tqdm(loader, desc=name):
            inputs = batch["input_ids"].to(device)
            masks = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            _, logits = model(inputs, masks, labels)
            logits = logits.float()
            probs = logits.softmax(-1)
            local_predictions = []
            for level, offset in enumerate(OFFSETS):
                target = labels[:, level]
                local_target = target - offset
                local_logits = logits[:, level, offset:offset + 256]
                local_probs = local_logits.softmax(-1)
                target_prob_full[level] += probs[:, level].gather(1, target[:, None]).sum().cpu()
                target_prob_local[level] += local_probs.gather(1, local_target[:, None]).sum().cpu()
                nll_full[level] += (-probs[:, level].gather(1, target[:, None]).clamp_min(1e-30).log()).sum().cpu()
                nll_local[level] += (-local_probs.gather(1, local_target[:, None]).clamp_min(1e-30).log()).sum().cpu()
                entropy_full[level] += (-(probs[:, level] * probs[:, level].clamp_min(1e-30).log()).sum(-1)).sum().cpu()
                entropy_local[level] += (-(local_probs * local_probs.clamp_min(1e-30).log()).sum(-1)).sum().cpu()
                rank = 1 + (local_logits > local_logits.gather(1, local_target[:, None])).sum(-1)
                local_ranks[level].extend(rank.cpu().tolist())
                pred = local_logits.argmax(-1)
                local_predictions.append(pred)
                per_position_correct[level] += pred.eq(local_target).sum().cpu()
                full_pred = logits[:, level].argmax(-1)
                correct_range = full_pred.ge(offset) & full_pred.lt(offset + 256)
                full_argmax_correct[level] += correct_range.sum().cpu()
                full_argmax_wrong[level] += (~correct_range).sum().cpu()
                mass[level, 0] += probs[:, level, offset:offset + 256].sum().cpu()
                other_ranges = [i for i in range(3) if i != level]
                mass[level, 1] += sum(probs[:, level, OFFSETS[i]:OFFSETS[i] + 256].sum().cpu() for i in other_ranges)
                mass[level, 2] += probs[:, level, 0].sum().cpu()
            predictions = torch.stack(local_predictions, 1)
            local_targets = labels - torch.tensor(OFFSETS, device=device)
            prefix1 += predictions[:, :1].eq(local_targets[:, :1]).all(-1).sum().item()
            prefix2 += predictions[:, :2].eq(local_targets[:, :2]).all(-1).sum().item()
            prefix3 += predictions.eq(local_targets).all(-1).sum().item()

            generated = model.generate(inputs, masks, num_beams=10)[:, 1:].reshape(inputs.size(0), 10, 3)
            generated_cpu = generated.cpu()
            for rank in range(10):
                seq = generated_cpu[:, rank]
                for level, offset in enumerate(OFFSETS):
                    beam_level_valid[rank, level] += (seq[:, level].ge(offset) & seq[:, level].lt(offset + 256)).sum()
                valid = torch.ones(seq.size(0), dtype=torch.bool)
                raw = torch.empty_like(seq)
                for level, offset in enumerate(OFFSETS):
                    valid &= seq[:, level].ge(offset) & seq[:, level].lt(offset + 256)
                    raw[:, level] = seq[:, level] - offset
                beam_full_valid[rank] += valid.sum()
                beam_known[rank] += sum(valid[i] and tuple(raw[i].tolist()) in known_sids for i in range(seq.size(0)))
            for user_beams in generated_cpu.tolist():
                unique = len({tuple(seq) for seq in user_beams})
                unique_per_user.append(unique)
                duplicate_per_user.append(10 - unique)
            count += inputs.size(0)

    positions = []
    for level in range(3):
        ranks = np.asarray(local_ranks[level])
        positions.append({
            "position": f"c{level}", "correct_level_mass": float(mass[level, 0] / count),
            "other_level_mass": float(mass[level, 1] / count), "token0_mass": float(mass[level, 2] / count),
            "full_argmax_correct_level_rate": float(full_argmax_correct[level] / count),
            "full_argmax_wrong_level_rate": float(full_argmax_wrong[level] / count),
            "target_probability_full": float(target_prob_full[level] / count),
            "target_probability_local": float(target_prob_local[level] / count),
            "target_nll_full": float(nll_full[level] / count), "target_nll_local": float(nll_local[level] / count),
            "entropy_full": float(entropy_full[level] / count), "entropy_local": float(entropy_local[level] / count),
            "token_accuracy_local": float(per_position_correct[level] / count),
            "rank_mrr": float((1 / ranks).mean()), "rank_mean": float(ranks.mean()),
            "rank_median": float(np.median(ranks)), "rank_top1": float((ranks <= 1).mean()),
            "rank_top5": float((ranks <= 5).mean()), "rank_top10": float((ranks <= 10).mean()),
        })
    groups = {"beam1": [0], "beam2_5": list(range(1, 5)), "beam6_10": list(range(5, 10)), "all": list(range(10))}
    beam = {}
    for group, ranks in groups.items():
        denom = count * len(ranks)
        beam[group] = {
            "level_valid_by_position": [float(beam_level_valid[ranks, level].sum() / denom) for level in range(3)],
            "full_level_valid": float(beam_full_valid[ranks].sum() / denom),
            "known_sid": float(beam_known[ranks].sum() / denom),
            "unknown_sid": float((beam_full_valid[ranks].sum() - beam_known[ranks].sum()) / denom),
        }
    return {
        "name": name, "checkpoint": str(Path(checkpoint).resolve()), "samples": count,
        "positions": positions, "prefix_accuracy": {"c0": prefix1/count, "c0_c1": prefix2/count, "full": prefix3/count},
        "beam": beam, "unique_sequences_per_user": quantiles(unique_per_user),
        "duplicate_beams_per_user": quantiles(duplicate_per_user),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--name", default="checkpoint")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", default=96, type=int)
    args = parser.parse_args()
    sid_path = Path("out/tiger/amazon/beauty/rqkmeans/semantic_ids.pt")
    artifact = torch.load(sid_path, map_location="cpu", weights_only=False)
    known_sids = {tuple(row) for row in artifact["sem_ids"].tolist()}
    dataset = AmazonSeqDataset(
        root="dataset/amazon", split="beauty", train_test_split="valid", max_seq_len=50,
        add_disambiguation=False, semantic_id_path=str(sid_path), target_n_layers=3,
    )
    result = diagnose(args.name, args.checkpoint, dataset, known_sids,
                      torch.device("cuda" if torch.cuda.is_available() else "cpu"), args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
