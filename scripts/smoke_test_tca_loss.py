"""One-real-batch smoke test for Strategy-B full-vocabulary TCA."""

import json
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from genrec.data.amazon import AmazonSeqDataset
from genrec.models.tiger import Tiger
from genrec.trainers.tca_loss import LEVEL_OFFSETS, TCATeacherCache, compute_tca_full_vocab_loss
from genrec.trainers.tiger_trainer import CollateFn

MODEL_CONFIG = {
    "num_layers": 4, "num_decoder_layers": 4, "d_model": 128, "d_ff": 1024,
    "num_heads": 6, "d_kv": 64, "dropout_rate": 0.1, "vocab_size": 769,
    "pad_token_id": 0, "eos_token_id": 0, "feed_forward_proj": "relu", "sem_id_dim": 3,
}


def main():
    torch.manual_seed(42)
    sid_path = Path("out/tiger/amazon/beauty/rqkmeans/semantic_ids.pt")
    cache_path = Path("out/tiger/amazon/beauty/tca/tca_teacher_cache.pt")
    dataset = AmazonSeqDataset(
        root="dataset/amazon", split="beauty", train_test_split="train", max_seq_len=50,
        add_disambiguation=False, semantic_id_path=str(sid_path), target_n_layers=3,
    )
    batch = next(iter(DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=0,
        collate_fn=CollateFn(256, 3, 3, 150),
    )))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = TCATeacherCache.load_and_validate(
        cache_path, dataset="beauty", num_samples=len(dataset), num_items=12101,
        sid_layers=3, codebook_size=256, sid_artifact_path=sid_path,
    )
    sample_mismatches = cache.validate_dataset(dataset)
    teacher = cache.batch(batch["sample_indices"], batch["sample_keys"], device)
    model = Tiger(MODEL_CONFIG).to(device).train()
    labels = batch["labels"].to(device)
    baseline_loss, logits = model(
        batch["input_ids"].to(device), batch["attention_mask"].to(device), labels,
    )
    alpha_zero, teacher_full, mixed = compute_tca_full_vocab_loss(
        logits, labels, teacher, alpha=0, return_targets=True
    )
    alpha_point_one = compute_tca_full_vocab_loss(logits, labels, teacher, alpha=0.1)

    hard_logits = logits.detach().float().clone().requires_grad_(True)
    tca_logits = logits.detach().float().clone().requires_grad_(True)
    hard_ce = F.cross_entropy(hard_logits.flatten(0, 1), labels.flatten())
    tca_zero_detached = compute_tca_full_vocab_loss(tca_logits, labels, teacher, alpha=0)
    hard_ce.backward()
    tca_zero_detached.backward()
    gradient_max_abs_diff = float((hard_logits.grad - tca_logits.grad).abs().max())

    toy_logits = torch.zeros(1, 3, 769, device=device, requires_grad=True)
    toy_labels = torch.tensor([[1, 257, 513]], device=device)
    toy_teacher = torch.zeros(1, 3, 256, device=device)
    toy_teacher[..., 3] = 1
    compute_tca_full_vocab_loss(toy_logits, toy_labels, toy_teacher, alpha=0.1).backward()
    wrong_level_gradients = {
        "token_0": float(toy_logits.grad[0, 0, 0]),
        "c1_range_min": float(toy_logits.grad[0, 0, 257:513].min()),
        "c2_range_min": float(toy_logits.grad[0, 0, 513:769].min()),
    }

    probs = logits.detach().float().softmax(-1)
    probability_mass = []
    for level, offset in enumerate(LEVEL_OFFSETS):
        correct = probs[:, level, offset : offset + 256].sum(-1).mean()
        token_zero = probs[:, level, 0].mean()
        probability_mass.append({
            "position": level, "correct_level": float(correct),
            "wrong_level": float(1 - correct - token_zero), "token_0": float(token_zero),
        })

    model.zero_grad(set_to_none=True)
    alpha_point_one.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    gradients_finite = bool(gradients) and all(torch.isfinite(g).all() for g in gradients)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "tca_full_vocab_smoke.pt"
        torch.save(model.state_dict(), checkpoint)
        restored = Tiger(MODEL_CONFIG)
        restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)

    result = {
        "device": str(device), "cache_shape": list(teacher.shape),
        "logits_shape": list(logits.shape), "expanded_cf_target_shape": list(teacher_full.shape),
        "mixed_target_shape": list(mixed.shape), "cache_hash": cache.cache_sha256,
        "sid_hash": cache.sid_sha256, "sample_key_mismatches": sample_mismatches,
        "baseline_outputs_loss": float(baseline_loss), "tca_alpha0_loss": float(alpha_zero),
        "alpha0_absolute_difference": float((baseline_loss.float() - alpha_zero).abs()),
        "alpha0_gradient_max_absolute_difference": gradient_max_abs_diff,
        "wrong_level_gradients_c0_toy": wrong_level_gradients,
        "tca_alpha_point_one_loss": float(alpha_point_one),
        "alpha_point_one_differs": not torch.allclose(baseline_loss.float(), alpha_point_one),
        "probability_mass": probability_mass, "loss_finite": bool(torch.isfinite(alpha_point_one)),
        "backward": True, "gradients_finite": gradients_finite, "checkpoint_strict_reload": True,
    }
    if result["alpha0_absolute_difference"] >= 1e-6 or gradient_max_abs_diff >= 1e-6:
        raise AssertionError(result)
    required = ("alpha_point_one_differs", "loss_finite", "backward", "gradients_finite", "checkpoint_strict_reload")
    if sample_mismatches or not all(result[key] for key in required):
        raise AssertionError(result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
