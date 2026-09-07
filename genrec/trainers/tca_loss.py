"""Train-only TCA cache validation and TIGER collaborative objectives."""

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F


LEVEL_OFFSETS = (1, 257, 513)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TCATeacherCache:
    def __init__(self, artifact, cache_sha256: str):
        self.artifact = artifact
        self.probabilities = artifact["cf_token_probs"]
        self.sample_keys = artifact["sample_keys"]
        self.cache_sha256 = cache_sha256
        self.sid_sha256 = artifact["sid"]["artifact_sha256"]

    @classmethod
    def load_and_validate(
        cls,
        cache_path,
        *,
        dataset,
        num_samples,
        num_items,
        sid_layers,
        codebook_size,
        sid_artifact_path,
    ):
        artifact = torch.load(cache_path, map_location="cpu", weights_only=False)
        expected = {
            "format_version": 1,
            "dataset": dataset,
            "num_samples": num_samples,
            "num_items": num_items,
            "sid_layers": sid_layers,
            "codebook_size": codebook_size,
        }
        mismatches = {
            key: (artifact.get(key), value)
            for key, value in expected.items()
            if artifact.get(key) != value
        }
        probabilities = artifact.get("cf_token_probs")
        expected_shape = (num_samples, sid_layers, codebook_size)
        if not isinstance(probabilities, torch.Tensor) or tuple(probabilities.shape) != expected_shape:
            mismatches["cf_token_probs.shape"] = (
                None if not isinstance(probabilities, torch.Tensor) else tuple(probabilities.shape),
                expected_shape,
            )
        if probabilities is not None and probabilities.dtype != torch.float16:
            mismatches["cf_token_probs.dtype"] = (probabilities.dtype, torch.float16)
        keys = artifact.get("sample_keys")
        if not isinstance(keys, list) or len(keys) != num_samples:
            mismatches["sample_keys"] = (None if keys is None else len(keys), num_samples)
        current_sid_hash = sha256_file(sid_artifact_path)
        cached_sid_hash = artifact.get("sid", {}).get("artifact_sha256")
        if cached_sid_hash != current_sid_hash:
            mismatches["sid.artifact_sha256"] = (cached_sid_hash, current_sid_hash)
        if mismatches:
            raise ValueError(f"TCA cache metadata mismatch: {mismatches}")
        return cls(artifact, sha256_file(cache_path))

    def batch(self, sample_indices, current_keys, device):
        indices = sample_indices.tolist()
        expected_keys = [self.sample_keys[index] for index in indices]
        if expected_keys != list(current_keys):
            mismatch_positions = [
                position
                for position, (expected, actual) in enumerate(zip(expected_keys, current_keys))
                if expected != actual
            ]
            raise ValueError(
                f"TCA cache sample-key mismatch at batch positions {mismatch_positions[:10]}"
            )
        return self.probabilities[sample_indices].to(device=device, dtype=torch.float32)

    def validate_dataset(self, dataset):
        """Validate every ordered cache row before the first optimizer step."""
        from genrec.data.tca_teacher_cache import canonical_item_id, sample_key

        if len(dataset) != len(self.sample_keys):
            raise ValueError("TCA cache/dataset length mismatch")
        mismatches = 0
        for index, sample in enumerate(dataset.samples):
            current_key = sample_key(
                [canonical_item_id(item) for item in sample["history"]],
                canonical_item_id(sample["target"]),
            )
            mismatches += current_key != self.sample_keys[index]
        if mismatches:
            raise ValueError(f"TCA cache has {mismatches} dataset-order sample-key mismatches")
        return mismatches


def level_logits_and_local_labels(logits, labels):
    if logits.ndim != 3 or logits.shape[1] != 3 or logits.shape[2] != 769:
        raise ValueError(f"Expected TIGER logits [B,3,769], got {tuple(logits.shape)}")
    if labels.shape != logits.shape[:2]:
        raise ValueError("TIGER label shape mismatch")
    level_logits = torch.stack(
        [logits[:, level, offset : offset + 256] for level, offset in enumerate(LEVEL_OFFSETS)],
        dim=1,
    )
    offsets = torch.tensor(LEVEL_OFFSETS, device=labels.device, dtype=labels.dtype)
    local_labels = labels - offsets.unsqueeze(0)
    if torch.any(local_labels < 0) or torch.any(local_labels >= 256):
        raise ValueError("TIGER label is outside its level-specific token range")
    return level_logits, local_labels


def compute_tca_loss(logits, labels, teacher_probs, *, alpha=0.1, temperature=4.5):
    if not 0 <= alpha <= 1:
        raise ValueError("tca_alpha must be in [0,1]")
    if temperature <= 0:
        raise ValueError("tca_temperature must be positive")
    level_logits, local_labels = level_logits_and_local_labels(logits.float(), labels)
    if teacher_probs.shape != (*labels.shape, 256):
        raise ValueError("Teacher probability shape mismatch")
    teacher_probs = teacher_probs.detach().to(device=logits.device, dtype=torch.float32)
    if not torch.isfinite(teacher_probs).all() or torch.any(teacher_probs < 0):
        raise ValueError("Teacher probabilities must be finite and non-negative")
    one_hot = F.one_hot(local_labels, num_classes=256).float()
    soft_target = (1 - alpha) * one_hot + alpha * teacher_probs
    if not torch.allclose(
        soft_target.sum(-1), torch.ones_like(soft_target[..., 0]), atol=5e-3, rtol=0
    ):
        raise ValueError("TCA soft target is not normalized")
    log_q = torch.where(
        soft_target > 0,
        soft_target.log(),
        torch.full_like(soft_target, float("-inf")),
    )
    scaled_logits = level_logits / temperature
    loss = -torch.logsumexp(scaled_logits + log_q, dim=-1)
    loss = loss + torch.logsumexp(scaled_logits, dim=-1)
    if not torch.isfinite(loss).all():
        raise FloatingPointError("TCA objective produced NaN/Inf")
    return loss.mean()


def compute_position_ce(logits, labels):
    """Hard CE over the 256 valid tokens at each SID position."""
    level_logits, local_labels = level_logits_and_local_labels(logits.float(), labels)
    return F.cross_entropy(level_logits.flatten(0, 1), local_labels.flatten())


def expand_teacher_to_full_vocab(teacher_probs, labels, vocab_size=769):
    """Embed per-level CF distributions into TIGER's full vocabulary."""
    if teacher_probs.shape != (*labels.shape, 256):
        raise ValueError("Teacher probability shape mismatch")
    if labels.ndim != 2 or labels.shape[1] != len(LEVEL_OFFSETS):
        raise ValueError("Expected TIGER labels [B,3]")
    teacher_probs = teacher_probs.detach().to(device=labels.device, dtype=torch.float32)
    if not torch.isfinite(teacher_probs).all() or torch.any(teacher_probs < 0):
        raise ValueError("Teacher probabilities must be finite and non-negative")
    if not torch.allclose(
        teacher_probs.sum(-1), torch.ones_like(teacher_probs[..., 0]), atol=5e-3, rtol=0
    ):
        raise ValueError("Teacher probabilities are not normalized")
    # The validated cache is float16, so restore exact per-row normalization in
    # float32 before constructing a strict probability target.
    teacher_probs = teacher_probs / teacher_probs.sum(-1, keepdim=True)
    full = torch.zeros(*labels.shape, vocab_size, device=labels.device, dtype=torch.float32)
    for level, offset in enumerate(LEVEL_OFFSETS):
        full[:, level, offset : offset + 256] = teacher_probs[:, level]
    if not torch.allclose(full.sum(-1), torch.ones_like(full[..., 0]), atol=5e-3, rtol=0):
        raise ValueError("Expanded teacher target is not normalized")
    return full


def compute_tca_full_vocab_loss(logits, labels, teacher_probs, *, alpha=0.1,
                                temperature=1.0, return_targets=False):
    """Full-769 soft CE preserving original TIGER cross-level negatives.

    This is a Semantic-ID TIGER adaptation, not the released TCA log-mixture
    objective. The primary controlled experiment requires temperature 1.
    """
    if logits.ndim != 3 or tuple(logits.shape[:2]) != tuple(labels.shape) or logits.shape[-1] != 769:
        raise ValueError(f"Expected logits [B,3,769] and matching labels, got {tuple(logits.shape)}")
    if not 0 <= alpha <= 1:
        raise ValueError("tca_alpha must be in [0,1]")
    if temperature != 1.0:
        raise ValueError("tca_full_vocab requires temperature=1.0")
    if torch.any(labels < 0) or torch.any(labels >= logits.shape[-1]):
        raise ValueError("TIGER label is outside the full vocabulary")
    teacher_full = expand_teacher_to_full_vocab(teacher_probs, labels, logits.shape[-1])
    hard_one_hot = F.one_hot(labels, num_classes=logits.shape[-1]).float()
    soft_target = (1 - alpha) * hard_one_hot + alpha * teacher_full
    if torch.any(soft_target < 0) or not torch.allclose(
        soft_target.sum(-1), torch.ones_like(soft_target[..., 0]), atol=1e-6, rtol=0
    ):
        raise ValueError("Full-vocabulary soft target is invalid")
    loss = -(soft_target * F.log_softmax(logits.float(), dim=-1)).sum(-1).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Full-vocabulary TCA objective produced NaN/Inf")
    return (loss, teacher_full, soft_target) if return_targets else loss


def select_training_loss(baseline_loss, *, objective, logits=None, labels=None,
                         teacher_probs=None, alpha=0.1, temperature=1.0):
    """Select an explicit objective while preserving the exact baseline object."""
    if objective == "baseline":
        return baseline_loss
    if objective == "position_ce":
        return compute_position_ce(logits, labels)
    if objective == "tca":
        if teacher_probs is None:
            raise ValueError("TCA teacher probabilities are required")
        return compute_tca_loss(
            logits, labels, teacher_probs, alpha=alpha, temperature=temperature
        )
    if objective == "tca_full_vocab":
        if teacher_probs is None:
            raise ValueError("TCA teacher probabilities are required")
        return compute_tca_full_vocab_loss(
            logits, labels, teacher_probs, alpha=alpha, temperature=temperature
        )
    raise ValueError(f"Unknown training_objective: {objective}")
