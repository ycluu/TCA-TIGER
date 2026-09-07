import random

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from genrec.trainers.tca_loss import (
    TCATeacherCache,
    compute_tca_full_vocab_loss,
    compute_position_ce,
    compute_tca_loss,
    expand_teacher_to_full_vocab,
    level_logits_and_local_labels,
    select_training_loss,
)


def _artifact(tmp_path, keys=("duplicate", "duplicate", "third")):
    sid = tmp_path / "sid.pt"
    sid.write_bytes(b"sid")
    import hashlib

    sid_hash = hashlib.sha256(b"sid").hexdigest()
    cache_path = tmp_path / "cache.pt"
    artifact = {
        "format_version": 1,
        "dataset": "beauty",
        "num_samples": 3,
        "num_items": 12101,
        "sid_layers": 3,
        "codebook_size": 256,
        "sid": {"artifact_sha256": sid_hash},
        "sample_keys": list(keys),
        "cf_token_probs": torch.full((3, 3, 256), 1 / 256, dtype=torch.float16),
    }
    torch.save(artifact, cache_path)
    return cache_path, sid


def test_cache_metadata_and_duplicate_keys(tmp_path):
    cache_path, sid = _artifact(tmp_path)
    cache = TCATeacherCache.load_and_validate(
        cache_path, dataset="beauty", num_samples=3, num_items=12101,
        sid_layers=3, codebook_size=256, sid_artifact_path=sid,
    )
    assert cache.sample_keys[:2] == ["duplicate", "duplicate"]
    with pytest.raises(ValueError, match="metadata mismatch"):
        TCATeacherCache.load_and_validate(
            cache_path, dataset="beauty", num_samples=4, num_items=12101,
            sid_layers=3, codebook_size=256, sid_artifact_path=sid,
        )


def test_index_survives_shuffle_and_key_is_asserted(tmp_path):
    cache_path, sid = _artifact(tmp_path, ("zero", "one", "two"))
    cache = TCATeacherCache.load_and_validate(
        cache_path, dataset="beauty", num_samples=3, num_items=12101,
        sid_layers=3, codebook_size=256, sid_artifact_path=sid,
    )
    rows = [{"index": i, "key": key} for i, key in enumerate(cache.sample_keys)]
    loader = DataLoader(rows, batch_size=3, shuffle=True)
    batch = next(iter(loader))
    result = cache.batch(batch["index"], batch["key"], "cpu")
    assert result.shape == (3, 3, 256)
    bad_keys = list(batch["key"])
    bad_keys[0] = "bad"
    with pytest.raises(ValueError, match="sample-key mismatch"):
        cache.batch(batch["index"], bad_keys, "cpu")


def test_full_dataset_order_key_validation(tmp_path):
    from genrec.data.tca_teacher_cache import sample_key

    class FakeDataset:
        samples = [
            {"history": [0], "target": 1},
            {"history": [0], "target": 1},
            {"history": [1], "target": 2},
        ]

        def __len__(self):
            return len(self.samples)

    keys = [sample_key([1], 2), sample_key([1], 2), sample_key([2], 3)]
    cache_path, sid = _artifact(tmp_path, keys)
    cache = TCATeacherCache.load_and_validate(
        cache_path, dataset="beauty", num_samples=3, num_items=12101,
        sid_layers=3, codebook_size=256, sid_artifact_path=sid,
    )
    assert cache.validate_dataset(FakeDataset()) == 0
    cache.sample_keys[1] = "wrong"
    with pytest.raises(ValueError, match="dataset-order"):
        cache.validate_dataset(FakeDataset())


def test_level_slices_and_label_offsets():
    logits = torch.arange(769.0).view(1, 1, 769).expand(1, 3, -1).clone()
    labels = torch.tensor([[1, 257, 768]])
    sliced, local = level_logits_and_local_labels(logits, labels)
    assert torch.equal(sliced[0, 0], torch.arange(1.0, 257.0))
    assert torch.equal(sliced[0, 1], torch.arange(257.0, 513.0))
    assert torch.equal(sliced[0, 2], torch.arange(513.0, 769.0))
    assert torch.equal(local, torch.tensor([[0, 0, 255]]))


def test_soft_target_objective_toy_zero_probs_and_alpha_zero():
    logits = torch.zeros(1, 3, 769)
    logits[0, 0, 1] = 2.0
    labels = torch.tensor([[1, 257, 513]])
    teacher = torch.zeros(1, 3, 256)
    teacher[..., 0] = 1
    loss = compute_tca_loss(logits, labels, teacher, alpha=0.1, temperature=1.0)
    expected_positions = [
        -2.0 + torch.log(torch.exp(torch.tensor(2.0)) + 255),
        torch.log(torch.tensor(256.0)),
        torch.log(torch.tensor(256.0)),
    ]
    assert torch.allclose(loss, torch.stack(expected_positions).mean())
    alpha_zero = compute_tca_loss(logits, labels, teacher, alpha=0, temperature=1.0)
    slices, local = level_logits_and_local_labels(logits, labels)
    assert torch.allclose(alpha_zero, F.cross_entropy(slices.flatten(0, 1), local.flatten()))
    assert torch.isfinite(loss)


def test_baseline_path_returns_existing_loss_object():
    baseline = torch.tensor(3.25, requires_grad=True)
    assert select_training_loss(baseline, objective="baseline") is baseline


def test_position_ce_equals_alpha_zero_but_nontrivial_tca_differs():
    torch.manual_seed(42)
    logits = torch.randn(4, 3, 769)
    labels = torch.tensor([[1, 257, 513], [256, 512, 768], [7, 300, 700], [99, 400, 600]])
    teacher = torch.softmax(torch.randn(4, 3, 256), dim=-1)
    position_loss = compute_position_ce(logits, labels)
    alpha_zero = compute_tca_loss(logits, labels, teacher, alpha=0, temperature=1)
    tca_loss = compute_tca_loss(logits, labels, teacher, alpha=0.1, temperature=1)
    assert abs(float(position_loss - alpha_zero)) < 1e-6
    assert not torch.allclose(position_loss, tca_loss)
    assert torch.isfinite(tca_loss)


def test_position_ce_does_not_require_or_access_cache():
    logits = torch.randn(2, 3, 769)
    labels = torch.tensor([[1, 257, 513], [2, 258, 514]])
    loss = select_training_loss(
        torch.tensor(-1.0), objective="position_ce", logits=logits, labels=labels
    )
    assert torch.isfinite(loss)


def test_full_vocab_teacher_mapping_and_targets():
    labels = torch.tensor([[1, 257, 513], [256, 512, 768]])
    teacher = torch.zeros(2, 3, 256)
    teacher[:, :, 7] = 1
    full = expand_teacher_to_full_vocab(teacher, labels)
    assert full.shape == (2, 3, 769)
    assert torch.equal(full[0, 0].nonzero().flatten(), torch.tensor([8]))
    assert torch.equal(full[0, 1].nonzero().flatten(), torch.tensor([264]))
    assert torch.equal(full[0, 2].nonzero().flatten(), torch.tensor([520]))
    assert torch.all(full[..., 0] == 0)
    assert torch.allclose(full.sum(-1), torch.ones(2, 3))
    loss, teacher_full, mixed = compute_tca_full_vocab_loss(
        torch.randn(2, 3, 769), labels, teacher, alpha=0.1, return_targets=True
    )
    assert torch.isfinite(loss)
    assert torch.allclose(teacher_full.sum(-1), torch.ones(2, 3))
    assert torch.allclose(mixed.sum(-1), torch.ones(2, 3), atol=1e-6)
    assert torch.all(mixed >= 0)
    for level, offset in enumerate((1, 257, 513)):
        outside = torch.ones(769, dtype=torch.bool)
        outside[offset : offset + 256] = False
        assert torch.all(mixed[:, level, outside] == 0)


def test_full_vocab_alpha_zero_loss_and_gradients_equal_hard_ce():
    torch.manual_seed(42)
    labels = torch.tensor([[1, 257, 513], [256, 512, 768], [7, 300, 700]])
    teacher = torch.softmax(torch.randn(3, 3, 256), dim=-1)
    logits_hard = torch.randn(3, 3, 769, requires_grad=True)
    logits_tca = logits_hard.detach().clone().requires_grad_(True)
    hard_loss = F.cross_entropy(logits_hard.flatten(0, 1), labels.flatten())
    tca_loss = compute_tca_full_vocab_loss(logits_tca, labels, teacher, alpha=0)
    assert abs(float(hard_loss - tca_loss)) < 1e-6
    hard_loss.backward()
    tca_loss.backward()
    assert torch.allclose(logits_hard.grad, logits_tca.grad, atol=1e-7, rtol=1e-6)


def test_full_vocab_wrong_level_and_token_zero_gradients():
    logits = torch.zeros(1, 3, 769, requires_grad=True)
    labels = torch.tensor([[1, 257, 513]])
    teacher = torch.zeros(1, 3, 256)
    teacher[..., 3] = 1
    loss = compute_tca_full_vocab_loss(logits, labels, teacher, alpha=0.1)
    loss.backward()
    assert logits.grad[0, 0, 0] > 0
    assert torch.all(logits.grad[0, 0, 257:513] > 0)
    assert torch.all(logits.grad[0, 0, 513:769] > 0)


def test_full_vocab_alpha_point_one_changes_loss_and_baseline_needs_no_cache():
    torch.manual_seed(7)
    logits = torch.randn(2, 3, 769)
    labels = torch.tensor([[1, 257, 513], [2, 258, 514]])
    teacher = torch.softmax(torch.randn(2, 3, 256), dim=-1)
    hard = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    collaborative = compute_tca_full_vocab_loss(logits, labels, teacher, alpha=0.1)
    assert not torch.allclose(hard, collaborative)
    baseline = torch.tensor(2.0)
    assert select_training_loss(baseline, objective="baseline") is baseline
