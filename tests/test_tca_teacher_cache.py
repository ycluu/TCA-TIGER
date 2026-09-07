import torch

from genrec.data.tca_teacher_cache import (
    SIDTrie,
    aggregate_item_probabilities,
    canonical_item_id,
    exact_prefix_distributions,
    sample_key,
    tiger_item_index,
)


def test_item_id_roundtrip():
    assert all(tiger_item_index(canonical_item_id(i)) == i for i in range(12101))


def test_trie_construction_and_prefix_filtering():
    trie = SIDTrie()
    mapping = {1: [12, 31, 7], 2: [12, 31, 91], 3: [12, 83, 22], 4: [47, 9, 16]}
    for item, tokens in mapping.items():
        trie.insert(tokens, item)
    assert trie.pairs([]) == ([12, 12, 12, 47], [1, 2, 3, 4])
    assert trie.pairs([12]) == ([31, 31, 83], [1, 2, 3])
    assert trie.pairs([12, 31]) == ([7, 91], [1, 2])
    assert trie.pairs([99]) == ([], [])


def test_scatter_sum_and_normalization():
    logits = torch.log(torch.tensor([[0.1, 0.2, 0.3, 0.4]]))
    codes = torch.tensor([[5, 5, 7, 9]])
    result = aggregate_item_probabilities(logits, codes)
    assert torch.allclose(result.sum(-1), torch.ones(1))
    assert torch.allclose(result[0, [5, 7, 9]], torch.tensor([0.3, 0.3, 0.4]))


def test_exact_prefix_target_support_and_collision_mass():
    sids = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 3, 4], [5, 6, 7]])
    targets = sids[[0, 3]]
    logits = torch.zeros(2, 4)
    result = exact_prefix_distributions(logits, sids, targets)
    assert result.shape == (2, 3, 256)
    assert torch.allclose(result.sum(-1), torch.ones(2, 3))
    assert result[0, 2, 2] == 1  # both collided items contribute to the same c2 branch
    assert all(result[:, level].gather(1, targets[:, level : level + 1]).gt(0).all() for level in range(3))


def test_sample_key_is_deterministic_and_order_sensitive():
    assert sample_key([1, 2], 3) == sample_key([1, 2], 3)
    assert sample_key([1, 2], 3) != sample_key([2, 1], 3)
    assert sample_key([1, 2], 3) != sample_key([1, 2], 4)
