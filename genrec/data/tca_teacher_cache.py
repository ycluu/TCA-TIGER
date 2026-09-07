"""Core utilities for exact full-catalog TCA teacher-cache generation."""

import hashlib
import json
from collections import defaultdict

import torch


def canonical_item_id(tiger_item_index: int) -> int:
    return tiger_item_index + 1


def tiger_item_index(canonical_id: int) -> int:
    return canonical_id - 1


def sample_key(history, target: int, split: str = "train") -> str:
    payload = {"split": split, "history": list(history), "target": int(target)}
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


class TrieNode:
    def __init__(self):
        self.children = {}
        self.next_tokens = []
        self.item_ids = []


class SIDTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, tokens, canonical_id: int):
        node = self.root
        for token in tokens:
            node.next_tokens.append(int(token))
            node.item_ids.append(int(canonical_id))
            node = node.children.setdefault(int(token), TrieNode())

    def pairs(self, prefix):
        node = self.root
        for token in prefix:
            node = node.children.get(int(token))
            if node is None:
                return [], []
        return node.next_tokens, node.item_ids


def build_sid_trie(raw_sids: torch.Tensor) -> SIDTrie:
    trie = SIDTrie()
    for row, sid in enumerate(raw_sids.tolist()):
        trie.insert([sid[0] + 1, sid[1] + 257, sid[2] + 513], row + 1)
    return trie


def aggregate_item_probabilities(item_logits, next_raw_codes, codebook_size=256):
    probabilities = torch.softmax(item_logits.float(), dim=-1)
    output = torch.zeros(
        item_logits.shape[0], codebook_size, dtype=torch.float32, device=item_logits.device
    )
    return output.scatter_add_(1, next_raw_codes, probabilities)


def exact_prefix_distributions(cf_logits, raw_sids, target_sids):
    """Return `[B,3,256]` full-catalog conditional distributions in FP32."""
    batch_size, num_items = cf_logits.shape
    if raw_sids.shape != (num_items, 3):
        raise ValueError("SID/catalog shape mismatch")
    output = torch.empty(batch_size, 3, 256, device=cf_logits.device, dtype=torch.float32)
    for level in range(3):
        if level == 0:
            masked_logits = cf_logits.float()
        else:
            matches = torch.ones(
                batch_size, num_items, dtype=torch.bool, device=cf_logits.device
            )
            for prefix_level in range(level):
                matches &= raw_sids[:, prefix_level].unsqueeze(0).eq(
                    target_sids[:, prefix_level].unsqueeze(1)
                )
            masked_logits = cf_logits.float().masked_fill(~matches, float("-inf"))
        codes = raw_sids[:, level].unsqueeze(0).expand(batch_size, -1)
        output[:, level] = aggregate_item_probabilities(masked_logits, codes)
    return output


def duplicate_sid_stats(raw_sids):
    groups = defaultdict(list)
    for item_id, sid in enumerate(raw_sids.tolist(), start=1):
        groups[tuple(sid)].append(item_id)
    duplicate_groups = [items for items in groups.values() if len(items) > 1]
    return len(duplicate_groups), duplicate_groups
