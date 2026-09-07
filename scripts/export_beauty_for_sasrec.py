#!/usr/bin/env python3
"""Export TIGER's canonical Amazon Beauty sequences for an external SASRec++ run.

The TIGER loader uses a zero-based array index for items.  The corresponding
public/canonical ItemID stored in the item-embedding artifact is one-based, so
this exporter writes ``internal_index + 1`` and reserves 0 for padding.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd
import torch


REVIEW_FILENAME = "reviews_Beauty_5.json.gz"
EMBEDDING_FILENAME = "item_emb_sentence-t5-base.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/amazon"))
    parser.add_argument(
        "--semantic-id-path",
        type=Path,
        default=Path("out/tiger/amazon/beauty/rqkmeans/semantic_ids.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser.parse_args()


def iter_reviews(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def load_canonical_sequences(review_path: Path) -> tuple[list[str], list[list[int]], dict[str, int]]:
    """Exactly reproduce AmazonSeqDataset._load_sequences before SID encoding."""
    item_index_by_asin: dict[str, int] = {}
    interactions_by_user: OrderedDict[str, list[tuple[int, int]]] = OrderedDict()

    for review in iter_reviews(review_path):
        asin = review.get("asin")
        reviewer_id = review.get("reviewerID")
        timestamp = review.get("unixReviewTime", 0)
        if not asin or not reviewer_id:
            continue
        if asin not in item_index_by_asin:
            item_index_by_asin[asin] = len(item_index_by_asin)
        interactions_by_user.setdefault(reviewer_id, []).append(
            (timestamp, item_index_by_asin[asin])
        )

    reviewer_ids: list[str] = []
    sequences: list[list[int]] = []
    for reviewer_id, timestamped_items in interactions_by_user.items():
        # Stable sort matches TIGER: raw-file order breaks equal-timestamp ties.
        timestamped_items.sort(key=lambda pair: pair[0])
        internal_indices = [item_index for _, item_index in timestamped_items]
        if len(internal_indices) >= 5:  # Exact current TIGER eligibility rule.
            reviewer_ids.append(reviewer_id)
            sequences.append([item_index + 1 for item_index in internal_indices])

    return reviewer_ids, sequences, item_index_by_asin


def verify_item_alignment(
    item_index_by_asin: dict[str, int], embedding_path: Path, semantic_id_path: Path
) -> tuple[int, dict[str, Any]]:
    item_ids = pd.read_parquet(embedding_path, columns=["ItemID"])["ItemID"].tolist()
    expected_ids = list(range(1, len(item_index_by_asin) + 1))
    assert item_ids == expected_ids, (
        "Embedding ItemID order is not the canonical 1..N order expected from the "
        "raw-review first-seen ASIN mapping"
    )

    artifact = torch.load(semantic_id_path, map_location="cpu", weights_only=False)
    required = {"sem_ids", "num_items", "n_layers", "quantizer"}
    assert isinstance(artifact, dict) and required <= artifact.keys(), "Invalid SID artifact"
    sem_ids = artifact["sem_ids"]
    assert sem_ids.ndim == 2
    assert sem_ids.shape[0] == len(expected_ids)
    assert int(artifact["num_items"]) == len(expected_ids)
    return int(sem_ids.shape[0]), artifact


def main() -> None:
    args = parse_args()
    review_path = args.dataset_root / "raw" / "beauty" / REVIEW_FILENAME
    embedding_path = args.dataset_root / "processed" / "beauty" / EMBEDDING_FILENAME
    for path in (review_path, embedding_path, args.semantic_id_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    reviewer_ids, sequences, item_index_by_asin = load_canonical_sequences(review_path)
    semantic_item_count, sid_artifact = verify_item_alignment(
        item_index_by_asin, embedding_path, args.semantic_id_path
    )

    assert sequences and all(len(sequence) >= 5 for sequence in sequences)
    canonical_interactions = sum(map(len, sequences))
    canonical_items = set(range(1, len(item_index_by_asin) + 1))
    recommended_items = {item for sequence in sequences for item in sequence}
    assert recommended_items <= canonical_items
    items_without_semantic_id = sum(item > semantic_item_count for item in recommended_items)
    assert items_without_semantic_id == 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sequence_path = args.output_dir / "beauty_sasrec.txt"
    metadata_path = args.output_dir / "beauty_sasrec_meta.json"
    split_path = args.output_dir / "beauty_sasrec_split_reference.pt"

    exported_interactions = 0
    with sequence_path.open("w", encoding="utf-8", newline="\n") as stream:
        for user_id, sequence in enumerate(sequences, start=1):
            for item_id in sequence:
                stream.write(f"{user_id} {item_id}\n")
                exported_interactions += 1
    assert exported_interactions == canonical_interactions

    split_reference = {
        user_id: {
            "train_items": sequence[:-2],
            "validation_item": sequence[-2],
            "test_item": sequence[-1],
            "train_length": len(sequence) - 2,
        }
        for user_id, sequence in enumerate(sequences, start=1)
    }
    torch.save(split_reference, split_path)

    lengths = [len(sequence) for sequence in sequences]
    num_users = len(sequences)
    num_items = len(item_index_by_asin)
    metadata: dict[str, Any] = {
        "dataset": "Amazon Beauty",
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions": canonical_interactions,
        "train_interactions": canonical_interactions - 2 * num_users,
        "validation_interactions": num_users,
        "test_interactions": num_users,
        "padding_idx": 0,
        "item_id_min": 1,
        "item_id_max": num_items,
        "user_id_min": 1,
        "user_id_max": num_users,
        "split": "leave-two-out",
        "validation_rule": "second-last interaction",
        "test_rule": "last interaction",
        "sequence_order": "same as TIGER canonical sequence",
        "min_sequence_length": min(lengths),
        "mean_sequence_length": statistics.fmean(lengths),
        "median_sequence_length": statistics.median(lengths),
        "max_sequence_length": max(lengths),
        "items_without_semantic_id": items_without_semantic_id,
        "exported_interactions": exported_interactions,
        "canonical_interactions": canonical_interactions,
        "exported_equals_canonical": exported_interactions == canonical_interactions,
        "source_reviews": str(review_path),
        "source_item_embeddings": str(embedding_path),
        "source_semantic_ids": str(args.semantic_id_path),
        "semantic_id_quantizer": str(sid_artifact["quantizer"]),
        "semantic_id_layers": int(sid_artifact["n_layers"]),
        "item_id_protocol": (
            "TIGER zero-based semantic_ids row index + 1; matches embedding ItemID and reserves 0"
        ),
        "user_id_protocol": (
            "stable 1-based enumeration in TIGER's retained-user order; TIGER sample hash IDs are "
            "non-canonical and are not used"
        ),
        "raw_reviewer_ids_retained": len(reviewer_ids),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"sequence_path={sequence_path.resolve()}")
    print(f"metadata_path={metadata_path.resolve()}")
    print(f"split_path={split_path.resolve()}")


if __name__ == "__main__":
    main()
