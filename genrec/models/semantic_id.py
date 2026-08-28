"""Shared semantic-ID artifacts and evaluation helpers.

The artifact deliberately stores assignments separately from a quantizer model.
TIGER only needs the stable ``item_index -> [code_0, ...]`` mapping, while a
quantizer checkpoint is useful for assigning future items.
"""
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Protocol

import numpy as np
import torch


class SemanticIdQuantizer(Protocol):
    """Minimal interface implemented by all non-neural SID quantizers."""

    n_layers: int
    codebook_size: int

    def fit(self, embeddings: np.ndarray) -> "SemanticIdQuantizer": ...
    def assign(self, embeddings: np.ndarray) -> Any: ...
    def reconstruct(self, embeddings: np.ndarray) -> np.ndarray: ...
    def save(self, path: str) -> None: ...


def sid_metrics(sem_ids: np.ndarray, embeddings: np.ndarray | None = None,
                reconstruction: np.ndarray | None = None) -> Dict[str, float | int | list]:
    """Compute collision, codebook-usage, and optional reconstruction metrics."""
    sem_ids = np.asarray(sem_ids, dtype=np.int64)
    if sem_ids.ndim != 2 or len(sem_ids) == 0:
        raise ValueError("sem_ids must be a non-empty [N, n_layers] array")

    keys = [tuple(row.tolist()) for row in sem_ids]
    counts = Counter(keys)
    total = len(keys)
    result: Dict[str, float | int | list] = {
        "total_items": total,
        "unique_sids": len(counts),
        "collision_rate": 1.0 - len(counts) / total,
        "max_collision": max(counts.values()),
        "layer_usage": [int(np.unique(sem_ids[:, i]).size) for i in range(sem_ids.shape[1])],
    }
    if embeddings is not None and reconstruction is not None:
        error = np.asarray(embeddings) - np.asarray(reconstruction)
        result["reconstruction_mse"] = float(np.mean(error ** 2))
        result["reconstruction_l2"] = float(np.linalg.norm(error, axis=1).mean())
    return result


def save_sid_artifact(path: str, sem_ids: np.ndarray, *, quantizer: str,
                      codebook_size: int, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Persist a portable, versioned semantic-ID mapping for downstream models."""
    sem_ids = np.asarray(sem_ids, dtype=np.int64)
    if sem_ids.ndim != 2 or np.any(sem_ids < 0):
        raise ValueError("sem_ids must be a non-negative [N, n_layers] array")
    payload = {
        "format_version": 1,
        "quantizer": quantizer,
        "codebook_size": int(codebook_size),
        "n_layers": int(sem_ids.shape[1]),
        "num_items": int(sem_ids.shape[0]),
        "sem_ids": torch.from_numpy(sem_ids),
        "metadata": metadata or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def load_sid_artifact(path: str) -> Dict[str, Any]:
    """Load and validate a SID artifact created by :func:`save_sid_artifact`."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"format_version", "quantizer", "codebook_size", "n_layers", "num_items", "sem_ids"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Invalid SID artifact {path}; missing {sorted(missing)}")
    sem_ids = payload["sem_ids"]
    if not isinstance(sem_ids, torch.Tensor) or sem_ids.ndim != 2:
        raise ValueError(f"Invalid SID artifact {path}; sem_ids must be rank 2")
    if sem_ids.shape != (payload["num_items"], payload["n_layers"]):
        raise ValueError(f"Invalid SID artifact {path}; metadata shape does not match sem_ids")
    return payload
