"""OPQ-rotated residual K-Means.

This is an alternating Procrustes implementation: fit residual K-Means in a
rotated space, then update the orthogonal rotation to align inputs with their
quantized reconstruction.  With full-dimensional Euclidean K-Means the method
can be close to rotation-invariant; it is kept as an explicit ablation rather
than assumed to improve a baseline.
"""
from typing import NamedTuple

import numpy as np
import torch

from genrec.models.rqkmeans import RqKmeans, RqKmeansOutput


class OpqRqKmeansOutput(NamedTuple):
    sem_ids: np.ndarray


class OpqRqKmeans:
    """Learn an orthogonal pre-rotation and then apply residual K-Means."""

    def __init__(self, n_layers: int = 3, codebook_size: int = 256,
                 random_state: int = 42, max_iter: int = 100, batch_size: int = 4096,
                 backend: str = "auto", n_init: int = 10, dead_reinit: bool = True,
                 opq_iterations: int = 3):
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.backend = backend
        self.n_init = n_init
        self.dead_reinit = dead_reinit
        self.opq_iterations = opq_iterations
        self.rotation: np.ndarray | None = None
        self.rqkmeans: RqKmeans | None = None

    def _new_rqkmeans(self) -> RqKmeans:
        return RqKmeans(self.n_layers, self.codebook_size, self.random_state,
                         self.max_iter, self.batch_size, self.backend,
                         self.n_init, self.dead_reinit)

    def fit(self, embeddings: np.ndarray) -> "OpqRqKmeans":
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("embeddings must have shape [N, D]")
        self.rotation = np.eye(x.shape[1], dtype=np.float32)

        for iteration in range(self.opq_iterations):
            quantizer = self._new_rqkmeans().fit(x @ self.rotation)
            rotated_reconstruction = quantizer.reconstruct(x @ self.rotation)
            # Orthogonal Procrustes: argmin_R ||X R - Q||_F.
            u, _, vt = np.linalg.svd(x.T @ rotated_reconstruction, full_matrices=False)
            self.rotation = (u @ vt).astype(np.float32)
            mse = np.mean((x @ self.rotation - rotated_reconstruction) ** 2)
            print(f"OPQ iteration {iteration + 1}/{self.opq_iterations}: rotated MSE={mse:.6f}")

        self.rqkmeans = self._new_rqkmeans().fit(x @ self.rotation)
        return self

    def assign(self, embeddings: np.ndarray) -> OpqRqKmeansOutput:
        if self.rotation is None or self.rqkmeans is None:
            raise RuntimeError("Must call fit() or load() before assign()")
        output: RqKmeansOutput = self.rqkmeans.assign(np.asarray(embeddings, dtype=np.float32) @ self.rotation)
        return OpqRqKmeansOutput(sem_ids=output.sem_ids)

    def reconstruct(self, embeddings: np.ndarray) -> np.ndarray:
        if self.rotation is None or self.rqkmeans is None:
            raise RuntimeError("Must call fit() or load() before reconstruct()")
        rotated = np.asarray(embeddings, dtype=np.float32) @ self.rotation
        return self.rqkmeans.reconstruct(rotated) @ self.rotation.T

    def save(self, path: str) -> None:
        if self.rotation is None or self.rqkmeans is None:
            raise RuntimeError("Cannot save an unfitted quantizer")
        torch.save({
            "type": "opq_rqkmeans", "n_layers": self.n_layers,
            "codebook_size": self.codebook_size, "rotation": self.rotation,
            "centroids": self.rqkmeans.centroids, "opq_iterations": self.opq_iterations,
        }, path)
        print(f"Saved OPQ-RQ-KMeans to {path}")

    @classmethod
    def load(cls, path: str) -> "OpqRqKmeans":
        state = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(n_layers=state["n_layers"], codebook_size=state["codebook_size"],
                  opq_iterations=state.get("opq_iterations", 0))
        obj.rotation = np.asarray(state["rotation"], dtype=np.float32)
        obj.rqkmeans = RqKmeans(n_layers=obj.n_layers, codebook_size=obj.codebook_size)
        obj.rqkmeans.centroids = [np.asarray(c, dtype=np.float32) for c in state["centroids"]]
        return obj
