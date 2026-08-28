"""RQ-OPQ semantic-ID quantizer.

RQ-KMeans first produces coarse, hierarchical semantic codes.  OPQ then
encodes the *remaining* residual with independent subspace codebooks, so the
tail information is retained instead of discarded.
"""
from typing import NamedTuple

import numpy as np
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans

from genrec.models.rqkmeans import RqKmeans


class RqOpqOutput(NamedTuple):
    sem_ids: np.ndarray  # [N, n_rq_layers + opq_subspaces]


class RqOpq:
    """Residual K-Means followed by OPQ of the final residual.

    The complete SID is a generation target for TIGER: the first
    ``n_rq_layers`` tokens encode coarse residual-quantization semantics and
    the following ``opq_subspaces`` tokens encode the remaining residual.
    """

    def __init__(self, n_layers: int = 3, codebook_size: int = 256,
                 opq_subspaces: int = 4, opq_codebook_size: int | None = None,
                 opq_iterations: int = 3, random_state: int = 42,
                 max_iter: int = 100, batch_size: int = 4096,
                 backend: str = "auto", n_init: int = 10, dead_reinit: bool = True):
        self.n_rq_layers = n_layers
        self.opq_subspaces = opq_subspaces
        self.codebook_size = codebook_size
        self.opq_codebook_size = opq_codebook_size or codebook_size
        if self.opq_codebook_size != self.codebook_size:
            raise ValueError("RQ-OPQ currently requires equal RQ and OPQ codebook sizes for Tiger offsets")
        self.n_layers = n_layers + opq_subspaces
        # Kept in artifacts so downstream consumers know that every RQ and OPQ
        # token belongs in the decoder target, not merely in encoder context.
        self.target_n_layers = self.n_layers
        self.opq_iterations = opq_iterations
        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.backend = backend
        self.n_init = n_init
        self.dead_reinit = dead_reinit
        self.rqkmeans = RqKmeans(n_layers=n_layers, codebook_size=codebook_size,
                                 random_state=random_state, max_iter=max_iter,
                                 batch_size=batch_size, backend=backend,
                                 n_init=n_init, dead_reinit=dead_reinit)
        self.rotation: np.ndarray | None = None
        self.opq_centroids: list[np.ndarray] = []

    def _residual(self, embeddings: np.ndarray) -> np.ndarray:
        return np.asarray(embeddings, dtype=np.float32) - self.rqkmeans.reconstruct(embeddings)

    def _fit_subspace_kmeans(self, rotated: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        dimension = rotated.shape[1]
        if dimension % self.opq_subspaces:
            raise ValueError(f"embedding dim {dimension} must be divisible by opq_subspaces={self.opq_subspaces}")
        sub_dim = dimension // self.opq_subspaces
        n_items = len(rotated)
        backend = self.backend if self.backend != "auto" else (
            "full" if n_items * self.opq_codebook_size < 5e7 else "minibatch"
        )
        centroids, reconstruction = [], np.zeros_like(rotated)
        for subspace in range(self.opq_subspaces):
            start, end = subspace * sub_dim, (subspace + 1) * sub_dim
            x = rotated[:, start:end]
            if backend == "full":
                model = KMeans(n_clusters=self.opq_codebook_size, init="k-means++",
                               n_init=self.n_init, max_iter=self.max_iter,
                               random_state=self.random_state)
            else:
                model = MiniBatchKMeans(n_clusters=self.opq_codebook_size,
                                        n_init=self.n_init, max_iter=self.max_iter,
                                        batch_size=min(self.batch_size, n_items),
                                        random_state=self.random_state)
            codes = model.fit_predict(x)
            center = model.cluster_centers_.astype(np.float32, copy=True)
            centroids.append(center)
            reconstruction[:, start:end] = center[codes]
        return centroids, reconstruction

    def fit(self, embeddings: np.ndarray) -> "RqOpq":
        x = np.asarray(embeddings, dtype=np.float32)
        print("Stage 1/2: fitting RQ-KMeans")
        self.rqkmeans.fit(x)
        residual = self._residual(x)
        print(f"Stage 2/2: fitting OPQ on final RQ residual (mean L2={np.linalg.norm(residual, axis=1).mean():.6f})")
        self.rotation = np.eye(x.shape[1], dtype=np.float32)
        for iteration in range(self.opq_iterations):
            rotated = residual @ self.rotation
            _, reconstruction = self._fit_subspace_kmeans(rotated)
            u, _, vt = np.linalg.svd(residual.T @ reconstruction, full_matrices=False)
            self.rotation = (u @ vt).astype(np.float32)
            print(f"  OPQ iteration {iteration + 1}/{self.opq_iterations}: "
                  f"MSE={np.mean((residual @ self.rotation - reconstruction) ** 2):.6f}")
        self.opq_centroids, _ = self._fit_subspace_kmeans(residual @ self.rotation)
        return self

    def _assign_opq(self, residual: np.ndarray) -> np.ndarray:
        if self.rotation is None or not self.opq_centroids:
            raise RuntimeError("Must call fit() or load() before assign()")
        rotated = residual @ self.rotation
        sub_dim = rotated.shape[1] // self.opq_subspaces
        codes = np.zeros((len(rotated), self.opq_subspaces), dtype=np.int64)
        for subspace, centroids in enumerate(self.opq_centroids):
            start, end = subspace * sub_dim, (subspace + 1) * sub_dim
            x = rotated[:, start:end]
            distances = (x ** 2).sum(axis=1, keepdims=True) + (centroids ** 2).sum(axis=1)[None, :] - 2 * x @ centroids.T
            codes[:, subspace] = distances.argmin(axis=1)
        return codes

    def assign(self, embeddings: np.ndarray) -> RqOpqOutput:
        x = np.asarray(embeddings, dtype=np.float32)
        rq_codes = self.rqkmeans.assign(x).sem_ids
        opq_codes = self._assign_opq(self._residual(x))
        return RqOpqOutput(sem_ids=np.concatenate((rq_codes, opq_codes), axis=1))

    def reconstruct(self, embeddings: np.ndarray) -> np.ndarray:
        x = np.asarray(embeddings, dtype=np.float32)
        rq_reconstruction = self.rqkmeans.reconstruct(x)
        opq_codes = self._assign_opq(x - rq_reconstruction)
        dimension = x.shape[1]
        sub_dim = dimension // self.opq_subspaces
        rotated = np.zeros_like(x)
        for subspace, centroids in enumerate(self.opq_centroids):
            start, end = subspace * sub_dim, (subspace + 1) * sub_dim
            rotated[:, start:end] = centroids[opq_codes[:, subspace]]
        return rq_reconstruction + rotated @ self.rotation.T

    def save(self, path: str) -> None:
        if self.rotation is None or not self.opq_centroids:
            raise RuntimeError("Cannot save an unfitted RQ-OPQ quantizer")
        torch.save({
            "type": "rqopq", "n_rq_layers": self.n_rq_layers,
            "codebook_size": self.codebook_size, "opq_subspaces": self.opq_subspaces,
            "opq_codebook_size": self.opq_codebook_size, "opq_iterations": self.opq_iterations,
            "rotation": self.rotation, "rq_centroids": self.rqkmeans.centroids,
            "opq_centroids": self.opq_centroids,
        }, path)

    @classmethod
    def load(cls, path: str) -> "RqOpq":
        state = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(n_layers=state["n_rq_layers"], codebook_size=state["codebook_size"],
                  opq_subspaces=state["opq_subspaces"], opq_codebook_size=state["opq_codebook_size"],
                  opq_iterations=state["opq_iterations"])
        obj.rotation = np.asarray(state["rotation"], dtype=np.float32)
        obj.rqkmeans.centroids = [np.asarray(c, dtype=np.float32) for c in state["rq_centroids"]]
        obj.opq_centroids = [np.asarray(c, dtype=np.float32) for c in state["opq_centroids"]]
        return obj
