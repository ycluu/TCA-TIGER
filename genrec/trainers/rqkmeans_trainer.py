"""
RQ-KMeans Fitting Script.

Fits Residual Quantized K-Means on item embeddings from AmazonItemDataset.
Much faster than training RQ-VAE (~10-20 minutes vs hours).

Usage:
    python genrec/trainers/rqkmeans_trainer.py config/onerec/amazon/rqkmeans.gin --split beauty
"""
import os
import gin
import numpy as np

from genrec.data.amazon import AmazonItemDataset
from genrec.models.rqkmeans import RqKmeans
from genrec.models.rqopq import RqOpq
from genrec.models.semantic_id import save_sid_artifact, sid_metrics
from genrec.modules.utils import parse_config, get_run_split
from genrec.trainers.trainer_utils import set_seed, save_run_results


@gin.configurable
def fit_rqkmeans(
    dataset=AmazonItemDataset,
    dataset_folder: str = "dataset/amazon",
    encoder_model_name: str = "./models_hub/Qwen3-Embedding-8B",
    n_layers: int = 3,
    codebook_size: int = 8192,
    save_dir_root: str = "out/onerec/amazon/{split}/rqkmeans",
    random_state: int = 42,
    max_iter: int = 100,
    batch_size: int = 4096,
    quantizer_type: str = "rqkmeans",
    opq_iterations: int = 3,
    opq_subspaces: int = 4,
    seed: int = 42,
):
    """
    Fit RQ-KMeans on item embeddings.

    1. Load AmazonItemDataset (generates embeddings if not cached)
    2. Fit RqKmeans on all embeddings
    3. Report collision rate
    4. Save centroids
    """
    _run_config = dict(locals())
    set_seed(seed)
    print(f"=== Semantic-ID Quantizer Fitting ({quantizer_type}) ===")
    print(f"  encoder: {encoder_model_name}")
    print(f"  codebook_size: {codebook_size}, n_layers: {n_layers}")
    print(f"  save_dir: {save_dir_root}")

    # Load embeddings
    print("\n--- Loading embeddings ---")
    item_dataset = dataset(
        root=dataset_folder,
        train_test_split="all",
        encoder_model_name=encoder_model_name,
    )
    embeddings = item_dataset.embeddings.astype(np.float32)
    print(f"Loaded {len(embeddings)} embeddings, dim={embeddings.shape[1]}")

    # Fit RQ-KMeans
    print("\n--- Fitting RQ-KMeans ---")
    quantizer_cls = {
        "rqkmeans": RqKmeans,
        "rqopq": RqOpq,
    }.get(quantizer_type)
    if quantizer_cls is None:
        raise ValueError("quantizer_type must be 'rqkmeans' or 'rqopq'")
    kwargs = dict(
        n_layers=n_layers, codebook_size=codebook_size, random_state=random_state,
        max_iter=max_iter, batch_size=batch_size,
    )
    if quantizer_type == "rqopq":
        kwargs["opq_iterations"] = opq_iterations
        kwargs["opq_subspaces"] = opq_subspaces
    rqkmeans = quantizer_cls(**kwargs).fit(embeddings)

    # Report collision rate
    print("\n--- Collision Statistics ---")
    assignments = rqkmeans.assign(embeddings).sem_ids
    stats = sid_metrics(assignments, embeddings, rqkmeans.reconstruct(embeddings))
    print(f"  Total items: {stats['total_items']}")
    print(f"  Unique SIDs: {stats['unique_sids']}")
    print(f"  Collision rate: {stats['collision_rate']:.4%}")
    print(f"  Max collision group: {stats['max_collision']}")

    # Save
    os.makedirs(save_dir_root, exist_ok=True)
    save_path = os.path.join(save_dir_root, f"{quantizer_type}.pt")
    rqkmeans.save(save_path)
    artifact_path = os.path.join(save_dir_root, "semantic_ids.pt")
    save_sid_artifact(
        artifact_path, assignments, quantizer=quantizer_type,
        codebook_size=codebook_size,
        metadata={"split": get_run_split(), "encoder_model_name": encoder_model_name,
                  "quantizer_checkpoint": save_path, "metrics": stats,
                  "target_n_layers": getattr(rqkmeans, "target_n_layers", rqkmeans.n_layers)},
    )
    print(f"Saved SID artifact to {artifact_path}")
    print(f"\n=== Done. Saved to {save_path} ===")

    save_run_results(
        save_dir=save_dir_root,
        model=quantizer_type,
        split=get_run_split(),
        seed=seed,
        metrics={
            "collision_rate": float(stats['collision_rate']),
            "unique_sids": int(stats['unique_sids']),
            "total_items": int(stats['total_items']),
            "max_collision": int(stats['max_collision']),
            "reconstruction_mse": float(stats["reconstruction_mse"]),
            "reconstruction_l2": float(stats["reconstruction_l2"]),
            "layer_usage": stats["layer_usage"],
        },
        config=_run_config,
    )


if __name__ == "__main__":
    parse_config()
    fit_rqkmeans()
