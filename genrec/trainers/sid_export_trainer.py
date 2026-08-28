"""Export an RQ-VAE checkpoint into the common TIGER SID artifact format."""
import os

import gin
import numpy as np
import torch

from genrec.data.amazon import AmazonItemDataset
from genrec.models.rqvae import RqVae
from genrec.models.semantic_id import save_sid_artifact, sid_metrics
from genrec.modules.normalize import l2norm
from genrec.modules.utils import get_run_split, parse_config
from genrec.trainers.trainer_utils import save_run_results


@gin.configurable
def export_rqvae_sids(
    dataset=AmazonItemDataset,
    dataset_folder: str = "dataset/amazon",
    encoder_model_name: str = "sentence-transformers/sentence-t5-base",
    checkpoint_path: str = "out/tiger/amazon/{split}/rqvae/checkpoint_epoch_4999.pt",
    save_dir_root: str = "out/tiger/amazon/{split}/rqvae",
    input_dim: int = 768,
    embed_dim: int = 32,
    hidden_dims=(512, 256, 128, 64),
    codebook_size: int = 256,
    n_layers: int = 3,
    export_batch_size: int = 4096,
    seed: int = 42,
):
    """Export RQ-VAE assignments after training without retraining it."""
    split = get_run_split()
    checkpoint_path = checkpoint_path.format(split=split)
    save_dir_root = save_dir_root.format(split=split)
    item_dataset = dataset(root=dataset_folder, train_test_split="all", encoder_model_name=encoder_model_name)
    embeddings = np.asarray(item_dataset.embeddings, dtype=np.float32)
    model = RqVae(input_dim=input_dim, embed_dim=embed_dim, hidden_dims=list(hidden_dims),
                  codebook_size=codebook_size, codebook_kmeans_init=False,
                  n_layers=n_layers, n_cat_features=0, commitment_weight=0.25)
    model.load_pretrained(checkpoint_path)
    model.eval()
    sem_ids_batches = []
    reconstruction_batches = []
    with torch.no_grad():
        for start in range(0, len(embeddings), export_batch_size):
            batch = torch.from_numpy(embeddings[start:start + export_batch_size])
            output = model.get_semantic_ids(batch)
            # RqVaeOutput is already [batch, n_layers]. The decoder path and
            # L2 normalization exactly match the n_cat_features=0 training path.
            reconstruction = l2norm(model.decode(output.embeddings.sum(dim=-1)))
            sem_ids_batches.append(output.sem_ids.cpu().numpy())
            reconstruction_batches.append(reconstruction.cpu().numpy())
    sem_ids = np.concatenate(sem_ids_batches, axis=0).astype(np.int64)
    reconstruction = np.concatenate(reconstruction_batches, axis=0)
    stats = sid_metrics(sem_ids, embeddings, reconstruction)
    stats["rqvae_reconstruction_loss"] = float(
        model.reconstruction_loss(torch.from_numpy(reconstruction), torch.from_numpy(embeddings)).mean().item()
    )
    artifact_path = os.path.join(save_dir_root, "semantic_ids.pt")
    save_sid_artifact(artifact_path, sem_ids, quantizer="rqvae", codebook_size=codebook_size,
                      metadata={"split": split, "checkpoint_path": checkpoint_path, "metrics": stats})
    print(f"Saved RQ-VAE SID artifact to {artifact_path}")
    save_run_results(save_dir=save_dir_root, model="rqvae_sid_export", split=split,
                     seed=seed, metrics=stats, config=dict(locals()))


if __name__ == "__main__":
    parse_config()
    export_rqvae_sids()
