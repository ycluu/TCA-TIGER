"""
TIGER trainer (HF T5-based).

Pipeline:
- Flat offset-encoded codebook tokens (token = code + codebook_idx * codebook_size + 1).
- HF T5ForConditionalGeneration via genrec.models.tiger.Tiger.
- Plain Adam (default) or cosine schedule with linear warmup.
- Early stopping on validation Recall@10.
"""

import os
from pathlib import Path
import gin
import torch
import wandb

from genrec.models.tiger import Tiger
from genrec.modules.utils import parse_config, setup_logger, get_run_split
from genrec.data.schemas import SeqData
from genrec.trainers.trainer_utils import setup_wandb, set_seed, save_run_results
from genrec.trainers.tca_loss import TCATeacherCache, select_training_loss
from genrec.trainers.validation_selection import ValidationSelection
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict


def t5_collate(
    batch: List[SeqData],
    codebook_size: int = 256,
    input_sem_id_dim: int = 3,
    target_sem_id_dim: int = 3,
    max_history_tokens: int = 60,  # max_seq_len * sem_id_dim (e.g. 20*3=60)
) -> Dict[str, torch.Tensor]:
    """
    Convert SeqData to flat T5 format with offset encoding.

    Offset encoding: token = code + codebook_idx * codebook_size + 1
    (0 is reserved for padding)
    """
    B = len(batch)

    input_ids = torch.zeros(B, max_history_tokens, dtype=torch.long)
    attention_mask = torch.zeros(B, max_history_tokens, dtype=torch.long)
    label_list = []

    for i, sample in enumerate(batch):
        raw_ids = sample.item_ids  # flat list: [c0, c1, c2, c0, c1, c2, ...]
        offset_ids = []
        for j, code in enumerate(raw_ids):
            cb_idx = j % input_sem_id_dim
            offset_ids.append(code + cb_idx * codebook_size + 1)

        L = len(offset_ids)
        if L > max_history_tokens:
            offset_ids = offset_ids[-max_history_tokens:]
            L = max_history_tokens

        start = max_history_tokens - L
        input_ids[i, start:] = torch.tensor(offset_ids, dtype=torch.long)
        attention_mask[i, start:] = 1

        target_ids = sample.target_ids
        offset_target = []
        for j, code in enumerate(target_ids):
            cb_idx = j % target_sem_id_dim
            offset_target.append(code + cb_idx * codebook_size + 1)
        label_list.append(offset_target)

    labels = torch.tensor(label_list, dtype=torch.long)
    raw_targets = torch.tensor([s.target_ids for s in batch], dtype=torch.long)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'raw_targets': raw_targets,
        'sample_indices': torch.tensor([s.sample_index for s in batch], dtype=torch.long),
        'sample_keys': [s.sample_key for s in batch],
    }


class CollateFn:
    """
    Module-level picklable collate wrapper.
    Windows spawn multiprocessing requires collate_fn to be picklable,
    which means no lambdas or functools.partial of local functions.
    A top-level callable class is always picklable.
    """
    def __init__(self, codebook_size: int, input_sem_id_dim: int, target_sem_id_dim: int, max_history_tokens: int):
        self.codebook_size = codebook_size
        self.input_sem_id_dim = input_sem_id_dim
        self.target_sem_id_dim = target_sem_id_dim
        self.max_history_tokens = max_history_tokens

    def __call__(self, batch):
        return t5_collate(batch, self.codebook_size, self.input_sem_id_dim,
                          self.target_sem_id_dim, self.max_history_tokens)


def calculate_pos_index(preds, labels, maxk=20):
    """Calculate position index of ground truth in predictions (vectorized)."""
    labels_expanded = labels.unsqueeze(1).expand_as(preds)  # (B, maxk, seq_len)
    match = (preds == labels_expanded).all(dim=-1)  # (B, maxk)
    first_match_mask = match.int().cumsum(dim=1) == 1
    return match & first_match_mask


def recall_at_k(pos_index, k):
    return pos_index[:, :k].sum(dim=1).float().mean().item()


def ndcg_at_k(pos_index, k):
    ranks = torch.arange(1, pos_index.shape[-1] + 1).float()
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0))
    return dcg[:, :k].sum(dim=1).mean().item()


@gin.configurable
def train(
    epochs=200,
    batch_size=256,
    infer_batch_size=96,
    learning_rate=1e-4,
    weight_decay=0.0,
    dataset_folder="dataset/amazon",
    save_dir_root="out/tiger/",
    dataset=None,
    wandb_logging=True,
    wandb_project="tiger_training",
    wandb_log_interval=10,
    # Model config
    num_layers=4,
    num_decoder_layers=4,
    d_model=128,
    d_ff=1024,
    num_heads=6,
    d_kv=64,
    dropout_rate=0.1,
    feed_forward_proj='relu',
    # Data config
    codebook_size=256,
    sem_id_dim=3,
    max_seq_len=20,
    pretrained_rqvae_path="./out/tiger/amazon/{split}/rqvae/checkpoint_epoch_4999.pt",
    semantic_id_path=None,
    # Scheduler config
    lr_schedule="none",  # "none" or "cosine"
    num_warmup_steps=0,
    # Eval config
    beam_size=30,
    early_stop_patience=10,
    eval_test_every_epoch=2,
    eval_valid_every_n_epochs=1,
    seed=42,
    # Explicit objective selection. The default is the untouched HF T5 CE path.
    training_objective="baseline",
    tca_alpha=0.1,
    tca_temperature=1.0,
    tca_objective="official_log_mixture",
    tca_cache_path=None,
    selection_metric="recall10",
    preference_cache_path=None,
    phase2_initialization_path=None,
    dpo_beta=0.1,
    lambda_pref=0.1,
    preference_sample_ratio=1.0,
    preference_microbatch_size=32,
):
    _run_config = dict(locals())
    phase2 = training_objective == "tca_full_vocab_dpo"
    if phase2:
        if not preference_cache_path or not phase2_initialization_path:
            raise ValueError('Phase2 requires explicit preference cache and epoch111 initialization')
        if (tca_alpha,tca_temperature,dpo_beta,lambda_pref,preference_sample_ratio) != (.1,1.,.1,.1,1.):
            raise ValueError('Phase2 first controlled run requires fixed alpha/tau/beta/lambda/ratio')
        if selection_metric != 'ndcg10' or early_stop_patience != 10:
            raise ValueError('Phase2 must select by Validation NDCG@10 with patience10')
        target_dir = Path(save_dir_root).resolve()
        if target_dir == Path(phase2_initialization_path).resolve().parent or (target_dir.exists() and any(target_dir.iterdir())):
            raise ValueError('Phase2 requires a clean new output directory; refusing overwrite')
    selection = ValidationSelection(selection_metric)
    set_seed(seed)
    logger = setup_logger(save_dir_root, name="tiger")
    logger.info("Selection metric: Validation %s", selection.metric)
    logger.info("Early stopping metric: Validation %s", selection.metric)
    logger.info("Test during training: enabled, diagnostic only (period=%s)", eval_test_every_epoch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_dataset = dataset(
        root=dataset_folder,
        train_test_split="train",
        max_seq_len=max_seq_len,
        pretrained_rqvae_path=pretrained_rqvae_path,
        semantic_id_path=semantic_id_path,
    )
    valid_dataset = dataset(
        root=dataset_folder,
        train_test_split="valid",
        max_seq_len=max_seq_len,
        pretrained_rqvae_path=pretrained_rqvae_path,
        semantic_id_path=semantic_id_path,
    )
    test_dataset = dataset(
        root=dataset_folder,
        train_test_split="test",
        max_seq_len=max_seq_len,
        pretrained_rqvae_path=pretrained_rqvae_path,
        semantic_id_path=semantic_id_path,
    )

    input_sem_id_dim = train_dataset.sid_n_layers
    target_sem_id_dim = train_dataset.sid_target_n_layers
    if target_sem_id_dim != sem_id_dim:
        raise ValueError(f"SID artifact generates {target_sem_id_dim} target layers, but train.sem_id_dim={sem_id_dim}")
    if train_dataset.sid_codebook_size != codebook_size:
        raise ValueError(f"SID artifact codebook_size={train_dataset.sid_codebook_size}, "
                         f"but train.codebook_size={codebook_size}")

    if training_objective not in {"baseline", "position_ce", "tca", "tca_full_vocab", "tca_full_vocab_dpo"}:
        raise ValueError(f"Unknown training_objective: {training_objective}")
    tca_cache = None
    if training_objective in {"tca", "tca_full_vocab", "tca_full_vocab_dpo"}:
        expected_tca_objective = (
            "official_log_mixture" if training_objective == "tca" else "conventional_soft_ce"
        )
        if tca_objective != expected_tca_objective:
            raise ValueError(f"Unsupported TCA objective: {tca_objective}")
        if not tca_cache_path:
            raise ValueError("training_objective=tca requires tca_cache_path")
        tca_cache = TCATeacherCache.load_and_validate(
            tca_cache_path,
            dataset="beauty",
            num_samples=len(train_dataset),
            num_items=12101,
            sid_layers=sem_id_dim,
            codebook_size=codebook_size,
            sid_artifact_path=semantic_id_path,
        )
        sample_mismatches = tca_cache.validate_dataset(train_dataset)
        logger.info(
            "training_objective=%s loss_space=%s temperature=%s "
            "alpha=%s collaborative_target=true objective=%s cache=%s "
            "cache_hash=%s sid_hash=%s sample_mismatches=%s",
            training_objective,
            "256_per_position" if training_objective == "tca" else "769_full_vocab",
            tca_temperature, tca_alpha, tca_objective, Path(tca_cache_path).resolve(),
            tca_cache.cache_sha256, tca_cache.sid_sha256, sample_mismatches,
        )
    elif training_objective == "position_ce":
        logger.info(
            "training_objective=position_ce loss_space=256_per_position "
            "temperature=1 collaborative_target=false"
        )
    else:
        logger.info(
            "training_objective=baseline loss_space=769 temperature=1 "
            "collaborative_target=false"
        )

    max_history_tokens = max_seq_len * input_sem_id_dim

    vocab_size = codebook_size * input_sem_id_dim + 1
    model_config = {
        'num_layers': num_layers, 'num_decoder_layers': num_decoder_layers,
        'd_model': d_model, 'd_ff': d_ff, 'num_heads': num_heads, 'd_kv': d_kv,
        'dropout_rate': dropout_rate, 'vocab_size': vocab_size, 'pad_token_id': 0,
        'eos_token_id': 0, 'feed_forward_proj': feed_forward_proj,
        'sem_id_dim': target_sem_id_dim,
    }
    if wandb_logging:
        setup_wandb(project=wandb_project, config={**model_config, 'lr': learning_rate,
                    'batch_size': batch_size, 'max_seq_len': max_seq_len,
                    'input_sem_id_dim': input_sem_id_dim, 'beam_size': beam_size})

    # CollateFn is a module-level picklable class, safe for num_workers > 0 on Windows.
    collate_fn = CollateFn(codebook_size, input_sem_id_dim, target_sem_id_dim, max_history_tokens)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=infer_batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=infer_batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
    )

    logger.info(f"Train: {len(train_loader)} batches, Valid: {len(valid_loader)}, Test: {len(test_loader)}")

    # Model
    model = Tiger(model_config).to(device)
    preference_cache = None
    if phase2:
        from genrec.trainers.preference_loss import PreferenceCache, initialize_policy, golden_test, preference_loss
        preference_cache = PreferenceCache.load(preference_cache_path,train_dataset,semantic_id_path,phase2_initialization_path)
        initialize_policy(model,phase2_initialization_path)
        logger.info('Phase2 initialization: Phase1 epoch111; reference: offline cached epoch111 logP')
        probe_ids = torch.where(preference_cache.a['has_pair'])[0][:16].tolist()
        probe = collate_fn([train_dataset[i] for i in probe_ids])
        probe_pairs = preference_cache.batch(probe['sample_indices'],probe['sample_keys'],device)
        golden_loss, golden_diag = golden_test(model,probe,probe_pairs,device)
        logger.info('DPO initialization golden PASS: loss=%s diagnostics=%s',golden_loss,golden_diag)
    total_params, emb_params = model.num_parameters
    logger.info(f"Device: {device}, Params: {total_params:,} (emb: {emb_params:,})")

    # Optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # LR scheduler
    scheduler = None
    if lr_schedule == "cosine" and epochs > 0:
        total_steps = len(train_loader) * epochs
        if num_warmup_steps > 0:
            warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=num_warmup_steps)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - num_warmup_steps)
            scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[num_warmup_steps])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        logger.info(f"Using cosine schedule, warmup={num_warmup_steps}, total_steps={total_steps}")
    else:
        logger.info("Using plain Adam, no scheduler")

    # Early stopping state

    def evaluate(loader, desc="Eval"):
        model.eval()
        all_recalls = {5: [], 10: []}
        all_ndcgs = {5: [], 10: []}

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            for batch in tqdm(loader, desc=desc):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']  # offset-encoded targets
                B_cur = input_ids.size(0)

                preds = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=beam_size,
                )
                # Remove decoder_start_token (first token)
                preds = preds[:, 1:]
                # Reshape: (B * beam_size, sem_id_dim) -> (B, beam_size, sem_id_dim)
                preds = preds.reshape(B_cur, beam_size, -1).cpu()

                pos_index = calculate_pos_index(preds, labels, maxk=beam_size)
                for k in [5, 10]:
                    all_recalls[k].append(recall_at_k(pos_index, k))
                    all_ndcgs[k].append(ndcg_at_k(pos_index, k))

        metrics = {}
        for k in [5, 10]:
            metrics[f'Recall@{k}'] = sum(all_recalls[k]) / len(all_recalls[k])
            metrics[f'NDCG@{k}'] = sum(all_ndcgs[k]) / len(all_ndcgs[k])
        model.train()
        return metrics

    # Training loop
    scaler = torch.amp.GradScaler('cuda')
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        preference_logs = {}
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.float16):
                baseline_loss, logits = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
            teacher_probs = None
            if training_objective in {"tca", "tca_full_vocab", "tca_full_vocab_dpo"}:
                teacher_probs = tca_cache.batch(
                    batch['sample_indices'], batch['sample_keys'], device=device
                )
            loss = select_training_loss(
                baseline_loss,
                objective="tca_full_vocab" if phase2 else training_objective,
                logits=logits,
                labels=labels,
                teacher_probs=teacher_probs,
                alpha=tca_alpha,
                temperature=tca_temperature,
            )
            if phase2:
                pair = preference_cache.batch(batch['sample_indices'],batch['sample_keys'],device,preference_sample_ratio)
                with torch.amp.autocast('cuda',dtype=torch.float16):
                    dpo_loss, diagnostics = preference_loss(model,input_ids,attention_mask,pair,
                        beta=dpo_beta,microbatch_size=preference_microbatch_size)
                base_loss = loss
                loss = base_loss + lambda_pref*dpo_loss
                fields = dict(train_total_loss=float(loss.detach()),train_tca_loss=float(base_loss.detach()),
                              train_dpo_loss=float(dpo_loss.detach()),**diagnostics)
                for key,value in fields.items():
                    preference_logs[key] = preference_logs.get(key,0.)+value
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch} - loss: {avg_loss:.4f}")

        log_dict = {"epoch": epoch, "train/loss": avg_loss}
        if phase2:
            phase2_metrics = {k:v/len(train_loader) for k,v in preference_logs.items()}
            logger.info('Phase2 losses/diagnostics: %s',phase2_metrics)
            log_dict.update(phase2_metrics)

        # Valid evaluation (skip non-eval epochs)
        if (epoch + 1) % eval_valid_every_n_epochs == 0 or epoch == 0:
            valid_metrics = evaluate(valid_loader, desc=f"Valid (Epoch {epoch})")
            logger.info(f"Epoch {epoch} - Valid: {valid_metrics}")

            for k, v in valid_metrics.items():
                log_dict[f"eval/valid_{k}"] = v

            # Only the configured validation metric decides saving and patience.
            if selection.update(epoch, valid_metrics):

                # Save best model
                os.makedirs(save_dir_root, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir_root, "best_model.pt"))
                logger.info(f"New best Valid {selection.metric}={selection.best_value:.8f} at epoch {epoch}")

                # Test evaluation on improvement
                test_metrics = evaluate(test_loader, desc=f"Test (Epoch {epoch})")
                selection.record_selected_test(epoch, test_metrics)
                logger.info(f"Epoch {epoch} - Test: {test_metrics}")
                for k, v in test_metrics.items():
                    log_dict[f"eval/test_{k}"] = v
            else:
                logger.info(f"No improvement in Valid {selection.metric}. Counter: {selection.counter}/{early_stop_patience}")

                # Periodic test metrics are diagnostic only and are never used
                # for model selection or early stopping.
                if (epoch + 1) % eval_test_every_epoch == 0:
                    test_metrics = evaluate(test_loader, desc=f"Test (Epoch {epoch})")
                    logger.info(f"Epoch {epoch} - Test: {test_metrics}")
                    for k, v in test_metrics.items():
                        log_dict[f"eval/test_{k}"] = v

        if wandb_logging:
            wandb.log(log_dict)

        if selection.counter >= early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch}. Best epoch: {selection.best_epoch}")
            break

    logger.info(f"Training done. Best Valid {selection.metric}={selection.best_value:.8f} at epoch {selection.best_epoch}")
    save_run_results(
        save_dir=save_dir_root, model="tiger", split=get_run_split(), seed=seed,
        metrics=selection.results(),
        config=_run_config,
    )

    if wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    parse_config()
    train()
