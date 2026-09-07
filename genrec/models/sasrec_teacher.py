"""Minimal, inference-only SASRec++ teacher used for offline TCA cache generation."""

from types import SimpleNamespace

import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units: int, dropout_rate: float):
        super().__init__()
        self.linear1 = torch.nn.Linear(hidden_units, hidden_units)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_units, hidden_units)
        self.dropout2 = torch.nn.Dropout(dropout_rate)

    def forward(self, inputs):
        return self.dropout2(self.linear2(self.relu(self.dropout1(self.linear1(inputs)))))


class SASRecTeacher(torch.nn.Module):
    """Architecture-compatible copy of the trained local Beauty SASRec++ model."""

    def __init__(self, num_items: int, config: dict, device: torch.device):
        super().__init__()
        args = SimpleNamespace(**config)
        self.num_items = num_items
        self.device_ref = device
        self.norm_first = args.norm_first
        self.use_time_gap = args.use_time_gap
        self.item_emb = torch.nn.Embedding(num_items + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        if self.use_time_gap:
            self.time_gap_emb = torch.nn.Embedding(
                args.time_gap_buckets, args.hidden_units, padding_idx=0
            )
        self.emb_dropout = torch.nn.Dropout(args.dropout_rate)
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        for _ in range(args.num_blocks):
            self.attention_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.attention_layers.append(
                torch.nn.MultiheadAttention(
                    args.hidden_units,
                    args.num_heads,
                    dropout=args.dropout_rate,
                    batch_first=True,
                )
            )
            self.forward_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.forward_layers.append(
                PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            )
        causal_mask = torch.triu(
            torch.ones(args.maxlen, args.maxlen, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def log2feats(self, log_seqs: torch.Tensor) -> torch.Tensor:
        log_seqs = log_seqs.to(device=self.device_ref, dtype=torch.long)
        batch_size, seq_len = log_seqs.shape
        padding_mask = log_seqs.eq(0)
        seqs = self.item_emb(log_seqs) * (self.item_emb.embedding_dim**0.5)
        positions = torch.arange(1, seq_len + 1, device=self.device_ref).unsqueeze(0)
        positions = positions.expand(batch_size, -1).masked_fill(padding_mask, 0)
        seqs = self.emb_dropout(seqs + self.pos_emb(positions))
        seqs = seqs.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        attention_mask = self.causal_mask[:seq_len, :seq_len]
        for i, attention_layer in enumerate(self.attention_layers):
            if self.norm_first:
                attention_inputs = self.attention_layernorms[i](seqs)
                mha_outputs, _ = attention_layer(
                    attention_inputs,
                    attention_inputs,
                    attention_inputs,
                    attn_mask=attention_mask,
                    key_padding_mask=padding_mask,
                    need_weights=False,
                )
                mha_outputs = mha_outputs.masked_fill(padding_mask.unsqueeze(-1), 0.0)
                seqs = seqs + mha_outputs
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = attention_layer(
                    seqs,
                    seqs,
                    seqs,
                    attn_mask=attention_mask,
                    key_padding_mask=padding_mask,
                    need_weights=False,
                )
                mha_outputs = mha_outputs.masked_fill(padding_mask.unsqueeze(-1), 0.0)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))
            seqs = seqs.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return self.last_layernorm(seqs).masked_fill(padding_mask.unsqueeze(-1), 0.0)

    def user_states(self, histories: torch.Tensor) -> torch.Tensor:
        return self.log2feats(histories)[:, -1, :]


EXPECTED_BEAUTY_CONFIG = {
    "dataset": "beauty",
    "maxlen": 50,
    "hidden_units": 50,
    "num_blocks": 2,
    "num_heads": 1,
    "norm_first": True,
    "use_time_gap": False,
    "loss_type": "ce",
    "seed": 42,
}


def load_beauty_teacher(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint or "args" not in checkpoint:
        raise ValueError("Expected SASRec++ checkpoint with model_state_dict and args")
    config = dict(checkpoint["args"])
    mismatches = {
        key: (config.get(key), expected)
        for key, expected in EXPECTED_BEAUTY_CONFIG.items()
        if config.get(key) != expected
    }
    state = checkpoint["model_state_dict"]
    expected_shape = (12102, 50)
    if tuple(state.get("item_emb.weight", torch.empty(0)).shape) != expected_shape:
        mismatches["item_emb.weight.shape"] = (
            tuple(state.get("item_emb.weight", torch.empty(0)).shape),
            expected_shape,
        )
    if mismatches:
        raise ValueError(f"SASRec++ checkpoint/config mismatch: {mismatches}")
    model = SASRecTeacher(12101, config, torch.device(device))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, config
