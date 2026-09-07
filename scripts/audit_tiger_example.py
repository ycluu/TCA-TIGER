"""Print one deterministic Beauty sample and scored original-TIGER beams."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from genrec.data.amazon import AmazonSeqDataset
from genrec.models.tiger import Tiger
from genrec.trainers.tiger_trainer import t5_collate


CONFIG = {
    "num_layers": 4, "num_decoder_layers": 4, "d_model": 128, "d_ff": 1024,
    "num_heads": 6, "d_kv": 64, "dropout_rate": 0.1, "vocab_size": 769,
    "pad_token_id": 0, "eos_token_id": 0, "feed_forward_proj": "relu", "sem_id_dim": 3,
}
OFFSETS = (1, 257, 513)


def scored_beams(model, input_ids, attention_mask, num_beams=10, max_length=4):
    core = model.model
    encoder_outputs = core.encoder(input_ids=input_ids, attention_mask=attention_mask)
    hidden = encoder_outputs[0].expand(num_beams, -1, -1).contiguous()
    mask = attention_mask.expand(num_beams, -1).contiguous()
    decoder = torch.zeros(num_beams, 1, dtype=torch.long, device=input_ids.device)
    scores = torch.zeros(1, num_beams, device=input_ids.device)
    scores[:, 1:] = -1e9
    cache = None
    for _ in range(max_length - 1):
        output = core(
            encoder_outputs=(hidden,), attention_mask=mask,
            decoder_input_ids=decoder if cache is None else decoder[:, -1:],
            use_cache=True, past_key_values=cache,
        )
        cache = output.past_key_values
        candidates = F.log_softmax(output.logits[:, -1].float(), -1).view(1, num_beams, -1)
        top_scores, top_indices = (candidates + scores.unsqueeze(-1)).view(1, -1).topk(num_beams)
        beam_indices = (top_indices // 769).reshape(-1)
        tokens = (top_indices % 769).reshape(-1, 1)
        decoder = torch.cat([decoder[beam_indices], tokens], -1)
        cache = core._reorder_cache(cache, beam_indices)
        scores = top_scores
    return decoder, scores.reshape(-1)


def main():
    sid_path = Path("out/tiger/amazon/beauty/rqkmeans/semantic_ids.pt")
    checkpoint = Path("out/tiger/amazon/beauty/rqkmeans/tiger/best_model.pt")
    sid_artifact = torch.load(sid_path, map_location="cpu", weights_only=False)
    known = {tuple(row) for row in sid_artifact["sem_ids"].tolist()}
    dataset = AmazonSeqDataset(
        root="dataset/amazon", split="beauty", train_test_split="valid", max_seq_len=50,
        add_disambiguation=False, semantic_id_path=str(sid_path), target_n_layers=3,
    )
    data = dataset.samples[0]
    sample = dataset[0]
    batch = t5_collate([sample], 256, 3, 3, 150)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Tiger(CONFIG).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
        sequences, scores = scored_beams(
            model, batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )
    labels = batch["labels"][0].tolist()
    beams = []
    for rank, (sequence, score) in enumerate(zip(sequences.cpu().tolist(), scores.cpu().tolist()), 1):
        generated = sequence[1:]
        valid = all(OFFSETS[i] <= generated[i] < OFFSETS[i] + 256 for i in range(3))
        raw = [generated[i] - OFFSETS[i] for i in range(3)] if valid else None
        beams.append({
            "rank": rank, "sequence_score": score, "generated_vocabulary_ids": generated,
            "level_interpretation": [
                next((f"c{i}" for i, offset in enumerate(OFFSETS) if offset <= token < offset + 256),
                     "special-0" if token == 0 else "invalid") for token in generated
            ],
            "raw_sid": raw, "known_sid": valid and tuple(raw) in known,
            "matches_target_sid": generated == labels,
        })
    sem = dataset.sem_ids_list
    offset = lambda sid: [sid[0] + 1, sid[1] + 257, sid[2] + 513]
    result = {
        "sample_index": 0,
        "canonical_history_items": [item + 1 for item in data["history"]],
        "internal_history_items": data["history"],
        "history_raw_sids": [sem[item] for item in data["history"]],
        "history_offset_tokens": [offset(sem[item]) for item in data["history"]],
        "target_canonical_item": data["target"] + 1,
        "target_internal_item": data["target"],
        "target_raw_sid": sem[data["target"]],
        "target_decoder_labels": labels,
        "decoder_input_ids_after_shift_right": [0] + labels[:-1],
        "beams": beams,
    }
    output = Path("out/tiger/amazon/beauty/rqkmeans/tiger/original_concrete_example.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
