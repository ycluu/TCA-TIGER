"""Offline train-only preference preparation; never imported by the trainer."""
import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from genrec.data.amazon import AmazonSeqDataset
from genrec.data.tca_teacher_cache import sample_key
from genrec.models.sasrec_teacher import load_beauty_teacher
from genrec.trainers.tca_loss import sha256_file, compute_tca_full_vocab_loss
from genrec.trainers.tiger_trainer import CollateFn
from scripts.audit_tiger_generation import load_model, OFFSETS
from scripts.generate_tca_teacher_cache import left_pad_history

ROOT = Path('out/tiger/amazon/beauty')
OUT = ROOT / 'preference'
CP = ROOT / 'rqkmeans/tiger_tca_full_vocab_ndcg_select/best_model.pt'
SID = ROOT / 'rqkmeans/semantic_ids.pt'
EXPECTED = '17d6ae7b8914ee43176757669e0a592b229a9ad2c41d9225b9f607e73a950870'
CACHE = OUT / 'train_preference_pairs.pt'


def stats(values):
    x = np.asarray(values, dtype=float)
    return {'count': int(x.size), **({k: float(v) for k, v in zip(
        ['mean', 'median', 'p10', 'p25', 'p75', 'p90'],
        [x.mean(), np.median(x), *np.percentile(x, [10, 25, 75, 90])])} if x.size else {})}


def reverse_map(raw):
    result = defaultdict(list)
    for item, row in enumerate(raw.tolist(), 1):
        result[tuple(c + o for c, o in zip(row, OFFSETS))].append(item)
    return dict(result)


def select_pair(sids, rewards, logp, reverse):
    """No target argument: rewards and explicit reference scores ONLY."""
    eligible = [i for i, sid in enumerate(sids) if len(reverse.get(tuple(sid), [])) == 1]
    pairs = [(i, j) for i in eligible for j in eligible
             if tuple(sids[i]) != tuple(sids[j]) and rewards[i] > rewards[j] and logp[i] < logp[j]]
    if not pairs:
        return None
    return min(pairs, key=lambda ij: (-(float(rewards[ij[0]]) - float(rewards[ij[1]])),
        -(float(logp[ij[1]]) - float(logp[ij[0]])), ij[0], -ij[1],
        tuple(sids[ij[0]]), tuple(sids[ij[1]])))


def dpo(w, l, rw, rl, beta):
    return -F.logsigmoid(beta * ((w - l) - (rw.detach() - rl.detach())))


def sequence_logp(model, x, mask, labels):
    labels = labels.contiguous()
    _, logits = model(x, mask, labels)
    return logits.float().log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1).sum(-1)


def setup():
    assert sha256_file(CP) == EXPECTED
    tca = torch.load(ROOT / 'tca/tca_teacher_cache.pt', map_location='cpu', weights_only=False)
    assert sha256_file(SID) == tca['sid']['artifact_sha256'] == 'ad3941ced7e03bae9ebe1ab68085b24983769abbde255d3617fae8a55072fe74'
    assert sha256_file(tca['teacher']['checkpoint_path']) == tca['teacher']['checkpoint_sha256'] == 'e285d915ad5032282116eb6382edf7eb7dd4da1887c3729da4e2df30ef8ba8ae'
    ds = AmazonSeqDataset(root='dataset/amazon', split='beauty', train_test_split='train',
        max_seq_len=50, add_disambiguation=False, semantic_id_path=str(SID), target_n_layers=3)
    assert len(ds) == 131413
    # Positional leakage proof, not item-set exclusion: repeats can be legitimate.
    index = 0
    for full in ds.sequences:
        train = full[:-2]
        for pos in range(1, len(train)):
            s = ds.samples[index]
            assert s['history'] == train[:pos] and s['target'] == train[pos]
            assert tca['sample_keys'][index] == sample_key([i+1 for i in s['history']], s['target']+1)
            index += 1
    assert index == len(ds)
    raw = torch.load(SID, map_location='cpu', weights_only=False)['sem_ids'].long()
    return ds, tca, reverse_map(raw)


def validate(a, keys, reverse):
    assert a['format_version'] == 1 and a['dataset'] == 'beauty'
    assert a['split'] == 'train' and a['num_samples'] == len(keys)
    assert a['generator']['epoch'] == 111 and a['generator']['beam_size'] == 10
    assert a['sid']['collision_policy'] == 'skip'
    assert a['pair_policy']['name'] == 'max_teacher_margin_inversion'
    assert a['sample_keys'] == keys
    assert a['generator']['checkpoint_sha256'] == EXPECTED == sha256_file(CP)
    assert a['sid']['sha256'] == sha256_file(SID)
    assert a['teacher']['checkpoint_sha256'] == sha256_file(a['teacher']['checkpoint_path'])
    assert torch.equal(a['sample_indices'], torch.arange(len(keys)))
    assert a['has_pair'].shape == (len(keys),) and a['has_pair'].dtype == torch.bool
    for name in ['chosen_sid', 'rejected_sid']:
        assert a[name].shape == (len(keys),3) and a[name].dtype == torch.long
    for name in ['chosen_reward', 'rejected_reward', 'ref_logp_chosen', 'ref_logp_rejected',
                 'chosen_beam_rank', 'rejected_beam_rank']:
        assert a[name].shape == (len(keys),)
    for name in ['chosen_reward', 'rejected_reward', 'ref_logp_chosen', 'ref_logp_rejected']:
        assert torch.isfinite(a[name]).all()
    for i in range(len(keys)):
        if not a['has_pair'][i]:
            for name in ['chosen_sid', 'rejected_sid', 'chosen_reward', 'rejected_reward',
                         'ref_logp_chosen', 'ref_logp_rejected', 'chosen_beam_rank', 'rejected_beam_rank']:
                assert (a[name][i] == 0).all()
            continue
        w, l = tuple(a['chosen_sid'][i].tolist()), tuple(a['rejected_sid'][i].tolist())
        assert w != l and len(reverse.get(w, [])) == len(reverse.get(l, [])) == 1
        assert a['chosen_reward'][i] > a['rejected_reward'][i]
        assert a['ref_logp_chosen'][i] < a['ref_logp_rejected'][i]
        assert 1 <= a['chosen_beam_rank'][i] <= 10 and 1 <= a['rejected_beam_rank'][i] <= 10


def generate():
    if CACHE.exists():
        raise FileExistsError(f'Refusing overwrite: {CACHE}')
    start = time.perf_counter()
    ds, tca, reverse = setup()
    model = load_model(CP, 'cuda')
    teacher, _ = load_beauty_teacher(tca['teacher']['checkpoint_path'], 'cuda')
    n = len(ds)
    a = {'format_version': 1, 'dataset': 'beauty', 'split': 'train', 'num_samples': n,
         'generator': {'checkpoint_path': str(CP.resolve()), 'checkpoint_sha256': EXPECTED,
                       'epoch': 111, 'beam_size': 10, 'generation_precision': 'fp16 autocast',
                       'reference_precision': 'fp32, TF32 disabled', 'batch_size': 96},
         'teacher': {'name': 'sasrecpp', **tca['teacher']},
         'sid': {'sha256': sha256_file(SID), 'collision_policy': 'skip',
                 'convention': 'offset vocabulary IDs: raw SID + [1,257,513]'},
         'pair_policy': {'name': 'max_teacher_margin_inversion',
                         'tie_break': 'teacher margin desc; inversion magnitude desc; chosen rank asc; rejected rank desc; numeric SID asc'},
         'sample_keys': tca['sample_keys'], 'sample_indices': torch.arange(n),
         'has_pair': torch.zeros(n, dtype=torch.bool), 'no_pair_sentinel': 0}
    for k in ['chosen', 'rejected']:
        a[k+'_sid'] = torch.zeros(n, 3, dtype=torch.long)
        a[k+'_reward'] = torch.zeros(n)
        a['ref_logp_'+k] = torch.zeros(n)
        a[k+'_beam_rank'] = torch.zeros(n, dtype=torch.long)
    unique_counts, exposure = [], np.zeros(3, dtype=np.int64)
    sanity = {k: {'chosen': 0, 'rejected': 0, 'neither': 0, 'pairs': 0} for k in ['all', 'unique_target']}
    loader = DataLoader(ds, batch_size=96, shuffle=False, collate_fn=CollateFn(256,3,3,150))
    cursor = 0
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for batch in loader:
            x, mask = batch['input_ids'].cuda(), batch['attention_mask'].cuda()
            b = len(x)
            assert batch['sample_indices'].tolist() == list(range(cursor, cursor+b))
            assert list(batch['sample_keys']) == a['sample_keys'][cursor:cursor+b]
            with torch.amp.autocast('cuda', dtype=torch.float16):
                gen = model.generate(x, mask, num_beams=10)[:,1:].reshape(b,10,3)
            y = gen.reshape(-1,3).contiguous()
            xx, mm = x.repeat_interleave(10,0), mask.repeat_interleave(10,0)
            lp = torch.cat([sequence_logp(model, xx[j:j+96], mm[j:j+96], y[j:j+96])
                            for j in range(0,len(y),96)]).reshape(b,10).cpu().numpy()
            histories = torch.tensor([left_pad_history(s['history']) for s in ds.samples[cursor:cursor+b]], device='cuda')
            scores = (teacher.user_states(histories).float() @ teacher.item_emb.weight[1:].float().T).cpu().numpy()
            g = gen.cpu().tolist()
            assert np.isfinite(lp).all() and np.isfinite(scores).all()
            for u in range(b):
                counts = [len(reverse.get(tuple(s), [])) for s in g[u]]
                unique_counts.append(len({tuple(s) for s,c in zip(g[u],counts) if c == 1}))
                exposure += [sum(c == 1 for c in counts), sum(c > 1 for c in counts), sum(c == 0 for c in counts)]
                rewards = [float(scores[u,reverse[tuple(s)][0]-1]) if c == 1 else 0. for s,c in zip(g[u],counts)]
                pair = select_pair(g[u], rewards, lp[u], reverse)
                assert pair == select_pair(g[u], rewards, lp[u], reverse)
                if pair is None:
                    continue
                row = cursor+u
                a['has_pair'][row] = True
                for name, k in zip(['chosen','rejected'],pair):
                    a[name+'_sid'][row] = torch.tensor(g[u][k])
                    a[name+'_reward'][row] = rewards[k]
                    a['ref_logp_'+name][row] = float(lp[u,k])
                    a[name+'_beam_rank'][row] = k+1
                # Target inspected only AFTER immutable selection.
                target = tuple(batch['labels'][u].tolist())
                for group in ['all'] + (['unique_target'] if len(reverse[target]) == 1 else []):
                    field = 'chosen' if tuple(g[u][pair[0]]) == target else 'rejected' if tuple(g[u][pair[1]]) == target else 'neither'
                    sanity[group][field] += 1
                    sanity[group]['pairs'] += 1
            cursor += b
            if cursor % 960 == 0 or cursor == n:
                print(f'Train {cursor}/{n}; elapsed {time.perf_counter()-start:.1f}s', flush=True)
    validate(a, tca['sample_keys'], reverse)
    h = a['has_pair']
    summary = {'samples': n, 'pairs': int(h.sum()), 'coverage': float(h.float().mean()),
        'unique_candidates': stats(unique_counts), 'exposure_unique_collision_unknown': exposure.tolist(),
        'teacher_margin': stats((a['chosen_reward']-a['rejected_reward'])[h]),
        'reference_margin': stats((a['ref_logp_chosen']-a['ref_logp_rejected'])[h]),
        'chosen_rank': stats(a['chosen_beam_rank'][h]), 'rejected_rank': stats(a['rejected_beam_rank'][h]),
        'target_sanity': sanity, 'wall_seconds': time.perf_counter()-start,
        'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20, 'gpu': torch.cuda.get_device_name()}
    a['diagnostics'] = summary
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(a, CACHE)
    summary.update(cache_path=str(CACHE.resolve()), cache_bytes=CACHE.stat().st_size, cache_sha256=sha256_file(CACHE))
    (OUT/'generation_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def weights_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def audit():
    ds, tca, reverse = setup()
    a = torch.load(CACHE, map_location='cpu', weights_only=False)
    validate(a, tca['sample_keys'], reverse)
    summary = json.loads((OUT/'generation_summary.json').read_text())
    assert sha256_file(CACHE) == summary['cache_sha256']
    model = load_model(CP, 'cuda')  # eval mode removes dropout at pi_theta == pi_ref.
    before = weights_hash(model)
    params = [p for p in model.parameters() if p.requires_grad]
    pool = torch.where(a['has_pair'])[0]
    ids = pool[torch.randperm(len(pool), generator=torch.Generator().manual_seed(2026))[:48]]
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for ids_batch in ids.split(16):
        batch = CollateFn(256,3,3,150)([ds[int(i)] for i in ids_batch])
        x, mask, labels = [batch[k].cuda() for k in ['input_ids','attention_mask','labels']]
        tick = time.perf_counter()
        _, z = model(x, mask, labels)
        base = compute_tca_full_vocab_loss(z, labels, tca['cf_token_probs'][ids_batch].cuda().float(), alpha=.1, temperature=1)
        gb = torch.autograd.grad(base, params)
        torch.cuda.synchronize()
        base_time = time.perf_counter()-tick
        nb = sum(g.double().square().sum() for g in gb).sqrt()
        for beta in [.05,.1,.2]:
            tick = time.perf_counter()
            w = sequence_logp(model,x,mask,a['chosen_sid'][ids_batch].cuda())
            l = sequence_logp(model,x,mask,a['rejected_sid'][ids_batch].cuda())
            rw, rl = a['ref_logp_chosen'][ids_batch].cuda(), a['ref_logp_rejected'][ids_batch].cuda()
            error = max(float((w-rw).abs().max()),float((l-rl).abs().max()))
            assert error < 5e-5, error
            loss = dpo(w,l,rw,rl,beta).mean()
            assert abs(float(loss)-np.log(2)) < 1e-5
            gd = torch.autograd.grad(loss,params)
            nd = sum(g.double().square().sum() for g in gd).sqrt()
            dot = sum((b.double()*d.double()).sum() for b,d in zip(gb,gd))
            torch.cuda.synchronize()
            row = dict(indices=ids_batch.tolist(), beta=beta, base_loss=float(base), dpo_loss=float(loss),
                max_reference_error=error, max_logit=float((beta*((w-l)-(rw-rl))).abs().max()),
                tca_norm=float(nb), dpo_norm=float(nd), cosine=float(dot/(nb*nd)),
                base_seconds=base_time, dpo_seconds=time.perf_counter()-tick)
            assert all(np.isfinite(row[k]) for k in ['base_loss','dpo_loss','tca_norm','dpo_norm','cosine'])
            rows.append(row)
    assert weights_hash(model) == before and sha256_file(CP) == EXPECTED
    result = {'rows': rows, 'mode': 'eval FP32; all parameters; no optimizer; three seeded batches of 16',
        'weights_unchanged': True, 'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
        'synthetic': [{'beta': b, 'delta_change': d, 'logit': b*d,
                       'loss': float(F.softplus(torch.tensor(-b*d))),
                       'derivative_delta': -b/(1+np.exp(b*d))} for b in [.05,.1,.2] for d in [-1.,-.1,0.,.1,1.]]}
    result['provenance'] = {'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'seed': 2026, 'cache_sha256': sha256_file(CACHE),
        'files': {str(p): sha256_file(p) for p in [CP, SID, ROOT/'tca/tca_teacher_cache.pt',
            Path('scripts/phase2a_preference.py'), Path('genrec/data/amazon.py'),
            Path('genrec/models/tiger.py'), Path('genrec/modules/t5.py'),
            Path('genrec/trainers/tca_loss.py'), Path('genrec/trainers/tiger_trainer.py')]}}
    (OUT/'scale_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['generate','audit'])
    args = parser.parse_args()
    generate() if args.mode == 'generate' else audit()
