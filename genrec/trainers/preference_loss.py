"""Fixed offline DPO: no teacher, reference model or candidate generation."""
from pathlib import Path
import hashlib
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from genrec.data.tca_teacher_cache import sample_key
from genrec.trainers.tca_loss import sha256_file

PHASE1_SHA = '17d6ae7b8914ee43176757669e0a592b229a9ad2c41d9225b9f607e73a950870'
PAIR_SHA = 'd2b6f27c9f4618311b18caeb6d1ab856901d4d6e5f1a7d02326a31981fd3ea93'
TEACHER_SHA = 'e285d915ad5032282116eb6382edf7eb7dd4da1887c3729da4e2df30ef8ba8ae'


class PreferenceCache:
    def __init__(self, artifact):
        self.a = artifact

    @classmethod
    def load(cls, path, dataset, sid_path, initialization):
        if sha256_file(path) != PAIR_SHA or sha256_file(initialization) != PHASE1_SHA:
            raise ValueError('Frozen preference cache / initialization SHA256 mismatch')
        a = torch.load(path,map_location='cpu',weights_only=False)
        cls.validate_metadata(a,sha256_file(sid_path))
        if len(dataset) != 131413 or dataset.train_test_split != 'train':
            raise ValueError('Preference cache is exclusively for the canonical train dataset')
        keys = [sample_key([i+1 for i in s['history']],s['target']+1) for s in dataset.samples]
        if keys != a['sample_keys'] or not torch.equal(a['sample_indices'],torch.arange(len(keys))):
            raise ValueError('Preference ordered dataset identity mismatch')
        raw = torch.load(sid_path,map_location='cpu',weights_only=False)['sem_ids'].long()
        counts = {}
        for row in (raw+torch.tensor([1,257,513])).tolist():
            key = tuple(row); counts[key] = counts.get(key,0)+1
        h = a['has_pair']
        if h.dtype != torch.bool or h.shape != (131413,) or int(h.sum()) != 130758:
            raise ValueError('Preference coverage/shape mismatch')
        for name in ['chosen','rejected']:
            if a[name+'_sid'].shape != (131413,3) or a[name+'_sid'].dtype != torch.long:
                raise ValueError('Invalid candidate SID tensor')
            if any(counts.get(tuple(row),0) != 1 for row in a[name+'_sid'][h].tolist()):
                raise ValueError('Non-unique/unknown SID in preference pair')
            for field in [name+'_reward','ref_logp_'+name]:
                if a[field].shape != (131413,) or not torch.isfinite(a[field]).all():
                    raise ValueError('Invalid preference scores')
        if not (a['chosen_reward'][h] > a['rejected_reward'][h]).all() or not (
                a['ref_logp_chosen'][h] < a['ref_logp_rejected'][h]).all():
            raise ValueError('Preference inversion violation')
        if (a['chosen_sid'][h] == a['rejected_sid'][h]).all(-1).any():
            raise ValueError('Identical preference candidates')
        return cls(a)

    @staticmethod
    def validate_metadata(a, sid_sha):
        checks = [a.get('format_version') == 1, a.get('dataset') == 'beauty',a.get('split') == 'train',
            a.get('num_samples') == 131413,a.get('generator',{}).get('checkpoint_sha256') == PHASE1_SHA,
            a.get('generator',{}).get('epoch') == 111,a.get('generator',{}).get('beam_size') == 10,
            a.get('teacher',{}).get('checkpoint_sha256') == TEACHER_SHA,
            a.get('sid',{}).get('sha256') == sid_sha,a.get('sid',{}).get('collision_policy') == 'skip',
            a.get('pair_policy',{}).get('name') == 'max_teacher_margin_inversion']
        if not all(checks):
            raise ValueError('Preference metadata mismatch')

    def batch(self, indices, keys, device, ratio=1.0):
        ids = indices.cpu().long()
        if ids.ndim != 1 or ids.min() < 0 or ids.max() >= len(self.a['sample_keys']):
            raise ValueError('Preference sample index out of range')
        if [self.a['sample_keys'][i] for i in ids.tolist()] != list(keys):
            raise ValueError('Preference sample-key mismatch')
        if not 0 <= ratio <= 1:
            raise ValueError('preference_sample_ratio must be in [0,1]')
        result = {k:self.a[k][ids].detach().to(device) for k in
                  ['has_pair','chosen_sid','rejected_sid','ref_logp_chosen','ref_logp_rejected']}
        if ratio < 1:
            # Stable sample-index + key hashing, never reward/target-based selection.
            selected = [int.from_bytes(hashlib.sha256(f'{i}:{k}'.encode()).digest()[:8],'big') / 2**64 < ratio
                        for i,k in zip(ids.tolist(),keys)]
            result['has_pair'] &= torch.tensor(selected,device=device)
        return result


def token_logp(logits, labels):
    if logits.shape != (*labels.shape,769) or labels.shape[-1] != 3:
        raise ValueError('Expected [B,3,769] candidate logits and [B,3] labels')
    return F.log_softmax(logits.float(),-1).gather(-1,labels.unsqueeze(-1)).squeeze(-1).sum(-1)


def dpo_values(w,l,rw,rl,beta=.1):
    return -F.logsigmoid(beta*((w-l)-(rw.detach()-rl.detach())))


def preference_loss(model, x, mask, pair, *, beta=.1, microbatch_size=32, recompute=True):
    if microbatch_size < 1 or beta <= 0:
        raise ValueError('Positive microbatch and beta required')
    keep = pair['has_pair'].bool()
    n = int(keep.sum())
    if n == 0:
        return x.new_zeros((),dtype=torch.float32), {'pair_coverage_in_batch':0.,'preference_accuracy':0.,
            'mean_policy_preference_margin':0.,'mean_reference_preference_margin':0.,'mean_dpo_logit':0.,
            'max_abs_dpo_logit':0.}
    xx = torch.cat([x[keep],x[keep]])
    mm = torch.cat([mask[keep],mask[keep]])
    yy = torch.cat([pair['chosen_sid'][keep],pair['rejected_sid'][keep]]).contiguous()
    def score(a,b,c):
        _, z = model(input_ids=a,attention_mask=b,labels=c.contiguous())
        return token_logp(z,c)
    scores = []
    for start in range(0,len(yy),microbatch_size):
        args = (xx[start:start+microbatch_size],mm[start:start+microbatch_size],yy[start:start+microbatch_size])
        # Merely splitting forward graphs does NOT bound retained activation memory.
        # Non-reentrant checkpointing recomputes each chunk during the single backward,
        # retaining its autocast context and dropout RNG state without changing reduction.
        scores.append(checkpoint(score,*args,use_reentrant=False,preserve_rng_state=True)
                      if recompute and torch.is_grad_enabled() else score(*args))
    w,l = torch.cat(scores).split(n)
    rw,rl = pair['ref_logp_chosen'][keep].float(),pair['ref_logp_rejected'][keep].float()
    values = dpo_values(w,l,rw,rl,beta)
    loss = values.sum()/len(x)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite DPO loss')
    delta,ref = (w-l).detach(),(rw-rl).detach()
    return loss, {'pair_coverage_in_batch':n/len(x),'preference_accuracy':float((delta>0).float().mean()),
        'mean_policy_preference_margin':float(delta.mean()),'mean_reference_preference_margin':float(ref.mean()),
        'mean_dpo_logit':float((beta*(delta-ref)).mean()),'max_abs_dpo_logit':float((beta*(delta-ref)).abs().max())}


def initialize_policy(model,path):
    if sha256_file(path) != PHASE1_SHA:
        raise ValueError('Phase2 requires frozen epoch111 initialization')
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)


def golden_test(model,batch,pair,device):
    old_mode = model.training
    model.eval()
    try:
        with torch.no_grad(),torch.amp.autocast(device.type,enabled=False):
            loss,diag = preference_loss(model,batch['input_ids'].to(device),batch['attention_mask'].to(device),
                                       pair,microbatch_size=16,recompute=False)
        conditional = float(loss)/diag['pair_coverage_in_batch']
        if abs(conditional - 0.69314718056) > 1e-5 or diag['max_abs_dpo_logit'] > 1e-5:
            raise ValueError(f'DPO initialization golden test failed: {conditional}, {diag}')
        return conditional,diag
    finally:
        model.train(old_mode)
