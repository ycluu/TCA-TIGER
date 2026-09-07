"""Read-only full-validation TCA effect audit; no training or cache regeneration."""
import ast
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.audit_tiger_generation import load_model, OFFSETS
from scripts.generate_tca_teacher_cache import left_pad_history
from genrec.data.amazon import AmazonSeqDataset
from genrec.data.tca_teacher_cache import exact_prefix_distributions
from genrec.models.sasrec_teacher import load_beauty_teacher
from genrec.trainers.tca_loss import TCATeacherCache, sha256_file
from genrec.trainers.tiger_trainer import CollateFn, calculate_pos_index, recall_at_k, ndcg_at_k

ROOT = Path('out/tiger/amazon/beauty/rqkmeans')
OUT = ROOT / 'tca_effect_audit'


def stats(x):
    return {'mean': float(np.mean(x)), 'median': float(np.median(x))}


def logs(folder):
    rows = {}
    for path in sorted(folder.glob('*.log')):
        for line in path.read_text().splitlines():
            m = re.search(r'Epoch (\d+) - (loss|Valid|Test): (.*)', line)
            if m:
                epoch, kind, value = m.groups()
                rows.setdefault(int(epoch), {})[kind] = float(value) if kind == 'loss' else ast.literal_eval(value)
    return rows


def main():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sid_path = ROOT / 'semantic_ids.pt'
    raw = torch.load(sid_path, weights_only=False, map_location='cpu')['sem_ids'].long()
    dataset = AmazonSeqDataset(root='dataset/amazon', split='beauty', train_test_split='valid',
        max_seq_len=50, add_disambiguation=False, semantic_id_path=str(sid_path), target_n_layers=3)
    assert len(dataset) == 22363
    for sample, sequence in zip(dataset.samples, dataset.sequences):
        assert sample['target'] == sequence[-2]
        assert sample['history'] == sequence[:-2]
    cache_path = Path('out/tiger/amazon/beauty/tca/tca_teacher_cache.pt')
    cache = TCATeacherCache.load_and_validate(cache_path, dataset='beauty', num_samples=131413,
        num_items=12101, sid_layers=3, codebook_size=256, sid_artifact_path=sid_path)
    teacher_path = Path(cache.artifact['teacher']['checkpoint_path'])
    assert sha256_file(teacher_path) == cache.artifact['teacher']['checkpoint_sha256']
    teacher, _ = load_beauty_teacher(teacher_path, device)
    q_parts = []
    raw_device = raw.to(device)
    with torch.no_grad():
        for start in tqdm(range(0, len(dataset), 256), desc='Validation SASRec'):
            samples = dataset.samples[start:start+256]
            histories = torch.tensor([left_pad_history(s['history']) for s in samples], device=device)
            targets = torch.tensor([s['target'] for s in samples], device=device)
            scores = teacher.user_states(histories).float() @ teacher.item_emb.weight[1:].float().T
            q = exact_prefix_distributions(scores, raw_device, raw_device[targets])
            assert torch.isfinite(q).all() and torch.allclose(q.sum(-1), torch.ones_like(q[..., 0]), atol=1e-5)
            q_parts.append(q.cpu())
    q_all = torch.cat(q_parts)
    del teacher
    result = {'samples': len(dataset), 'teacher_hash': sha256_file(teacher_path),
        'teacher_path': str(teacher_path), 'cache_hash': cache.cache_sha256, 'sid_hash': cache.sid_sha256,
        'models': {}, 'logs': {}, 'epoch83': 'EPOCH83_CHECKPOINT_NOT_AVAILABLE'}
    del cache
    known = set(map(tuple, (raw + torch.tensor(OFFSETS)).tolist()))
    per_model = {}
    for name, folder in [('Original', ROOT/'tiger'), ('TCA-73', ROOT/'tiger_tca_full_vocab')]:
        checkpoint = folder/'best_model.pt'
        model = load_model(checkpoint, device)
        chunks = {k: [] for k in ['rank','pf','pl','correct','mass','kl','ce','js','over5','over10','qtarget','qrank','hit10','ndcg10','prefix']}
        batch_metrics = {f'{m}@{k}': [] for m in ['Recall','NDCG'] for k in [5,10]}
        legal, known_counts = np.zeros(10), np.zeros(10)
        cursor = 0
        loader = DataLoader(dataset, batch_size=96, shuffle=False, num_workers=0, collate_fn=CollateFn(256,3,3,150))
        with torch.no_grad():
            for batch in tqdm(loader, desc=name):
                inputs, masks, labels = [batch[k].to(device) for k in ['input_ids','attention_mask','labels']]
                b = len(labels)
                q = q_all[cursor:cursor+b].to(device)
                with torch.amp.autocast('cuda', enabled=device.type=='cuda', dtype=torch.float16):
                    _, logits = model(inputs,masks,labels)
                    generated = model.generate(inputs,masks,num_beams=10)[:,1:].reshape(b,10,3)
                logits = logits.float()
                full = logits.softmax(-1)
                local = torch.stack([logits[:,l,o:o+256] for l,o in enumerate(OFFSETS)],1)
                lp = local.log_softmax(-1)
                p = lp.exp()
                y = labels-torch.tensor(OFFSETS,device=device)
                ranks = 1+(local>local.gather(-1,y[...,None])).sum(-1)
                correct = local.argmax(-1)==y
                qlog = q.clamp_min(1e-30).log()
                middle_log = ((q+p)/2).clamp_min(1e-30).log()
                cm = torch.stack([full[:,l,o:o+256].sum(-1) for l,o in enumerate(OFFSETS)],1)
                zero = full[:,:,0]
                values = {'rank':ranks, 'pf':full.gather(-1,labels[...,None]).squeeze(-1),
                    'pl':p.gather(-1,y[...,None]).squeeze(-1), 'correct':correct,
                    'mass':torch.stack([cm,1-cm-zero,zero],-1),
                    'kl':(q*(qlog-lp)).sum(-1), 'ce':-(q*lp).sum(-1),
                    'js':.5*((q*(qlog-middle_log)).sum(-1)+(p*(lp-middle_log)).sum(-1)),
                    'qtarget':q.gather(-1,y[...,None]).squeeze(-1),
                    'qrank':1+(q>q.gather(-1,y[...,None])).sum(-1),
                    'prefix':correct.int().cumprod(-1)}
                for k in [5,10]:
                    # Stable tie order; zero-probability teacher tokens are excluded.
                    qi=torch.argsort(q,dim=-1,descending=True,stable=True)[...,:k]
                    pi=torch.argsort(p,dim=-1,descending=True,stable=True)[...,:k]
                    support=q.gather(-1,qi)>0
                    common=(qi[...,None]==pi[...,None,:]).any(-1)&support
                    values[f'over{k}']=common.sum(-1)/support.sum(-1).clamp_min(1)
                pos=calculate_pos_index(generated,labels,maxk=10).cpu()
                values['hit10']=pos.any(-1)
                values['ndcg10']=(pos/torch.log2(torch.arange(2,12).float())).sum(-1)
                for k in [5,10]:
                    batch_metrics[f'Recall@{k}'].append(recall_at_k(pos,k))
                    batch_metrics[f'NDCG@{k}'].append(ndcg_at_k(pos,k))
                for key,value in values.items(): chunks[key].append(value.cpu().numpy())
                g=generated.cpu().numpy()
                valid=((g>=np.array(OFFSETS))&(g<np.array(OFFSETS)+256)).all(-1)
                legal+=valid.sum(0)
                known_counts+=np.array([[tuple(s) in known for s in beams] for beams in g]).sum(0)
                cursor+=b
        a={k:np.concatenate(v) for k,v in chunks.items()}
        per_model[name]=a
        rows=[]
        for l in range(3):
            r=a['rank'][:,l]
            row={'level':l,'rank':{'mrr':float((1/r).mean()),**stats(r),**{f'top{k}':float((r<=k).mean()) for k in [1,5,10]}},
                 'mass':a['mass'][:,l].astype(np.float64).mean(0).tolist()}
            for key in ['pf','pl','kl','ce','js','over5','over10','qtarget','qrank','correct','prefix']:
                row[key]=stats(a[key][:,l])
            row['teacher_top_rates']={f'top{k}':float((a['qrank'][:,l]<=k).mean()) for k in [1,5,10]}
            rows.append(row)
        result['models'][name]={'checkpoint':str(checkpoint.resolve()),'sha256':sha256_file(checkpoint),
            'selection':json.loads((folder/'results.json').read_text())['metrics'], 'levels':rows,
            'beam':{'level_valid@1':legal[0]/cursor,'level_valid_all':legal.mean()/cursor,
                    'known@1':known_counts[0]/cursor,'known_all':known_counts.mean()/cursor},
            'validation_existing_evaluator':{k:float(np.mean(v)) for k,v in batch_metrics.items()},
            'sample_mean_recall10':float(a['hit10'].mean())}
        result['logs'][name]=logs(folder)
        del model
        if name=='TCA-73' and legal.mean()/cursor < .99:
            OUT.mkdir(exist_ok=True)
            (OUT/'pathology.json').write_text(json.dumps(result,indent=2))
            raise RuntimeError('Material beam legality pathology; stopping')
    result['buckets']=[]
    for l in range(3):
        support=per_model['Original']['qtarget'][:,l]
        for label,mask in [('low',support<.1),('medium',(support>=.1)&(support<.5)),('high',support>=.5)]:
            entry={'level':l,'bucket':label,'count':int(mask.sum()),'models':{}}
            for name,a in per_model.items():
                entry['models'][name]={k:float(a[k][mask,l].mean()) for k in ['rank','pf','pl','correct','prefix']} if mask.any() else {}
                if mask.any():
                    entry['models'][name]['exact']=float(a['prefix'][mask,2].mean())
                    entry['models'][name]['mrr']=float((1/a['rank'][mask,l]).mean())
                    entry['models'][name]['recall10']=float(a['hit10'][mask].mean())
            result['buckets'].append(entry)
    rng=np.random.default_rng(42)
    result['bootstrap']={}
    for metric in ['hit10','ndcg10']:
        d=per_model['TCA-73'][metric].astype(float)-per_model['Original'][metric].astype(float)
        estimates=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)]
        result['bootstrap'][metric]={'delta':float(d.mean()),'paired_95ci':np.percentile(estimates,[2.5,97.5]).tolist()}
    OUT.mkdir(exist_ok=True)
    (OUT/'diagnostics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='logs'},indent=2))


if __name__=='__main__': main()
