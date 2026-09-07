"""Validation-only candidate/reward audit. No pair training or recommender changes."""
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import rankdata
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.audit_tiger_generation import load_model, OFFSETS
from scripts.generate_tca_teacher_cache import left_pad_history
from genrec.data.amazon import AmazonSeqDataset
from genrec.models.sasrec_teacher import load_beauty_teacher
from genrec.trainers.tca_loss import sha256_file
from genrec.trainers.tiger_trainer import CollateFn

ROOT = Path('out/tiger/amazon/beauty/rqkmeans')
OUT = ROOT/'preference_candidate_audit'


def stats(x):
    a=np.asarray(x,dtype=np.float64)
    if not a.size: return {'count':0}
    return {'count':int(a.size),'mean':float(a.mean()),'median':float(np.median(a)),
            **{f'p{k}':float(np.percentile(a,k)) for k in [10,25,75,90,95,99]},'max':float(a.max())}


def capture_beams(model, x, mask):
    # Observe actual local beam scores at return without replacing decoding.
    scores=[]
    code=model.model._beam_search.__func__.__code__
    previous=sys.getprofile()
    def observe(frame,event,arg):
        if event=='return' and frame.f_code is code:
            scores.append(frame.f_locals['beam_scores'].detach().clone())
    sys.setprofile(observe)
    try: generated=model.generate(x,mask,num_beams=10)
    finally: sys.setprofile(previous)
    assert len(scores)==1
    return generated[:,1:].reshape(-1,10,3), scores[0]


def generate():
    start=time.perf_counter()
    cp=ROOT/'tiger_tca_full_vocab_ndcg_select/best_model.pt'
    digest=sha256_file(cp)
    assert digest=='17d6ae7b8914ee43176757669e0a592b229a9ad2c41d9225b9f607e73a950870'
    device=torch.device('cuda')
    model=load_model(cp,device)
    sid=ROOT/'semantic_ids.pt'
    raw=torch.load(sid,weights_only=False,map_location='cpu')['sem_ids'].long()
    reverse=defaultdict(list)
    for i,row in enumerate(raw.tolist(),1): reverse[tuple(np.array(row)+np.array(OFFSETS))].append(i)
    cache=torch.load('out/tiger/amazon/beauty/tca/tca_teacher_cache.pt',weights_only=False,map_location='cpu')
    teacher_path=cache['teacher']['checkpoint_path']
    assert sha256_file(sid)==cache['sid']['artifact_sha256']
    assert sha256_file(teacher_path)==cache['teacher']['checkpoint_sha256']
    teacher,_=load_beauty_teacher(teacher_path,device)
    del cache
    dataset=AmazonSeqDataset(root='dataset/amazon',split='beauty',train_test_split='valid',max_seq_len=50,
        add_disambiguation=False,semantic_id_path=str(sid),target_n_layers=3)
    assert len(dataset)==22363
    for s,seq in zip(dataset.samples,dataset.sequences):
        assert s['target']==seq[-2] and s['history']==seq[:-2]
    loader=DataLoader(dataset,batch_size=96,shuffle=False,num_workers=0,collate_fn=CollateFn(256,3,3,150))
    arrays=defaultdict(list)
    ambiguity=defaultdict(list)
    times=defaultdict(float)
    cursor=0
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for batch in tqdm(loader,desc='Validation candidates'):
            x=batch['input_ids'].to(device); mask=batch['attention_mask'].to(device)
            b=len(x)
            torch.cuda.synchronize(); tick=time.perf_counter()
            with torch.amp.autocast('cuda',dtype=torch.float16):
                gen,beam_scores=capture_beams(model,x,mask)
            torch.cuda.synchronize(); times['generate']+=time.perf_counter()-tick
            # Reuse only existing forward; full FP32 log probabilities avoid half softmax rounding.
            tick=time.perf_counter()
            repeated=x.repeat_interleave(10,0); repeated_mask=mask.repeat_interleave(10,0)
            cand=gen.reshape(-1,3).contiguous(); scores=[]
            for j in range(0,len(cand),96):
                _,z=model(repeated[j:j+96],repeated_mask[j:j+96],cand[j:j+96])
                scores.append(z.log_softmax(-1).gather(-1,cand[j:j+96,:,None]).squeeze(-1).sum(-1))
            lp=torch.cat(scores).reshape(b,10)
            torch.cuda.synchronize(); times['candidate_forward']+=time.perf_counter()-tick
            tick=time.perf_counter()
            histories=torch.tensor([left_pad_history(s['history']) for s in dataset.samples[cursor:cursor+b]],device=device)
            item_scores=(teacher.user_states(histories).float()@teacher.item_emb.weight[1:].float().T).cpu().numpy().astype(np.float64)
            times['teacher']+=time.perf_counter()-tick
            g=gen.cpu().numpy(); group_size=np.zeros((b,10),int)
            rewards={k:np.full((b,10),np.nan) for k in ['max','mean','lse','mass']}
            normalizer=logsumexp(item_scores,axis=1)
            for u in range(b):
                for k in range(10):
                    ids=reverse.get(tuple(g[u,k]),[])
                    group_size[u,k]=len(ids)
                    if not ids: continue
                    s=item_scores[u,np.array(ids)-1]
                    rewards['max'][u,k]=s.max(); rewards['mean'][u,k]=s.mean()
                    rewards['lse'][u,k]=logsumexp(s)
                    rewards['mass'][u,k]=np.exp(s-normalizer[u]).sum()
                    if len(ids)>1:
                        for name,value in [('max',s.max()),('min',s.min()),('range',np.ptp(s)),
                            ('std',s.std()),('top1_top2',np.sort(s)[-1]-np.sort(s)[-2])]: ambiguity[name].append(value)
            for k,value in {'generated':g,'beam_scores':beam_scores.cpu().numpy(),'logp':lp.cpu().numpy(),
                'group_size':group_size,'target':batch['labels'].numpy(),'normalizer':normalizer,**rewards}.items(): arrays[k].append(value)
            cursor+=b
    data={k:np.concatenate(v) for k,v in arrays.items()}
    sizes=np.array([len(v) for v in reverse.values()]); collision=sizes[sizes>1]
    meta={'checkpoint':str(cp.resolve()),'sha256':digest,'teacher_sha256':sha256_file(teacher_path),
        'sid_sha256':sha256_file(sid),'samples':len(dataset),'catalog':{'items':len(raw),'unique_sids':len(reverse),
        'collision_groups':len(collision),'collision_items':int(collision.sum()),'participating_item_rate':float(collision.sum()/len(raw)),
        'duplicate_excess_rate':float((len(raw)-len(reverse))/len(raw)), 'collision_group_size':stats(collision),'all_group_size':stats(sizes)},
        'ambiguity':{k:stats(v) for k,v in ambiguity.items()},'timings_seconds':dict(times),
        'wall_seconds':time.perf_counter()-start,'gpu':torch.cuda.get_device_name(),
        'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'model_parameters':sum(p.numel() for p in model.parameters())}
    data['target_size']=np.array([len(reverse[tuple(row)]) for row in data['target']])
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'validation_candidates.npz',**data)
    (OUT/'inference_metadata.json').write_text(json.dumps(meta,indent=2))
    return data,meta


def analyze(d,meta):
    n=len(d['logp']); size=d['group_size']; lp=d['logp']
    result=dict(meta)
    result['exposure']={'unique':float((size==1).mean()),'collision':float((size>1).mean()),'unknown':float((size==0).mean()),
        **{f'collision_top{k}':float((size[:,:k]>1).mean()) for k in [1,5,10]},
        'users_collision0':float(((size>1).sum(-1)==0).mean()),'users_collision1':float(((size>1).sum(-1)==1).mean()),
        'users_collision2plus':float(((size>1).sum(-1)>=2).mean()),'target_collision':float((d['target_size']>1).mean()),
        'users_at_least2_unique':float(((size==1).sum(-1)>=2).mean())}
    matches=(d['generated']==d['target'][:,None,:]).all(-1)
    result['duplicate_sequences']=sum(10-len(set(map(tuple,g))) for g in d['generated'])
    result['beam_vs_logp']={'identical_full_order_rate':float((np.argsort(-lp,axis=1,kind='stable')==np.arange(10)).all(-1).mean()),
        'top1_agreement':float((lp.argmax(-1)==0).mean()),'absolute_score_difference':stats(np.abs(lp-d['beam_scores'])),
        'inverted_pair_gap':stats([lp[u,j]-lp[u,i] for u in range(n) for i,j in itertools.combinations(range(10),2) if lp[u,j]>lp[u,i]])}
    metrics={}
    for label,mask in [('unique_target',d['target_size']==1),('collision_target',d['target_size']>1),('all',np.ones(n,bool))]:
        metrics[label]={'users':int(mask.sum())}
        for k in [5,10]:
            metrics[label][f'Recall@{k}']=float(matches[mask,:k].any(-1).mean())
            metrics[label][f'NDCG@{k}']=float((matches[mask,:k]/np.log2(np.arange(2,k+2))).sum(-1).mean())
    result['target_metrics_user_mean']=metrics
    policies={'U':('max',size==1),'Skip':('max',size==1),'Max':('max',size>0),
        'Mean':('mean',size>0),'LSE':('lse',size>0),'Mass':('mass',size>0)}
    policy_results={}; choices={}
    for policy,(key,masks) in policies.items():
        if policy=='Skip': continue
        counts=[]; overlaps=[]; correlations=[]; top1=[]; extreme=defaultdict(list)
        disagree_counts=[]; all_margins=[]; all_gaps=[]; closest_margins=[]; closest_gaps=[]
        user_max_margin=[]; target_ranks=[]; target_beams=[]; target_unique_ranks=[]; winners=[]; losers=[]
        for u in range(n):
            ids=np.flatnonzero(masks[u]); counts.append(len(ids)); r=d[key][u]
            if len(ids)<2:
                winners.append(-1);losers.append(-1);disagree_counts.append(0);user_max_margin.append(0);continue
            mo=ids[np.argsort(-lp[u,ids],kind='stable')]; ro=ids[np.argsort(-r[ids],kind='stable')]
            winner,loser=ro[0],ro[-1];winners.append(int(winner));losers.append(int(loser))
            top1.append(mo[0]==winner); k=min(3,len(ids));overlaps.append(len(set(mo[:k])&set(ro[:k]))/k)
            ar=rankdata(lp[u,ids]);br=rankdata(r[ids])
            if ar.std()>0 and br.std()>0: correlations.append(np.corrcoef(ar,br)[0,1])
            for k,value in [('winner_beam',winner+1),('loser_beam',loser+1),('teacher_margin',r[winner]-r[loser]),
                ('model_margin',lp[u,winner]-lp[u,loser])]:extreme[k].append(value)
            pairs=[]
            for i,j in itertools.combinations(ids,2):
                dt=r[i]-r[j];dm=lp[u,i]-lp[u,j]
                if dt*dm<0: pairs.append((abs(float(dm)),abs(float(dt))))
            disagree_counts.append(len(pairs)); user_max_margin.append(max([p[1] for p in pairs],default=0))
            all_gaps.extend([-p[0] for p in pairs]);all_margins.extend([p[1] for p in pairs])
            if pairs:
                gap,margin=min(pairs,key=lambda p:(p[0],-p[1]))
                closest_gaps.append(-gap);closest_margins.append(margin)
            target=np.flatnonzero(matches[u]&masks[u])
            if len(target):
                tid=target[0];rank=1+sum(r[ids]>r[tid]);target_ranks.append(rank);target_beams.append(tid+1)
                if d['target_size'][u]==1: target_unique_ranks.append(rank)
        threshold=float(np.percentile(all_margins,75)) if all_margins else None
        result_ranks=lambda rr: {**stats(rr),**{f'top{k}':float((np.asarray(rr)<=k).mean()) if rr else None for k in [1,3,5]}}
        policy_results[policy]={'mean_candidates':float(np.mean(counts)), 'users_with_extrema_pair':len(top1),
            'extrema_coverage':len(top1)/n,'users_with_disagreement':int(np.count_nonzero(disagree_counts)),
            'disagreement_coverage':float(np.count_nonzero(disagree_counts)/n),'disagreement_pairs_per_user':float(np.mean(disagree_counts)),
            'strong_threshold_p75_teacher_margin':threshold,'strong_user_coverage':float((np.array(user_max_margin)>=threshold).mean()) if threshold else 0,
            'top1_model_teacher_agreement':float(np.mean(top1)),'top3_overlap':float(np.mean(overlaps)),
            'spearman':float(np.mean(correlations)),'extrema':{k:stats(v) for k,v in extreme.items()},
            'all_disagreements':{'teacher_margin':stats(all_margins),'model_margin':stats(all_gaps)},
            'closest_disagreement':{'teacher_margin':stats(closest_margins),'model_margin':stats(closest_gaps)},
            'target_in_pool_reward_rank':result_ranks(target_ranks),'target_in_pool_beam_rank':stats(target_beams),
            'unique_target_reward_rank':result_ranks(target_unique_ranks),
            'winner_is_target_all_users':float(sum(w>=0 and matches[u,w] for u,w in enumerate(winners))/n)}
        choices[policy]=(np.array(winners),np.array(losers))
    policy_results['Skip']=policy_results['U']
    result['policies']=policy_results
    result['stability']={}
    for a,b in itertools.combinations(['Max','Mean','LSE','Mass'],2):
        wa,la=choices[a];wb,lb=choices[b];mask=(wa>=0)&(wb>=0)
        result['stability'][a+' vs '+b]={'winner':float((wa[mask]==wb[mask]).mean()),
            'loser':float((la[mask]==lb[mask]).mean()),'pair':float(((wa[mask]==wb[mask])&(la[mask]==lb[mask])).mean())}
    u=size==1;known=size>0
    result['probability_identity']={'max_error_logmass_vs_lse_minus_logZ':float(np.abs(np.log(d['mass'][known])-(d['lse']-d['normalizer'][:,None])[known]).max()),
        'unique_max_error':float(np.abs(np.log(d['mass'][u])-(d['max']-d['normalizer'][:,None])[u]).max())}
    result['skip_loss_of_pairs']={'extrema_user_fraction':(policy_results['Max']['users_with_extrema_pair']-policy_results['U']['users_with_extrema_pair'])/n,
        'disagreement_user_fraction':(policy_results['Max']['users_with_disagreement']-policy_results['U']['users_with_disagreement'])/n}
    (OUT/'diagnostics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    if '--analyze-only' in sys.argv:
        analyze(dict(np.load(OUT/'validation_candidates.npz')),json.loads((OUT/'inference_metadata.json').read_text()))
    else:
        analyze(*generate())
