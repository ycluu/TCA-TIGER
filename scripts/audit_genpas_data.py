"""Read-only GenPAS distribution audit; no trainer integration or saved training pairs.

Imports only official analysis functions, never their Test-reading main functions.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, str(Path('tmp/genpas_deps').resolve()))
from genrec.data.amazon import AmazonSeqDataset

OUT = Path('out/tiger/amazon/beauty/genpas_data_audit')
SID = Path('out/tiger/amazon/beauty/rqkmeans/semantic_ids.pt')


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path('external/GenPAS/data_analysis') / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summary(values, weights=None):
    values = np.asarray(values)
    if weights is None:
        return {'n':len(values),'mean':float(values.mean()),'min':int(values.min()),'max':int(values.max()),
                **{f'p{q}':float(np.percentile(values,q)) for q in [10,25,50,75,90]}}
    weights = np.asarray(weights,float)
    order = np.argsort(values,kind='stable')
    v=values[order]; w=weights[order]; c=np.cumsum(w)/w.sum()
    return {'support_rows':len(v),'weight_sum':float(w.sum()),'mean':float(np.average(v,weights=w)),
            **{f'p{q}':int(v[min(np.searchsorted(c,q/100),len(v)-1)]) for q in [10,25,50,75,90]}}


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    start=time.perf_counter()
    kl=load_module('official_genpas_kl','get_kl.py')
    ad=load_module('official_genpas_ad','get_align_disc.py')
    sid_hash=hashlib.sha256(SID.read_bytes()).hexdigest()
    assert sid_hash=='ad3941ced7e03bae9ebe1ab68085b24983769abbde255d3617fae8a55072fe74'
    # The shared raw loader is required for the existing canonical sequence mapping.
    # Never instantiate Test, access sequence[-1], or load official bundled *_seq.pkl.
    ds=AmazonSeqDataset(root='dataset/amazon',split='beauty',train_test_split='valid',max_seq_len=50,
                        add_disambiguation=False,semantic_id_path=str(SID),target_n_layers=3)
    train=[list(s['history']) for s in ds.samples]
    validation=[s['history']+[s['target']] for s in ds.samples]
    del ds
    raw=torch.load(SID,map_location='cpu',weights_only=False)['sem_ids'].long().numpy()
    symbols=raw[:,0]*256**2+raw[:,1]*256+raw[:,2]
    sid_train=[[int(symbols[i]) for i in s] for s in train]
    sid_valid=[[int(symbols[i]) for i in s] for s in validation]
    lengths=np.array([len(s) for s in train]); pair_counts=lengths-1
    N=int(pair_counts.sum()); assert N==131413 and len(train)==22363
    hist=np.concatenate([np.arange(1,len(s)) for s in train])
    target_items=np.array([i for s in train for i in s[1:]])
    val_items=np.array([s[-1] for s in validation])
    result={'seed':42,'users':len(train),'train_interactions':int(lengths.sum()),'current_pairs':N,
            'samples_per_user':summary(pair_counts),'target_position_one_based':summary(hist+1),
            'current_history_raw':summary(hist),'current_history_model50':summary(np.minimum(hist,50)),
            'validation_history_raw':summary(lengths),'validation_history_model50':summary(np.minimum(lengths,50)),
            'current_steps_batch256_drop_last':N//256,'current_rows_dropped_per_epoch':N%256,
            'sid_sha256':sid_hash,'official_commit':'7803b541f646237084ac01692accc8e97053491b',
            'by_train_sequence_length':{str(L):{'users':int((lengths==L).sum()),'pairs':int(((lengths==L)*pair_counts).sum())}
                                       for L in sorted(set(lengths))},'kl':{},'genpas_expected':{},'alignment':{}}
    arrays={}
    for name,codes in [('c0',raw[:,0]),('c1',raw[:,1]),('c2',raw[:,2]),('full_sid',symbols)]:
        tr=[[int(codes[i]) for i in s] for s in train]
        va=[[int(codes[i]) for i in s] for s in validation]
        # Train+Validation symbol support replaces forbidden Test-derived universe.
        universe=sorted(set(v for s in tr+va for v in s))
        p=kl.compute_validation_distribution(va,universe,False,1e-10)
        q=kl.compute_alpha_beta_distribution(tr,universe,1.,0.,1e-10)
        target_counts=Counter(codes[target_items].tolist())
        target_support=set(target_counts)
        counts=np.array([target_counts[v] for v in universe],float)
        cur=counts/counts.sum(); cur+=1e-10; cur/=cur.sum()
        assert np.allclose(cur,q,rtol=0,atol=1e-14)
        result['kl'][name]={'support':len(universe),'current':kl.kl_divergence(p,cur),
            'genpas':kl.kl_divergence(p,q),'relative_change':0.,'max_probability_difference':float(abs(cur-q).max()),
            'validation_self':kl.kl_divergence(p,p),'reverse_current':kl.kl_divergence(cur,p),
            'validation_targets_absent_train':int(sum(v not in target_support for v in codes[val_items]))}
        arrays[name+'_support']=np.array(universe); arrays[name+'_valid']=p; arrays[name+'_current']=cur; arrays[name+'_genpas']=q
    # Exact expected length distribution under gamma=1, at uncapped / repo / project caps.
    for label,cap in [('uncapped',int(hist.max())),('repo24',24),('project50',50),('analysis_default20',20)]:
        weights=Counter(); earliest=0.; support=0
        for h in hist:
            m=min(int(h),cap); norm=m*(m+1)/2
            for L in range(1,m+1): weights[L]+=(m-L+1)/norm
            earliest+=1/norm; support+=m
        result['genpas_expected'][label]={'history':summary(list(weights),list(weights.values())),
            'position_indexed_window_support':support,'support_multiplier':support/N,
            'same_earliest_window_probability':earliest/N,'target_position_mean':float((hist+1).mean()),
            'expected_samples_if_budget_N':N,'sample_multiplier_if_budget_N':1.0}
        arrays[label+'_length']=np.array(sorted(weights)); arrays[label+'_length_expected_counts']=np.array([weights[x] for x in sorted(weights)])
    # One fixed, budget-matched diagnostic draw: input sampling only, no output training cache.
    rng=random.Random(42); sampled_h=[]
    for h in hist:
        m=min(int(h),50)
        j=rng.choices(range(m),weights=range(1,m+1),k=1)[0]
        sampled_h.append(m-j)
    result['fixed_draw_project50']={'N':len(sampled_h),'history':summary(sampled_h),
         'changed_history_rows':int((np.array(sampled_h)!=np.minimum(hist,50)).sum()),
         'same_targets':True,'same_user_pair_counts':True}
    print('Distribution summaries complete; exact official SID-symbol alignment follows.',flush=True)
    OUT.mkdir(parents=True)
    # Partial results are a diagnostic checkpoint, not a training artifact.
    (OUT/'distributions.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(OUT/'distributions.npz',**arrays)
    for cap in [20,24,50]:
        result['alignment'][str(cap)]={}
        for name,gamma in [('current',float('-inf')),('genpas',1.)]:
            tick=time.perf_counter()
            mapping,total=ad.build_target2inputs(sid_train,1.,0.,gamma,cap,use_tqdm=False)
            sums=ad.finalize_target_weights(mapping)
            supported=sum(s[-1] in sums for s in sid_valid)
            print(f'A/D cap={cap} {name}: {sum(map(len,mapping.values()))} weighted windows, supported validation={supported}',flush=True)
            A,D=ad.evaluate(sid_valid,mapping,sums,cap,20,42,'lev',use_tqdm=False)
            result['alignment'][str(cap)][name]={'A':A,'D':D,'A_over_D':A/D,'supported_validation':supported,
                'skipped_validation':len(sid_valid)-supported,'sum_weights':total,'seconds':time.perf_counter()-tick}
            print(result['alignment'][str(cap)][name],flush=True)
            del mapping,sums
            (OUT/'alignment_progress.json').write_text(json.dumps(result['alignment'],indent=2))
    assert hashlib.sha256(SID.read_bytes()).hexdigest()==sid_hash
    result['wall_seconds']=time.perf_counter()-start
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
