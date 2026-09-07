"""Additional read-only numeric/target sanity checks on validation audit arrays."""
import json
import itertools
import numpy as np
from scripts.audit_preference_candidates import OUT, stats


def main():
    d=dict(np.load(OUT/'validation_candidates.npz'))
    n=len(d['logp']); same=[]; baseline=[]; random=[]; chosen=[]; rejected=[]
    robust_users=0; threshold=json.loads((OUT/'diagnostics.json').read_text())['beam_vs_logp']['inverted_pair_gap']['p90']
    for u in range(n):
        ids=np.flatnonzero(d['group_size'][u]==1)
        if len(ids):
            same.append(np.array_equal(ids[np.argsort(-d['max'][u,ids],kind='stable')],ids[np.argsort(-d['mass'][u,ids],kind='stable')]))
        matches=(d['generated'][u]==d['target'][u]).all(-1)
        target=np.flatnonzero(matches&(d['group_size'][u]==1))
        if len(target):
            baseline.append(1+np.sum(d['logp'][u,ids]>d['logp'][u,target[0]]))
            random.append(1/len(ids))
        pairs=[]
        for i,j in itertools.combinations(ids,2):
            dt=d['max'][u,i]-d['max'][u,j];dm=d['logp'][u,i]-d['logp'][u,j]
            if dt*dm<0:
                c,r=(i,j) if dt>0 else (j,i)
                pairs.append((abs(dm),c,r))
        robust_users+=any(x[0]>threshold for x in pairs)
        if pairs:
            _,c,r=min(pairs)
            chosen.append(bool(matches[c]));rejected.append(bool(matches[r]))
    result={'unique_raw_vs_probability_identical_order_rate':float(np.mean(same)),
        'unique_target_model_rank':{**stats(baseline),**{f'top{k}':float((np.array(baseline)<=k).mean()) for k in [1,3,5]}},
        'unique_target_random_top1_baseline':float(np.mean(random)),
        'closest_U_pair_chosen_target_count':int(sum(chosen)), 'closest_U_pair_rejected_target_count':int(sum(rejected)),
        'numeric_sensitivity_threshold':threshold,'U_disagreement_coverage_above_numeric_threshold':robust_users/n}
    (OUT/'sanity.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
