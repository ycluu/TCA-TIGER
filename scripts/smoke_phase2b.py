"""Golden tests and exactly five disposable optimization steps; no held-out data."""
import json
import time
from pathlib import Path
import torch
from genrec.data.amazon import AmazonSeqDataset
from genrec.trainers.tiger_trainer import CollateFn
from genrec.trainers.tca_loss import TCATeacherCache, select_training_loss, sha256_file
from genrec.trainers.preference_loss import PreferenceCache, initialize_policy, golden_test, preference_loss, PHASE1_SHA
from genrec.models.tiger import Tiger
from scripts.audit_tiger_generation import MODEL_CONFIG

ROOT=Path('out/tiger/amazon/beauty')
OUT=ROOT/'preference/phase2b_smoke'
CP=ROOT/'rqkmeans/tiger_tca_full_vocab_ndcg_select/best_model.pt'
SID=ROOT/'rqkmeans/semantic_ids.pt'


def base_forward(model,batch,teacher,amp=False):
    x,mask,y=[batch[k].cuda() for k in ['input_ids','attention_mask','labels']]
    with torch.amp.autocast('cuda',enabled=amp,dtype=torch.float16):
        original,z=model(x,mask,y)
    return select_training_loss(original,objective='tca_full_vocab',logits=z,labels=y,
                                teacher_probs=teacher,alpha=.1,temperature=1.)


def main():
    if OUT.exists():
        raise FileExistsError(f'Refusing smoke artifact overwrite: {OUT}')
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    ds=AmazonSeqDataset(root='dataset/amazon',split='beauty',train_test_split='train',max_seq_len=50,
        add_disambiguation=False,semantic_id_path=str(SID),target_n_layers=3)
    cache=PreferenceCache.load(ROOT/'preference/train_preference_pairs.pt',ds,SID,CP)
    tca=TCATeacherCache.load_and_validate(ROOT/'tca/tca_teacher_cache.pt',dataset='beauty',num_samples=len(ds),
        num_items=12101,sid_layers=3,codebook_size=256,sid_artifact_path=SID)
    tca.validate_dataset(ds)
    collate=CollateFn(256,3,3,150)
    model=Tiger(MODEL_CONFIG).cuda()
    initialize_policy(model,CP)
    model.eval()
    before=torch.load(ROOT/'preference/phase2b_before.pt',weights_only=True)
    b=collate([ds[i] for i in before['indices']])
    teacher=tca.batch(b['sample_indices'],b['sample_keys'],'cuda')
    preservation={}
    with torch.no_grad():
        for amp in [False,True]:
            now=float(base_forward(model,b,teacher,amp))
            preservation[str(amp)]={'before':before[str(amp)],'after':now,'difference':now-before[str(amp)]}
            assert abs(now-before[str(amp)])<1e-7
    pair=cache.batch(b['sample_indices'],b['sample_keys'],'cuda')
    golden,diag=golden_test(model,b,pair,torch.device('cuda'))
    print('Preservation/golden PASS',preservation,golden,flush=True)
    params=list(model.parameters())
    norms=[]
    prior=json.loads((ROOT/'preference/scale_audit.json').read_text())
    for row in [r for r in prior['rows'] if r['beta']==.1]:
        b=collate([ds[i] for i in row['indices']])
        teacher=tca.batch(b['sample_indices'],b['sample_keys'],'cuda')
        pair=cache.batch(b['sample_indices'],b['sample_keys'],'cuda')
        gb=torch.autograd.grad(base_forward(model,b,teacher),params)
        pref,_=preference_loss(model,b['input_ids'].cuda(),b['attention_mask'].cuda(),pair,microbatch_size=32)
        gd=torch.autograd.grad(pref,params)
        nb=sum(g.double().square().sum() for g in gb).sqrt()
        nd=sum(g.double().square().sum() for g in gd).sqrt()
        cos=sum((g.double()*h.double()).sum() for g,h in zip(gb,gd))/(nb*nd)
        assert float(.1*nd/nb)<.1
        norms.append({'tca':float(nb),'dpo':float(nd),'weighted_ratio':float(.1*nd/nb),'cosine':float(cos)})
    del gb,gd
    # Actual model pair: derivative of its relative margin along -grad(DPO) is positive.
    one={k:v[:1] for k,v in pair.items()}
    from genrec.trainers.preference_loss import token_logp,dpo_values
    x,mask=b['input_ids'][:1].cuda(),b['attention_mask'][:1].cuda()
    _,zw=model(x,mask,one['chosen_sid']); _,zl=model(x,mask,one['rejected_sid'])
    w,l=token_logp(zw,one['chosen_sid']),token_logp(zl,one['rejected_sid'])
    delta=w-l
    real_loss=dpo_values(w,l,one['ref_logp_chosen'],one['ref_logp_rejected']).mean()
    gradient_delta=torch.autograd.grad(delta.sum(),params,retain_graph=True)
    gradient_dpo=torch.autograd.grad(real_loss,params)
    direction=-sum((g.double()*h.double()).sum() for g,h in zip(gradient_delta,gradient_dpo))
    assert direction>0
    del gradient_delta,gradient_dpo
    # Fresh initialization after diagnostic gradients, before any optimizer step.
    initialize_policy(model,CP)
    ids=torch.randperm(len(ds),generator=torch.Generator().manual_seed(42))[:256].tolist()
    b=collate([ds[i] for i in ids])
    pair=cache.batch(b['sample_indices'],b['sample_keys'],'cuda')
    teacher=tca.batch(b['sample_indices'],b['sample_keys'],'cuda')
    x,mask=b['input_ids'].cuda(),b['attention_mask'].cuda()
    def probe():
        model.eval()
        with torch.no_grad():
            loss,d=preference_loss(model,x,mask,pair,microbatch_size=32)
        return {'dpo':float(loss),**d}
    initial=probe()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=0.)
    scaler=torch.amp.GradScaler('cuda')
    steps=[]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for step in range(5):
        model.train(); optimizer.zero_grad()
        tick=time.perf_counter()
        base=base_forward(model,b,teacher,amp=True)
        with torch.amp.autocast('cuda',dtype=torch.float16):
            pref,d=preference_loss(model,x,mask,pair,beta=.1,microbatch_size=32)
        loss=base+.1*pref
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        scaler.step(optimizer); scaler.update()
        torch.cuda.synchronize()
        steps.append({'step':step+1,'total':float(loss.detach()),'tca':float(base.detach()),
            'dpo':float(pref.detach()),'seconds':time.perf_counter()-tick,**d})
        print(steps[-1],flush=True)
    peak_alloc=torch.cuda.max_memory_allocated()/2**20
    peak_reserved=torch.cuda.max_memory_reserved()/2**20
    final=probe()
    assert final['mean_policy_preference_margin']>initial['mean_policy_preference_margin'], (initial,final)
    OUT.mkdir(parents=True)
    torch.save(model.state_dict(),OUT/'smoke_only.pt')
    saved={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(torch.load(OUT/'smoke_only.pt',weights_only=True,map_location='cpu'),strict=True)
    assert all(torch.equal(v.cpu(),saved[k]) for k,v in model.state_dict().items())
    reloaded=probe()
    assert reloaded==final
    assert sha256_file(CP)==PHASE1_SHA
    result={'preservation':preservation,'golden_dpo':golden,'golden_diag':diag,'norms':norms,
        'real_pair_margin_directional_derivative':float(direction),'steps':steps,'initial':initial,'final':final,
        'batch_size':256,'preference_microbatch_size':32,'peak_allocated_mib':peak_alloc,
        'peak_reserved_mib':peak_reserved,'strict_reload':True,'phase1_hash_unchanged':True}
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
