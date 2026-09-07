import ast
import inspect
import math
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader
from genrec.trainers import preference_loss as module
from genrec.trainers.preference_loss import PreferenceCache, preference_loss, dpo_values, token_logp


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.z=torch.nn.Parameter(torch.randn(3,769)*.01)
        self.calls=[]

    def forward(self,input_ids,attention_mask,labels):
        self.calls.append(len(labels))
        return self.z.sum()*0,self.z.unsqueeze(0).expand(len(labels),-1,-1)


class DPOTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.model=Toy()
        self.x=torch.ones(4,6,dtype=torch.long)
        self.p={'has_pair':torch.tensor([True,False,True,False]),
                'chosen_sid':torch.tensor([[1,257,513]]*4),
                'rejected_sid':torch.tensor([[2,258,514]]*4),
                'ref_logp_chosen':torch.full((4,),-4.),'ref_logp_rejected':torch.full((4,),-3.)}

    def test_metadata_fail_loud(self):
        a={'format_version':1,'dataset':'beauty','split':'train','num_samples':131413,
           'generator':{'checkpoint_sha256':module.PHASE1_SHA,'epoch':111,'beam_size':10},
           'teacher':{'checkpoint_sha256':module.TEACHER_SHA},'sid':{'sha256':'sid','collision_policy':'skip'},
           'pair_policy':{'name':'max_teacher_margin_inversion'}}
        PreferenceCache.validate_metadata(a,'sid')
        for field,value in [('dataset','sports'),('split','valid'),('num_samples',1),('format_version',2)]:
            with self.assertRaises(ValueError):
                PreferenceCache.validate_metadata({**a,field:value},'sid')
        for field in ['generator','teacher','sid','pair_policy']:
            with self.assertRaises(ValueError):
                PreferenceCache.validate_metadata({**a,field:{}},'sid')
        with patch.object(module,'sha256_file',return_value='wrong'),self.assertRaises(ValueError):
            PreferenceCache.load('mock',None,None,'mock')

    def cache(self):
        return PreferenceCache({**self.p,'sample_keys':['dup','b','dup','d']})

    def test_shuffled_dataloader_index_retrieval(self):
        cache=self.cache()
        for ids in DataLoader(list(range(4)),batch_size=3,shuffle=True,generator=torch.Generator().manual_seed(4)):
            rows=cache.batch(ids,[cache.a['sample_keys'][i] for i in ids],'cpu')
            self.assertTrue(torch.equal(rows['chosen_sid'],self.p['chosen_sid'][ids]))
            self.assertTrue(torch.equal(rows['has_pair'],self.p['has_pair'][ids]))

    def test_key_mismatch(self):
        with self.assertRaises(ValueError):
            self.cache().batch(torch.tensor([0]),['wrong'],'cpu')

    def test_no_pair_base_only_no_forward(self):
        self.p['has_pair'].fill_(False)
        loss,_=preference_loss(self.model,self.x,self.x,self.p)
        base=self.model.z.square().mean()
        self.assertEqual(float(base+.1*loss),float(base))
        self.assertEqual(self.model.calls,[])

    def test_pairs_and_full_batch_reduction(self):
        loss,d=preference_loss(self.model,self.x,self.x,self.p,recompute=False)
        z=self.model.z.unsqueeze(0)
        w=token_logp(z,self.p['chosen_sid'][:1]); l=token_logp(z,self.p['rejected_sid'][:1])
        expected=dpo_values(w,l,torch.tensor([-4.]),torch.tensor([-3.]))[0]*2/4
        torch.testing.assert_close(loss,expected)
        self.assertEqual(d['pair_coverage_in_batch'],.5)
        self.assertEqual(sum(self.model.calls),4) # not 8: no-pair rows excluded
        base=self.model.z.square().mean()
        self.assertGreater(float(base+.1*loss),float(base))

    def test_formula_beta(self):
        loss=dpo_values(torch.tensor(-4.),torch.tensor(-3.),torch.tensor(-5.),torch.tensor(-3.),.1)
        self.assertAlmostEqual(float(loss),math.log1p(math.exp(-.1)),places=6)

    def test_log2(self):
        w,l=torch.tensor(-5.),torch.tensor(-3.)
        self.assertAlmostEqual(float(dpo_values(w,l,w,l)),math.log(2),places=6)

    def test_direction_and_detached_reference(self):
        w,l,rw,rl=[torch.tensor(-2.,requires_grad=True) for _ in range(4)]
        dpo_values(w,l,rw,rl).backward()
        self.assertLess(w.grad.item(),0); self.assertGreater(l.grad.item(),0)
        self.assertIsNone(rw.grad); self.assertIsNone(rl.grad)

    def test_cache_reference_detached(self):
        c=self.cache(); c.a['ref_logp_chosen'].requires_grad_()
        self.assertFalse(c.batch(torch.tensor([0]),['dup'],'cpu')['ref_logp_chosen'].requires_grad)

    def test_three_positions_full769_denominator(self):
        z=torch.zeros(1,3,769,requires_grad=True)
        y=torch.tensor([[1,257,513]])
        lp=token_logp(z,y)
        self.assertAlmostEqual(float(lp),-3*math.log(769),places=5)
        lp.backward()
        self.assertTrue((z.grad[0,:,0]<0).all())
        self.assertLess(z.grad[0,0,700].item(),0) # wrong level remains in denominator
        self.assertTrue((z.grad[0,torch.arange(3),y[0]]>0).all())

    def test_microbatch_loss_and_grad_equivalent(self):
        outcomes=[]
        for size,recompute in [(1,False),(32,False),(1,True),(32,True)]:
            loss,_=preference_loss(self.model,self.x,self.x,self.p,microbatch_size=size,recompute=recompute)
            grad=torch.autograd.grad(loss,self.model.z)[0]
            outcomes.append((loss.detach(),grad))
        for loss,grad in outcomes[1:]:
            torch.testing.assert_close(loss,outcomes[0][0])
            torch.testing.assert_close(grad,outcomes[0][1])

    def test_ratio_fixed_deterministic(self):
        c=self.cache(); ids=torch.arange(4); keys=c.a['sample_keys']
        one=c.batch(ids,keys,'cpu',1.)
        self.assertTrue(torch.equal(one['has_pair'],self.p['has_pair']))
        self.assertFalse(c.batch(ids,keys,'cpu',0.)['has_pair'].any())
        self.assertTrue(torch.equal(c.batch(ids,keys,'cpu',.5)['has_pair'],c.batch(ids,keys,'cpu',.5)['has_pair']))

    def test_checkpoint_preserves_dropout_rng(self):
        class DropoutToy(Toy):
            def forward(self,input_ids,attention_mask,labels):
                _,z=super().forward(input_ids,attention_mask,labels)
                return z.sum()*0,torch.nn.functional.dropout(z,p=.2,training=True)
        model=DropoutToy()
        values=[]
        for recompute in [False,True]:
            torch.manual_seed(7)
            loss,_=preference_loss(model,self.x,self.x,self.p,microbatch_size=2,recompute=recompute)
            values.append((loss.detach(),torch.autograd.grad(loss,model.z)[0]))
        torch.testing.assert_close(values[0][0],values[1][0])
        torch.testing.assert_close(values[0][1],values[1][1])

    def test_finite_extremes(self):
        w=torch.tensor([-1e6,1e6],requires_grad=True)
        loss=dpo_values(w,torch.zeros(2),torch.zeros(2),torch.zeros(2)).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertTrue(torch.isfinite(w.grad).all())

    def test_no_runtime_teacher_reference_generation(self):
        source=inspect.getsource(module)
        self.assertNotIn('load_beauty_teacher',source)
        self.assertNotIn('SASRecTeacher',source)
        self.assertNotIn('Tiger(',source)
        self.assertNotIn('.generate(',source)
        tree=ast.parse(Path('genrec/trainers/tiger_trainer.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='Tiger']
        self.assertEqual(len(calls),1)

    def test_eval_never_reads_preference(self):
        source=Path('genrec/trainers/tiger_trainer.py').read_text()
        tree=ast.parse(source)
        evaluate=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
        text=ast.get_source_segment(source,evaluate)
        for word in ['preference_cache','ref_logp','SASRec','preference_loss']:
            self.assertNotIn(word,text)

    def test_strict_initialization(self):
        with patch.object(module,'sha256_file',return_value=module.PHASE1_SHA),patch.object(
                module.torch,'load',return_value=self.model.state_dict()),patch.object(
                self.model,'load_state_dict',wraps=self.model.load_state_dict) as load:
            module.initialize_policy(self.model,'fake')
            self.assertTrue(load.call_args.kwargs['strict'])

    def test_bad_golden_stops_and_restores_mode(self):
        self.model.train()
        b={'input_ids':self.x,'attention_mask':self.x}
        with self.assertRaises(ValueError):
            module.golden_test(self.model,b,self.p,torch.device('cpu'))
        self.assertTrue(self.model.training)


if __name__=='__main__':
    unittest.main()
