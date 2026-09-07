"""CPU mathematical/selection tests; real provenance and FP32 checks live in audit."""
import inspect
import math
import unittest
from unittest.mock import patch

import torch

from scripts.phase2a_preference import dpo, reverse_map, select_pair, validate, EXPECTED
from genrec.data.tca_teacher_cache import sample_key


class PreferenceTests(unittest.TestCase):
    def setUp(self):
        self.sids = [(1,257,513),(2,258,514),(3,259,515)]
        self.reverse = {s:[i+1] for i,s in enumerate(self.sids)}

    def test_reverse_mapping_canonical_offsets(self):
        self.assertEqual(reverse_map(torch.tensor([[0,0,0],[0,0,0],[1,1,1]])),
                         {(1,257,513):[1,2],(2,258,514):[3]})

    def test_collision_and_unknown_skipped(self):
        reverse = {self.sids[0]:[1,2],self.sids[1]:[3]}
        self.assertIsNone(select_pair(self.sids,[100,2,99],[-3,-1,-2],reverse))

    def test_max_teacher_margin_not_closest_model_gap(self):
        self.assertEqual(select_pair(self.sids,[5,2,0],[-3,-1,-2],self.reverse),(0,2))

    def test_teacher_margin_strict(self):
        self.assertIsNone(select_pair(self.sids,[1,1,1],[-3,-2,-1],self.reverse))

    def test_reference_inversion_strict(self):
        self.assertIsNone(select_pair(self.sids,[3,2,1],[-1,-1,-1],self.reverse))

    def test_agreeing_extrema_not_used(self):
        self.assertIsNone(select_pair(self.sids,[3,2,1],[-1,-2,-3],self.reverse))

    def test_chosen_rejected_distinct(self):
        self.assertIsNone(select_pair([self.sids[0]]*2,[2,1],[-2,-1],self.reverse))

    def test_tiebreak_inversion_magnitude(self):
        self.assertEqual(select_pair(self.sids,[3,1,1],[-3,-2,-1],self.reverse),(0,2))

    def test_tiebreak_rank_and_deterministic(self):
        for _ in range(20):
            self.assertEqual(select_pair(self.sids,[3,1,1],[-3,-1,-1],self.reverse),(0,2))

    def test_no_target_in_selection_signature(self):
        self.assertEqual(list(inspect.signature(select_pair).parameters),['sids','rewards','logp','reverse'])

    def test_sample_keys_include_history_target_and_split(self):
        base = sample_key([1,2],3,split='train')
        self.assertEqual(base,sample_key([1,2],3,split='train'))
        for h,t,s in [([2,1],3,'train'),([1,2],4,'train'),([1,2],3,'valid'),([1,2],3,'test')]:
            self.assertNotEqual(base,sample_key(h,t,split=s))

    def test_dpo_initialization_log2(self):
        for beta in [.05,.1,.2]:
            w,l = torch.tensor([-5.,-6.]),torch.tensor([-3.,-2.])
            torch.testing.assert_close(dpo(w,l,w,l,beta),torch.full((2,),math.log(2)))

    def test_gradient_direction_and_reference_detached(self):
        w,l,rw,rl = [torch.tensor(-3.,requires_grad=True) for _ in range(4)]
        dpo(w,l,rw,rl,.1).backward()
        self.assertLess(w.grad.item(),0)
        self.assertGreater(l.grad.item(),0)
        self.assertIsNone(rw.grad); self.assertIsNone(rl.grad)

    def test_extreme_logits_finite(self):
        w=torch.tensor([-1e6,1e6],requires_grad=True)
        loss=dpo(w,torch.zeros(2),torch.zeros(2),torch.zeros(2),.2).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertTrue(torch.isfinite(w.grad).all())

    def test_beta_gradient_linear_at_initialization(self):
        for beta in [.05,.1,.2]:
            w=torch.tensor(0.,requires_grad=True)
            dpo(w,torch.tensor(0.),torch.tensor(0.),torch.tensor(0.),beta).backward()
            self.assertAlmostEqual(w.grad.item(),-beta/2,places=7)

    def test_cache_integrity_rejects_tampering(self):
        a = dict(format_version=1, dataset='beauty', split='train', num_samples=1, sample_keys=['key'],
                 generator={'checkpoint_sha256':EXPECTED,'epoch':111,'beam_size':10},
                 sid={'sha256':EXPECTED,'collision_policy':'skip'},
                 pair_policy={'name':'max_teacher_margin_inversion'},
                 teacher={'checkpoint_sha256':EXPECTED,'checkpoint_path':'mock'},
                 sample_indices=torch.arange(1), has_pair=torch.tensor([True]),
                 chosen_sid=torch.tensor([self.sids[0]]), rejected_sid=torch.tensor([self.sids[1]]),
                 chosen_reward=torch.tensor([2.]), rejected_reward=torch.tensor([1.]),
                 ref_logp_chosen=torch.tensor([-3.]), ref_logp_rejected=torch.tensor([-1.]),
                 chosen_beam_rank=torch.tensor([2]), rejected_beam_rank=torch.tensor([1]))
        with patch('scripts.phase2a_preference.sha256_file', return_value=EXPECTED):
            validate(a,['key'],self.reverse)
            for key,value in [('split','valid'),('sample_keys',['wrong']),
                              ('generator',{**a['generator'],'checkpoint_sha256':'wrong'}),
                              ('sid',{**a['sid'],'sha256':'wrong'}),
                              ('chosen_reward',torch.tensor([float('nan')])),
                              ('sample_indices',torch.tensor([1]))]:
                with self.subTest(key=key), self.assertRaises(AssertionError):
                    validate({**a,key:value},['key'],self.reverse)


if __name__ == '__main__':
    unittest.main()
