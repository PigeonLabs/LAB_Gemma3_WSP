import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from wsp.study import Journal, atomic_json, digest, gate
from wsp.circuits import oracle, serialized, modified_table
from wsp.policies import allowed_combinations, views, raw_scores, plans
from wsp.prompts import prompt
from wsp.statistics import arm_scores
from wsp.preservation import preservation_prompts


def example():
    rng=random.Random(33)
    table=[[rng.uniform(-2,2) for _ in range(5)] for _ in range(16)]
    return {'id':'test','layer':2,'table':table,'units':list(range(16)),
            'gate_norm':[1+i/10 for i in range(16)],'up_norm':[2+i/10 for i in range(16)],
            'down_norm':[.5+i/10 for i in range(16)],'oracle':oracle(table)}


class StudyTests(unittest.TestCase):
    def test_four_point_truth(self):
        a=oracle([[1.,0.,2.,0.,1.]])
        self.assertEqual(a['truth'],'A'); self.assertEqual(a['influence_B'],0.)
        b=oracle([[0.,1.,0.,2.,1.]])
        self.assertEqual(b['truth'],'B')

    def test_all_subsets_preserve_scalar_function_and_wire_truth(self):
        c=example(); before=oracle(c['table'])
        for mask in range(16):
            table=modified_table(c,[i for i in range(4) if mask&(1<<i)])
            np.testing.assert_allclose(oracle(table)['outputs'],before['outputs'],rtol=1e-12,atol=1e-14)
            self.assertEqual(oracle(serialized(table))['truth'],before['truth'])

    def test_budget_matches_actual_full_tensor_difference(self):
        torch.manual_seed(2); u=torch.randn(16,9,dtype=torch.float64); d=torch.randn(9,16,dtype=torch.float64)
        allowed,ref=allowed_combinations(u.norm(dim=1).tolist(),d.norm(dim=0).tolist())
        for ids,norm in allowed[:20]:
            expected=math.sqrt(float(((7*u[list(ids)])**2).sum()+((-.875*d[:,list(ids)])**2).sum()))
            self.assertAlmostEqual(expected,norm,places=10)
            self.assertLessEqual(abs(norm/ref-1),.05)

    def test_information_views_and_derangement(self):
        c=example(); info,_,_,perm=views(c,4101)
        self.assertEqual(set(info['S']),{'norms'}); self.assertEqual(info['R'],{})
        self.assertEqual(set(info['W']),{'table'}); self.assertEqual(set(info['M']),{'table'})
        self.assertEqual(sorted(perm),list(range(16)))
        self.assertTrue(all(i!=p for i,p in enumerate(perm)))
        np.testing.assert_allclose(raw_scores(info['M']['table'],'A'),np.array(raw_scores(info['W']['table'],'A'))[perm])

    def test_plans_have_fixed_budget_and_no_result_feedback(self):
        c=example(); first=plans(c)
        self.assertEqual(first,plans(c)); self.assertEqual(len(first),12)
        for p in first:
            self.assertEqual(len(set(p['units'])),4); self.assertEqual(p['factors'],[8.]*4)
            self.assertLessEqual(abs(p['edit_norm']/p['reference_norm']-1),.05)
        c['unrelated_observer_feedback']='must not be used'
        self.assertEqual(first,plans(c))

    def test_canonical_prompts_identical_under_gauge_edit(self):
        c=example(); edited=modified_table(c,[1,4,8,11])
        self.assertEqual(prompt(c['table'],defense='canonical'),prompt(edited,defense='canonical'))
        self.assertNotEqual(prompt(c['table']),prompt(edited))

    def test_prompts_are_blind_and_prose_not_prefilled(self):
        text=prompt(example()['table'],prose=True)
        self.assertNotIn('weight_informed',text); self.assertNotIn('target label',text)
        self.assertIn('UNKNOWN',text); self.assertNotIn('ANSWER: A',text)

    def test_journal_resume_provenance_and_immutability(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); journal=Journal(folder,{'a':1},chunk_size=2)
            for i in range(5): journal.add({'trial_id':str(i),'x':i})
            self.assertEqual(len(list(folder.glob('chunk-*.jsonl'))),3)
            resumed=Journal(folder,{'a':1},chunk_size=2)
            self.assertEqual(len(resumed.rows),5)
            resumed.add({'trial_id':'0','x':0})
            with self.assertRaises(ValueError): resumed.add({'trial_id':'0','x':9})
            with self.assertRaises(FileExistsError): Journal(folder,{'a':2})

    def test_gate_cannot_be_overridden_by_failed_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); atomic_json(root/'calibration_summary.json',{'status':'stopped_baseline_failure'})
            with patch('wsp.study.PUBLIC',root),patch('wsp.study.require_freeze',return_value={}):
                with self.assertRaises(RuntimeError): gate()

    def test_success_requires_original_correct_and_preservation(self):
        rows=[]
        for i,(base,edited,preserved) in enumerate([('A','B',True),('B','B',True),('A','B',False),('A','UNKNOWN',True)]):
            common={'circuit_id':str(i),'layer':0,'wording':0,'truth':'A','target':'B'}
            rows += [{**common,'arm':'O','answer':base,'preserved':True},
                     {**common,'arm':'W','answer':edited,'preserved':preserved}]
        scores=arm_scores(rows)
        self.assertEqual(list(scores.values()),[1.,0.,0.,0.])
        self.assertEqual(np.mean(list(scores.values())),.25)

    def test_new_preservation_suite_size(self):
        values=preservation_prompts(); self.assertEqual(len(values),32)
        self.assertEqual(len(set(values)),32)


if __name__=='__main__': unittest.main()
