import ast
import copy
import math
from pathlib import Path
import tempfile
import unittest
from dream_rsi.qwen4b_curriculum_c import TRAIN,DEV
from dream_rsi.qwen4b_validation import build_suite,grid_paths,oracle,load_suite,promotion_gate

class VerifiedCurriculumTests(unittest.TestCase):
    def test_training_and_dev_ids_unique_and_disjoint(self):
        a=[x.task_id for x in TRAIN];b=[x.task_id for x in DEV]
        self.assertEqual(len(a),32);self.assertEqual(len(b),8)
        self.assertEqual(len(set(a)),len(a));self.assertEqual(len(set(b)),len(b));self.assertFalse(set(a)&set(b))
    def test_reference_programs_compile(self):
        for ex in TRAIN:
            self.assertTrue(ex.cases)
            f=[x for x in ast.parse(ex.solution).body if isinstance(x,ast.FunctionDef)]
            self.assertEqual([x.name for x in f],['solve'])
    def test_known_mislabeled_grid_fixtures(self):
        self.assertEqual(grid_paths([4,3,[[1,1],[2,1]]]),2)
        self.assertEqual(grid_paths([3,4,[[0,1]]]),4)
        self.assertEqual(grid_paths([4,4,[[1,1],[2,2]]]),4)
    def test_independent_grid_oracle_matches_open_grid_combinatorics(self):
        for r in range(1,6):
            for c in range(1,6):self.assertEqual(grid_paths([r,c,[]]),math.comb(r+c-2,r-1))
    def test_known_oracle_results(self):
        for kind,arg,expected in [('rotate',[[1,2,3],-1],[2,3,1]),('window',[[1,4,2],2],[4,4]),
          ('merge',[[1,2],[2,4],[8,9]],[[1,4],[8,9]]),('product',[0,2,3],[6,0,0]),
          ('coins',[[1,3,4],6],2),('coins',[[2],3],-1),('unique','pwwkew',3),
          ('spiral',[[1,2,3],[4,5,6]],[1,2,3,6,5,4]),('subarrays',[[1,-1,0],0],3),
          ('next',[2,1,3],[3,3,-1]),('rle','aab',[['a',2],['b',1]]),('brackets','{[(])}',False)]:
            with self.subTest(kind=kind):self.assertEqual(oracle(kind,arg),expected)
    def test_suite_generation_reproducible(self):
        a=build_suite(91730261);self.assertEqual(a,build_suite(91730261))
        self.assertNotEqual(a,build_suite(21));self.assertEqual(len(a),12)
        self.assertEqual(sum(len(t['cases']) for t in a),240)
    def test_suite_hash_mismatch_rejected(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'suite.json';p.write_text(json.dumps(build_suite(5)))
            with self.assertRaises(ValueError):load_suite(p,'0'*64)
    def test_metamorphic_product_and_rotation(self):
        for n in range(1,9):
            x=list(range(1,n+1));self.assertEqual(oracle('rotate',[x,n]),x)
            self.assertEqual(oracle('product',x),[math.prod(x)//v for v in x])

class PromotionTests(unittest.TestCase):
    def rows(self,counts):return [{'task_id':str(i),'result':{'passed':v,'total':20}} for i,v in enumerate(counts)]
    def test_no_gain_is_not_a_pass(self):
        r=self.rows([20]*12);self.assertFalse(promotion_gate(r,r)[0]['passed'])
    def test_two_task_gain_can_pass(self):
        b=self.rows([20]*10+[0,0]);a=self.rows([20]*12)
        self.assertTrue(promotion_gate(b,a)[0]['passed'])
    def test_complete_task_regression_blocks(self):
        b=self.rows([20]*9+[0]*3);a=self.rows([0]+[20]*11)
        self.assertFalse(promotion_gate(b,a)[0]['passed'])
    def test_changed_case_counts_fail(self):
        a=self.rows([10]*12);b=copy.deepcopy(a);b[0]['result']['total']=30
        with self.assertRaises(ValueError):promotion_gate(a,b)
    def test_nonfinite_counts_rejected(self):
        a=self.rows([10]*12);b=copy.deepcopy(a);b[0]['result']['passed']=float('inf')
        with self.assertRaises(ValueError):promotion_gate(a,b)

if __name__=='__main__':unittest.main()

class GraderIntegrityTests(unittest.TestCase):
    def test_harmless_underscore_names_allowed(self):
        from dream_rsi.coding_tasks import extract_code
        for code in ['def solve(data):\n    return [0 for _ in data]\n',
                     'def solve(data):\n    _count=len(data)\n    return _count\n']:
            self.assertEqual(extract_code(code),code.strip())
    def test_dunder_names_and_private_imports_still_blocked(self):
        from dream_rsi.coding_tasks import extract_code
        for code in ['def solve(data): return __builtins__',
                     'from collections import _sys\ndef solve(data): return data',
                     'from math import __dict__\ndef solve(data): return data']:
            with self.assertRaises(ValueError):extract_code(code)
    def test_candidate_harness_has_no_gold_answers(self):
        from dream_rsi.coding_tasks import HARNESS
        self.assertNotIn('expected',HARNESS)
        self.assertNotIn('obj["cases"]',HARNESS)
        self.assertIn('obj["inputs"]',HARNESS)
    def test_forged_pass_flags_cannot_promote(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from dream_rsi.coding_tasks import CodingTask,grade
        fake=SimpleNamespace(returncode=0,stdout='{"passed":[true]}')
        with patch('dream_rsi.coding_tasks.subprocess.run',return_value=fake):
            r=grade(CodingTask('test','x',(([1],999),)),'def solve(data): return 0','python@sha256:'+'a'*64)
        self.assertEqual(r['passed'],0)
    def test_wrong_values_do_not_pass(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from dream_rsi.coding_tasks import CodingTask,grade
        fake=SimpleNamespace(returncode=0,stdout='{"results":[{"ok":true,"value":0}]}')
        with patch('dream_rsi.coding_tasks.subprocess.run',return_value=fake):
            r=grade(CodingTask('test','x',(([1],999),)),'def solve(data): return 0','python@sha256:'+'a'*64)
        self.assertEqual(r['passed'],0)
