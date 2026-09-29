import unittest
from dream_rsi.qwen4b_protected_repair import projected_delta,minimal_repairs
from dream_rsi.coding_tasks import CodingTask
class ProjectionTests(unittest.TestCase):
    def test_removes_anchor_components(self):
        import torch
        d=torch.tensor([1.,2.,3.]);a=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
        out,metrics=projected_delta(d,a)
        self.assertTrue(torch.allclose(out,torch.tensor([0.,0.,3.]),atol=1e-6))
        self.assertEqual(metrics['rank'],2)
    def test_correlated_and_zero_anchors(self):
        import torch
        d=torch.tensor([2.,3.]);a=torch.tensor([[1.,0.],[2.,0.],[0.,0.]])
        out,_=projected_delta(d,a)
        self.assertTrue(torch.allclose(out,torch.tensor([0.,3.]),atol=1e-6))
    def test_random_projection(self):
        import torch
        torch.manual_seed(7);d=torch.randn(1000);a=torch.randn(8,1000)
        out,_=projected_delta(d,a)
        self.assertLess(float((a@out).abs().max()),1e-4)
        self.assertLessEqual(float(out.norm()),float(d.norm())+1e-5)
    def test_empty_basis_identity(self):
        import torch
        d=torch.tensor([1.,2.]);out,_=projected_delta(d,torch.empty(0,2))
        self.assertTrue(torch.equal(out,d))
    def test_bad_input_rejected(self):
        import torch
        for d,a in [(torch.tensor([float('nan')]),torch.ones(1,1)),(torch.ones(2),torch.ones(3,1))]:
            with self.assertRaises(ValueError):projected_delta(d,a)
class MinimalRepairTests(unittest.TestCase):
    def test_exact_edits_preserve_remaining_text(self):
        tasks=(CodingTask('dev_range','range',()),CodingTask('dev_sum','sum',()))
        source=['def solve(data):\n    k = data[-1]\n    nums = data[:-1]\n    return nums\n','def solve(data):\n    return [sum(data[i::2]) for i in range(2)] if data else []\n']
        records=[{'task_id':t.task_id,'result':{'passed':0,'total':2},'response':s} for t,s in zip(tasks,source)]
        out=minimal_repairs(tasks,records)
        self.assertIn('nums, k = data',out[0]['solution']);self.assertTrue(out[0]['solution'].endswith('    return nums\n'))
        self.assertIn('sum(data) - x for x in data',out[1]['solution'])
    def test_unexpected_program_rejected(self):
        task=CodingTask('dev_range','range',())
        with self.assertRaises(ValueError):minimal_repairs([task],[{'task_id':'dev_range','result':{'passed':0,'total':2},'response':'different'}])
if __name__=='__main__':unittest.main()
