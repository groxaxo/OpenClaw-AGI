import unittest
from dream_rsi.qwen4b_token_margins import first_divergence,choose_margins

class MarginTests(unittest.TestCase):
    def test_first_difference(self):
        self.assertEqual(first_divergence([2,3,4],[2,5,4]),(1,3,5))
    def test_equal_or_prefix_not_fake_contrast(self):
        self.assertIsNone(first_divergence([2,3],[2,3]))
        self.assertIsNone(first_divergence([2,3],[2]))
    def test_bad_tokens(self):
        for x in ([True],[-1],[1.2]):
            with self.assertRaises(ValueError):first_divergence(x,[1])
    def test_weakest_margin_and_real_competitor(self):
        import torch
        logits=torch.tensor([[5.,4.,0.],[7.,0.,3.],[0.,1.1,1.]])
        chosen=torch.tensor([0,0,1])
        r=choose_margins(logits,chosen,count=1)
        self.assertEqual((r[0]['position'],r[0]['chosen_token'],r[0]['competitor_token']),(2,1,2))
    def test_counterexample_included_even_not_weak(self):
        import torch
        r=choose_margins(torch.tensor([[7.,1.,2.],[5.,4.,0.]]),torch.tensor([0,0]),
                         count=1,counterexamples=([1,0],))
        self.assertEqual(len(r),2)
        self.assertEqual(r[1]['sources'],['prior_public_regression'])
        self.assertEqual(r[1]['position'],0)
    def test_duplicate_contrasts_pooled(self):
        import torch
        r=choose_margins(torch.tensor([[2.,1.]]),torch.tensor([0]),count=1,counterexamples=([1],[1]))
        self.assertEqual(len(r),1)
    def test_nonfinite_empty_shape_rejected(self):
        import torch
        with self.assertRaises(ValueError):choose_margins(torch.tensor([[float('nan'),0.]]),torch.tensor([0]))
        with self.assertRaises(ValueError):choose_margins(torch.zeros(0,2),torch.tensor([],dtype=torch.int64))
        with self.assertRaises(ValueError):choose_margins(torch.zeros(2,2),torch.tensor([0]))
    def test_invalid_vocab(self):
        import torch
        with self.assertRaises(ValueError):choose_margins(torch.zeros(1,2),torch.tensor([2]))
        with self.assertRaises(ValueError):choose_margins(torch.zeros(1,2),torch.tensor([0]),counterexamples=([4],))
    def test_contrast_backward_moves_expected_logits(self):
        import torch
        logits=torch.tensor([1.,2.,3.],requires_grad=True)
        (logits[2]-logits[1]).backward()
        self.assertEqual(logits.grad.tolist(),[0.,-1.,1.])

if __name__=='__main__':unittest.main()
