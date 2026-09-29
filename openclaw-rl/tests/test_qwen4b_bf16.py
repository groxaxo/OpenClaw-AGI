import unittest
from dream_rsi.qwen4b_bf16_guarded import accept_public_update
class GuardedTests(unittest.TestCase):
    def test_improvement_with_guards_passes(self):
        self.assertTrue(accept_public_update(True,dict(passed_cases=4,total_cases=10),dict(passed_cases=5,total_cases=10)))
    def test_plateau_allowed_without_claiming_improvement(self):
        self.assertTrue(accept_public_update(True,dict(passed_cases=4,total_cases=10),dict(passed_cases=4,total_cases=10)))
    def test_guard_veto_overrides_gain(self):
        self.assertFalse(accept_public_update(False,dict(passed_cases=4,total_cases=10),dict(passed_cases=10,total_cases=10)))
    def test_regression_rejected(self):
        self.assertFalse(accept_public_update(True,dict(passed_cases=4,total_cases=10),dict(passed_cases=3,total_cases=10)))
    def test_changed_or_malformed_counts_rejected(self):
        for new in [dict(passed_cases=5,total_cases=11),dict(passed_cases=True,total_cases=10),dict(passed_cases=11,total_cases=10)]:
            with self.subTest(new=new),self.assertRaises(ValueError):
                accept_public_update(True,dict(passed_cases=4,total_cases=10),new)
if __name__=='__main__':unittest.main()
