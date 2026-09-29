import unittest
from types import SimpleNamespace
from dream_rsi.coding_tasks import CodingTask
from dream_rsi.qwen4b_replay_verification import verification_tasks,make_target,complete_tasks_preserved

class ReplayVerificationTests(unittest.TestCase):
 def test_verbatim_fenced_replay(self):
  raw='```python\ndef solve(x):\n    return x\n```\n'
  self.assertEqual(make_target(raw,True,'unused'),raw)
 def test_verbatim_whitespace(self):
  raw='def solve(x): return x\n\n'
  self.assertEqual(make_target(raw,True,'unused'),raw)
 def test_repair_keeps_fenced_style(self):
  self.assertTrue(make_target('```python\nwrong',False,'def solve(x): return x').startswith('```python\n'))
 def test_alias_groups_inherit_public_verification(self):
  code='def solve(x): return x+1'
  a=SimpleNamespace(task_id='a',description='first wording',solution=code,cases=((1,2),))
  b=SimpleNamespace(task_id='b',description='second wording',solution=code,cases=((1,2),))
  dev=CodingTask('dev','second wording',((3,4),(5,6)))
  result=verification_tasks([a,b],[dev])
  self.assertEqual(len(result['a'].cases),3)
  self.assertEqual(result['a'].cases,result['b'].cases)
 def test_other_reference_program_not_mixed(self):
  a=SimpleNamespace(task_id='a',description='inc',solution='def solve(x): return x+1',cases=((1,2),))
  b=SimpleNamespace(task_id='b',description='dec',solution='def solve(x): return x-1',cases=((1,0),))
  tasks=verification_tasks([a,b],[CodingTask('dev','dec',((3,2),))])
  self.assertEqual(len(tasks['a'].cases),1)
  self.assertEqual(len(tasks['b'].cases),2)
 def test_complete_task_loss_rejected_even_with_other_gain(self):
  before=[dict(task_id='a',result=dict(passed=2,total=2)),dict(task_id='b',result=dict(passed=0,total=4))]
  after=[dict(task_id='a',result=dict(passed=1,total=2)),dict(task_id='b',result=dict(passed=4,total=4))]
  self.assertFalse(complete_tasks_preserved(before,after))
 def test_complete_task_retention(self):
  rows=[dict(task_id='a',result=dict(passed=2,total=2))]
  self.assertTrue(complete_tasks_preserved(rows,rows))
 def test_task_set_changes_rejected(self):
  with self.assertRaises(ValueError):complete_tasks_preserved([dict(task_id='a',result=dict(passed=1,total=1))],[])
 def test_empty_success_not_replayed(self):
  with self.assertRaises(ValueError):make_target('',True,'def solve(x): return x')
if __name__=='__main__':unittest.main()
