import contextlib
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dream_rsi.external_judge import (ExternalJudgeGate, JudgeError, canonical,
    parse_verdict, extract_response, digest)
from dream_rsi import Candidate, DreamRSIController, PolicyConfig


class ExternalJudgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'repo/openclaw-rl/dream_rsi').mkdir(parents=True)
        (self.root / 'repo/openclaw-rl/unsloth_qlora_trainer.py').write_text('# trainer\n')
        (self.root / 'repo/openclaw-rl/dream_rsi/README.md').write_text('Bounded experiment\n')
        self.gate = ExternalJudgeGate(self.root/'state', self.root/'repo')

    def response(self, provider, work, prompt, schema):
        return {'text': canonical({'review_id':work.name,'verdict':'OK','reason':'checked','blocking_issues':[]})}

    def review(self, deterministic=True):
        with patch.object(self.gate, '_invoke', side_effect=self.response):
            return self.gate.review('before_training', {'cycle':0}, deterministic_ok=deterministic)

    def test_both_required_judges_approve(self):
        result = self.review()
        self.assertTrue(result.approved)
        self.assertEqual(set(result.verdicts), {'muse','glm'})

    def test_deterministic_veto_overrides_both_ok(self):
        self.assertFalse(self.review(False).approved)

    def test_one_negative_is_not_bypassed(self):
        def answer(provider,work,prompt,schema):
            result = self.response(provider,work,prompt,schema)
            if provider == 'muse':
                data=json.loads(result['text']);data['verdict']='NOT_OK';data['blocking_issues']=['regression']
                result['text']=canonical(data)
            return result
        with patch.object(self.gate,'_invoke',side_effect=answer):
            result=self.gate.review('before_training',{},deterministic_ok=True)
        self.assertFalse(result.approved)
        self.assertEqual(result.verdicts['glm']['verdict'],'OK')

    def test_timeout_veto(self):
        with patch.object(self.gate,'_invoke',side_effect=JudgeError('timeout')):
            self.assertFalse(self.gate.review('before_training',{},deterministic_ok=True).approved)

    def test_stale_response_id_veto(self):
        with patch.object(self.gate,'_invoke',return_value={'text':canonical({'review_id':'old','verdict':'OK','reason':'fine','blocking_issues':[]})}):
            self.assertFalse(self.gate.review('before_training',{},deterministic_ok=True).approved)

    def test_receipt_single_use(self):
        r=self.review();self.gate.consume(r,'before_training',{'cycle':0})
        with self.assertRaises(FileExistsError): self.gate.consume(r,'before_training',{'cycle':0})

    def test_receipt_cannot_authorize_other_stage(self):
        with self.assertRaises(JudgeError):self.gate.consume(self.review(),'checkpoint_promotion',{'cycle':0})

    def test_receipt_cannot_authorize_other_evidence(self):
        with self.assertRaises(JudgeError):self.gate.consume(self.review(),'before_training',{'cycle':1})

    def test_source_change_invalidates_approval(self):
        r=self.review();(self.root/'repo/openclaw-rl/dream_rsi/new.py').write_text('changed=True\n')
        with self.assertRaises(JudgeError):self.gate.consume(r,'before_training',{'cycle':0})

    def test_receipt_edit_invalidates_approval(self):
        r=self.review();(self.root/'state'/r.review_id/'approval.json').write_text('{}')
        with self.assertRaises(JudgeError):self.gate.consume(r,'before_training',{'cycle':0})

    def test_negative_cannot_be_consumed(self):
        with self.assertRaises(JudgeError):self.gate.consume(self.review(False),'before_training',{'cycle':0})

    def test_schema_rejects_ambiguous_verdicts(self):
        for text in ['OK', '```json\n{}\n```', '{"verdict":"NOT_OK","verdict":"OK"}',
                     canonical({'review_id':'id','verdict':'OK','reason':'x','blocking_issues':['problem']})]:
            with self.subTest(text=text), self.assertRaises(JudgeError):parse_verdict(text,'id')

    def test_muse_terminal_required_and_model_checked(self):
        configured={'payload_type':'run.model.configured','payload':{'model_id':'muse-spark-1.3'}}
        final={'payload_type':'run.terminal.completed','payload':{'terminal':'completed','text':'{}'}}
        output='\n'.join(map(canonical,[configured,final]))
        self.assertEqual(extract_response('muse',output,'muse-spark-1.3'),'{}')
        with self.assertRaises(JudgeError):extract_response('muse',output,'other-model')
        with self.assertRaises(JudgeError):extract_response('muse',canonical(configured),'muse-spark-1.3')

    def test_opencode_requires_normal_stop(self):
        text={'type':'text','part':{'text':'{}'}}
        stop={'type':'step_finish','part':{'reason':'stop'}}
        self.assertEqual(extract_response('glm',canonical(text)+'\n'+canonical(stop),'glm'),'{}')
        stop['part']['reason']='length'
        with self.assertRaises(JudgeError):extract_response('glm',canonical(text)+'\n'+canonical(stop),'glm')

    def test_provider_config_is_bounded(self):
        for providers in [(),('echo',),('muse','muse')]:
            with self.assertRaises(ValueError):ExternalJudgeGate(self.root/'x',self.root/'repo',providers=providers)

    def test_nonfinite_evidence_fails_before_judges(self):
        with patch.object(self.gate,'_invoke') as invoke, self.assertRaises(ValueError):
            self.gate.review('before_training',{'loss':float('nan')},deterministic_ok=True)
        invoke.assert_not_called()

    def test_policy_promotion_can_be_vetoed(self):
        c=DreamRSIController(self.root/'policy',target_batch_size=4,evolve_interval=1,
                            holdout_pools=4,min_holdout_pairs=4,promotion_judge=lambda evidence:False)
        rows=[('s0',1,0),('s0',2,1),('s1',1,-1),('s1',2,1),('s2',1,-1),('s2',2,-1),('s3',1,-1),('s3',2,0)]
        for n in range(6):
            c.trace.append_pool(n,[Candidate(f'{n}-{i}',f'{n}-{session}',turn,float(score)) for i,(session,turn,score) in enumerate(rows)])
        challenger=PolicyConfig(version=2,exploit_fraction=1,explore_fraction=0,recover_fraction=0,depth_penalty=0)
        with patch('dream_rsi.core.mutate_policy',return_value=[challenger]):event=c.maybe_evolve(5)
        self.assertTrue(event['gate']['promote'])
        self.assertFalse(event['promoted'])
        self.assertEqual(c.policy.version,1)


class LossTests(unittest.TestCase):
    def test_kl_uses_frozen_reference(self):
        import torch
        from dream_rsi.losses import clipped_policy_loss
        current=torch.tensor([[-1.0]],requires_grad=True)
        old=current.detach().clone();ref=torch.tensor([[-2.0]])
        loss,metrics=clipped_policy_loss(current,old,ref,torch.tensor([0.]),torch.ones_like(current))
        self.assertGreater(metrics['reference_kl'],0.3)
        loss.backward();self.assertNotEqual(float(current.grad),0.)

    def test_equal_reference_has_zero_kl(self):
        import torch
        from dream_rsi.losses import clipped_policy_loss
        p=torch.tensor([[-1.,-2.]],requires_grad=True)
        loss,m=clipped_policy_loss(p,p.detach(),p.detach(),torch.tensor([1.]),torch.ones_like(p))
        self.assertEqual(m['reference_kl'],0.)
        loss.backward();self.assertTrue(torch.isfinite(p.grad).all())

    def test_invalid_loss_inputs_rejected(self):
        import torch
        from dream_rsi.losses import clipped_policy_loss
        p=torch.tensor([[-1.]])
        with self.assertRaises(ValueError):clipped_policy_loss(p,p,p,torch.tensor([1.]),torch.zeros_like(p))
        with self.assertRaises(ValueError):clipped_policy_loss(p,p,p,torch.tensor([float('inf')]),torch.ones_like(p))


class CodingTaskTests(unittest.TestCase):
    def test_disjoint_task_splits(self):
        from dream_rsi.coding_tasks import TRAIN_TASKS,HOLDOUT_TASKS
        self.assertFalse({x.task_id for x in TRAIN_TASKS}&{x.task_id for x in HOLDOUT_TASKS})
        self.assertEqual(len(TRAIN_TASKS),6);self.assertEqual(len(HOLDOUT_TASKS),6)

    def test_unsafe_code_rejected(self):
        from dream_rsi.coding_tasks import extract_code
        for code in ['import os\ndef solve(x): return os.environ','def solve(x): return open("/etc/passwd").read()',
                     'def solve(x): return x.__class__']:
            with self.assertRaises(ValueError):extract_code(code)

    def test_unpinned_docker_image_rejected(self):
        from dream_rsi.coding_tasks import grade,TRAIN_TASKS
        with self.assertRaises(ValueError):grade(TRAIN_TASKS[0],'def solve(x): return []','python:latest')


class ModelPreparationTests(unittest.TestCase):
    def test_large_embedding_stays_bf16_and_base_freezes(self):
        import torch
        from dream_rsi.model_utils import prepare_frozen_kbit_model
        class Model(torch.nn.Module):
            is_loaded_in_4bit=True
            def __init__(self):
                super().__init__()
                self.embedding=torch.nn.Embedding(1024,1024,dtype=torch.bfloat16)
                self.norm=torch.nn.LayerNorm(1024,dtype=torch.bfloat16)
            def enable_input_require_grads(self):self.input_grad_enabled=True
            def gradient_checkpointing_enable(self, **kwargs):self.checkpointing=kwargs
        model=prepare_frozen_kbit_model(Model())
        self.assertEqual(model.embedding.weight.dtype,torch.bfloat16)
        self.assertEqual(model.norm.weight.dtype,torch.float32)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertTrue(model.input_grad_enabled)
        self.assertFalse(model.checkpointing['gradient_checkpointing_kwargs']['use_reentrant'])

if __name__ == '__main__':unittest.main()
