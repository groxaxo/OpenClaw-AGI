import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from dream_rsi.coding_tasks import CodingTask
from dream_rsi.external_judge import canonical
from dream_rsi.qwen4b_chain import (validate_parent, file_sha256, expand_public_development,
                                   build_fresh_confirmation, lineage_summary)
from dream_rsi.qwen4b_repair import TRAIN, DEV

A='a'*64
B='b'*64
C='c'*64

class ChainTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.adapter=self.root/'adapter';self.adapter.mkdir()
        (self.adapter/'adapter_model.safetensors').write_bytes(b'test-only-adapter')
        config={'r':8,'lora_alpha':16,'peft_type':'LORA',
                'target_modules':['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']}
        (self.adapter/'adapter_config.json').write_text(json.dumps(config))
        self.parent={'status':'PASS','model':'Qwen/Qwen3.5-4B','cold_reproduction_passed':True,
          'adapter_file_sha256':file_sha256(self.adapter/'adapter_model.safetensors'),
          'adapter_tensor_state_sha256':A,'adapter_path':str(self.adapter)}
        self.record=self.root/'parent.json';self.save()
    def save(self):self.record.write_text(json.dumps(self.parent))
    def test_verified_parent_load(self):
        self.assertEqual(validate_parent(self.record)['successful_generation'],1)
    def test_no_parent_without_cold_reproduction(self):
        self.parent['cold_reproduction_passed']=False;self.save()
        with self.assertRaises(ValueError):validate_parent(self.record)
    def test_rejected_parent_not_accepted(self):
        self.parent['status']='NOT_OK';self.save()
        with self.assertRaises(ValueError):validate_parent(self.record)
    def test_parent_bytes_bound_to_approval(self):
        (self.adapter/'adapter_model.safetensors').write_bytes(b'changed')
        with self.assertRaises(ValueError):validate_parent(self.record)
    def test_parent_adapter_config_checked(self):
        (self.adapter/'adapter_config.json').write_text('{}')
        with self.assertRaises(ValueError):validate_parent(self.record)
    def test_retired_suite_is_deduplicated(self):
        base=[CodingTask('dev','prompt',(([1],2),))]
        result=expand_public_development(base,[{'description':'prompt','cases':[[[1],2],[[2],3]]}])
        self.assertEqual(len(result[0].cases),2)
    def test_conflicting_answers_fail_closed(self):
        base=[CodingTask('dev','prompt',(([1],2),))]
        with self.assertRaises(ValueError):expand_public_development(base,[{'description':'prompt','cases':[[[1],99]]}])
    def test_unmatched_retired_prompt_not_silently_ignored(self):
        with self.assertRaises(ValueError):expand_public_development(DEV,[])
    def test_fresh_case_inputs_exclude_public(self):
        suite=build_fresh_confirmation(926317,TRAIN,DEV,count=5)
        public={canonical(arg) for t in TRAIN+DEV for arg,_ in t.cases}
        self.assertEqual(len(suite),12)
        for row in suite:
            keys=[canonical(arg) for arg,_ in row['cases']]
            self.assertEqual(len(keys),5);self.assertEqual(len(set(keys)),5)
            self.assertFalse(public & set(keys))
        self.assertEqual(suite,build_fresh_confirmation(926317,TRAIN,DEV,count=5))
    def good(self,parent=A,candidate=B):
        return {'parent_sha256':parent,'candidate_sha256':candidate,'status':'PASS',
          'promotion_approved':True,'final_external_review_ok':True,'cold_reproduction_passed':True,
          'gate':{'passed':True,'case_gain':.1,'final_case_accuracy':.95,
                  'complete_task_wins':2,'complete_task_losses':0}}
    def test_optimizer_steps_are_not_promotions(self):
        self.assertEqual(lineage_summary(A,[])['successful_promotions'],1)
    def test_verified_child_advances_tip(self):
        s=lineage_summary(A,[self.good()]);self.assertEqual(s['successful_promotions'],2)
        self.assertEqual(s['accepted_tip_sha256'],B)
    def test_rejection_keeps_parent(self):
        s=lineage_summary(A,[{'parent_sha256':A,'status':'NOT_OK'}])
        self.assertEqual(s['successful_promotions'],1);self.assertEqual(s['accepted_tip_sha256'],A)
    def test_multiple_base_wins_not_a_lineage(self):
        with self.assertRaises(ValueError):lineage_summary(A,[self.good(),self.good(candidate=C)])
    def test_missing_cold_check_never_counts(self):
        row=self.good();row['cold_reproduction_passed']=False
        with self.assertRaises(ValueError):lineage_summary(A,[row])
    def test_same_adapter_never_counts(self):
        with self.assertRaises(ValueError):lineage_summary(A,[self.good(candidate=A)])
    def test_continuation_compares_parent_not_disabled_adapter(self):
        source=(Path(__file__).parents[1]/'dream_rsi/qwen4b_continue.py').read_text()
        ast.parse(source)
        self.assertNotIn('with model.disable_adapter()',source)
        self.assertIn('set_peft_model_state_dict(model,parent_state)',source)
        self.assertIn('initial_hash!=parent["adapter_tensor_state_sha256"]',source)
        self.assertIn('torch.cuda.device_count()!=1',source)

if __name__=='__main__':unittest.main()
