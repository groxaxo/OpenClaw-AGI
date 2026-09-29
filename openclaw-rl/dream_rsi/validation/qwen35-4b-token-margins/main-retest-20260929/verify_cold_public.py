"""Reload fixed public checkpoints; no training, selection, or final-suite access."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

p = Path(sys.argv[1]).resolve()
prior = Path('/home/op/dream-qwen4b-margins.FCF7pJ')
from dream_rsi.external_judge import ExternalJudgeGate, digest, canonical
from dream_rsi.qwen4b_chain import validate_parent, expand_public_development, file_sha256
from dream_rsi.qwen4b_repair import DEV as INITIAL_DEV, GUARDS
from dream_rsi.coding_tasks import grade
reports = list(p.glob('live-attempts/*/run/report.json'))
if len(reports) != 1:
    raise SystemExit('Exactly one completed training rerun is required')
run = reports[0].parent
report = json.loads(reports[0].read_text())
manifest = json.loads((run / 'manifest.json').read_text())
checker = ExternalJudgeGate(p / 'cold-attestation', p / 'repo', providers=('muse',))
if digest(checker.sources()) != manifest['code_sha256']:
    raise SystemExit('Source differs from the tested training source')
parent = validate_parent(p / 'parent-approval.json')
public = json.loads((p / 'retired-parent-suite.json').read_text())
tasks = expand_public_development(INITIAL_DEV, public)
original = json.loads((prior / 'development-candidate.json').read_text())
if file_sha256(Path(original['adapter_path']) / 'adapter_model.safetensors') != original['adapter_file_sha256']:
    raise SystemExit('Original development adapter bytes changed')
import torch
from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
if torch.cuda.device_count() != 1:
    raise SystemExit('Exactly one CUDA GPU must be visible')
torch.cuda.set_device(0)
torch.set_num_threads(4)
torch.manual_seed(20260929)
start = time.monotonic()
base_path = (p / 'model-path.txt').read_text().strip()
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    base_path, local_files_only=True, trust_remote_code=False,
    dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa')
for parameter in model.parameters():
    parameter.requires_grad_(False)
model.enable_input_require_grads()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    task_type='CAUSAL_LM'))
tok = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
image = (p / 'docker-image.txt').read_text().strip()

def tensor_hash():
    h = hashlib.sha256()
    for name, value in sorted(get_peft_model_state_dict(model).items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def evaluate(suite, expected, label):
    wanted = {row['task_id']: row for row in expected}
    if set(wanted) != {task.task_id for task in suite}:
        raise RuntimeError('Public task set changed')
    rows = []
    model.eval()
    for task in suite:
        prompt = tok.apply_chat_template([{'role':'user','content':
            'Write a Python function solve(data). Return only executable Python code, no explanation and no I/O. ' + task.description}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tok(prompt, return_tensors='pt').to('cuda:0')
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False,
                max_new_tokens=manifest['max_new_tokens'], use_cache=True,
                pad_token_id=tok.eos_token_id)
        response = tok.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        sha = hashlib.sha256(response.encode()).hexdigest()
        result = grade(task, response, image)
        old = wanted[task.task_id]
        rows.append({'task_id':task.task_id, 'result':result, 'response_sha256':sha,
            'output_hash_matched':sha == old['response_sha256'], 'grading_matched':result == old['result']})
        print(label, task.task_id, result['passed'], '/', result['total'],
              'exact=', rows[-1]['output_hash_matched'], flush=True)
    return rows

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

old_run = prior / 'run-001'
old_report = json.loads((old_run / 'report.json').read_text())
old_epoch = old_report['history'][-1]
new_epoch = report['history'][-1]
last_accepted = next((x for x in reversed(report['history']) if x['proposal_accepted']), None)
new_dev = (read_jsonl(run / f"responses-DEV{last_accepted['epoch']}-scale{last_accepted['accepted_scale']}.jsonl")
           if last_accepted else json.loads((run / 'dev-baseline.json').read_text()))
specifications = [
    ('accepted_parent', Path(parent['adapter_path']), parent['adapter_tensor_state_sha256'],
     json.loads((old_run / 'dev-baseline.json').read_text()), json.loads((old_run / 'guard-baseline.json').read_text())),
    ('saved_development_candidate', Path(original['adapter_path']), original['adapter_tensor_state_sha256'],
     read_jsonl(old_run / 'responses-DEV3-scale1.0.jsonl'), old_epoch['guard_records']),
    ('fresh_training_candidate', run / f"epoch-{new_epoch['epoch']}", new_epoch['adapter_sha256'],
     new_dev, new_epoch['guard_records']),
]
results = {}
for label, folder, expected_sha, expected_dev, expected_guards in specifications:
    set_peft_model_state_dict(model, load_file(folder / 'adapter_model.safetensors'))
    if tensor_hash() != expected_sha:
        raise RuntimeError('Loaded adapter identity mismatch: ' + label)
    rows = evaluate(tasks, expected_dev, label + '/DEV')
    guards = evaluate(GUARDS, expected_guards, label + '/GUARDS')
    results[label] = {'adapter_sha256':expected_sha, 'dev':rows, 'guards':guards,
        'passed_cases':sum(r['result']['passed'] for r in rows),
        'total_cases':sum(r['result']['total'] for r in rows),
        'complete_families':sum(r['result']['passed'] == r['result']['total'] for r in rows)}
all_rows = [row for value in results.values() for row in value['dev'] + value['guards']]
exact = all(row['output_hash_matched'] and row['grading_matched'] for row in all_rows)
summary = {'status':'PASS' if exact else 'FAIL_REPRODUCTION',
    'scope':'cold public-output reproduction of fixed adapters; not model promotion',
    'source_sha256':manifest['code_sha256'], 'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
    'visible_gpu_count':torch.cuda.device_count(), 'new_training_steps':0,
    'final_confirmation_read':False, 'new_promotions':0, 'outputs_checked':len(all_rows),
    'all_recorded_hashes_reproduced':exact,
    'verifier_sha256':file_sha256(__file__), 'results':results,
    'peak_allocated_gib':torch.cuda.max_memory_allocated(0)/2**30,
    'elapsed_seconds':round(time.monotonic()-start,3)}
with (p / 'evidence/cold-public-retest.json').open('x') as f:
    f.write(canonical(summary) + '\n')
print(canonical({k:v for k,v in summary.items() if k != 'results'}), flush=True)
if not exact:
    raise SystemExit(2)
