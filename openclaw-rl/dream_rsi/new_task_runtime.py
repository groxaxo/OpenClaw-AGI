"""Shared single-GPU runtime for new-task adaptation and locked cold checks."""
from __future__ import annotations
import hashlib, json, os, random
from pathlib import Path
from .coding_tasks import grade
from .external_judge import canonical


def state_hash(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def counts(rows):
    if not rows: raise ValueError('empty evaluation')
    if len({r['task_id'] for r in rows})!=len(rows): raise ValueError('duplicate evaluation task')
    passed=sum(r['result']['passed'] for r in rows); total=sum(r['result']['total'] for r in rows)
    return {'passed_cases':passed,'total_cases':total,'case_accuracy':passed/total,
            'complete_tasks':sum(r['result']['passed']==r['result']['total'] for r in rows),'task_count':len(rows)}


def preserved(before,after):
    old={r['task_id']:r['result'] for r in before}; new={r['task_id']:r['result'] for r in after}
    if set(old)!=set(new): return False
    return all(new[k]['total']==v['total'] and new[k]['passed']>=v['passed'] for k,v in old.items())


def paired(before,after):
    old={r['task_id']:r['result'] for r in before};new={r['task_id']:r['result'] for r in after}
    if set(old)!=set(new): raise ValueError('paired task mismatch')
    rows=[]
    for key in sorted(old):
        a,b=old[key],new[key]
        if a['total']!=b['total']: raise ValueError('paired case-count mismatch')
        rows.append({'task_id':key,'before_passed':a['passed'],'after_passed':b['passed'],'total':a['total'],
                     'newly_complete':a['passed']<a['total'] and b['passed']==b['total'],
                     'lost_complete':a['passed']==a['total'] and b['passed']<b['total']})
    return {'before':counts(before),'after':counts(after),
            'case_gain':counts(after)['case_accuracy']-counts(before)['case_accuracy'],
            'newly_complete':sum(r['newly_complete'] for r in rows),'lost_complete':sum(r['lost_complete'] for r in rows),'tasks':rows}


class Runtime:
    def __init__(self, model_path, max_new_tokens, image, seed=20260929):
        import torch
        from transformers import Qwen3_5ForConditionalGeneration,AutoTokenizer
        from peft import get_peft_model,LoraConfig
        if torch.cuda.device_count()!=1: raise RuntimeError('Exactly one visible GPU required')
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8': raise RuntimeError('required deterministic CUDA workspace is missing')
        if '@sha256:' not in image: raise ValueError('digest-pinned sandbox required')
        torch.cuda.set_device(0); torch.set_num_threads(4);random.seed(seed);torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if torch.cuda.mem_get_info(0)[0]<12*2**30: raise RuntimeError('selected GPU lacks 12 GiB headroom')
        model_path=Path(model_path).resolve();cfg=json.loads((model_path/'config.json').read_text())
        if cfg.get('model_type')!='qwen3_5' or cfg['text_config'].get('hidden_size')!=2560 or model_path.name!='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a':
            raise ValueError('expected pinned Qwen3.5-4B BF16 snapshot')
        model=Qwen3_5ForConditionalGeneration.from_pretrained(str(model_path),local_files_only=True,trust_remote_code=False,
                dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
        for p in model.parameters():p.requires_grad_(False)
        model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.0,
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
        self.tokenizer=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True)
        self.params=[p for p in self.model.parameters() if p.requires_grad]
        self.max_new_tokens,self.image=max_new_tokens,image
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0)

    def snapshot(self):
        from peft import get_peft_model_state_dict
        return {k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(self.model).items()}

    def restore(self,state,expected=None):
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(self.model,state)
        actual=state_hash(self.snapshot())
        if expected is not None and actual!=expected: raise RuntimeError('adapter tensor identity mismatch')
        return actual

    def load(self,path,expected):
        from safetensors.torch import load_file
        return self.restore(load_file(Path(path)/'adapter_model.safetensors'),expected)

    def prompt_ids(self,description):
        text=self.tokenizer.apply_chat_template([{'role':'user','content':'Write a Python function solve(data). Return only executable Python code, no explanation and no I/O. '+description}],
              tokenize=False,add_generation_prompt=True,enable_thinking=False)
        return self.tokenizer(text,add_special_tokens=False)['input_ids']

    def evaluate(self,tasks,label,output=None):
        import torch
        rows=[];self.model.eval()
        for task in tasks:
            pids=self.prompt_ids(task.description)
            if len(pids)+self.max_new_tokens>1536: raise RuntimeError('evaluation context budget')
            ids=torch.tensor([pids],device='cuda:0')
            with torch.inference_mode():
                out=self.model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,
                        max_new_tokens=self.max_new_tokens,use_cache=True,pad_token_id=self.tokenizer.eos_token_id)
            aids=out[0,len(pids):].tolist(); response=self.tokenizer.decode(aids,skip_special_tokens=True)
            row={'task_id':task.task_id,'response':response,'response_sha256':hashlib.sha256(response.encode()).hexdigest(),
                 'prompt_token_ids':pids,'response_token_ids':aids,'result':grade(task,response,self.image),
                 'generated_tokens':len(aids),'hit_token_limit':len(aids)>=self.max_new_tokens}
            rows.append(row)
            print(label,task.task_id,row['result']['passed'],'/',row['result']['total'],flush=True)
        if output:
            with Path(output).open('x') as f:f.write(canonical(rows)+'\n')
        return rows
