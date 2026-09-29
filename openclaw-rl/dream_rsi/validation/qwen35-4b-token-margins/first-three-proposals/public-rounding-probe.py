"""Read-only public token precision probe. No training or confirmation access."""
from pathlib import Path
import json,sys,time,hashlib
p=Path(sys.argv[1]).resolve()
from dream_rsi.qwen4b_chain import validate_parent
from dream_rsi.external_judge import canonical
parent=validate_parent(p/'parent-approval.json')
rows=json.loads((p/'run-001/dev-baseline.json').read_text())+json.loads((p/'run-001/guard-baseline.json').read_text())
byid={r['task_id']:r for r in rows}
anchors=json.loads((p/'run-001/margin-anchors-1.json').read_text())
targets=[x for x in anchors if 'prior_public_regression' in x['sources']]
if len(targets)>8:raise RuntimeError('probe budget exceeded')
import torch
from transformers import Qwen3_5ForConditionalGeneration
from peft import LoraConfig,get_peft_model,set_peft_model_state_dict,get_peft_model_state_dict
from safetensors.torch import load_file
if torch.cuda.device_count()!=1:raise RuntimeError('exactly one visible GPU required')
torch.set_num_threads(4);torch.cuda.set_device(0);torch.manual_seed(20260929)
start=time.monotonic()
model=Qwen3_5ForConditionalGeneration.from_pretrained((p/'model-path.txt').read_text().strip(),
 local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
for param in model.parameters():param.requires_grad_(False)
model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,
 target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
set_peft_model_state_dict(model,load_file(Path(parent['adapter_path'])/'adapter_model.safetensors'))
h=hashlib.sha256()
for name,t in sorted(get_peft_model_state_dict(model).items()):
 h.update(name.encode());h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
assert h.hexdigest()==parent['adapter_tensor_state_sha256']
model.eval();head=model.get_output_embeddings();cache={}
def capture(module,args):
 cache['last_hidden']=args[0][0,-1].detach()
handle=head.register_forward_pre_hook(capture)
def capture_logits(module,args,output):
 cache['last_head_output']=output[0,-1].detach()
output_handle=head.register_forward_hook(capture_logits)
results=[]
for anchor in targets:
 row=byid[anchor['task_id']];pos=anchor['position']
 ids=torch.tensor([row['prompt_token_ids']],device='cuda:0',dtype=torch.long)
 with torch.inference_mode():
  result=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,
    max_new_tokens=pos+1,use_cache=True,pad_token_id=model.config.text_config.eos_token_id,
    return_dict_in_generate=True,output_scores=True)
 assert result.sequences[0,ids.shape[1]:].tolist()==row['response_token_ids'][:pos+1], 'parent prefix failed cold match'
 chosen,other=anchor['chosen_token'],anchor['competitor_token']
 score=result.scores[-1][0];hidden=cache['last_hidden'];raw=cache['last_head_output']
 # Recompute only two output projections with original frozen values. This
 # is a precision diagnostic, not a change to decoding or an alternative model.
 w=head.weight[[chosen,other]].detach()
 full_precision=(w.double()*hidden.double()).sum(dim=-1)
 if head.bias is not None:full_precision+=head.bias[[chosen,other]].detach().double()
 selected_score=float(score[chosen]);other_score=float(score[other])
 record={'task_id':anchor['task_id'],'position':pos,'chosen_token':chosen,'competitor_token':other,
  'actual_generated_chosen_token':int(result.sequences[0,-1]),'parent_prefix_reproduced':True,
  'teacher_forced_bf16_margin':anchor['teacher_forced_margin'],
  'actual_cached_decode_margin':selected_score-other_score,
  'actual_cached_decode_scores':[selected_score,other_score],
  'actual_cached_raw_head_scores':[float(raw[chosen]),float(raw[other])],
  'actual_cached_raw_head_margin':float(raw[chosen].float()-raw[other].float()),
  'same_hidden_and_weights_fp64_dot_products':full_precision.tolist(),
  'same_hidden_and_weights_fp64_contrast':float(full_precision[0]-full_precision[1])}
 results.append(record);print(canonical(record),flush=True)
handle.remove();output_handle.remove()
report={'status':'PASS','purpose':'public numerical diagnosis only; no training, no configuration change, no confirmation outcomes',
 'parent_sha256':parent['adapter_tensor_state_sha256'],'new_optimizer_steps':0,'visible_gpu_count':1,
 'observations':results,'elapsed_seconds':round(time.monotonic()-start,3)}
with (p/'evidence/public-rounding-diagnostic.json').open('x') as f:f.write(canonical(report)+'\n')
