"""Single-GPU continuation from an approved Qwen3.5-4B BF16 LoRA checkpoint.

Only public TRAIN/DEV may steer training. The pre-registered sealed suite is
evaluated for accepted parent and candidate only after the DEV-selected checkpoint is
fixed. Its result cannot trigger retries or hyperparameter changes within this
run. Muse max is a fail-closed reviewer before every optimizer step and before
promotion.
"""
from __future__ import annotations
import argparse
import ast
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading
import time

from .coding_tasks import CodingTask, extract_code, grade
from .external_judge import ExternalJudgeGate, JudgeError, canonical, digest
from .qwen4b_repair import TRAIN, DEV as INITIAL_DEV, GUARDS, choose_target
from .qwen4b_protected_repair import minimal_repairs, projected_delta
from .qwen4b_chain import validate_parent, expand_public_development, file_sha256, development_ready
from .qwen4b_validation import load_suite, promotion_gate
from .qwen4b_replay_verification import verification_tasks, make_target, complete_tasks_preserved
from .qwen4b_token_margins import choose_margins

def accept_public_update(guards_passed, old, new):
    """Never trade public guard regressions for aggregate DEV improvements."""
    for value in (old,new):
        for key in ("passed_cases","total_cases"):
            if type(value.get(key)) is not int:raise ValueError("invalid public result")
        if not 0<=value["passed_cases"]<=value["total_cases"] or value["total_cases"]<=0:
            raise ValueError("invalid public result counts")
    if old["total_cases"]!=new["total_cases"]:raise ValueError("public case count changed")
    return guards_passed is True and new["passed_cases"]>=old["passed_cases"]

def args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path",required=True)
    p.add_argument("--parent-approval",required=True)
    p.add_argument("--public-counterexamples",required=True)
    p.add_argument("--public-counterexamples-sha256",required=True)
    p.add_argument("--retired-parent-suite",required=True)
    p.add_argument("--retired-parent-sha256",required=True)
    p.add_argument("--sealed-final",required=True)
    p.add_argument("--sealed-sha256",required=True)
    p.add_argument("--docker-image",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--test-report",required=True)
    p.add_argument("--max-epochs",type=int,default=6)
    p.add_argument("--min-epochs",type=int,default=2)
    p.add_argument("--learning-rate",type=float,default=5e-5)
    p.add_argument("--max-new-tokens",type=int,default=512)
    p.add_argument("--judge-timeout",type=int,default=300)
    p.add_argument("--max-wall-seconds",type=int,default=2400)
    p.add_argument("--seed",type=int,default=20260929)
    a=p.parse_args()
    if not 1<=a.min_epochs<=a.max_epochs<=6:p.error("epochs require 1<=min<=max<=6")
    if not 1e-6<=a.learning_rate<=5e-4:p.error("learning rate outside bounded range")
    if not 96<=a.max_new_tokens<=512:p.error("max new tokens must be 96..512")
    if "@sha256:" not in a.docker_image:p.error("evaluator image must be digest-pinned")
    if not 120<=a.max_wall_seconds<=3600:p.error("wall budget must be 120..3600")
    if not 1<=a.judge_timeout<=600:p.error("judge timeout must be 1..600")
    return a

def main():
    a=args()
    import torch
    import torch.nn.functional as F
    from transformers import Qwen3_5ForConditionalGeneration,AutoTokenizer,BitsAndBytesConfig
    from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
    from safetensors.torch import load_file
    from .model_utils import prepare_frozen_kbit_model

    if torch.cuda.device_count()!=1:
        raise RuntimeError("single-GPU runner requires exactly one visible CUDA GPU")
    torch.cuda.set_device(0); torch.set_num_threads(4)
    repo=Path(__file__).resolve().parents[2]
    out=Path(a.output).resolve(); out.mkdir(parents=True,exist_ok=False,mode=0o700)
    start=time.monotonic(); stopped=threading.Event()
    def event(stage, **data):
        row={"stage":stage,"elapsed_seconds":round(time.monotonic()-start,3),**data}
        with (out/"events.jsonl").open("a") as f:f.write(canonical(row)+"\n")
        print(canonical(row),flush=True)
    event("started",visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    def watchdog():
        while not stopped.wait(1):
            if time.monotonic()-start>a.max_wall_seconds or shutil.disk_usage(out).free<4*2**30:
                try:(out/"budget-failure.txt").write_text("wall or disk budget exceeded\n")
                finally:os._exit(124)
    threading.Thread(target=watchdog,daemon=True).start()
    def write(name,value):
        with (out/name).open("x") as f:f.write(canonical(value)+"\n")

    sealed=Path(a.sealed_final).resolve()
    if sealed.with_suffix(sealed.suffix+".consumed.json").exists():raise RuntimeError("confirmation already consumed")
    sealed_bytes=sealed.read_bytes()
    if hashlib.sha256(sealed_bytes).hexdigest()!=a.sealed_sha256:
        raise RuntimeError("sealed suite digest mismatch")
    # Check the digest now; construct no final tasks and compute no final
    # outcomes until the candidate checkpoint has been fixed using DEV.
    final_task_count=12
    parent=validate_parent(a.parent_approval)
    counterexample_path=Path(a.public_counterexamples).resolve()
    if file_sha256(counterexample_path)!=a.public_counterexamples_sha256:
        raise RuntimeError("public counterexamples hash mismatch")
    historical_rows=json.loads(counterexample_path.read_text())
    if not isinstance(historical_rows,list):raise RuntimeError("malformed public counterexamples")
    for row in historical_rows:
        if set(row)!={"task_id","response","source_sha256"} or not isinstance(row["response"],str):
            raise RuntimeError("counterexample schema mismatch")
    retired=Path(a.retired_parent_suite)
    if file_sha256(retired)!=a.retired_parent_sha256:
        raise RuntimeError("retired parent suite hash mismatch")
    DEV=expand_public_development(INITIAL_DEV,json.loads(retired.read_text()))
    ids=[{x.task_id for x in TRAIN},{x.task_id for x in DEV}]
    if ids[0]&ids[1]:raise RuntimeError("training/development ID overlap")

    judge=ExternalJudgeGate(out/"judges",repo,providers=("muse",),timeout_s=a.judge_timeout)
    source_hash=digest(judge.sources())
    tests=json.loads(Path(a.test_report).read_text())
    if not tests.get("passed") or tests.get("code_sha256")!=source_hash:raise RuntimeError("stale tests")
    model_path=Path(a.model_path).resolve(); cfg=json.loads((model_path/"config.json").read_text());tc=cfg["text_config"]
    if cfg.get("model_type")!="qwen3_5" or tc.get("hidden_size")!=2560 or tc.get("num_hidden_layers")!=32:
        raise RuntimeError("expected Qwen3.5-4B")
    manifest={"backend":"single_gpu_qwen35_4b_token_margin_repair","model":"Qwen/Qwen3.5-4B",
        "revision":model_path.name,"visible_gpu_count":1,"train_examples":2,
        "dev_tasks":len(DEV),"guard_tasks":len(GUARDS),"sealed_tasks":final_task_count,"sealed_sha256":a.sealed_sha256,
        "learning_rate":a.learning_rate,"min_epochs":a.min_epochs,"max_epochs":a.max_epochs,
        "max_new_tokens":a.max_new_tokens,"code_sha256":source_hash,"tests":tests,
        "docker_image":a.docker_image,
        "adaptive_rule":"proposed deltas1,.5,.25; require named guards, zero loss of any incumbent complete DEV task, nondecreasing aggregate DEV; otherwise revert and halve LR; confirmation used only after candidate fixed",
        "validation_scope":"known prompt and algorithm families with new generated final inputs; not unseen-prompt, unseen-algorithm, or broad coding generalization",
        "training_method":"supervised BF16 LoRA on two verified public repairs with token-margin protection; not weight-level recursive algorithm discovery",
        "experiment_round":"approved-parent-token-margin-repair",
        "optimizer_strategy":"AdamW displacement projected away from selected chosen-minus-competitor token-logit gradients; optimizer reset each proposal",
        "training_scope":"two minimal known-task repairs only; 10 solved public DEV plus 3 guards define projection anchors",
        "projection_is_not_guarantee":"all proposals still require actual public guard and complete-task preservation",
        "parent_adapter_tensor_sha256":parent["adapter_tensor_state_sha256"],
        "parent_adapter_file_sha256":parent["adapter_file_sha256"],
        "parent_approval_file_sha256":parent["approval_file_sha256"],
        "parent_successful_generation":parent["successful_generation"],
        "target_successful_generation":parent["successful_generation"]+1,
        "comparator":"immediate approved parent, never original base",
        "retired_parent_suite_sha256":a.retired_parent_sha256,
        "repair_weight":1.0,
        "base_precision":"frozen original BF16 weights",
        "warm_start_status":"load exact previously approved BF16 LoRA tensor state",
        "repair_method":"preserve parent response formatting and edit only the known wrong input unpack / sum expression; equal weights; no unrelated SFT repairs",
        "development_scope":"previous public DEV plus published parent confirmation are merged and deduplicated as DEVELOPMENT; prompts/algorithms overlap training",
        "prior_round_status":"one verified successful generation; prior failed experiments retained separately",
        "new_scope":"known-task program correctness on newly generated final inputs; no unseen-task or broad-capability claim; do not pool trials",
        "guard_rule":"no per-task case-count regression on closed interval, bracket and coin development guards",
        "max_wall_seconds":a.max_wall_seconds,"max_eval_context_tokens":1280,"max_training_tokens":768,
        "max_optimizer_updates":a.max_epochs,"physical_cuda_device":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "promotion_gate":{"min_case_gain":0.08,"min_final_case_accuracy":0.75,
                          "min_complete_task_wins":2,"max_complete_task_losses":0}}
    manifest.update(protection_method="individual low-margin token contrasts plus first-divergence public regression tokens; first-order only",
        margin_tokens_per_answer=3, public_counterexamples_sha256=a.public_counterexamples_sha256,
        max_protection_anchors=96,
        controlled_change="token-level contrast gradients replace response-mean/top3-NLL-average gradients; repair targets, initial LR, parent comparator and promotion gate unchanged")
    write("manifest.json",manifest)
    pre_evidence={"manifest":manifest,"proposed_action":
        "single-GPU bounded BF16 LoRA SFT; sealed suite is pre-registered and cannot steer training"}
    pre=judge.review("preflight",pre_evidence,deterministic_ok=True)
    if not pre.approved:raise JudgeError("preflight veto")
    judge.consume(pre,"preflight",pre_evidence)

    event("preflight_approved",review_id=pre.review_id)
    free,_=torch.cuda.mem_get_info(0)
    if free<12*2**30:raise RuntimeError("selected GPU lacks 12 GiB headroom")
    torch.manual_seed(a.seed)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(str(model_path),local_files_only=True,
        trust_remote_code=False,dtype=torch.bfloat16,device_map={"":0},attn_implementation="sdpa")
    for parameter in model.parameters():parameter.requires_grad_(False)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.0,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM"))
    tok=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=a.learning_rate,weight_decay=0.0)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0)
    runtime={"gpu":torch.cuda.get_device_name(0),"initial_allocated_gib":torch.cuda.memory_allocated(0)/2**30,
             "trainable_parameters":sum(p.numel() for p in params)}
    write("runtime.json",runtime)
    event("model_loaded",**runtime)

    def adapter_hash():
        h=hashlib.sha256()
        for name,t in sorted(get_peft_model_state_dict(model).items()):
            h.update(name.encode());h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    # Revalidate the accepted artifact immediately before loading. Loading uses
    # the same rank/targets as the parent, then validates every tensor via hash.
    validate_parent(a.parent_approval)
    parent_state=load_file(Path(parent["adapter_path"])/"adapter_model.safetensors")
    set_peft_model_state_dict(model,parent_state)
    initial_hash=adapter_hash()
    if initial_hash!=parent["adapter_tensor_state_sha256"]:
        raise RuntimeError("loaded adapter is not the approved parent")
    parent_state={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
    write("parent-loaded.json",{"parent_sha256":initial_hash,"successful_generation":parent["successful_generation"],
                                "parent_file_sha256":parent["adapter_file_sha256"],"matched":True})
    event("approved_parent_loaded",parent_sha256=initial_hash,generation=parent["successful_generation"])

    def prompt(desc):
        return tok.apply_chat_template([{"role":"user","content":
            "Write a Python function solve(data). Return only executable Python code, no explanation and no I/O. "+desc}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
    def evaluate(tasks,label,quiet=False):
        records=[];model.eval()
        for task in tasks:
            event("evaluation_task",phase=label,task_id=task.task_id)
            text=prompt(task.description);inputs=tok(text,return_tensors="pt").to("cuda:0")
            if inputs["input_ids"].shape[1]+a.max_new_tokens>1280:raise RuntimeError("eval context budget")
            with torch.inference_mode():
                out_ids=model.generate(**inputs,do_sample=False,max_new_tokens=a.max_new_tokens,use_cache=True,
                    pad_token_id=tok.eos_token_id)
            response=tok.decode(out_ids[0,inputs["input_ids"].shape[1]:],skip_special_tokens=True)
            result=grade(task,response,a.docker_image)
            rec={"task_id":task.task_id,"result":result,
                 "response_sha256":hashlib.sha256(response.encode()).hexdigest(),"response_chars":len(response),"response":response,
                 "prompt_token_ids":inputs["input_ids"][0].tolist(),
                 "response_token_ids":out_ids[0,inputs["input_ids"].shape[1]:].tolist()}
            records.append(rec)
            with (out/("responses-"+label+".jsonl")).open("a") as f:
                f.write(canonical({**rec,"response":response})+"\n")
            if not quiet:print(label,task.task_id,result["passed"],"/",result["total"],flush=True)
        return records
    def score(records):
        p=sum(x["result"]["passed"] for x in records);t=sum(x["result"]["total"] for x in records)
        c=sum(x["result"]["passed"]==x["result"]["total"] for x in records)
        return {"passed_cases":p,"total_cases":t,"case_accuracy":p/t,"complete_tasks":c,"task_count":len(records)}
    def loss_for(ex, protect_tail=False):
        pids=tok(prompt(ex.description),add_special_tokens=False)["input_ids"]
        aids=tok(ex.solution+tok.eos_token,add_special_tokens=False)["input_ids"]
        if len(pids)+len(aids)>768:raise RuntimeError("training context budget exceeded")
        ids=torch.tensor([pids+aids],device="cuda:0");n=len(aids)
        logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,
                     logits_to_keep=n+1).logits[:,:-1]
        if logits.shape[1]!=n:raise RuntimeError("SFT alignment")
        losses = F.cross_entropy(logits.reshape(-1,logits.shape[-1]).float(), ids[:,-n:].reshape(-1), reduction="none")
        return losses.topk(min(3, losses.numel())).values.mean() if protect_tail else losses.mean()

    dev0=evaluate(DEV,"DEV0");dev0s=score(dev0);write("dev-baseline.json",dev0)
    guard0=evaluate(GUARDS,"GUARD0");write("guard-baseline.json",guard0)
    def guards_ok(rows):
        baseline={x["task_id"]:x["result"] for x in guard0}
        return all(x["result"]["total"]==baseline[x["task_id"]]["total"] and
                   x["result"]["passed"]>=baseline[x["task_id"]]["passed"] for x in rows)
    # Only observed public defects are used for these minimal, executable repairs.
    from types import SimpleNamespace
    repair_data=minimal_repairs(DEV,dev0)
    by_task={task.task_id:task for task in DEV}
    for row in repair_data:
        verification=grade(by_task[row["task_id"]],row["solution"],a.docker_image)
        if verification["passed"]!=verification["total"]:
            raise RuntimeError("unverified minimal repair")
        row["verification"]=verification
    write("repair-dataset.json",repair_data)
    dataset_summary={"examples":len(repair_data),"sha256":digest(repair_data),"source":"minimal verified public edits"}
    write("repair-dataset-summary.json",dataset_summary)
    event("repair_dataset_fixed",**dataset_summary)
    def flat_values():
        return torch.cat([p.detach().reshape(-1) for p in params])
    def flat_gradients():
        return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1) for p in params]).clone()
    def copy_flat(vector):
        at=0
        with torch.no_grad():
            for p in params:
                n=p.numel();p.copy_(vector[at:at+n].view_as(p));at+=n
        if at!=vector.numel():raise RuntimeError("flat adapter dimension mismatch")
    def protected_responses(dev_rows,guard_rows):
        return [row for row in tuple(dev_rows)+tuple(guard_rows)
                if row["result"]["passed"]==row["result"]["total"]]
    def margin_gradient(row,anchor):
        # Exact generated token prefix; no whitespace/fence normalization.
        pos=anchor["position"]
        seq=row["prompt_token_ids"]+row["response_token_ids"][:pos]
        ids=torch.tensor([seq],dtype=torch.long,device="cuda:0")
        logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,
                     logits_to_keep=1).logits[0,-1].float()
        contrast=logits[anchor["chosen_token"]]-logits[anchor["competitor_token"]]
        contrast.backward()
        return flat_gradients()
    def margin_anchors(row):
        aids=row["response_token_ids"];n=len(aids)
        ids=torch.tensor([row["prompt_token_ids"]+aids],device="cuda:0")
        targets=torch.tensor(aids,device="cuda:0")
        with torch.no_grad():
            logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,
                         logits_to_keep=n+1).logits[0,:-1]
            if logits.shape[0]!=n:raise RuntimeError("margin response alignment")
            prior=[tok(r["response"],add_special_tokens=False)["input_ids"] for r in historical_rows
                   if r["task_id"]==row["task_id"]]
            anchors=choose_margins(logits,targets,count=3,counterexamples=prior)
        return anchors
    def snapshot():
        return {key:value.detach().cpu().clone() for key,value in get_peft_model_state_dict(model).items()}
    incumbent_state=snapshot();incumbent_dev=dev0;incumbent_guard=guard0
    best_acc=dev0s["case_accuracy"];best_epoch=0;history=[]
    for epoch in range(1,a.max_epochs+1):
        event("training_epoch_gradients",epoch=epoch)
        model.train();opt.zero_grad(set_to_none=True);losses=[]
        incumbent_flat=flat_values()
        anchor_gradients=[];anchor_records=[]
        for row in protected_responses(incumbent_dev,incumbent_guard):
            for anchor in margin_anchors(row):
                if len(anchor_records)>=96:raise RuntimeError("margin anchor budget exceeded")
                opt.zero_grad(set_to_none=True)
                grad=margin_gradient(row,anchor)
                anchor_records.append({"task_id":row["task_id"],**anchor,
                                       "gradient_norm":float(grad.norm())})
                anchor_gradients.append(grad)
        write(f"margin-anchors-{epoch}.json",anchor_records)
        opt.zero_grad(set_to_none=True)
        anchor_matrix=torch.stack(anchor_gradients)
        del anchor_gradients
        for row in repair_data:
            loss=loss_for(SimpleNamespace(**row))/len(repair_data)
            loss.backward();losses.append(float(loss.detach())*len(repair_data))
        norm=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
        evidence={"manifest":manifest,"epoch":epoch,"mean_sft_loss":sum(losses)/len(losses),
            "gradient_norm":float(norm),"actual_learning_rate":opt.param_groups[0]["lr"],"dev_before":dev0s if not history else history[-1]["dev"],
            "repair_dataset":dataset_summary,
            "projection_anchor_rows":int(anchor_matrix.shape[0]),
            "projection_scope":"individual chosen-minus-competitor logits: 3 weakest margins per solved answer plus historical first divergences; first-order only, actual tests required",
            "margin_anchors_sha256":digest(anchor_records),"public_counterexamples_sha256":a.public_counterexamples_sha256,
            "adapter_before":adapter_hash(),
            "guard_before":guard0 if not history else history[-1]["guard_records"],
            "proposed_action":"one minimal-repair AdamW proposal, projected into public token-margin gradient nullspace, then full/half/quarter public line search; confirmation absent"}
        ok=math.isfinite(float(norm)) and float(norm)>0 and all(math.isfinite(x) for x in losses)
        approval=judge.review("before_training",evidence,deterministic_ok=ok)
        if not approval.approved:
            opt.zero_grad(set_to_none=True);raise JudgeError("optimizer step vetoed")
        judge.consume(approval,"before_training",evidence)
        opt.step();opt.zero_grad(set_to_none=True)
        event("optimizer_updated",epoch=epoch,review_id=approval.review_id)
        current_hash=adapter_hash()
        if current_hash==initial_hash:raise RuntimeError("adapter unchanged")
        projected,projection_metrics=projected_delta(flat_values()-incumbent_flat,anchor_matrix)
        del anchor_matrix
        event("projection_measured",epoch=epoch,**projection_metrics)
        copy_flat(incumbent_flat+projected)
        del projected,incumbent_flat
        proposal=snapshot();accepted=False;accepted_scale=0.0;line_search=[]
        for scale in (1.0,0.5,0.25):
            blended={key:incumbent_state[key]+scale*(proposal[key]-incumbent_state[key]) for key in proposal}
            set_peft_model_state_dict(model,blended)
            trial_guard=evaluate(GUARDS,f"GUARD{epoch}-scale{scale}")
            passed=guards_ok(trial_guard)
            entry={"scale":scale,"guards_preserved":passed}
            if passed:
                trial_dev=evaluate(DEV,f"DEV{epoch}-scale{scale}")
                entry["dev"]=score(trial_dev)
                entry["incumbent_complete_tasks_preserved"]=complete_tasks_preserved(incumbent_dev,trial_dev)
                if entry["incumbent_complete_tasks_preserved"] and accept_public_update(passed,score(incumbent_dev),entry["dev"]):
                    accepted=True;accepted_scale=scale
                    incumbent_state=snapshot();incumbent_dev=trial_dev;incumbent_guard=trial_guard
            line_search.append(entry)
            if accepted:break
        if not accepted:set_peft_model_state_dict(model,incumbent_state)
        # Projection changes Adam's proposal; reset moments after every proposal.
        lr=opt.param_groups[0]["lr"]*(0.5 if not accepted else 1.0)
        opt=torch.optim.AdamW(params,lr=max(1e-6,lr),weight_decay=0.0)
        dev=incumbent_dev;guard=incumbent_guard;ds=score(dev);guard_passed=guards_ok(guard)
        current_hash=adapter_hash()
        ck=out/f"epoch-{epoch}";ck.mkdir(exist_ok=False);model.save_pretrained(ck,safe_serialization=True)
        before=adapter_hash();state=load_file(ck/"adapter_model.safetensors");set_peft_model_state_dict(model,state)
        if adapter_hash()!=before:raise RuntimeError("checkpoint roundtrip mismatch")
        item={"epoch":epoch,"dev":ds,"mean_sft_loss":sum(losses)/len(losses),
              "gradient_norm":float(norm),"adapter_sha256":current_hash,
              "review_id":approval.review_id,"checkpoint_roundtrip_exact":True,
              "guard_passed":guard_passed,"guard_records":guard,"proposal_accepted":accepted,"accepted_scale":accepted_scale,"line_search":line_search,"projection":projection_metrics,"margin_anchor_count":len(anchor_records)}
        history.append(item);write(f"epoch-{epoch}.json",item)
        if guard_passed and ds["case_accuracy"]>best_acc+1e-12:
            best_acc=ds["case_accuracy"];best_epoch=epoch
        if epoch>=a.min_epochs and development_ready(dev0s,ds):break

    if best_epoch==0 or not development_ready(dev0s,score(incumbent_dev)):
        write("report.json",{"validation_status":"NOT_OK","reason":"public development did not meet unchanged promotion prerequisites",
            "dev_baseline":dev0s,"history":history,"sealed_post_evaluated":False,
            "parent_sha256":initial_hash,"successful_promotions_before":parent["successful_generation"],
            "comparator":"approved_parent","promotion_approved":False})
        raise SystemExit(3)
    state=load_file(out/f"epoch-{best_epoch}"/"adapter_model.safetensors");set_peft_model_state_dict(model,state)
    # Consume this suite once across run directories. A failed evaluation is
    # not a license to re-test another checkpoint against the same final cases.
    with sealed.with_suffix(sealed.suffix+".consumed.json").open("x") as f:
        f.write(canonical({"run":str(out),"candidate_sha256":adapter_hash(),"code_sha256":source_hash})+"\n")
    final_tasks=load_suite(sealed,a.sealed_sha256)
    final_ids={x.task_id for x in final_tasks}
    if final_ids&(ids[0]|ids[1]):raise RuntimeError("final task IDs overlap")
    event("confirmation_candidate_fixed",epoch=best_epoch,adapter_sha256=adapter_hash())
    selected_state=snapshot()
    selected_sha256=adapter_hash()
    set_peft_model_state_dict(model,parent_state)
    if adapter_hash()!=initial_hash:raise RuntimeError("confirmation comparator is not accepted parent")
    sealed0=evaluate(final_tasks,"SEALED0",quiet=True)
    write("sealed-baseline.json",sealed0)
    set_peft_model_state_dict(model,selected_state)
    if adapter_hash()!=selected_sha256:raise RuntimeError("confirmation candidate changed")
    sealed1=evaluate(final_tasks,"SEALED1",quiet=True);write("sealed-post.json",sealed1)
    b=score(sealed0);q=score(sealed1);before={x["task_id"]:x for x in sealed0};after={x["task_id"]:x for x in sealed1}
    wins=losses=0;per=[]
    for task_id in sorted(before):
        br=before[task_id]["result"];ar=after[task_id]["result"]
        bc=br["passed"]==br["total"];ac=ar["passed"]==ar["total"]
        wins+=ac and not bc;losses+=bc and not ac
        per.append({"task_id":task_id,"before_passed":br["passed"],"after_passed":ar["passed"],
                    "total":br["total"],"complete_before":bc,"complete_after":ac})
    gate,b,q=promotion_gate(sealed0,sealed1)
    selected_guard=next(x["guard_records"] for x in history if x["epoch"]==best_epoch)
    gate["development_guards_passed"]=guards_ok(selected_guard)
    gate["passed"]=gate["passed"] and gate["development_guards_passed"]
    pe={"manifest":manifest,"best_epoch_selected_on_dev_only":best_epoch,"dev_baseline":dev0s,
        "best_dev_accuracy":best_acc,"sealed_before":b,"sealed_after":q,"gate":gate,
        "per_task":per,"candidate_sha256":adapter_hash(),
        "guard_baseline":guard0,"selected_guard":selected_guard,
        "parent_sha256":initial_hash,"parent_successful_generation":parent["successful_generation"],
        "proposed_action":"accept child only if it improves over the immediate accepted parent; no serving deployment"}
    decision=judge.review("checkpoint_promotion",pe,deterministic_ok=bool(gate["passed"]))
    if decision.approved:judge.consume(decision,"checkpoint_promotion",pe)
    report={"validation_status":"PASS" if decision.approved else "NOT_OK","model":"Qwen/Qwen3.5-4B",
        "revision":model_path.name,"best_epoch":best_epoch,
        "parent_sha256":initial_hash,"candidate_sha256":adapter_hash(),
        "parent_successful_generation":parent["successful_generation"],
        "comparator":"approved_parent","successful_promotions_before":parent["successful_generation"],"dev_baseline":dev0s,"best_dev_accuracy":best_acc,
        "sealed_before":b,"sealed_after":q,"gate":gate,"promotion_approved":decision.approved,
        "promotion_review_id":decision.review_id,"per_task":per,"history":history,
        "memory":{"peak_allocated_gib":torch.cuda.max_memory_allocated(0)/2**30,
                  "peak_reserved_gib":torch.cuda.max_memory_reserved(0)/2**30},
        "production_deployed":False,"elapsed_seconds":round(time.monotonic()-start,3)}
    final=judge.review("final_review",{"manifest":manifest,"report":report,
        "proposed_action":"certify bounded single-GPU 4B run; no deployment"},deterministic_ok=decision.approved)
    report["total_elapsed_seconds"]=round(time.monotonic()-start,3)
    report["final_external_review_ok"]=final.approved;report["final_review_id"]=final.review_id
    if not final.approved:report["validation_status"]="NOT_OK"
    report["candidate_promotion_passed_pending_cold_reproduction"]=report["validation_status"]=="PASS"
    write("report.json",report);print(canonical(report),flush=True)
    stopped.set()
    if report["validation_status"]!="PASS":raise SystemExit(4)

if __name__=="__main__":
    try:
        main()
    except BaseException as exc:
        # Never leave a crash looking like a successful experiment.
        import sys
        try:
            raw=sys.argv[sys.argv.index("--output")+1];folder=Path(raw)
            if folder.exists() and not (folder/"failure.json").exists():
                (folder/"failure.json").write_text(canonical({"status":"NOT_OK","error_type":type(exc).__name__,"error":str(exc)})+"\n")
        except (ValueError,IndexError,OSError):pass
        raise
