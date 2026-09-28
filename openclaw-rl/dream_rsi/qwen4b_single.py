"""Single-GPU Qwen3.5-4B dev-guided supervised QLoRA and in-domain promotion.

Only public TRAIN/DEV may steer training. The pre-registered sealed suite is
evaluated for base and candidate only after the DEV-selected checkpoint is
fixed. Its result cannot trigger retries or hyperparameter changes within this
run. Muse max is a fail-closed reviewer before every optimizer step and before
promotion.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading
import time

from .coding_tasks import CodingTask, grade
from .external_judge import ExternalJudgeGate, JudgeError, canonical, digest
from .qwen4b_curriculum_c import TRAIN, DEV
from .qwen4b_validation import load_suite, promotion_gate

def args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path",required=True)
    p.add_argument("--sealed-final",required=True)
    p.add_argument("--sealed-sha256",required=True)
    p.add_argument("--docker-image",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--test-report",required=True)
    p.add_argument("--max-epochs",type=int,default=6)
    p.add_argument("--min-epochs",type=int,default=2)
    p.add_argument("--learning-rate",type=float,default=3e-5)
    p.add_argument("--max-new-tokens",type=int,default=256)
    p.add_argument("--judge-timeout",type=int,default=300)
    p.add_argument("--max-wall-seconds",type=int,default=2400)
    p.add_argument("--seed",type=int,default=20260929)
    a=p.parse_args()
    if not 1<=a.min_epochs<=a.max_epochs<=6:p.error("epochs require 1<=min<=max<=6")
    if not 1e-6<=a.learning_rate<=5e-4:p.error("learning rate outside bounded range")
    if not 96<=a.max_new_tokens<=256:p.error("max new tokens must be 96..256")
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

    sealed=Path(a.sealed_final).resolve(); sealed_bytes=sealed.read_bytes()
    if hashlib.sha256(sealed_bytes).hexdigest()!=a.sealed_sha256:
        raise RuntimeError("sealed suite digest mismatch")
    # Check the digest now; construct no final tasks and compute no final
    # outcomes until the candidate checkpoint has been fixed using DEV.
    final_task_count=12
    ids=[{x.task_id for x in TRAIN},{x.task_id for x in DEV}]
    if ids[0]&ids[1]:raise RuntimeError("training/development ID overlap")

    judge=ExternalJudgeGate(out/"judges",repo,providers=("muse",),timeout_s=a.judge_timeout)
    source_hash=digest(judge.sources())
    tests=json.loads(Path(a.test_report).read_text())
    if not tests.get("passed") or tests.get("code_sha256")!=source_hash:raise RuntimeError("stale tests")
    model_path=Path(a.model_path).resolve(); cfg=json.loads((model_path/"config.json").read_text());tc=cfg["text_config"]
    if cfg.get("model_type")!="qwen3_5" or tc.get("hidden_size")!=2560 or tc.get("num_hidden_layers")!=32:
        raise RuntimeError("expected Qwen3.5-4B")
    manifest={"backend":"single_gpu_qwen35_4b_qlora_sft","model":"Qwen/Qwen3.5-4B",
        "revision":model_path.name,"visible_gpu_count":1,"train_examples":len(TRAIN),
        "dev_tasks":len(DEV),"sealed_tasks":final_task_count,"sealed_sha256":a.sealed_sha256,
        "learning_rate":a.learning_rate,"min_epochs":a.min_epochs,"max_epochs":a.max_epochs,
        "max_new_tokens":a.max_new_tokens,"code_sha256":source_hash,"tests":tests,
        "docker_image":a.docker_image,
        "adaptive_rule":"training and checkpoint choice use DEV only; final base/challenger evaluation happens once after candidate selection",
        "validation_scope":"in-domain coding smoke with new prompt wording and inputs, not unseen-algorithm generalization",
        "training_method":"supervised QLoRA on reference solutions; not weight-level recursive algorithm discovery",
        "max_wall_seconds":a.max_wall_seconds,"max_eval_context_tokens":512,"max_training_tokens":768,
        "max_optimizer_updates":a.max_epochs,"physical_cuda_device":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "promotion_gate":{"min_case_gain":0.08,"min_final_case_accuracy":0.75,
                          "min_complete_task_wins":2,"max_complete_task_losses":0}}
    write("manifest.json",manifest)
    pre_evidence={"manifest":manifest,"proposed_action":
        "single-GPU bounded QLoRA SFT; sealed suite is pre-registered and cannot steer training"}
    pre=judge.review("preflight",pre_evidence,deterministic_ok=True)
    if not pre.approved:raise JudgeError("preflight veto")
    judge.consume(pre,"preflight",pre_evidence)

    event("preflight_approved",review_id=pre.review_id)
    free,_=torch.cuda.mem_get_info(0)
    if free<7*2**30:raise RuntimeError("selected GPU lacks 7 GiB headroom")
    torch.manual_seed(a.seed)
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(str(model_path),local_files_only=True,
        trust_remote_code=False,quantization_config=quant,dtype=torch.bfloat16,device_map={"":0},
        attn_implementation="sdpa")
    model=prepare_frozen_kbit_model(model)
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
    initial_hash=adapter_hash()

    def prompt(desc):
        return tok.apply_chat_template([{"role":"user","content":
            "Write a Python function solve(data). Return only executable Python code, no explanation and no I/O. "+desc}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
    def evaluate(tasks,label,quiet=False):
        records=[];model.eval()
        for task in tasks:
            event("evaluation_task",phase=label,task_id=task.task_id)
            text=prompt(task.description);inputs=tok(text,return_tensors="pt").to("cuda:0")
            if inputs["input_ids"].shape[1]+a.max_new_tokens>512:raise RuntimeError("eval context budget")
            with torch.inference_mode():
                out_ids=model.generate(**inputs,do_sample=False,max_new_tokens=a.max_new_tokens,use_cache=True,
                    pad_token_id=tok.eos_token_id)
            response=tok.decode(out_ids[0,inputs["input_ids"].shape[1]:],skip_special_tokens=True)
            result=grade(task,response,a.docker_image)
            rec={"task_id":task.task_id,"result":result,
                 "response_sha256":hashlib.sha256(response.encode()).hexdigest(),"response_chars":len(response)}
            records.append(rec)
            with (out/("responses-"+label+".jsonl")).open("a") as f:
                f.write(canonical({**rec,"response":response})+"\n")
            if not quiet:print(label,task.task_id,result["passed"],"/",result["total"],flush=True)
        return records
    def score(records):
        p=sum(x["result"]["passed"] for x in records);t=sum(x["result"]["total"] for x in records)
        c=sum(x["result"]["passed"]==x["result"]["total"] for x in records)
        return {"passed_cases":p,"total_cases":t,"case_accuracy":p/t,"complete_tasks":c,"task_count":len(records)}
    def loss_for(ex):
        pids=tok(prompt(ex.description),add_special_tokens=False)["input_ids"]
        aids=tok(ex.solution+tok.eos_token,add_special_tokens=False)["input_ids"]
        if len(pids)+len(aids)>768:raise RuntimeError("training context budget exceeded")
        ids=torch.tensor([pids+aids],device="cuda:0");n=len(aids)
        logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,
                     logits_to_keep=n+1).logits[:,:-1]
        if logits.shape[1]!=n:raise RuntimeError("SFT alignment")
        return F.cross_entropy(logits.reshape(-1,logits.shape[-1]).float(),ids[:,-n:].reshape(-1))

    dev0=evaluate(DEV,"DEV0");dev0s=score(dev0);write("dev-baseline.json",dev0)
    best_acc=dev0s["case_accuracy"];best_epoch=0;history=[]
    for epoch in range(1,a.max_epochs+1):
        event("training_epoch_gradients",epoch=epoch)
        model.train();opt.zero_grad(set_to_none=True);losses=[]
        for ex in TRAIN:
            loss=loss_for(ex)/len(TRAIN);loss.backward();losses.append(float(loss.detach())*len(TRAIN))
        norm=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
        evidence={"manifest":manifest,"epoch":epoch,"mean_sft_loss":sum(losses)/len(losses),
            "gradient_norm":float(norm),"dev_before":dev0s if not history else history[-1]["dev"],
            "adapter_before":adapter_hash(),
            "proposed_action":"one optimizer step; sealed-suite result is deliberately absent"}
        ok=math.isfinite(float(norm)) and float(norm)>0 and all(math.isfinite(x) for x in losses)
        approval=judge.review("before_training",evidence,deterministic_ok=ok)
        if not approval.approved:
            opt.zero_grad(set_to_none=True);raise JudgeError("optimizer step vetoed")
        judge.consume(approval,"before_training",evidence)
        opt.step();opt.zero_grad(set_to_none=True)
        event("optimizer_updated",epoch=epoch,review_id=approval.review_id)
        current_hash=adapter_hash()
        if current_hash==initial_hash:raise RuntimeError("adapter unchanged")
        dev=evaluate(DEV,f"DEV{epoch}");ds=score(dev)
        ck=out/f"epoch-{epoch}";ck.mkdir(exist_ok=False);model.save_pretrained(ck,safe_serialization=True)
        before=adapter_hash();state=load_file(ck/"adapter_model.safetensors");set_peft_model_state_dict(model,state)
        if adapter_hash()!=before:raise RuntimeError("checkpoint roundtrip mismatch")
        item={"epoch":epoch,"dev":ds,"mean_sft_loss":sum(losses)/len(losses),
              "gradient_norm":float(norm),"adapter_sha256":current_hash,
              "review_id":approval.review_id,"checkpoint_roundtrip_exact":True}
        history.append(item);write(f"epoch-{epoch}.json",item)
        if ds["case_accuracy"]>best_acc+1e-12:
            best_acc=ds["case_accuracy"];best_epoch=epoch
        target=max(0.88,min(0.98,dev0s["case_accuracy"]+0.12))
        if epoch>=a.min_epochs and (best_acc>=target or (best_epoch>0 and epoch-best_epoch>=2)):break

    if best_epoch==0:
        write("report.json",{"validation_status":"NOT_OK","reason":"no DEV improvement",
            "dev_baseline":dev0s,"history":history,"sealed_post_evaluated":False})
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
    with model.disable_adapter():
        sealed0=evaluate(final_tasks,"SEALED0",quiet=True)
    write("sealed-baseline.json",sealed0)
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
    pe={"manifest":manifest,"best_epoch_selected_on_dev_only":best_epoch,"dev_baseline":dev0s,
        "best_dev_accuracy":best_acc,"sealed_before":b,"sealed_after":q,"gate":gate,
        "per_task":per,"candidate_sha256":adapter_hash(),
        "proposed_action":"accept experimental adapter only; do not alter existing service"}
    decision=judge.review("checkpoint_promotion",pe,deterministic_ok=bool(gate["passed"]))
    if decision.approved:judge.consume(decision,"checkpoint_promotion",pe)
    report={"validation_status":"PASS" if decision.approved else "NOT_OK","model":"Qwen/Qwen3.5-4B",
        "revision":model_path.name,"best_epoch":best_epoch,"dev_baseline":dev0s,"best_dev_accuracy":best_acc,
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
