"""Bounded all-GPU Qwen3.5-9B validation backend, independent of SGLang.

This tests real on-policy generation, immutable coding rewards, curriculum
selection, signed-reward PPO-style QLoRA updates, CLI vetoes and checkpoint
roundtrips. It is not an efficacy benchmark or an official Dream-RSI replica.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading
import time
from types import SimpleNamespace

from .coding_tasks import TRAIN_TASKS, HOLDOUT_TASKS, grade
from .core import DreamRSIController, holdout_gate
from .external_judge import ExternalJudgeGate, canonical, digest, JudgeError


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--docker-image", required=True)
    parser.add_argument("--test-report", required=True)
    parser.add_argument("--judges", choices=("muse", "glm", "both"), default="both")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-wall-seconds", type=int, default=1800)
    parser.add_argument("--judge-timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    if not 1 <= args.cycles <= 3 or not 32 <= args.max_new_tokens <= 256:
        parser.error("smoke budget: cycles 1..3; new tokens 32..256")
    if not 120 <= args.max_wall_seconds <= 3600:
        parser.error("wall budget must be 120..3600 seconds")
    if "@sha256:" not in args.docker_image:
        parser.error("Docker image must use a digest")
    return args


def main():
    args = parse_args()
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import (Qwen3_5ForConditionalGeneration, AutoTokenizer, BitsAndBytesConfig)
    from peft import (LoraConfig, get_peft_model, prepare_model_for_kbit_training,
                      get_peft_model_state_dict, set_peft_model_state_dict)
    from safetensors.torch import load_file
    from .losses import clipped_policy_loss
    from .model_utils import prepare_frozen_kbit_model

    rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if world != 3 or torch.cuda.device_count() != 3:
        raise RuntimeError("This validation recipe requires all three visible CUDA GPUs and three ranks")
    torch.cuda.set_device(local)
    torch.cuda.set_per_process_memory_fraction(0.58, local)
    torch.set_num_threads(3)
    dist.init_process_group("nccl", timeout=timedelta(seconds=args.max_wall_seconds),
                            device_id=torch.device("cuda", local))
    output, repo = Path(args.output).resolve(), Path(__file__).resolve().parents[2]
    start = time.monotonic()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
    dist.barrier()
    stopped = threading.Event()
    def watchdog():
        while not stopped.wait(1):
            if time.monotonic() - start > args.max_wall_seconds or shutil.disk_usage(output).free < 5 * 2**30:
                (output / f"rank-{rank}-budget-failure.txt").write_text("wall or disk budget exceeded\n")
                os._exit(124)
    threading.Thread(target=watchdog, daemon=True).start()

    def write(name, value):
        if rank == 0:
            with (output / name).open("x") as f:
                f.write(canonical(value) + "\n")

    def gather(value):
        result = [None] * world
        dist.all_gather_object(result, value)
        return result

    def broadcast(value):
        box = [value if rank == 0 else None]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    providers = ("muse", "glm") if args.judges == "both" else (args.judges,)
    judge = ExternalJudgeGate(output / "judges", repo, providers=providers, timeout_s=args.judge_timeout) if rank == 0 else None
    model_path = Path(args.model_path).resolve()
    revision = model_path.name
    config = json.loads((model_path / "config.json").read_text())
    tc = config.get("text_config", {})
    if (len(revision) != 40 or config.get("model_type") != "qwen3_5"
            or tc.get("hidden_size") != 4096 or tc.get("num_hidden_layers") != 32):
        raise RuntimeError("Expected pinned official Qwen3.5-9B snapshot, not another model")
    test_report = json.loads(Path(args.test_report).read_text())
    source_hash = broadcast(digest(judge.sources()) if rank == 0 else None)
    if not test_report.get("passed") or test_report.get("code_sha256") != source_hash:
        raise RuntimeError("Unit-test evidence does not match this exact source tree")
    manifest = {"backend": "native_transformers_qlora_ddp", "purpose": "bounded integration smoke, not production",
                "model": "Qwen/Qwen3.5-9B", "revision": revision, "world_size": world,
                "quantization": "NF4 double quantization; BF16 compute; rank-8 LoRA",
                "cycles": args.cycles, "max_new_tokens": args.max_new_tokens,
                "max_generated_tokens": (len(TRAIN_TASKS)*args.cycles + 2*len(HOLDOUT_TASKS))*args.max_new_tokens,
                "max_wall_seconds": args.max_wall_seconds, "min_free_disk_gib": 5,
                "judges": list(providers), "code_sha256": source_hash, "tests": test_report,
                "train_ids": [x.task_id for x in TRAIN_TASKS], "holdout_ids": [x.task_id for x in HOLDOUT_TASKS],
                "task_suite_sha256": digest([asdict(t) for t in TRAIN_TASKS + HOLDOUT_TASKS]),
                "docker_image": args.docker_image, "learning_rate": 1e-5,
                "promotion_rule": "single terminal comparison, alpha=.05, mean gain>.02, no per-task regression; external veto",
                "legacy_sglang_backend_exercised": False}
    write("manifest.json", manifest)
    if set(manifest["train_ids"]) & set(manifest["holdout_ids"]):
        raise RuntimeError("train/holdout overlap")

    torch.manual_seed(args.seed)
    # Stagger loading to bound host RAM/transient quantization pressure.
    for turn in range(world):
        if rank == turn:
            free, _ = torch.cuda.mem_get_info(local)
            if free < 13 * 2**30:
                raise RuntimeError(f"GPU {local} has insufficient headroom; leave existing jobs untouched")
            print(f"RANK {rank}: loading pinned Qwen3.5-9B", flush=True)
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                str(model_path), local_files_only=True, trust_remote_code=False,
                quantization_config=quant, dtype=torch.bfloat16, device_map={"": local},
                attn_implementation="sdpa")
            model = prepare_frozen_kbit_model(model)
            model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                task_type="CAUSAL_LM"))
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(local)
        dist.barrier()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    # Same pinned base + seed on all ranks; avoid re-broadcasting packed frozen
    # quantization tensors. Explicit adapter checks below verify initialization.
    ddp = DDP(model, device_ids=[local], output_device=local, broadcast_buffers=False,
              find_unused_parameters=False, init_sync=False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=0.0)

    def adapter_digest():
        h = hashlib.sha256()
        for name, tensor in sorted(get_peft_model_state_dict(model).items()):
            h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    initial_hashes = gather(adapter_digest())
    if len(set(initial_hashes)) != 1:
        raise RuntimeError("adapter initialization differs across ranks")
    runtime = gather({"rank": rank, "gpu": torch.cuda.get_device_name(local),
                      "initial_allocated_gib": torch.cuda.memory_allocated(local)/2**30,
                      "trainable_parameters": sum(p.numel() for p in parameters)})
    write("runtime.json", runtime)

    def generate_task(task, sample_seed, sample):
        torch.manual_seed(sample_seed)
        messages = [{"role": "user", "content": "Write a Python function solve(data). Return only executable Python code, no explanation and no I/O. " + task.description}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(local)
        prefix = inputs["input_ids"].shape[1]
        if prefix + args.max_new_tokens > 512:
            raise RuntimeError("context budget exceeded; no silent truncation")
        model.eval()
        options = {"do_sample": sample}
        if sample:
            # Raw distribution: matches unwarped old log-probs during update.
            options.update(temperature=1.0, top_p=1.0, top_k=0)
        t0 = time.monotonic()
        with torch.inference_mode():
            tokens = model.generate(**inputs, **options, max_new_tokens=args.max_new_tokens,
                                    use_cache=True, pad_token_id=tokenizer.eos_token_id)
        ids = tokens[0].tolist()
        response = tokenizer.decode(tokens[0, prefix:], skip_special_tokens=True)
        result = grade(task, response, args.docker_image)
        del inputs, tokens
        return {"task_id": task.task_id, "seed": sample_seed, "ids": ids, "prefix": prefix,
                "response": response, "result": result, "generation_seconds": round(time.monotonic()-t0, 3)}

    def evaluate(tasks, cycle, sample):
        local_records = []
        for i, task in enumerate(tasks):
            if i % world == rank:
                rec = generate_task(task, args.seed + cycle*1000 + i, sample)
                local_records.append(rec)
                print(f"RANK {rank}: {task.task_id} {rec['result']['passed']}/{rec['result']['total']}", flush=True)
        records = [r for chunk in gather(local_records) for r in chunk]
        return sorted(records, key=lambda r: r["task_id"])

    baseline = evaluate(HOLDOUT_TASKS, 0, False)
    write("baseline.json", baseline)
    controller = DreamRSIController(output / "curriculum", target_batch_size=world, pool_factor=2,
                                   evolve_interval=8, holdout_pools=4, min_holdout_pairs=4) if rank == 0 else None
    updates = []

    def response_lp(module, record):
        ids = torch.tensor([record["ids"]], dtype=torch.long, device=local)
        n = ids.shape[1] - record["prefix"]
        if n < 1:
            raise RuntimeError("no response tokens")
        logits = module(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                        logits_to_keep=n+1).logits[:, :-1]
        targets = ids[:, -n:]
        if logits.shape[1] != n:
            raise RuntimeError("response logit alignment mismatch")
        chunks = []
        for a in range(0, n, 16):
            block = logits[:, a:a+16].float()
            actual = block.gather(-1, targets[:, a:a+16, None]).squeeze(-1)
            chunks.append(actual - block.logsumexp(-1))
        return torch.cat(chunks, dim=1)

    for cycle in range(args.cycles):
        records = evaluate(TRAIN_TASKS, cycle+1, True)
        write(f"rollouts-{cycle}.json", records)
        chosen = None
        if rank == 0:
            samples = [SimpleNamespace(metadata={"session_id": r["task_id"], "turn": cycle+1,
                         "dream_node_id": f"{cycle}:{r['task_id']}", "has_next_state": True},
                         index=i, reward={"score": r["result"]["score"]}, status="completed", loss_mask=[1],
                         remove_sample=False, record=r) for i,r in enumerate(records)]
            chosen = [s.record for s in controller.select_samples(samples, cycle)]
        chosen = broadcast(chosen)
        if len(chosen) != world:
            raise RuntimeError("incomplete valid DDP batch")
        current_record = chosen[rank]
        model.eval()
        with torch.no_grad():
            old_lp = response_lp(model, current_record).detach()
            with model.disable_adapter():
                ref_lp = response_lp(model, current_record).detach()
        lp_checks = gather({"rank": rank, "finite": bool(torch.isfinite(old_lp).all() and torch.isfinite(ref_lp).all()),
                            "response_tokens": old_lp.numel()})
        evidence = {"manifest": manifest, "runtime": runtime, "cycle": cycle,
                    "proposed_optimizer_steps": 1, "completed_updates": updates,
                    "batch": [{"task_id": r["task_id"], "result": r["result"],
                               "record_sha256": digest(r)} for r in chosen],
                    "logprob_checks": lp_checks, "adapter_before": adapter_digest(),
                    "note": "Signed-reward zero-baseline PPO-style smoke, not group-normalized GRPO. Only LoRA is trainable. No checkpoint deployment."}
        ok = all(x["finite"] and x["response_tokens"] > 0 for x in lp_checks) and any(r["result"]["score"] != 0 for r in chosen)
        approval = judge.review("before_training", evidence, deterministic_ok=ok) if rank == 0 else None
        approved = broadcast(approval.approved if rank == 0 else None)
        if not approved:
            raise JudgeError("Training vetoed; optimizer not stepped")
        if rank == 0:
            judge.consume(approval, "before_training", evidence)
        dist.barrier()
        ddp.train()
        optimizer.zero_grad(set_to_none=True)
        lp = response_lp(ddp, current_record)
        adv = torch.tensor([current_record["result"]["score"]], dtype=torch.float32, device=local)
        loss, metrics = clipped_policy_loss(lp, old_lp, ref_lp, adv, torch.ones_like(lp))
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        norms = gather(float(gradient_norm))
        if any(not math.isfinite(n) or n <= 0 for n in norms):
            raise RuntimeError("non-finite or zero DDP gradient; no update")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del loss, lp, old_lp, ref_lp
        hashes = gather(adapter_digest())
        if len(set(hashes)) != 1 or hashes[0] == initial_hashes[0]:
            raise RuntimeError("adapter did not change consistently on every GPU")
        checkpoint = output / f"candidate-{cycle+1}"
        roundtrip = None
        if rank == 0:
            checkpoint.mkdir(exist_ok=False)
            model.save_pretrained(checkpoint, safe_serialization=True)
            before_reload = adapter_digest()
            state = load_file(checkpoint / "adapter_model.safetensors")
            set_peft_model_state_dict(model, state)
            roundtrip = adapter_digest() == before_reload
        if not broadcast(roundtrip):
            raise RuntimeError("saved adapter failed exact reload")
        update = {"cycle": cycle, "review_id": broadcast(approval.review_id if rank == 0 else None),
                  "per_rank_loss": gather(metrics), "gradient_norms": norms,
                  "adapter_hashes": hashes, "checkpoint_roundtrip_exact": True,
                  "checkpoint": checkpoint.name}
        updates.append(update)
        write(f"update-{cycle}.json", update)
        print(f"RANK {rank}: UPDATE {cycle+1} complete gradient_norm={float(gradient_norm):.6g}", flush=True)
        torch.cuda.empty_cache()

    # Exactly one terminal heldout comparison. No training conditions on these
    # post-training outcomes, no retrying a failed promotion within this run.
    final = evaluate(HOLDOUT_TASKS, 0, False)
    write("final-holdout.json", final)
    before = [r["result"]["passed"]/r["result"]["total"] for r in baseline]
    after = [r["result"]["passed"]/r["result"]["total"] for r in final]
    statistical = holdout_gate(before, after, min_pairs=6, min_mean_delta=.02, alpha=.05)
    no_regression = all(b >= a for a,b in zip(before, after))
    memory = gather({"rank": rank, "peak_allocated_gib": torch.cuda.max_memory_allocated(local)/2**30,
                     "peak_reserved_gib": torch.cuda.max_memory_reserved(local)/2**30})
    report = {"validation_status": "PASS", "backend": manifest["backend"], "model": manifest["model"],
              "revision": revision, "optimizer_updates": len(updates), "world_size": world,
              "updates": updates, "memory": memory, "heldout_before": before, "heldout_after": after,
              "statistical_gate": asdict(statistical), "no_regression": no_regression,
              "legacy_sglang_network_path_tested": False, "production_deployed": False,
              "elapsed_seconds": round(time.monotonic()-start, 3)}
    if rank == 0:
        promotion_evidence = {"manifest": manifest, "report": report,
                              "candidate_sha256": adapter_digest(),
                              "proposed_action": "accept experimental checkpoint; never publish to existing service"}
        decision = judge.review("checkpoint_promotion", promotion_evidence,
                                deterministic_ok=bool(statistical.promote and no_regression))
        if decision.approved:
            judge.consume(decision, "checkpoint_promotion", promotion_evidence)
            write("accepted-candidate.json", {"checkpoint": f"candidate-{args.cycles}", "review_id": decision.review_id})
        report["promotion_approved"] = decision.approved
        report["promotion_review_id"] = decision.review_id
        final_evidence = {"manifest": manifest, "report": report,
                          "proposed_action": "certify bounded end-to-end integration test, NOT efficacy or production readiness"}
        final_review = judge.review("final_review", final_evidence, deterministic_ok=True)
        report["final_external_review_ok"] = final_review.approved
        report["final_review_id"] = final_review.review_id
        if not final_review.approved:
            report["validation_status"] = "NOT_OK"
        write("report.json", report)
        print(canonical(report), flush=True)
    dist.barrier()
    stopped.set()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
