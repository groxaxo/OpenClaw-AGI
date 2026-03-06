"""OpenClaw-OPD Unsloth QLoRA Trainer (On-Policy Distillation)
=================================================================

Lightweight OPD trainer using Unsloth + QLoRA for reduced GPU VRAM.
Designed to run on 2x or 3x RTX 3090 (24 GB) GPUs.

The OPD algorithm trains the student policy to imitate a teacher signal
built from hindsight hints:

  A_t = log π_teacher(a_t | s + hint) - log π_student(a_t | s)

The student is the current QLoRA model; the teacher is also the QLoRA
model with a hindsight hint prepended to the prompt (the hint makes
the task easier, so the teacher has higher probability on good actions).

GPU Layout
----------
2x RTX 3090:
  TRAINING_DEVICES=0   -> Unsloth QLoRA student (training + teacher inference)
  ROLLOUT_DEVICES=1    -> sglang rollout server

3x RTX 3090:
  TRAINING_DEVICES=0,1 -> Unsloth QLoRA student (DDP via accelerate)
  ROLLOUT_DEVICES=2    -> sglang rollout server

Usage
-----
  # 2x RTX 3090:
  CUDA_VISIBLE_DEVICES=0 python unsloth_qlora_opd_trainer.py \\
      --hf-checkpoint /path/to/Qwen3-4B \\
      --save /path/to/qlora-opd-ckpt \\
      --sglang-host 0.0.0.0 --sglang-port 30001 \\
      --proxy-host 0.0.0.0 --proxy-port 30000

  # 3x RTX 3090 (DDP):
  CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \\
      unsloth_qlora_opd_trainer.py \\
      --hf-checkpoint /path/to/Qwen3-4B \\
      --save /path/to/qlora-opd-ckpt \\
      --sglang-host 0.0.0.0 --sglang-port 30001 \\
      --proxy-host 0.0.0.0 --proxy-port 30000
"""

import argparse
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import requests
import torch

# Re-use helpers from the RL trainer
from unsloth_qlora_trainer import (
    collate_samples,
    drain_output_queue,
    get_token_log_probs,
    push_lora_weights_to_sglang,
    start_sglang_server,
    wait_for_sglang_ready,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OpenClaw-OPD Unsloth QLoRA Trainer (2-3x RTX 3090)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── model / checkpoint ──────────────────────────────────────────────────
    p.add_argument("--hf-checkpoint", required=True)
    p.add_argument("--save", required=True)
    p.add_argument("--save-interval", type=int, default=1)
    p.add_argument("--max-seq-length", type=int, default=32768)

    # ── QLoRA / LoRA ─────────────────────────────────────────────────────────
    p.add_argument("--load-in-4bit", action="store_true", default=True)
    p.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--lora-target-modules", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])

    # ── OPD training ─────────────────────────────────────────────────────────
    p.add_argument("--rollout-batch-size", type=int, default=16)
    p.add_argument("--num-rollout", type=int, default=100_000_000)
    p.add_argument("--global-batch-size", type=int, default=16)
    p.add_argument("--mini-batch-size", type=int, default=2)
    p.add_argument("--rollout-max-response-len", type=int, default=8192)
    p.add_argument("--rollout-temperature", type=float, default=0.6)
    p.add_argument("--kl-coef", type=float, default=0.02,
                   help="KL divergence coefficient between student and reference")
    p.add_argument("--entropy-coef", type=float, default=0.0)
    p.add_argument("--eps-clip", type=float, default=0.2)
    p.add_argument("--eps-clip-high", type=float, default=0.28)
    p.add_argument("--distill-topk", type=int, default=0,
                   help="If > 0, use top-K logits distillation (OPD option B); "
                        "0 means token-level OPD (option A, default)")

    # ── optimizer ────────────────────────────────────────────────────────────
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.98)
    p.add_argument("--gradient-clip", type=float, default=1.0)

    # ── sglang ───────────────────────────────────────────────────────────────
    p.add_argument("--sglang-host", default="127.0.0.1")
    p.add_argument("--sglang-port", type=int, default=30001)
    p.add_argument("--sglang-tp", type=int, default=1)
    p.add_argument("--sglang-mem-fraction", type=float, default=0.85)
    p.add_argument("--sglang-context-length", type=int, default=32768)
    p.add_argument("--sglang-reasoning-parser", default="qwen3")
    p.add_argument("--served-model-name", default="qwen3-4b")
    p.add_argument("--update-weights-interval", type=int, default=1)

    # ── proxy server ─────────────────────────────────────────────────────────
    p.add_argument("--proxy-host", default="0.0.0.0")
    p.add_argument("--proxy-port", type=int, default=30000)

    # ── PRM / judge ──────────────────────────────────────────────────────────
    p.add_argument("--prm-enable", action="store_true")
    p.add_argument("--prm-host", default="127.0.0.1")
    p.add_argument("--prm-port", type=int, default=30002)
    p.add_argument("--prm-model-path", default=None)
    p.add_argument("--prm-m", type=int, default=3)
    p.add_argument("--prm-temperature", type=float, default=0.6)
    p.add_argument("--prm-max-new-tokens", type=int, default=8192)
    p.add_argument("--teacher-lp-max-concurrency", type=int, default=3)

    # ── misc ─────────────────────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run-name", default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# OPD loss (token-level)
# ---------------------------------------------------------------------------

def compute_opd_loss(
    student_log_probs: torch.Tensor,    # [B, T]  – current student
    teacher_log_probs: torch.Tensor,    # [B, T]  – teacher (with hint)
    ref_log_probs: torch.Tensor,        # [B, T]  – reference (base model)
    old_log_probs: torch.Tensor,        # [B, T]  – from rollout (old policy)
    loss_mask: torch.Tensor,            # [B, T]
    eps_clip: float = 0.2,
    eps_clip_high: float = 0.28,
    kl_coef: float = 0.02,
    entropy_coef: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    OPD Option A (token-level):

      A_t = log π_teacher(a_t | s+hint) - log π_student_old(a_t | s)

    Training uses a PPO-style clipped loss on these token-level advantages.
    """
    # Token-level advantages
    advantages = teacher_log_probs - old_log_probs  # [B, T]

    # PPO ratio: current_policy / old_policy
    log_ratio = student_log_probs - old_log_probs
    ratio = log_ratio.exp()

    surr1 = ratio * advantages
    surr2 = ratio.clamp(1.0 - eps_clip, 1.0 + eps_clip_high) * advantages
    pg_loss = -torch.min(surr1, surr2)

    # KL divergence (student vs reference base model)
    # low_var_kl = (ratio_ref - 1) - log_ratio_ref
    log_ratio_ref = student_log_probs - ref_log_probs
    ratio_ref = log_ratio_ref.exp()
    kl = (ratio_ref - 1) - log_ratio_ref

    entropy = -student_log_probs

    denom = loss_mask.sum().clamp(min=1)
    pg = (pg_loss * loss_mask).sum() / denom
    kl_mean = (kl * loss_mask).sum() / denom
    ent_mean = (entropy * loss_mask).sum() / denom

    loss = pg + kl_coef * kl_mean - entropy_coef * ent_mean

    metrics = {
        "loss/total": loss.item(),
        "loss/pg": pg.item(),
        "loss/kl": kl_mean.item(),
        "loss/entropy": ent_mean.item(),
        "advantage/mean": (advantages * loss_mask).sum().item() / denom.item(),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Teacher log-prob retrieval
# ---------------------------------------------------------------------------

def get_teacher_log_probs_for_sample(
    model,
    sample,
    tokenizer,
    max_seq_length: int,
    device: torch.device,
) -> list[float]:
    """
    Re-query the model with the hint-augmented prompt to get teacher log-probs.

    The OPD API server stores the teacher prompt in
    ``sample.train_metadata["teacher_prompt_ids"]``.  If that field is
    absent (e.g. the hint was empty), we fall back to the student prompt
    and return all-zero advantages.
    """
    meta = sample.train_metadata or {}
    teacher_prompt_ids = meta.get("teacher_prompt_ids")

    if teacher_prompt_ids is None:
        # No hint → return zeros (this sample will not contribute gradient)
        return [0.0] * (sample.response_length or 0)

    response_ids = sample.tokens[len(sample.tokens) - (sample.response_length or 0):]
    full_ids = teacher_prompt_ids + response_ids
    full_ids = full_ids[:max_seq_length]

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attn_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        lp_full = get_token_log_probs(model, input_ids, attn_mask)

    # Extract only response token positions
    prompt_len_teacher = len(teacher_prompt_ids)
    lp_response = lp_full[0, prompt_len_teacher:].cpu().tolist()

    # Pad or truncate to match response_ids length
    target_len = len(response_ids)
    lp_response = lp_response[:target_len]
    lp_response = lp_response + [0.0] * (target_len - len(lp_response))
    return lp_response


# ---------------------------------------------------------------------------
# Batch collation with teacher log-probs
# ---------------------------------------------------------------------------

def collate_opd_samples(
    samples,
    model,
    tokenizer,
    pad_id: int,
    max_seq_len: int,
    device: torch.device,
):
    """
    Like collate_samples but also computes teacher log-probs inline.

    Returns the standard batch dict plus ``teacher_log_probs`` [B, T].
    """
    # Compute teacher log-probs per sample (no grad)
    teacher_lps_per_sample: list[list[float]] = []
    for s in samples:
        lp = get_teacher_log_probs_for_sample(model, s, tokenizer, max_seq_len, device)
        teacher_lps_per_sample.append(lp)

    # Now build the standard batch
    batch = collate_samples(samples, pad_id, max_seq_len, device)

    # Build teacher_log_probs tensor aligned with full sequence
    all_teacher_lp = []
    max_len = batch["input_ids"].shape[1]

    for s, t_lp in zip(samples, teacher_lps_per_sample):
        prompt_len = len(s.tokens) - (s.response_length or 0)
        full_t_lp = [0.0] * prompt_len + t_lp
        # Pad/truncate to max_len
        full_t_lp = (full_t_lp + [0.0] * max_len)[:max_len]
        all_teacher_lp.append(full_t_lp)

    batch["teacher_log_probs"] = torch.tensor(
        all_teacher_lp, dtype=torch.float32, device=device
    )
    return batch


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Optional accelerate multi-GPU
    try:
        from accelerate import Accelerator
        accelerator = Accelerator()
        is_main = accelerator.is_main_process
        local_rank = accelerator.local_process_index
        device = accelerator.device
        logger.info("[opd_train] Using accelerate: rank=%d device=%s", local_rank, device)
    except ImportError:
        accelerator = None
        is_main = True
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load model with Unsloth QLoRA ──────────────────────────────────────
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise SystemExit(
            "[unsloth_qlora_opd_trainer] unsloth is not installed.\n"
            "  pip install 'unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git'"
        )

    logger.info("[opd_train] Loading %s (4bit=%s) …", args.hf_checkpoint, args.load_in_4bit)
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.hf_checkpoint,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    logger.info("[opd_train] Applying LoRA (r=%d, alpha=%d) …", args.lora_r, args.lora_alpha)
    model = FastLanguageModel.get_peft_model(
        base_model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    model.train()

    # ── 2. Optimizer ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )

    if accelerator is not None:
        model, optimizer = accelerator.prepare(model, optimizer)

    # ── 3. Start sglang rollout server ───────────────────────────────────────
    sglang_proc: subprocess.Popen | None = None
    if is_main:
        logger.info("[opd_train] Starting sglang on port %d (tp=%d) …",
                    args.sglang_port, args.sglang_tp)
        sglang_proc = start_sglang_server(
            model_path=args.hf_checkpoint,
            host=args.sglang_host,
            port=args.sglang_port,
            tp=args.sglang_tp,
            mem_fraction=args.sglang_mem_fraction,
            context_length=args.sglang_context_length,
            reasoning_parser=args.sglang_reasoning_parser,
            served_model_name=args.served_model_name,
        )
        if not wait_for_sglang_ready(args.sglang_host, args.sglang_port):
            sglang_proc.terminate()
            raise RuntimeError("sglang server failed to start.")

    # ── 4. Start OpenClaw OPD proxy ───────────────────────────────────────────
    proxy_args = argparse.Namespace(
        hf_checkpoint=args.hf_checkpoint,
        sglang_router_ip=args.sglang_host,
        sglang_router_port=args.sglang_port,
        prm_enable=args.prm_enable,
        prm_router_ip=args.prm_host if args.prm_enable else None,
        prm_router_port=args.prm_port if args.prm_enable else None,
        prm_model_path=args.prm_model_path,
        prm_m=args.prm_m,
        prm_temperature=args.prm_temperature,
        prm_max_new_tokens=args.prm_max_new_tokens,
        teacher_lp_max_concurrency=args.teacher_lp_max_concurrency,
    )

    output_queue: queue.Queue = queue.Queue(maxsize=200_000)
    submission_enabled = threading.Event()

    if is_main:
        try:
            from openclaw_opd_api_server import OpenClawOPDAPIServer
        except ImportError:
            raise SystemExit(
                "[unsloth_qlora_opd_trainer] Could not import openclaw_opd_api_server. "
                "Run this script from the openclaw-opd directory or add it to PYTHONPATH."
            )

        os.environ.setdefault("HOST", args.proxy_host)
        os.environ.setdefault("PORT", str(args.proxy_port))
        os.environ.setdefault("SERVED_MODEL_NAME", args.served_model_name)
        os.environ.setdefault(
            "OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY",
            str(args.teacher_lp_max_concurrency),
        )

        api_server = OpenClawOPDAPIServer(
            args=proxy_args,
            output_queue=output_queue,
            submission_enabled=submission_enabled,
        )
        api_server.start()
        logger.info("[opd_train] OPD proxy listening on %s:%d", args.proxy_host, args.proxy_port)

    # ── 5. WandB ─────────────────────────────────────────────────────────────
    wandb_run = None
    if is_main and args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )
        except ImportError:
            logger.warning("[opd_train] wandb not installed.")

    # ── 6. Training loop ──────────────────────────────────────────────────────
    sglang_base_url = f"http://{args.sglang_host}:{args.sglang_port}"
    grad_accum_steps = max(1, args.global_batch_size // args.mini_batch_size)
    temp_dir = tempfile.mkdtemp(prefix="qlora_opd_push_")

    logger.info("[opd_train] Starting OPD loop (rollout_batch=%d, grad_accum=%d) …",
                args.rollout_batch_size, grad_accum_steps)

    for rollout_id in range(args.num_rollout):
        # ── 6a. Collect rollout data ──────────────────────────────────────────
        if is_main:
            submission_enabled.set()
            logger.info("[opd_train] rollout %d: collecting %d samples …",
                        rollout_id, args.rollout_batch_size)
            samples = drain_output_queue(output_queue, target=args.rollout_batch_size)
            submission_enabled.clear()
        else:
            samples = []

        # Multi-GPU: broadcast samples from rank 0
        if accelerator is not None and accelerator.num_processes > 1:
            import torch.distributed as dist
            obj_list = [samples]
            dist.broadcast_object_list(obj_list, src=0)
            samples = obj_list[0]

        if not samples:
            logger.warning("[opd_train] rollout %d: no samples, skipping.", rollout_id)
            continue

        # ── 6b. Train one OPD update ──────────────────────────────────────────
        model.train()
        optimizer.zero_grad()

        total_loss = 0.0
        all_metrics: list[dict] = []
        step_count = 0

        for mb_start in range(0, len(samples), args.mini_batch_size):
            mb_samples = samples[mb_start: mb_start + args.mini_batch_size]
            unwrapped = accelerator.unwrap_model(model) if accelerator else model
            batch = collate_opd_samples(
                mb_samples, unwrapped, tokenizer,
                pad_id, args.max_seq_length, device,
            )

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            loss_mask = batch["loss_mask"]
            old_lp = batch["old_log_probs"]
            teacher_lp = batch["teacher_log_probs"]

            # Reference (base model, no LoRA)
            with torch.no_grad():
                try:
                    from peft import disable_adapter_layers, enable_adapter_layers
                    disable_adapter_layers(unwrapped)
                    ref_lp = get_token_log_probs(unwrapped, input_ids, attention_mask)
                    enable_adapter_layers(unwrapped)
                except ImportError:
                    ref_lp = old_lp  # fallback: KL → 0

            # Current student log-probs
            cur_lp = get_token_log_probs(model, input_ids, attention_mask)

            loss, metrics = compute_opd_loss(
                student_log_probs=cur_lp,
                teacher_log_probs=teacher_lp,
                ref_log_probs=ref_lp,
                old_log_probs=old_lp,
                loss_mask=loss_mask,
                eps_clip=args.eps_clip,
                eps_clip_high=args.eps_clip_high,
                kl_coef=args.kl_coef,
                entropy_coef=args.entropy_coef,
            )

            scaled = loss / grad_accum_steps
            if accelerator is not None:
                accelerator.backward(scaled)
            else:
                scaled.backward()

            total_loss += loss.item()
            all_metrics.append(metrics)
            step_count += 1

        if accelerator is not None:
            accelerator.clip_grad_norm_(model.parameters(), args.gradient_clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

        optimizer.step()

        if is_main and all_metrics:
            agg: dict[str, float] = {}
            for k in all_metrics[0]:
                agg[k] = sum(m[k] for m in all_metrics) / len(all_metrics)
            agg["rollout_id"] = rollout_id
            logger.info(
                "[opd_train] rollout %d | loss=%.4f pg=%.4f kl=%.4f adv=%.4f",
                rollout_id,
                agg.get("loss/total", 0),
                agg.get("loss/pg", 0),
                agg.get("loss/kl", 0),
                agg.get("advantage/mean", 0),
            )
            if wandb_run is not None:
                wandb_run.log(agg)

        # ── 6c. Save checkpoint ───────────────────────────────────────────────
        if is_main and (rollout_id + 1) % args.save_interval == 0:
            ckpt_dir = save_dir / f"step_{rollout_id + 1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model) if accelerator else model
            unwrapped.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            logger.info("[opd_train] Saved LoRA checkpoint → %s", ckpt_dir)

        # ── 6d. Push weights to sglang ────────────────────────────────────────
        if is_main and (rollout_id + 1) % args.update_weights_interval == 0:
            unwrapped = accelerator.unwrap_model(model) if accelerator else model
            push_lora_weights_to_sglang(unwrapped, sglang_base_url, temp_dir)

    # ── 7. Cleanup ────────────────────────────────────────────────────────────
    if is_main:
        final_dir = save_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model) if accelerator else model
        unwrapped.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        logger.info("[opd_train] Final LoRA checkpoint → %s", final_dir)

        if wandb_run is not None:
            wandb_run.finish()
        if sglang_proc is not None:
            sglang_proc.terminate()
            logger.info("[opd_train] sglang terminated.")

    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
