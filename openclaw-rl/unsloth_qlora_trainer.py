"""OpenClaw-RL Unsloth Qwen 3.5 Vision Trainer
==============================================

Lightweight RL trainer using Unsloth + QLoRA for reduced GPU VRAM.
Defaults to Unsloth Qwen 3.5 (4B) Vision and is designed to run on 2x or
3x RTX 3090 (24 GB) GPUs.

GPU Layout
----------
2x RTX 3090:
  TRAINING_DEVICES=0   -> Unsloth QLoRA actor (training + optional inference)
  ROLLOUT_DEVICES=1    -> sglang rollout server

3x RTX 3090:
  TRAINING_DEVICES=0,1 -> Unsloth QLoRA actor (multi-GPU DDP via accelerate)
  ROLLOUT_DEVICES=2    -> sglang rollout server

Usage
-----
  # 2x RTX 3090 (training on GPU 0, rollout on GPU 1):
  CUDA_VISIBLE_DEVICES=0 python unsloth_qlora_trainer.py \\
      --hf-checkpoint unsloth/Qwen3.5-4B \\
      --save /path/to/qlora-ckpt \\
      --sglang-host 0.0.0.0 --sglang-port 30001 \\
      --proxy-host 0.0.0.0 --proxy-port 30000

  # 3x RTX 3090 (training on GPUs 0-1, rollout on GPU 2):
  CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \\
      unsloth_qlora_trainer.py \\
      --hf-checkpoint unsloth/Qwen3.5-4B \\
      --save /path/to/qlora-ckpt \\
      --sglang-host 0.0.0.0 --sglang-port 30001 \\
      --proxy-host 0.0.0.0 --proxy-port 30000
"""

import argparse
import asyncio
import json
import logging
import math
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
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)

# ---------------------------------------------------------------------------
# Lazy-import guard for optional heavy dependencies
# ---------------------------------------------------------------------------

def _require(pkg: str, install_hint: str = ""):
    try:
        return __import__(pkg)
    except ImportError:
        hint = f"  pip install {install_hint or pkg}"
        raise SystemExit(f"[unsloth_qlora_trainer] Missing required package: {pkg}\n{hint}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OpenClaw-RL Unsloth Qwen 3.5 Vision Trainer (2-3x RTX 3090)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── model / checkpoint ──────────────────────────────────────────────────
    p.add_argument(
        "--hf-checkpoint",
        default="unsloth/Qwen3.5-4B",
        help="Path or HF hub ID for the base model (default: unsloth/Qwen3.5-4B)",
    )
    p.add_argument("--save", required=True,
                   help="Directory to save LoRA adapter checkpoints")
    p.add_argument("--save-interval", type=int, default=1,
                   help="Save LoRA adapter every N training steps")
    p.add_argument("--max-seq-length", type=int, default=32768,
                   help="Maximum sequence length (prompt + response)")

    # ── QLoRA / LoRA hyper-params ────────────────────────────────────────────
    p.add_argument("--load-in-4bit", action="store_true", default=True,
                   help="Load base model in 4-bit NF4 (default: True)")
    p.add_argument("--no-4bit", dest="load_in_4bit", action="store_false",
                   help="Disable 4-bit quantization (use bf16 instead)")
    p.add_argument("--lora-r", type=int, default=16,
                   help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=16,
                   help="LoRA alpha (scaling = alpha / r)")
    p.add_argument("--lora-dropout", type=float, default=0.0,
                   help="LoRA dropout")
    p.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=None,
        help="Optional explicit LoRA target modules. Leave unset to use Unsloth's Qwen 3.5 vision defaults.",
    )
    p.add_argument(
        "--finetune-vision-layers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LoRA on the vision encoder blocks.",
    )
    p.add_argument(
        "--finetune-language-layers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LoRA on the language-model blocks.",
    )
    p.add_argument(
        "--finetune-attention-modules",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LoRA on attention projections.",
    )
    p.add_argument(
        "--finetune-mlp-modules",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LoRA on MLP projections.",
    )

    # ── RL training ─────────────────────────────────────────────────────────
    p.add_argument("--rollout-batch-size", type=int, default=16,
                   help="Number of training samples to collect per rollout step")
    p.add_argument("--num-rollout", type=int, default=100_000_000,
                   help="Maximum number of rollout iterations")
    p.add_argument("--global-batch-size", type=int, default=16,
                   help="Global batch size for a training update")
    p.add_argument("--mini-batch-size", type=int, default=2,
                   help="Per-device mini-batch size for gradient accumulation")
    p.add_argument("--rollout-max-response-len", type=int, default=8192,
                   help="Maximum response length during rollout")
    p.add_argument("--rollout-temperature", type=float, default=0.6,
                   help="Sampling temperature for rollout")

    # ── GRPO loss ────────────────────────────────────────────────────────────
    p.add_argument("--eps-clip", type=float, default=0.2,
                   help="PPO lower clip epsilon")
    p.add_argument("--eps-clip-high", type=float, default=0.28,
                   help="PPO upper clip epsilon")
    p.add_argument("--kl-coef", type=float, default=0.02,
                   help="KL divergence coefficient")
    p.add_argument("--entropy-coef", type=float, default=0.0,
                   help="Entropy bonus coefficient")
    p.add_argument("--disable-rewards-normalization", action="store_true", default=True,
                   help="Disable per-batch reward normalization (GRPO style)")

    # ── optimizer ────────────────────────────────────────────────────────────
    p.add_argument("--lr", type=float, default=1e-6,
                   help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=0.1,
                   help="AdamW weight decay")
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.98)
    p.add_argument("--gradient-clip", type=float, default=1.0,
                   help="Gradient norm clipping")

    # ── rollout server (sglang) ───────────────────────────────────────────────
    p.add_argument("--sglang-host", default="127.0.0.1",
                   help="Host of the sglang rollout server")
    p.add_argument("--sglang-port", type=int, default=30001,
                   help="Port of the sglang rollout server")
    p.add_argument("--sglang-tp", type=int, default=1,
                   help="Tensor-parallel size for sglang")
    p.add_argument("--sglang-mem-fraction", type=float, default=0.85,
                   help="Memory fraction for sglang KV cache")
    p.add_argument("--sglang-context-length", type=int, default=32768,
                   help="Context length for sglang")
    p.add_argument("--sglang-reasoning-parser", default="qwen3",
                   help="Reasoning parser for sglang (e.g. qwen3)")
    p.add_argument("--sglang-tool-call-parser", default="qwen3_coder",
                   help="Tool-call parser for sglang when serving agent requests")
    p.add_argument("--served-model-name", default="qwen3.5-4b",
                   help="Model name served by sglang (OpenAI API compat.)")
    p.add_argument("--update-weights-interval", type=int, default=1,
                   help="Push updated weights to sglang every N rollout steps")

    # ── OpenClaw proxy server ────────────────────────────────────────────────
    p.add_argument("--proxy-host", default="0.0.0.0",
                   help="Host for the OpenClaw proxy (FastAPI) server")
    p.add_argument("--proxy-port", type=int, default=30000,
                   help="Port for the OpenClaw proxy server")

    # ── PRM (optional) ───────────────────────────────────────────────────────
    p.add_argument("--prm-enable", action="store_true",
                   help="Enable Process Reward Model scoring")
    p.add_argument("--prm-host", default="127.0.0.1")
    p.add_argument("--prm-port", type=int, default=30002)
    p.add_argument("--prm-model-path", default=None,
                   help="HF path for the PRM model (defaults to hf-checkpoint)")
    p.add_argument("--prm-m", type=int, default=3,
                   help="Number of PRM votes per sample")
    p.add_argument("--prm-temperature", type=float, default=0.6)
    p.add_argument("--prm-max-new-tokens", type=int, default=4096)

    # ── misc ─────────────────────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-run-name", default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# GRPO loss
# ---------------------------------------------------------------------------

def compute_grpo_loss(
    current_log_probs: torch.Tensor,   # [B, T]
    old_log_probs: torch.Tensor,        # [B, T]  (from rollout)
    ref_log_probs: torch.Tensor,        # [B, T]  (from frozen base / ref LoRA)
    advantages: torch.Tensor,           # [B]
    loss_mask: torch.Tensor,            # [B, T]  1 on response tokens, 0 on prompt
    eps_clip: float = 0.2,
    eps_clip_high: float = 0.28,
    kl_coef: float = 0.02,
    entropy_coef: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    Compute GRPO policy-gradient + KL loss.

    Returns (scalar loss, metrics_dict).
    """
    # ── policy-gradient ─────────────────────────────────────────────────────
    log_ratio = current_log_probs - old_log_probs
    ratio = log_ratio.exp()

    # Broadcast scalar advantage to every response token
    adv_expanded = advantages.unsqueeze(1)  # [B, 1]

    surr1 = ratio * adv_expanded
    surr2 = ratio.clamp(1.0 - eps_clip, 1.0 + eps_clip_high) * adv_expanded
    pg_loss = -torch.min(surr1, surr2)

    # ── KL divergence  ───────────────────────────────────────────────────────
    # low_var_kl: k3 estimator  = (r - 1) - log(r)  ≥ 0 everywhere
    kl = (ratio - 1) - log_ratio  # [B, T]

    # ── entropy bonus ────────────────────────────────────────────────────────
    entropy = -current_log_probs  # token-level entropy proxy

    # ── masked aggregate ─────────────────────────────────────────────────────
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
        "ratio/mean": (ratio * loss_mask).sum().item() / denom.item(),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Weight push to sglang
# ---------------------------------------------------------------------------

def push_lora_weights_to_sglang(
    model,
    sglang_base_url: str,
    temp_dir: str,
    api_key: str | None = None,
) -> bool:
    """
    Merge LoRA adapters into the base model and push all parameters to sglang
    via its HTTP ``update_weights_from_tensor`` endpoint.

    This is called after every ``update_weights_interval`` training steps so
    that the sglang rollout server always uses the latest policy weights.

    Returns True on success, False on failure.
    """
    try:
        import unsloth  # noqa: F401
    except ImportError:
        logger.warning("[weight_push] unsloth not available; skipping weight push.")
        return False

    logger.info("[weight_push] Merging LoRA and pushing weights to sglang …")
    t0 = time.time()

    # Temporarily merge LoRA into the base for parameter iteration
    # We use save_pretrained + reload to avoid modifying the training model.
    merge_path = os.path.join(temp_dir, "merged_for_push")
    try:
        model.save_pretrained_merged(merge_path, save_method="merged_16bit",
                                     tokenizer=None)
    except AttributeError:
        # Fallback: merge_and_unload then save
        merged = model.merge_and_unload()
        merged.save_pretrained(merge_path)

    # Reload with the most specific Transformers auto-loader available so the
    # merged model can expose all parameters (including vision blocks) on CPU.
    import transformers

    loader_candidates = [
        getattr(transformers, "AutoModelForImageTextToText", None),
        getattr(transformers, "AutoModelForVision2Seq", None),
        getattr(transformers, "AutoModelForCausalLM", None),
        getattr(transformers, "AutoModel", None),
    ]
    merged_cpu = None
    load_errors: list[str] = []
    for loader in [candidate for candidate in loader_candidates if candidate is not None]:
        try:
            merged_cpu = loader.from_pretrained(
                merge_path,
                torch_dtype=torch.float16,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
            break
        except Exception as exc:
            load_errors.append(f"{loader.__name__}: {exc}")

    if merged_cpu is None:
        raise RuntimeError(
            "[weight_push] Failed to reload merged model for SGLang weight push: "
            + "; ".join(load_errors)
        )

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{sglang_base_url}/update_weights_from_tensor"
    failed = 0
    for name, param in merged_cpu.named_parameters():
        tensor_bytes = param.data.numpy().tobytes()
        payload = {
            "parameter_name": name,
            "dtype": str(param.dtype).replace("torch.", ""),
            "shape": list(param.shape),
        }
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                files={"tensor": tensor_bytes},
                timeout=120,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("[weight_push] failed to push %s: %s", name, exc)
            failed += 1

    if failed:
        logger.warning("[weight_push] %d parameters failed to push.", failed)

    # Flush the KV cache so the next generation uses fresh weights
    try:
        requests.post(f"{sglang_base_url}/flush_cache", headers=headers, timeout=30)
    except Exception:
        pass

    elapsed = time.time() - t0
    logger.info("[weight_push] Done in %.1f s (%d params failed).", elapsed, failed)

    # Clean up temp merge directory
    import shutil
    shutil.rmtree(merge_path, ignore_errors=True)

    del merged_cpu
    torch.cuda.empty_cache()
    return failed == 0


# ---------------------------------------------------------------------------
# Sample dataset helper
# ---------------------------------------------------------------------------

class RolloutDataset(Dataset):
    """Flat list of Sample objects collected from the OpenClaw API server."""

    def __init__(self, samples):
        self.samples = samples  # list[Sample]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _pad_or_truncate(ids: list[int], max_len: int, pad_id: int) -> list[int]:
    ids = ids[:max_len]
    ids = ids + [pad_id] * (max_len - len(ids))
    return ids


def _merge_multimodal_train_inputs(chunks: list[dict | None]) -> dict | None:
    if not chunks:
        return None

    values_by_key: dict[str, list[torch.Tensor]] = {}
    for chunk in chunks:
        if not chunk:
            continue
        for key, value in chunk.items():
            if isinstance(value, torch.Tensor):
                values_by_key.setdefault(key, []).append(value)

    merged = {
        key: torch.cat(values, dim=0)
        for key, values in values_by_key.items()
        if values
    }
    return merged or None


def collate_samples(batch, pad_token_id: int, max_seq_len: int, device: torch.device):
    """
    Build tensors from a list of Sample objects.

    Returns a dict with:
      input_ids       [B, T]
      attention_mask  [B, T]
      loss_mask       [B, T]   – 1 on response tokens
      old_log_probs   [B, T]   – from rollout
      advantages      [B]
    """
    # Determine per-sample lengths
    all_ids, all_masks, all_loss_masks, all_lp, all_adv = [], [], [], [], []
    multimodal_chunks = [getattr(sample, "multimodal_train_inputs", None) for sample in batch]
    has_multimodal = [chunk is not None for chunk in multimodal_chunks]
    if any(has_multimodal) and not all(has_multimodal):
        raise ValueError(
            "Mixed multimodal and text-only samples in the same mini-batch are not supported. "
            "Group samples by modality before collation."
        )

    max_len = min(max(len(s.tokens) for s in batch), max_seq_len)

    for s, multimodal_train_inputs in zip(batch, multimodal_chunks, strict=True):
        tokens = _pad_or_truncate(s.tokens, max_len, pad_token_id)
        prompt_len = len(s.tokens) - (s.response_length or 0)

        # loss_mask: 1 on response tokens only (from Sample.loss_mask)
        if s.loss_mask is not None:
            resp_mask = _pad_or_truncate(s.loss_mask, s.response_length, 0)
        else:
            resp_mask = [1] * (s.response_length or 0)

        # Pad loss_mask to full sequence length
        full_mask = [0] * prompt_len + resp_mask
        full_mask = _pad_or_truncate(full_mask, max_len, 0)

        # attention_mask: 1 for real tokens, 0 for padding positions
        attn_mask = [1] * min(len(s.tokens), max_len) + [0] * max(0, max_len - len(s.tokens))

        # rollout log-probs (response tokens only, aligned to full sequence)
        lp = s.rollout_log_probs or [0.0] * (s.response_length or 0)
        lp = _pad_or_truncate(lp, s.response_length, 0.0)
        full_lp = [0.0] * prompt_len + lp
        full_lp = _pad_or_truncate(full_lp, max_len, 0.0)

        # advantage = reward score (no normalization per --disable-rewards-normalization)
        if isinstance(s.reward, dict):
            adv = float(s.reward.get("score", 0.0))
        else:
            adv = float(s.reward) if s.reward is not None else 0.0

        all_ids.append(tokens)
        all_masks.append(attn_mask)
        all_loss_masks.append(full_mask)
        all_lp.append(full_lp)
        all_adv.append(adv)

        if multimodal_train_inputs is not None:
            s.multimodal_train_inputs = multimodal_train_inputs

    batch_multimodal_inputs = _merge_multimodal_train_inputs(multimodal_chunks)
    if batch_multimodal_inputs is not None:
        batch_multimodal_inputs = {
            key: value.to(device)
            for key, value in batch_multimodal_inputs.items()
        }
    return {
        "input_ids": torch.tensor(all_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(all_masks, dtype=torch.long, device=device),
        "loss_mask": torch.tensor(all_loss_masks, dtype=torch.float32, device=device),
        "old_log_probs": torch.tensor(all_lp, dtype=torch.float32, device=device),
        "advantages": torch.tensor(all_adv, dtype=torch.float32, device=device),
        "multimodal_train_inputs": batch_multimodal_inputs,
    }


# ---------------------------------------------------------------------------
# Per-token log-prob helper
# ---------------------------------------------------------------------------

def get_token_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    multimodal_train_inputs: dict | None = None,
) -> torch.Tensor:
    """
    Run a forward pass and return per-token log-probs aligned with input_ids.

    Shape: [B, T] where position t holds log p(token_t | tokens_{<t}).
    The first position is always 0.0 (no prediction for the very first token).
    """
    forward_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if multimodal_train_inputs is not None:
        forward_kwargs.update(multimodal_train_inputs)
    outputs = model(**forward_kwargs)
    # logits: [B, T, V]
    logits = outputs.logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)  # [B, T, V]

    # Gather the log-prob of the actual next token
    # shifted: log p(t) = log_probs[:, t-1, token_t]
    # Result shape: [B, T-1]
    next_token_ids = input_ids[:, 1:]  # [B, T-1]
    gathered = log_probs[:, :-1, :].gather(
        dim=-1, index=next_token_ids.unsqueeze(-1)
    ).squeeze(-1)  # [B, T-1]

    # Prepend a zero so the tensor is [B, T] (aligned with input_ids)
    zeros = torch.zeros(gathered.shape[0], 1, device=gathered.device, dtype=gathered.dtype)
    return torch.cat([zeros, gathered], dim=1)  # [B, T]


# ---------------------------------------------------------------------------
# Reference log-prob helper (base model, no LoRA)
# ---------------------------------------------------------------------------

def get_ref_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    multimodal_train_inputs: dict | None = None,
) -> torch.Tensor:
    """
    Compute reference (base-model) log-probs by temporarily disabling LoRA.
    Works for PEFT LoRA models.
    """
    try:
        from peft import disable_adapter_layers, enable_adapter_layers
        disable_adapter_layers(model)
        with torch.no_grad():
            ref_lp = get_token_log_probs(
                model,
                input_ids,
                attention_mask,
                multimodal_train_inputs=multimodal_train_inputs,
            )
        enable_adapter_layers(model)
    except ImportError:
        # If PEFT is not available (plain Unsloth model without PEFT adapters),
        # fall back to current model log-probs.  The KL penalty term then
        # evaluates to 0 everywhere (ratio=1, log_ratio=0 ⟹ kl=0), effectively
        # disabling the KL regularisation for this batch.
        with torch.no_grad():
            ref_lp = get_token_log_probs(
                model,
                input_ids,
                attention_mask,
                multimodal_train_inputs=multimodal_train_inputs,
            )
    return ref_lp


# ---------------------------------------------------------------------------
# Rollout sample collection
# ---------------------------------------------------------------------------

def drain_output_queue(
    output_queue: queue.Queue,
    target: int,
    timeout_no_progress: float = 60.0,
):
    """
    Drain *target* completed sample groups from *output_queue*.

    Returns a flat list of Sample objects.
    """
    samples = []
    completed_groups: dict[int, list] = {}
    last_progress = time.time()

    while len(samples) < target:
        # Drain everything currently available
        while True:
            try:
                group_id, group = output_queue.get_nowait()
                completed_groups[group_id] = group
                last_progress = time.time()
            except queue.Empty:
                break

        # Move complete groups into the flat list
        for gid in sorted(list(completed_groups.keys())):
            if len(samples) >= target:
                break
            group = completed_groups.pop(gid)
            # Skip groups with aborted samples
            from slime.utils.types import Sample as SlimeSample
            if any(s.status == SlimeSample.Status.ABORTED for s in group):
                continue
            samples.extend(group)

        if time.time() - last_progress > timeout_no_progress:
            logger.warning(
                "[drain] still waiting for samples: %d/%d collected", len(samples), target
            )
            last_progress = time.time()

        if len(samples) < target:
            time.sleep(0.05)

    return samples[:target]


def iter_homogeneous_mini_batches(samples, mini_batch_size: int):
    multimodal_samples = [
        sample for sample in samples if getattr(sample, "multimodal_train_inputs", None) is not None
    ]
    text_only_samples = [
        sample for sample in samples if getattr(sample, "multimodal_train_inputs", None) is None
    ]

    for bucket in (multimodal_samples, text_only_samples):
        for start in range(0, len(bucket), mini_batch_size):
            yield bucket[start:start + mini_batch_size]


# ---------------------------------------------------------------------------
# sglang subprocess management
# ---------------------------------------------------------------------------

def start_sglang_server(
    model_path: str,
    host: str,
    port: int,
    tp: int = 1,
    mem_fraction: float = 0.85,
    context_length: int = 32768,
    reasoning_parser: str = "qwen3",
    tool_call_parser: str | None = "qwen3_coder",
    served_model_name: str = "model",
    env_extra: dict | None = None,
) -> subprocess.Popen:
    """
    Launch an sglang OpenAI-compatible server in a subprocess.

    When the environment variable ``SGLANG_CUDA_VISIBLE_DEVICES`` is set,
    the subprocess uses only those GPU indices for inference (e.g. the rollout
    GPU(s) separate from the training GPU(s)).  This is how the launch scripts
    isolate training and rollout onto different physical devices.

    The caller is responsible for calling proc.terminate() on exit.
    """
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--host", host,
        "--port", str(port),
        "--tp", str(tp),
        "--mem-fraction-static", str(mem_fraction),
        "--context-length", str(context_length),
        "--served-model-name", served_model_name,
    ]
    if reasoning_parser:
        cmd += ["--reasoning-parser", reasoning_parser]
    if tool_call_parser:
        cmd += ["--tool-call-parser", tool_call_parser]

    env = dict(os.environ)
    # Respect SGLANG_CUDA_VISIBLE_DEVICES to pin sglang to rollout GPUs
    sglang_devices = os.environ.get("SGLANG_CUDA_VISIBLE_DEVICES", "")
    if sglang_devices:
        env["CUDA_VISIBLE_DEVICES"] = sglang_devices
        logger.info("[sglang] Using CUDA_VISIBLE_DEVICES=%s for sglang", sglang_devices)
    if env_extra:
        env.update(env_extra)

    logger.info("[sglang] Launching: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, env=env)
    return proc


def wait_for_sglang_ready(host: str, port: int, timeout: float = 300.0) -> bool:
    """Poll sglang /health until it responds 200 or *timeout* is exceeded."""
    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                logger.info("[sglang] Server ready at %s:%d", host, port)
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    logger.error("[sglang] Server did NOT become ready within %.0f s", timeout)
    return False


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # ── 0. setup ─────────────────────────────────────────────────────────────
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
        logger.info("[train] Using accelerate: rank=%d device=%s", local_rank, device)
    except ImportError:
        accelerator = None
        is_main = True
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load model with Unsloth QLoRA ──────────────────────────────────────
    try:
        from unsloth import FastVisionModel
    except ImportError:
        raise SystemExit(
            "[unsloth_qlora_trainer] unsloth is not installed.\n"
            "  Install via: pip install 'unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git'\n"
            "  Or see https://github.com/unslothai/unsloth for your CUDA version."
        )

    logger.info("[train] Loading model %s (4bit=%s) …", args.hf_checkpoint, args.load_in_4bit)
    base_model, processor = FastVisionModel.from_pretrained(
        model_name=args.hf_checkpoint,
        max_seq_length=args.max_seq_length,
        dtype=None,          # auto-detect (bf16 on Ampere+)
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )
    tokenizer = getattr(processor, "tokenizer", processor)

    logger.info("[train] Applying LoRA adapters (r=%d, alpha=%d) …", args.lora_r, args.lora_alpha)
    peft_kwargs = {}
    if args.lora_target_modules:
        peft_kwargs["target_modules"] = args.lora_target_modules
    model = FastVisionModel.get_peft_model(
        base_model,
        finetune_vision_layers=args.finetune_vision_layers,
        finetune_language_layers=args.finetune_language_layers,
        finetune_attention_modules=args.finetune_attention_modules,
        finetune_mlp_modules=args.finetune_mlp_modules,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimised checkpointing
        random_state=args.seed,
        use_rslora=False,
        **peft_kwargs,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    FastVisionModel.for_training(model)
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
        logger.info("[train] Starting sglang rollout server (tp=%d) on port %d …",
                    args.sglang_tp, args.sglang_port)
        sglang_proc = start_sglang_server(
            model_path=args.hf_checkpoint,
            host=args.sglang_host,
            port=args.sglang_port,
            tp=args.sglang_tp,
            mem_fraction=args.sglang_mem_fraction,
            context_length=args.sglang_context_length,
            reasoning_parser=args.sglang_reasoning_parser,
            tool_call_parser=args.sglang_tool_call_parser,
            served_model_name=args.served_model_name,
        )
        if not wait_for_sglang_ready(args.sglang_host, args.sglang_port):
            sglang_proc.terminate()
            raise RuntimeError("sglang server failed to start.")

    # ── 4. Start OpenClaw API proxy server ───────────────────────────────────
    # Build a minimal Namespace that openclaw_api_server.OpenClawAPIServer expects.
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
    )

    output_queue: queue.Queue = queue.Queue(maxsize=200_000)
    submission_enabled = threading.Event()

    if is_main:
        # Import here so PYTHONPATH adjustments take effect
        try:
            from openclaw_api_server import OpenClawAPIServer
        except ImportError:
            raise SystemExit(
                "[unsloth_qlora_trainer] Could not import openclaw_api_server. "
                "Run this script from the openclaw-rl directory or add it to PYTHONPATH."
            )

        # Override environment variables consumed by OpenClawAPIServer
        os.environ.setdefault("HOST", args.proxy_host)
        os.environ.setdefault("PORT", str(args.proxy_port))
        os.environ.setdefault("SERVED_MODEL_NAME", args.served_model_name)

        api_server = OpenClawAPIServer(
            args=proxy_args,
            output_queue=output_queue,
            submission_enabled=submission_enabled,
        )
        api_server.start()
        logger.info("[train] OpenClaw proxy listening on %s:%d", args.proxy_host, args.proxy_port)

    # ── 5. WandB (optional) ───────────────────────────────────────────────────
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
            logger.warning("[train] wandb not installed; skipping.")

    # ── 6. Training loop ──────────────────────────────────────────────────────
    sglang_base_url = f"http://{args.sglang_host}:{args.sglang_port}"
    grad_accum_steps = max(1, args.global_batch_size // args.mini_batch_size)
    temp_dir = tempfile.mkdtemp(prefix="qlora_push_")

    logger.info("[train] Starting RL loop (rollout_batch=%d, global_bs=%d, accum=%d) …",
                args.rollout_batch_size, args.global_batch_size, grad_accum_steps)

    for rollout_id in range(args.num_rollout):
        # ── 6a. Collect rollout data ──────────────────────────────────────────
        if is_main:
            submission_enabled.set()    # allow sample collection
            logger.info("[train] rollout %d: collecting %d samples …",
                        rollout_id, args.rollout_batch_size)
            samples = drain_output_queue(output_queue, target=args.rollout_batch_size)
            submission_enabled.clear()  # pause while we train
        else:
            samples = []

        # Synchronise across processes if using multi-GPU DDP
        if accelerator is not None and accelerator.num_processes > 1:
            import torch.distributed as dist
            # broadcast_object_list handles variable-size objects in one collective op
            obj_list = [samples]
            dist.broadcast_object_list(obj_list, src=0)
            samples = obj_list[0]

        if not samples:
            logger.warning("[train] rollout %d: no samples collected, skipping.", rollout_id)
            continue

        # ── 6b. Train one GRPO update ─────────────────────────────────────────
        model.train()
        optimizer.zero_grad()

        # Split into mini-batches for gradient accumulation
        total_loss = 0.0
        all_metrics: list[dict] = []
        step_count = 0

        for mb_samples in iter_homogeneous_mini_batches(samples, args.mini_batch_size):
            batch = collate_samples(mb_samples, pad_id, args.max_seq_length, device)

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            loss_mask = batch["loss_mask"]
            old_lp = batch["old_log_probs"]
            advantages = batch["advantages"]
            multimodal_train_inputs = batch["multimodal_train_inputs"]

            # Reference log-probs (base model, LoRA disabled)
            with torch.no_grad():
                ref_lp = get_ref_log_probs(
                    model,
                    input_ids,
                    attention_mask,
                    multimodal_train_inputs=multimodal_train_inputs,
                )

            # Current policy log-probs
            cur_lp = get_token_log_probs(
                model,
                input_ids,
                attention_mask,
                multimodal_train_inputs=multimodal_train_inputs,
            )

            loss, metrics = compute_grpo_loss(
                current_log_probs=cur_lp,
                old_log_probs=old_lp,
                ref_log_probs=ref_lp,
                advantages=advantages,
                loss_mask=loss_mask,
                eps_clip=args.eps_clip,
                eps_clip_high=args.eps_clip_high,
                kl_coef=args.kl_coef,
                entropy_coef=args.entropy_coef,
            )

            scaled_loss = loss / grad_accum_steps
            if accelerator is not None:
                accelerator.backward(scaled_loss)
            else:
                scaled_loss.backward()

            total_loss += loss.item()
            all_metrics.append(metrics)
            step_count += 1

        # Gradient clipping and optimizer step
        if accelerator is not None:
            accelerator.clip_grad_norm_(model.parameters(), args.gradient_clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

        optimizer.step()

        # Log aggregated metrics
        if is_main and all_metrics:
            agg: dict[str, float] = {}
            for k in all_metrics[0]:
                agg[k] = sum(m[k] for m in all_metrics) / len(all_metrics)
            agg["loss/total_accumulated"] = total_loss / max(step_count, 1)
            agg["rollout_id"] = rollout_id

            logger.info(
                "[train] rollout %d | loss=%.4f pg=%.4f kl=%.4f ent=%.4f",
                rollout_id,
                agg.get("loss/total", 0),
                agg.get("loss/pg", 0),
                agg.get("loss/kl", 0),
                agg.get("loss/entropy", 0),
            )
            if wandb_run is not None:
                wandb_run.log(agg)

        # ── 6c. Save LoRA checkpoint ──────────────────────────────────────────
        if is_main and (rollout_id + 1) % args.save_interval == 0:
            ckpt_dir = save_dir / f"step_{rollout_id + 1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model) if accelerator else model
            unwrapped.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            logger.info("[train] Saved LoRA checkpoint to %s", ckpt_dir)

        # ── 6d. Push weights to sglang ────────────────────────────────────────
        if is_main and (rollout_id + 1) % args.update_weights_interval == 0:
            unwrapped = accelerator.unwrap_model(model) if accelerator else model
            push_lora_weights_to_sglang(unwrapped, sglang_base_url, temp_dir)

    # ── 7. Cleanup ────────────────────────────────────────────────────────────
    if is_main:
        # Final save
        final_dir = save_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model) if accelerator else model
        unwrapped.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        logger.info("[train] Final LoRA checkpoint saved to %s", final_dir)

        if wandb_run is not None:
            wandb_run.finish()

        if sglang_proc is not None:
            sglang_proc.terminate()
            logger.info("[train] sglang server terminated.")

    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
