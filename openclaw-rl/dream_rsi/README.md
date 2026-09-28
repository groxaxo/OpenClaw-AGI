# Dream-RSI controller for OpenClaw-RL

This module adds a **paper-inspired Dream-RSI meta-controller** above OpenClaw-RL's existing PRM + GRPO pipeline.

It is intentionally **not presented as Google's official implementation**. The official Dream-RSI code was still unreleased when this integration was authored. The parts implemented here are the reproducible ideas exposed by the paper: frozen replay, prefix/trace-only observability, an exploration/exploitation/recovery portfolio, efficiency-aware scoring, policy mutation, and a paired holdout promotion gate.

## What it changes

When `--dream-rsi-enable` is off (the default), training behavior is unchanged.

When enabled:

1. OpenClaw collects a candidate pool larger than the GRPO batch.
2. Each `Sample` already contains its PRM reward; OpenClaw now also preserves `session_id` and `turn` in `Sample.metadata`.
3. The Dream controller chooses the training batch using a versioned declarative search policy.
4. Candidate pools are persisted to an append-only, content-free `replay.jsonl` trace (no prompt or response text).
5. Every `N` rollouts the controller creates bounded challenger policies and evaluates them against frozen history.
6. A challenger must beat the incumbent on replay **and** pass a paired holdout gate on the newest pools before `policy.json` is atomically replaced.

This first integration evolves **the data/search curriculum**, not model code. The base model continues learning through the existing QLoRA/GRPO loop.

## Enable

Example for the existing 3×3090 recipe:

```bash
python openclaw-rl/unsloth_qlora_trainer.py \
  ...existing arguments... \
  --dream-rsi-enable \
  --dream-rsi-pool-factor 2.0 \
  --dream-rsi-evolve-interval 8 \
  --dream-rsi-mutations 24 \
  --dream-rsi-holdout-pools 4
```

State is written below `<save>/dream_rsi/`:

```text
dream_rsi/
├── policy.json            # active, bounded controller policy
├── policy_history.jsonl   # promotion/rejection audit trail
└── replay.jsonl           # frozen candidate pools, no conversation content
```

## Offline replay

```bash
cd openclaw-rl
python -m dream_rsi.cli /path/to/checkpoints/dream_rsi/replay.jsonl \
  --policy /path/to/checkpoints/dream_rsi/policy.json \
  --target 16 \
  --mutations 64
```

## Safety and reproducibility properties

- **Feature flagged:** zero runtime effect unless explicitly enabled.
- **No arbitrary generated code execution:** policy evolution is a constrained numeric configuration surface.
- **Deterministic replay:** same trace + same policy + same seed produces the same result.
- **No prompt/response retention:** replay stores only compact controller fields.
- **Fresh holdout:** the newest pools are withheld from challenger search and used only for promotion.
- **Atomic policy promotion:** `policy.json` is written through a temporary file + `os.replace`.
- **Rollback:** previous policy decisions remain in `policy_history.jsonl`.

## Why this shape

Dream-RSI's key insight is that an expensive live search can be frozen and replayed many times to improve the controller cheaply. OpenClaw-RL already supplies the expensive part—live agent trajectories and PRM evaluation—so this module reuses those signals rather than introducing a second judge stack.

The biggest difference from the paper is deliberate: this version evolves a constrained policy configuration instead of letting an LLM rewrite arbitrary Python. Once Google's official implementation is public, the replay and promotion interfaces here provide a stable place to test a code-evolving controller without coupling it directly to the trainer.
