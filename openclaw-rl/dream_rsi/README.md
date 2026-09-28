# Experimental replay-guided curriculum controller

This independent, default-off experiment applies ideas from Dream-RSI to **selection of already-scored training samples** in OpenClaw-RL. It is not Google's implementation and is not a faithful reproduction of the paper's online search algorithm.

## Scope and limitations

The controller sees the completed candidate pool, including PRM scores, before selecting a training batch. It performs retrospective curriculum selection, **not prefix-only discovery-tree replay**. It does not generate branches, schedule coding agents, rewrite policy source, or simulate unseen actions.

Policy mutations optimize a heuristic over PRM rewards and session diversity. A higher replay score is **not evidence of improved model capability**, training efficiency, or generalization. All pool candidates have already incurred their generation and judging costs. The legacy call/parallel terms in the surrogate are not measurements of saved inference calls and are constant across policies selecting the same batch size on the same pools.

The newest recorded pools are held out from challenger fitting. Training pools exclude their session IDs; heldout pools must also have mutually disjoint session IDs. A persisted evaluation cursor prevents holdout reuse, including after restart. These are **retrospective holdout pools, not fresh paired live runs** of the incumbent and challenger. The small sign-test gate is experimental; its per-cycle alpha does not establish a lifetime false-promotion bound across repeated cycles.

Before claiming benefits, run a separate matched-budget live comparison with fixed model/evaluator versions, heldout tasks, and task-success/cost measurements. The existing trainer still updates model weights and serves checkpoints independently of this controller's surrogate gate.

## Runtime behavior

With `--dream-rsi-enable` absent, the new sampler and replay evolution do not run. Sample metadata is still preserved. The accompanying PRM bug fix permits next-state scoring when disk conversation logging is disabled.

With the flag enabled:

1. Collect a larger PRM-scored pool and record compact observations.
2. Select valid, trainable samples using bounded policy parameters.
3. Exclude removed, failed, aborted and zero-loss-mask samples; never refill the batch with excluded samples. The trainer skips an undersized valid batch on all ranks.
4. Periodically fit deterministic policy mutations on historical pools and compare the best proposal on session-disjoint heldout pools.
5. Replace `policy.json` before switching the in-memory policy; record successful evolution decisions in the audit history.

PRM scoring must be enabled and its server reachable. Conversation text logging is not required. Controller traces omit prompt/response bodies and allowlist metadata to `has_next_state`; identifiers are retained, so traces are not claimed to be anonymized. Keep the state directory private. Use a single controller writer per state directory.

## Enable in an existing validated training configuration

Add these arguments to the existing trainer command, retaining its model, GPU, PRM endpoint and training arguments:

```text
--prm-enable
--dream-rsi-enable
--dream-rsi-pool-factor 2.0
--dream-rsi-evolve-interval 8
--dream-rsi-mutations 24
--dream-rsi-holdout-pools 4
--dream-rsi-min-holdout-pairs 4
```

For example, a batch target of 16 and pool factor 2 collects 32 candidates and selects at most 16 valid samples. This increases collection requirements; it is not a promise of lower total compute. Four heldout pools is only an experimental default, not a well-powered validation study.

State is under `<save>/dream_rsi/`:

```text
policy.json               Current bounded policy
policy_history.jsonl      Evolution decisions and metrics
replay.jsonl              Recorded candidate pools and evolution events
evaluation_state.json     Consumed holdout cursor
```

Policy files use atomic replacement. Malformed policy or trace input fails closed. Audit history supports inspection; there is no automatic production rollback service.

## CPU-only validation

From the repository root:

```bash
PYTHONPATH=openclaw-rl python3 -m unittest discover -s openclaw-rl/tests -p 'test_dream_rsi*.py' -v
python3 -m py_compile openclaw-rl/dream_rsi/*.py openclaw-rl/openclaw_api_server.py openclaw-rl/unsloth_qlora_trainer.py
```

See [VALIDATION.md](VALIDATION.md) for executed checks and remaining limits. No GitHub Actions workflow is needed.

## Offline replay

From `openclaw-rl`, passing paths to an existing recorded run:

```bash
python3 -m dream_rsi.cli /path/to/checkpoints/dream_rsi/replay.jsonl --policy /path/to/checkpoints/dream_rsi/policy.json --target 16 --mutations 64
```

The CLI reports retrospective metrics only and does not promote a policy. Targets and mutation counts must be positive.

## External CLI vetoes and native Qwen3.5-9B validation

`external_judge.py` invokes Muse Spark 1.3 with `--reasoning-effort max`
(and `--yolo`, but shell/writes/web disabled), or OpenCode GLM-5.3 with
`--variant max` and denied tool permissions. In `both` mode every required
reviewer must return a strict, evidence-bound `OK`. A rejection, timeout,
malformed output or stale source is a veto, never a reason to shop for another
approval. Review receipts are single-use. Deterministic checks remain required.

The **native validation backend**, `python -m dream_rsi.live_validation`, uses
all three GPUs as independent NF4/BF16 QLoRA replicas with synchronized LoRA
gradients. It exercises actual Qwen3.5-9B generation, immutable coding tests in
a digest-pinned unprivileged Docker container, replay-guided batch selection,
an external review before every optimizer step, adapter save/reload, and one
terminal matched-budget heldout comparison plus a separate release review.

This is an explicitly bounded integration smoke (at most three updates), not a
capability benchmark. Raw signed rewards are used as zero-baseline advantages;
this is PPO-style policy optimization, **not group-normalized GRPO**. Training
and heldout task IDs are disjoint. Holdout results are not used to retry or tune
within a run. Promotion requires the registered statistical gate, no task
regression, and all selected judges; otherwise the candidate is retained but
not activated. No existing serving process is restarted or overwritten.

This backend does **not** validate the legacy SGLang network/PRM path. The legacy
trainer now also requests CLI reviews before opted-in Dream updates, and cannot
publish updated serving weights without fresh checkpoint holdout evidence.
That path currently has no such evaluator, so publication fails closed. A fix
also makes its KL term depend on the frozen reference and uses PEFT's actual
`disable_adapter()` context rather than silently substituting the current model.

Run-specific commands, versions, reviewer outputs and measurements must be
recorded with the actual validation evidence. Do not treat a unit test or an
LLM opinion as evidence that a training run improved the model.

## Single-GPU Qwen3.5-4B continuation

`qwen4b_single.py` refuses to run unless exactly one CUDA device is visible.
It uses 32 independently tested reference solutions for supervised QLoRA and
an eight-task development set for checkpoint selection. This is **supervised
adaptation**, not a reproduction of Google's code-evolving Dream-RSI loop.

The new final suite contains 12 in-domain task families with 20 independently
computed small-input cases each. Several algorithm families overlap with the
training curriculum: it measures new inputs and reworded instructions, not
unseen-algorithm generalization or statistically significant broad capability.
Old unfinished/previously exposed final suites are not reused as confirmation.
A known old grid-path development label is corrected from 1 to 2 before runs.

The final suite is hashed before training and claimed once across run folders.
Both base and candidate are evaluated only after the development-selected
checkpoint is fixed. All generation budgets are matched. Promotion still
requires +0.08 case accuracy, final accuracy >=0.75, >=2 complete-task wins,
zero complete-task regressions, and an independent Muse max approval. Every
optimizer update requires a separate source/evidence-bound approval. A crash,
missing receipt or negative final review is never reported as PASS.
