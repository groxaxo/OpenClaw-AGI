# Single-GPU BF16 guarded continuation

## Status

This continuation is undergoing real validation. No promotion PASS is claimed
until report, independent reviewer receipts and cold-reload verification exist.

Prior NF4 runs remain rejected. The third verified-repair trial did not produce
a guard-preserving development improvement; its report and seven reviewer
receipts are preserved in validation/qwen35-4b-single/round3-rejected/.

## Fixed protocol

The new runner is qwen4b_bf16_guarded.py, with a separate one-GPU launcher.
It starts from the original frozen BF16 Qwen3.5-4B model, not from an unpromoted
quantized adapter. Rank-8 LoRA is trained using verified current answers and
canonicalized verified reference repairs; correct public guard answers receive
additional replay weight. Every optimizer update requires a fresh Muse Spark
1.3 max-reasoning approval tied to the actual code and evidence.

A public development-only line search tests full, half and quarter proposed
adapter deltas. It retains the largest change that preserves the named public
regression guards and does not reduce aggregate DEV case accuracy. Otherwise it
reverts the proposal and halves the learning rate. Adam state is reset after a
shrunk or reverted proposal. Best-checkpoint selection uses DEV only.

Base and candidate use the same 512-token deterministic generation budget.
Do not compare these absolute scores directly with earlier 256-token trials.
The confirmation suite is preregistered and evaluated once only after candidate
selection. Its algorithm/prompt families are known and trained; new case inputs
are not evidence of unseen-task or general coding capability gains.

The original engineering promotion gate is unchanged: at least +0.08 absolute
case accuracy, >=0.75 final accuracy, >=2 complete-task wins, zero complete-task
losses, public-guard preservation and independent Muse promotion approval.
A final review must also approve. No production model is automatically deployed.

This remains supervised adaptation with guarded acceptance, not Google's
code-evolving Dream-RSI reference implementation or a statistical significance
claim over a broad task population.

## Integrity and preflight

The original Hugging Face cache was missing before this continuation's first
attempt could train. The exact official revision was restored into a task-local
directory. Both weight shards matched their official LFS SHA-256 digests.
The failed startup is preserved and is not counted as a training run.

91 unit/regression tests and 58 reference programs passed. Single-GPU launcher
syntax and rejection of a multiple-GPU selector were checked. Runtime and test
configuration are tied to the source hash in the preregistration evidence.

No GitHub Actions, service restarts or model deletion were used. All model work
runs on one RTX 3090 on the authorized .51 host, reached through Desktop
Commander's Mac connection and SSH because the direct .51 agent is offline.
