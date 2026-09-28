# Qwen3.5-4B single-GPU validation record

## Method and scope

This is a native supervised QLoRA experiment on the official Qwen3.5-4B
checkpoint, not Google's official Dream-RSI implementation and not a broad
coding benchmark. Training uses reference solutions. Development results
select checkpoints; a separately preregistered in-domain suite confirms the
selected candidate under a matched generation budget.

The second distribution was chosen after an earlier rejected result. It is
reported separately, not pooled with the first distribution or presented as
an unbiased estimate across all coding tasks. Algorithm families are covered
by the training curriculum, and common edge cases can recur in independently
generated cases. The result does not establish unseen-algorithm generalization
or population-level statistical significance.

## Fixed controls

- Exactly one visible CUDA device for this experiment: physical GPU 0 on the
  owner's `.51` Ubuntu host. Unrelated processes are not stopped or restarted.
- Official model revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; both
  local weight shards matched their official Hugging Face SHA-256 hashes.
- NF4 double quantization, BF16 compute, rank-8/alpha-16 LoRA. Only adapters
  are updated; the base checkpoint remains unchanged.
- Greedy evaluation, at most 256 generated tokens, identical for base and
  candidate. Gold outputs remain in the trusted parent evaluator.
- Candidate code runs in a digest-pinned unprivileged Docker container with
  no network, no host mounts, a read-only filesystem, and resource limits.
- Each optimizer update requires a new evidence-bound Muse Spark 1.3 maximum-
  reasoning approval. Promotion and final review are separate veto gates.
- Final suites are hashed before training and claimed once across run folders.
  Both base and candidate confirmation runs happen only after checkpoint choice.
- No existing serving endpoint is updated; experimental approval is not deploy.

## First corrected trial: rejected

The first trial completed four real optimizer updates. Development improved
from 18/27 to 21/27; the selected checkpoint was epoch 2. Exact adapter reload
checks passed for every saved checkpoint.

The fixed final suite scored **235/240 for the base and 231/240 for the
candidate**, with one complete interval-merging task lost and no complete task
won. Deterministic promotion and independent Muse reviews returned NOT_OK.
The model was not promoted or deployed. All result JSON and review receipts
are preserved under `validation/qwen35-4b-single/round1-rejected/`.

That near-ceiling base result also could not support an absolute eight-point
improvement. The next round uses a different, explicitly harder distribution;
the old rejection is not relabeled or treated as successful confirmation.

## Second round: preregistered plan

- 46 verified reference training examples; 12 development tasks.
- Three development regression guards covering closed-interval merging,
  bracket matching and coin change. A candidate must preserve each guard's
  number of passed cases to be eligible for checkpoint selection.
- Learning rate 5e-5, at most six full-batch updates, seed 20260929.
- Final distribution: 12 task families, 20 cases each (240 cases).
- Fixed gate: at least +0.08 case accuracy, final accuracy >=0.75, at least
  two complete-task wins, zero complete-task losses, preserved development
  guard counts, independent Muse promotion OK, and final review OK.
- Source bundle SHA-256:
  `e9612ebadd64697ed7a13cb2fb072996989d12e3bac132527345caa62f9b8462`.
- Preregistered suite SHA-256:
  `00f7011c4e60ad63f1bf011aa57af2e0c3f19efd96fe90b23b87e919d21dfca1`.

Preflight checks: 82 automated tests, all 46 reference programs evaluated in
Docker, and 240 independent oracle/reference cross-checks passed. The original
code filter's rejection of benign underscore variable names and a mislabeled
grid-path fixture were fixed before relying on candidate results.

Second-round final measurements are not asserted here until its saved report,
review receipts and fresh-process reload verification are available.

## Second-round measured outcome: improved but rejected

The second round completed four approved updates and retained epoch 2 based on
development performance (27/39 before, 30/39 at the selected checkpoint).
Confirmation improved from **106/240 (44.17%) to 146/240 (60.83%)**: two complete
task wins, zero complete-task losses, and preserved development guards.

The +16.67-point gain was real on that fixed suite, but final accuracy was below
the unchanged 75% floor. The deterministic gate, Muse promotion review and final
review all rejected promotion. This is not a PASS and was not deployed. Reports
and review receipts are preserved under `round2-rejected/` alongside round 1.

## Third-round plan: verified repairs with self-replay

The third round starts from the unpromoted, development-selected round-2 adapter.
For 58 training prompts, it verifies the current model's responses. Correct
responses become self-replay targets (weight 1); failures use reference repairs
(weight 4). This fixed derived dataset is hashed before any optimizer update.
Learning rate is 2e-5, with at most six independently approved full-batch updates.

Previously exposed round-2 confirmation cases are now openly treated as
**development data**. Final round-3 inputs are separately generated and hashed
before the run. Prompt/algorithm templates overlap training and development;
this is known-task program-correctness adaptation, not unseen-prompt or unseen-
algorithm generalization. The original base remains the final comparator.

All promotion criteria and independent review requirements remain in force,
including the 75% floor. No third-round outcome is asserted before the actual
saved measurements, review receipts and a fresh-process reload check exist.
