# Parent-relative protected minimal-repair experiment

This continues the **approved generation-1** Qwen3.5-4B BF16 rank-8 LoRA
checkpoint, not an unpromoted intermediate and not the original unadapted base.
It is a known-prompt, in-domain supervised experiment, not Google's official
Dream-RSI implementation or evidence of broad recursive self-improvement.

## Why this separate trial exists

Both completed broad-repair continuations failed to improve public development:
372/435 cases, ten of twelve complete families. The weight-8 experiment previously
reported as resource-blocked was subsequently retried with adequate headroom;
all four proposals and their smaller alternatives were rejected. Its actual
report, parent-load proof and five reviewer receipts are retained under
`validation/qwen35-4b-chain/weight8-completed-rejected/`. It did not consume the
registered confirmation or replace the approved adapter.

## Proposed changes, fixed before execution

Two public failures remain: window max-minus-min uses the wrong input unpack,
and sum-excluding-self interprets 'every other' as alternating positions.
Their existing generated programs are minimally edited; all other characters,
including surrounding formatting, are retained. Both targets must pass all
existing public cases (32 and 33 respectively) before training.

Only those two equally weighted targets produce the repair loss. Each currently
correct public development answer and each regression guard contributes two
protected gradient directions: mean response negative log likelihood and the
mean of its three highest-negative-log-likelihood response tokens. For this
parent that gives 26 directions from 13 responses.

An AdamW displacement is projected off the span of these measured gradients.
Projection uses normalized gradient rows and a small Gram pseudoinverse with a
registered relative eigenvalue cutoff of 1e-7. This is only **first-order local
protection**, not a proof that generated outputs will remain correct. Actual
Docker-graded tests still decide whether any full, half or quarter displacement
can be retained. Complete development tasks and guards may not regress, and
aggregate development accuracy may not fall. Otherwise the proposal is reverted
and learning rate halved. Adam moments reset after each projected proposal.

The two changes (target selection and projection) are combined; this is not a
single-factor attribution of any gain. A controlled ablation would be needed
before crediting either component separately.

## Unchanged acceptance and resource constraints

The registered run uses one physical RTX 3090 (GPU 0 on .51), the same original
frozen BF16 base and existing rank-8 adapter configuration, initial LR 5e-5,
at most three independently reviewed optimizer proposals, and identical
512-token deterministic generation for parent/child evaluation. The inherited
`repair_weight` metadata/CLI field is unused by this equal-weight two-target
backend; it does not change the actual losses.

The fresh 240-case confirmation was already preregistered before all these
public development searches and remains inaccessible to checkpoint selection.
Exact public inputs were excluded when it was built. Its generated gold labels
were independently cross-checked before the earlier runs. Reusing an unconsumed
registration is not reusing observed holdout outcomes. It can be consumed once,
only after public results meet the unchanged prerequisites.

Promotion requires >=8 percentage points gain against the immediate approved
parent, >=75% child accuracy, >=2 new complete families, zero complete-family
losses, preserved guards, independent Muse max approval, final review, and cold
reproduction. These are engineering gates, not statistical significance claims;
240 cases do not represent 240 independent LLM tasks. A rejected proposal,
optimizer step or successful reproduction never increments generation count.

No production-serving state changes and no GitHub Actions are introduced. The
launcher refuses multiple GPU selectors, holds the shared exclusive experiment
lock, creates unique output directories and bounds runtime. Existing GPU jobs
are not killed. Any accepted child remains an experimental artifact.

## Reproduction entry point

`openclaw-rl/run_qwen4b_projected.sh` accepts explicit model, parent approval,
retired-public suite, fresh confirmation and digest-pinned Docker image paths.
`CHECK_ONLY=1` executes tests without training. Full runs require an unconsumed
confirmation; after a suite is consumed, cold reproduction may check only the
already-locked candidate and cannot select another checkpoint.

## Result

The current run is not yet a completed promotion. Results and the exact
successful-generation count will be recorded with the final evidence.
