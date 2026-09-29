# Qwen3.5-4B promotion lineage

## Verified starting point

Generation 1 is documented in `VALIDATION_BF16_GUARDED.md` and was merged to
`main` in PR #6 (`e917cffcc7c5d43b58a553fb736d9eaae4182fdd`). It scored
176/240 -> 204/240 against the original frozen BF16 base on its registered
confirmation suite, retained all 11 separate guard checks, passed Muse max
promotion/final reviews, and reproduced the exact outputs in a fresh process.
Its logical adapter SHA256 is
`4350f1d97ab1402eae61f24d501da6d3c4f26826ff2d2cc656dfdfa9d89e7b69`.

## What constitutes a successful next run

A continuation must load the exact **last approved adapter**, verify its file
and logical tensor hashes, and compare its child against that parent. Comparing
a succession of candidates only to the original base does not demonstrate
successive improvements. Optimizer steps, approved proposals, or repeated
reproduction checks do not increment the successful-generation count.

The unchanged engineering promotion gate requires >=0.08 absolute case-accuracy
gain, >=0.75 child accuracy, >=2 newly complete task families, zero lost complete
families, preserved development guards, and an independent Muse max approval.
The final audit must approve, and a new-process locked-checkpoint reproduction
must pass before the child becomes the next parent. Rejected/interrupted trials
remain in the record but never replace the accepted checkpoint.

The first continuation uses one visible GPU, BF16 + the existing rank-8 LoRA,
maximum four optimizer proposals, LR 2.5e-5, replay weight 4, reference repair
weight 12, and guard weight 16. Full/half/quarter proposal sizes are evaluated
only on public development; regressions are reverted. Every optimizer proposal
requires its own hash-bound, single-use Muse approval.

The published generation-1 confirmation cases are explicitly retired into
DEVELOPMENT, merged by exact prompt with prior development and deduplicated.
They are not claimed as unseen evidence. A new 12-family/240-case confirmation
is preregistered before training and excludes all exact public inputs. It is
consumed once, only after development selects a fixed candidate. The comparison
uses identical prompts, deterministic decoding and 512-token budgets for parent
and child. This measures known-task, in-domain program correctness, not broad
unseen-task generalization, population-level significance, or official Dream-RSI.

## Saturation and honest stopping

An eight-point minimum improvement becomes mathematically unavailable on a
fixed distribution above 92% parent accuracy. The runner must not lower that
threshold, reset the comparator to the original base, or rename reproductions
as new successes. Report the reached generation and the plateau/ceiling; any
new benchmark or acceptance policy must be a separately declared experiment.

Use `run_qwen4b_continuation.sh` with the explicit parent acceptance record,
pinned model, retired parent suite hash and fresh confirmation hash. The script
rejects multiple GPU selectors, locks its GPU, runs tests first, applies a hard
runtime limit, and does not deploy or alter an existing model server.

## Current state

Generation 1 is verified. The first continuation is prepared; no generation-2
success is claimed until its measured report, reviewer receipts and cold
reproduction are recorded. Machine-readable lineage evidence will be kept
under `validation/qwen35-4b-chain/`.
