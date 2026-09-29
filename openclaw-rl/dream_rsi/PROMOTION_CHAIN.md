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

## Measured continuation results

Generation 1 remains the accepted checkpoint. Its original confirmation result
was 176/240 -> 204/240 (73.33% -> 85.00%), two complete-family wins and no losses,
with cold reproduction; the next generation is not compared to that original
base again.

The first completed continuation loaded that approved checkpoint and executed
four separately Muse-approved optimizer proposals on GPU 0. Public development
remained **372/435 -> 372/435**, with ten of twelve families fully solved. The
first two proposals and all their line-search alternatives were rejected for
losing previously solved families. The third half-sized and fourth full-sized
updates preserved results but added no correct cases. They are not successful
promotions. The run returned `NOT_OK: no DEV improvement` before consuming the
registered confirmation suite. The approved parent was not overwritten.

A second bounded continuation is now testing one factor: reference-repair weight
8 instead of 12. Initial learning rate remains **2.5e-5**, not 5e-5; all other
model, comparator, GPU, line-search, epoch, and promotion rules are unchanged.
This variant is motivated by a read-only public-data diagnosis of interference
with already-correct RPN/LRU outputs, not by final-suite outcomes. Its result is
pending and is not included in the successful-generation count. Both variants
are public-development search within one heldout evaluation opportunity, not
two independent final-test trials.

An initial setup was interrupted before any optimizer step to align public
stopping with the fixed two-family promotion requirement. No confirmation
outcomes were consumed by that correction. Its record is retained separately.

Evidence: `validation/qwen35-4b-chain/`. Completed rejected trials retain their
reports, configuration, reviewer receipts, and acceptance-integrity audit.
The JSON lineage separates one verified successful generation, completed
rejections, setup interruptions, and pending work. An integrity-audit PASS means
the audit agrees with the rejection; it does not turn that model run into a PASS.

## Validation of this continuation implementation

120 unit/regression tests passed. All 58 reference programs passed 1,010 expanded
public case checks, and the 240 generated confirmation gold answers agreed with
independent reference programs. These are evaluator/data checks, not model
performance scores. The one-GPU launcher CHECK_ONLY path, syntax, compilation,
whitespace checks and multi-GPU-selector rejection passed. Executed code bundle:
`85995370317a1831aa65e657dff010bd0b44e365efeaba8aef6eaccda20817a7`.

No existing model-serving service was restarted or replaced. Model weights,
adapter binaries and raw CLI transcripts remain on the owner's machine.
