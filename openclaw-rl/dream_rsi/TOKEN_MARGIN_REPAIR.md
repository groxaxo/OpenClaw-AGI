# Token-margin continuation experiment

## Question

Can we repair the accepted Qwen3.5-4B checkpoint's two remaining public defects
without the LRU/RPN regressions seen in prior continuation attempts?

The previous implementation projected updates away from whole-response loss
and the average of three high-loss tokens. Those aggregate gradients do not
necessarily preserve the individual token choices that determine a program's
execution. In the retained rejected LRU answer, the first token divergence is
position 6: the correct answer begins by unpacking capacity; the rejected answer
begins by initializing the cache. This is a diagnostic, not proof that token 6
alone causes the later program error.

## Controlled change

The new runner is `qwen4b_margin_continue.py`. It keeps the original frozen BF16
base, exact accepted rank-8 adapter, two equal-weight minimal repairs, initial
learning rate 5e-5, full/half/quarter public line search, maximum three optimizer
proposals, deterministic 512-token generation, immediate-parent comparison, and
the original promotion gate.

Only the protection signals change. For each fully solved public answer and
regression guard, select three low-margin chosen-versus-best-alternative token
contrasts. Add first-divergence contrasts from already-public rejected answers.
The supporting prefix and chosen token come from the exact generated token IDs,
not normalized text. Historical alternative tokens are compared only at their
first matching-prefix divergence.

For a selected position t, the protected scalar is

    margin_t = logit(chosen_t | prefix_t) - logit(competitor_t | prefix_t).

Its gradient with respect to the existing LoRA parameters supplies an anchor
row. The same row-normalized pseudoinverse projection removes the AdamW proposal's
component along those rows. This is still first-order protection, NOT a guarantee
of preserved greedy decoding. Cached decoding and teacher-forced logits can
also differ numerically; measured margins, including any nonpositive values,
are recorded rather than hidden. Executable public tests remain authoritative.

Both the repair targets and counterexamples are public development data. No
confirmation outcomes are used for anchor selection. The already registered,
unconsumed 240-case suite is used only after development fixes a candidate.
A cold reproduction is required before a passing child advances the lineage.

## Safeguards

One visible GPU, exclusive shared experiment lock, maximum 96 anchor rows,
finite-value checks, exact parent/hash verification, distinct output directory,
12 GiB minimum free VRAM before loading, and fixed time/disk limits. Every real
optimizer proposal needs a fresh source/evidence-bound Muse Spark 1.3 max review.
The unchanged promotion gate requires +8 percentage points over the immediate
approved parent, at least 75% final accuracy, two newly complete families, zero
complete-family losses, and preserved guards. No server is restarted or deployed.

## Preregistration and checks

The execution manifest, counterexample digest, prior public token differences,
source/test report and final outcome are retained with the experiment evidence.
Initial tests: 136 unit/regression checks and both minimal repairs on all 65
relevant public cases. These counts validate implementation and targets, not
model improvement. This experiment uses GPU 1 because it was free; no other GPU
is made visible to the process and unrelated GPU 0/2 workloads are untouched.

## First registered result: partial development gain, no promotion

Three individually approved optimizer proposals completed on GPU 1. The accepted
parent reproduced 372/435 public cases and ten complete families. Proposal 1's
half-sized change and proposal 2's full change retained that score. Proposal 3's
full change reached **403/435 (92.64%), eleven complete families**, preserving all
previously complete families and all 11 separate guard cases. Sum-excluding-self
improved from 2/33 to 33/33 while LRU stayed 40/40. Window range remains 0/32.

This is NOT a promoted generation: the gain is 31/435 = 7.13 percentage points,
below the fixed eight-point threshold, and only one new family is complete.
The run returned NOT_OK without evaluating confirmation. Its checkpoint is
saved explicitly as DEVELOPMENT_ONLY, never as the accepted parent.

The projection used forty token-contrast anchors per proposal. Eleven first-
proposal teacher-forced margins were nonpositive. A separate no-training probe
reproduced the parent's three diagnostic prefixes exactly. LRU's contrast was
zero during full-answer scoring but +0.125 during actual cached decoding. RPN's
cached logits tied at 24.375, while a float64 dot product of those same frozen
weights and hidden values gives a -0.00611 contrast. Thus cached evaluation and
output rounding both matter. This diagnostic does not prove a causal fix or
justify changing the production model's precision.

Evidence, all four completed review receipts, the unpromoted checkpoint hashes
and the read-only numerical probe are under
`validation/qwen35-4b-token-margins/first-three-proposals/`.

A separately registered follow-up may continue public optimization from this
candidate, but must retain generation 1 as the final comparator. It must verify
the candidate's bytes, lineage and cold public outputs, request fresh approvals,
and leave the confirmation set untouched until all original prerequisites pass.
There are no new successful generations at this point.

## Merged-main retest — 29 September 2026

See [MAIN_E2E_RETEST.md](MAIN_E2E_RETEST.md). The actual three-proposal rerun
retained 372/435 rather than the earlier 403/435: training reproducibility is
not established. A separate cold process reproduced all 45 recorded response
hashes for the accepted parent, saved development candidate and rerun candidate;
the saved development adapter still scores 403/435. All 136 tests and execution
receipt checks passed. These results do not promote either candidate or consume
final confirmation. One verified successful generation remains accepted.
