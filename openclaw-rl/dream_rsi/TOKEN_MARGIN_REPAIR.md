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

## Result

The live result is pending. No new successful generation is claimed here until
the completed report, required reviews and cold reproduction support it.
This remains known-prompt, in-domain supervised research, not official Dream-RSI
or evidence of broad unseen-task generalization.
