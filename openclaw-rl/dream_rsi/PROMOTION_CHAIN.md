# Qwen3.5-4B promotion lineage

## Current verified count — 29 September 2026

**One verified successful generation remains the accepted tip. Four completed
continuations have not produced a promoted child. The newest continuation did
produce a nonregressing public-development improvement: 372/435 -> 403/435.**

The current machine-readable ledger is
`validation/qwen35-4b-chain/current-lineage.json`. Optimizer updates, public
acceptance and implementation-test success do not increment the generation count.

| Category | Count | Meaning |
|---|---:|---|
| Verified successful generations | 1 | Approved and cold-reproduced checkpoint from PR #6 |
| New successful generations | 0 | No continuation child passed the unchanged promotion gate |
| Completed training continuations | 4 | Weight-12, weight-8 retry, response-loss projection, token-margin projection |
| Optimizer proposals in completed continuations | 14 | 4 + 4 + 3 + 3; not successful generations |
| Resource-blocked launches | 1 | No model loading or optimizer step |
| Setup interruptions before training | 1 | Stopping-rule correction before an update |
| Command-blocked follow-up launches | 1 | Follow-up preparation/launch denied before execution |

The accepted adapter remains unchanged. The new 240-case confirmation has not
been consumed. The completed token-margin run and read-only numerical diagnostic
exited; the planned additional job did not start. No production inference service
was changed.

## Verified starting checkpoint

Generation 1 is documented in `VALIDATION_BF16_GUARDED.md` and merged in PR #6,
commit `e917cffcc7c5d43b58a553fb736d9eaae4182fdd`. Against the original frozen
BF16 Qwen3.5-4B base, it scored 176/240 -> 204/240 (73.33% -> 85.00%) on its
registered confirmation suite, with two complete-family wins, zero losses and
all 11 separate regression checks preserved. Muse promotion/final reviews and
fresh-process exact-output reproduction passed.

Logical adapter SHA-256:
`4350f1d97ab1402eae61f24d501da6d3c4f26826ff2d2cc656dfdfa9d89e7b69`.
Adapter file SHA-256:
`2f0816a757b0d2405578bb0238e8fab1459cbbc1134c2d7f80995f0a24cb31e1`.

The model revision is `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, with original
frozen BF16 weights and rank-8 LoRA. Its original confirmation is now public
DEVELOPMENT, not a reusable unseen test.

## Definition of a successful continuation

A candidate is compared against the exact last approved adapter, with file,
configuration and logical tensor identity verified. Repeated comparisons against
the original base do not establish successive improvements. An unpromoted public
search state can be retained, but it must never silently become the comparator
or count as an accepted generation.

The unchanged engineering gate requires >=0.08 absolute case-accuracy gain,
>=0.75 child accuracy, >=2 newly complete families, zero complete-family losses,
preserved regression guards, Muse max promotion approval, final review and a
fresh-process locked-checkpoint reproduction. Only then may a child become the
next parent. Rejected, interrupted and blocked attempts remain in the record.

The published generation-1 confirmation is merged by exact prompt into earlier
public development and deduplicated by input; conflicting gold answers fail
validation. The resulting distribution has 435 cases over 12 known families.
The approved parent scores 372/435 with 10/12 complete. These public results
must not be pooled with or substituted for its original 204/240 confirmation.

The new 12-family / 240-case confirmation was generated and hashed before these
searches. Exact public inputs were excluded and oracle answers cross-checked.
It is consumed once, only after development fixes a candidate. Parent and child
use identical prompts and deterministic 512-token generation. This measures
known-prompt, in-domain program correctness, not unseen-task generalization or
population-level significance.

## Completed continuation history

### 1. Broad verified repair, weight 12 — NOT_OK

Four Muse-approved proposals ran on GPU 0 from the accepted parent at initial
LR 2.5e-5. Early proposals lost correct RPN/LRU programs and were reverted.
Later nonregressing alternatives added no cases. Public results remained
372/435, 10/12 complete; confirmation was not consumed.

### 2. Broad verified repair, weight 8 — NOT_OK after resource retry

An initial launch stopped before loading because unrelated work left less than
the fixed 12 GiB headroom requirement. The later adequately resourced run did
execute four proposals. Only repair weight changed from 12 to 8; initial LR
remained 2.5e-5. Every proposal and its alternatives failed public acceptance,
leaving 372/435. The resource block and completed rejection are distinct records.

A prior diagnostic inferred LR 5e-5 from a source default. The executed receipts
establish 2.5e-5; the original diagnosis and explicit correction remain retained.
Completed evidence: `validation/qwen35-4b-chain/weight8-completed-rejected/`.

### 3. Response-loss projected minimal repairs — NOT_OK

Three proposals on GPU 0 used two minimally edited public responses and 26
correct-answer gradient directions, initial LR 5e-5. The first preserved scores
without gain. The second full proposal fixed sum-excluding-self (2/33 -> 33/33)
but broke LRU (40/40 -> 0/40). It and regressing alternatives were reverted.
The retained score remained 372/435; no confirmation was consumed.

Four Muse max reviews approved bounded execution, not promotion. The separate
receipt/lineage audit passed because it verified the NOT_OK outcome accurately.
Details: `PROTECTED_REPAIR.md` and
`validation/qwen35-4b-chain/projected-minimal-repair-rejected/`.
Executed source: `ca041bb0f80e4108d43de60aa329ecf37abde75b3196ddb917dcc42197d591b8`.

### 4. Individual token-margin protection — public gain, no promotion

This job used GPU 1 alone, because it was free. The accepted parent, two minimal
repair targets, initial LR 5e-5, full/half/quarter public line search, three-proposal
budget, decoding and promotion gate were unchanged. Protection instead used
individual chosen-minus-competitor token logits at weak decisions, plus first
divergences from previously exposed regression outputs.

The first half-sized and second full proposal retained 372/435. The third full
proposal reached **403/435 (92.64%), 11/12 complete families**, with all prior
complete families preserved and guards 11/11. Sum-excluding-self improved from
2/33 to 33/33; LRU stayed 40/40. Window range remained 0/32.

The gain is 31/435 = 7.13 percentage points and only one new complete family.
Both are below the unchanged prerequisites. The run correctly returned NOT_OK,
without final confirmation or model promotion. Its saved adapter is explicitly
DEVELOPMENT_ONLY, logical hash:
`e4c1e5fe16cc79013006d31a4d03f11bfdbe0df196549dbe152c982beee156e3`.
This does not replace generation 1.

Four Muse max reviews approved preflight and each real proposal. The run-source,
receipt, parent and anchor-dataset audits passed. All 15 parent public outputs
matched the prior run byte-for-byte. A separate read-only precision probe found
cached-versus-teacher-forced margin differences and a rounded RPN logit tie;
no precision setting or model weight was changed by that probe.

Details: `TOKEN_MARGIN_REPAIR.md` and
`validation/qwen35-4b-token-margins/first-three-proposals/`.
Executed source: `63e891855636eba1dd73d0876d4253ab03a0ef45bd7706c527be083c8f1b70c9`.

## Prepared follow-up: not launched

A local draft was prepared to permit at most two additional public optimization
proposals from the documented 403/435 development state. It requires explicit
DEVELOPMENT_ONLY metadata, adapter/configuration hashes, a matching original
accepted parent, and cold reproduction of the public candidate before updates.
The final comparison would still be against generation 1, never the unpromoted
intermediate. It may not reuse evaluated confirmation data or relax any gate.

The local draft passed 143 unit tests, but is not part of the completed tested
implementation in this change. Desktop Commander rejected the subsequent
compound registration/preparation/launch command with **Command not allowed**.
That command did not execute. No new preflight approval, model load, optimizer
proposal or confirmation evaluation occurred. The denial was not bypassed via
another device, tool, shell encoding or execution route.

This is BLOCKED_COMMAND, not a failed model trial or a successful run. Further
execution requires authorized command access and fresh source-bound evidence;
it must not be described as running or scheduled.

## Implementation validation and safeguards

The executed token-margin implementation passed **136 unit/regression tests**,
including nine new token-contrast checks. Both unchanged minimal repair programs
passed all 65 relevant public cases. Launcher syntax, multi-GPU selector rejection,
concurrent-lock rejection, compilation and git whitespace checks passed.
The separate local follow-up draft's 143 tests are not a live-training result.

The earlier broad runner's 58 references passed 1,010 expanded public checks,
and 240 generated final oracle answers agreed with independent references.
These data/evaluator checks are not independent model-performance samples.

One-GPU launchers use an exclusive shared experiment lock, unique output folders,
explicit parent identity, token/time/disk bounds and independent source/evidence-
bound approvals. Model weights, adapters and raw CLI transcripts remain on the
owner's machine; the repository stores reports, registrations, verdicts, hashes
and relevant outputs. No GitHub Actions or production activation is introduced.

## Interpretation

The latest experiment broke the flat public score without losing solved families,
but it did not meet promotion requirements. One successful generation is the
verified count so far, not a proven maximum. On a fixed distribution above 92%
parent accuracy, an eight-point minimum gain becomes mathematically unavailable;
that ceiling must be reported rather than concealed by changing comparators or
thresholds. The 92.64% development candidate is not the accepted comparator.

Any future extension must retain failures and total proposal counts, preserve
honest public/confirmation separation, and earn a new promotion through the
same accepted-parent comparison and cold reproduction.

## Merged-main retest — 29 September 2026

See [MAIN_E2E_RETEST.md](MAIN_E2E_RETEST.md). The actual three-proposal rerun
retained 372/435 rather than the earlier 403/435: training reproducibility is
not established. A separate cold process reproduced all 45 recorded response
hashes for the accepted parent, saved development candidate and rerun candidate;
the saved development adapter still scores 403/435. All 136 tests and execution
receipt checks passed. These results do not promote either candidate or consume
final confirmation. One verified successful generation remains accepted.
