# Qwen3.5-4B promotion lineage

## Current verified count — 29 September 2026

**One verified successful generation remains the accepted tip. No child has
passed promotion in the three completed continuations from that checkpoint.**
The current machine-readable ledger is `validation/qwen35-4b-chain/current-lineage.json`.
It supersedes the earlier pending/resource-blocked description of the weight-8
variant while retaining that historical launch separately.

| Category | Count | Meaning |
|---|---:|---|
| Verified successful generations total | 1 | Approved and cold-reproduced checkpoint from PR #6 |
| New successful generations | 0 | No continuation child met the unchanged promotion gate |
| Completed training continuations | 3 | Weight-12, weight-8 retry, projected minimal repair |
| Optimizer proposals in those continuations | 11 | 4 + 4 + 3; not successful generations |
| Resource-blocked launches | 1 | Approved preflight but no model loading or optimizer step |
| Setup interruptions before training | 1 | Stopping-rule correction before any update |

All completed continuations preserve the accepted parent on disk. The new
240-case confirmation is still registered and unconsumed. No experiment from
this continuation task remains running; no production inference service was
changed.

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

The model revision remains `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, with
original frozen BF16 weights and the existing rank-8 LoRA. The original parent
confirmation is now public DEVELOPMENT, not a reusable unseen test.

## Definition of a successful continuation

Each continuation must load the exact last approved adapter, validate its file,
configuration and logical tensor identity, and compare the child against that
immediate parent. Repeatedly comparing children with the original base cannot
establish a succession of improvements. Optimizer proposals, non-regressing
updates, independent execution approvals, and cold reproductions alone do not
increment the successful-generation count.

The unchanged engineering gate requires >=0.08 absolute case-accuracy gain,
>=0.75 child accuracy, >=2 newly complete families, zero complete-family losses,
regression guards preserved, Muse max promotion approval, final review and a
fresh-process locked-checkpoint reproduction. Only after all these pass may a
child become the next parent. Rejections and interruptions remain visible but
never replace an accepted checkpoint.

The published generation-1 confirmation is merged by exact prompt into earlier
public development and deduplicated by input; conflicting gold answers fail
validation. This produces 435 cases over 12 known families. The approved parent
scores 372/435, with 10/12 complete. These are a different public distribution
from the original 204/240 confirmation and must not be pooled or substituted.

The new 12-family / 240-case confirmation was generated and hashed before the
continuation searches. Exact public inputs were excluded and gold answers were
independently cross-checked. It is consumed once, only after public development
fixes a candidate; parent and child use identical prompts and deterministic
512-token generation. The measured scope is known-prompt, in-domain program
correctness, not unseen-task generalization or population-level significance.

## Completed continuations

### 1. Broad verified repair, weight 12 — NOT_OK

Four independently Muse-approved optimizer proposals ran on GPU 0, starting
from the accepted parent with initial LR 2.5e-5. Public development remained
372/435, 10/12 complete. The first proposals lost correct RPN/LRU programs and
were reverted. Later non-regressing alternatives added no correct cases.
The run stopped before confirmation with `NOT_OK: no DEV improvement`.

### 2. Broad verified repair, weight 8 — NOT_OK after resource retry

The initial launch was resource-blocked after preflight because unrelated GPU
work left less than the fixed 12 GiB headroom requirement. No model was loaded
and no optimizer step happened during that blocked launch.

A subsequent launch with sufficient headroom did complete four real proposals.
It changed only repair weight 12 -> 8 from the earlier broad-repair setting;
initial LR remained 2.5e-5. Every proposal and its tested smaller alternatives
failed public acceptance. The retained score stayed 372/435, 10/12 complete.
The outcome is now a completed model rejection, not an untested hypothesis.
The earlier blocked launch is still recorded separately.

A prior diagnostic had inferred LR 5e-5 from the source default. The executed
manifests/receipts establish 2.5e-5; the original diagnosis and correction remain
in the evidence. The diagnostic was a hypothesis, not an approval or causal proof.

Completed weight-8 evidence, including its five reviewer receipts and integrity
audit, is in `validation/qwen35-4b-chain/weight8-completed-rejected/`.

### 3. Projected minimal repairs — NOT_OK

The next job again loaded the exact accepted parent on GPU 0. It trained only
two minimally edited public responses and projected each proposed displacement
away from 26 measured correct-response loss-gradient directions. Initial LR
was 5e-5, with at most three proposals. The method and its first-order limitations
are documented in `PROTECTED_REPAIR.md`; all promotion thresholds stayed fixed.

The first full projected update retained all public results but gained nothing.
The second full proposal fixed sum-excluding-self (2/33 -> 33/33) while breaking
LRU (40/40 -> 0/40), so it was rejected; smaller trials and the third proposal
also lost solved families. Retained public results stayed 372/435, 10/12 complete.
The run exited 3 before consuming confirmation. No candidate was promoted.

Four actual Muse max reviews approved bounded execution: preflight and each
optimizer proposal. There is no promotion/final approval for this failed model
run. The receipt/lineage audit is PASS because it faithfully verifies the
NOT_OK outcome, not because the model improved.

Executed-source SHA-256:
`ca041bb0f80e4108d43de60aa329ecf37abde75b3196ddb917dcc42197d591b8`.
Evidence: `validation/qwen35-4b-chain/projected-minimal-repair-rejected/`.

## Implementation validation and safeguards

The current suite has 127 passing unit/regression tests, including projection
orthogonality, dependent/zero anchors and exact minimal-edit checks. Both repair
targets passed their 65 relevant public cases. Previously, the broad runner's
58 reference programs passed 1,010 expanded public case checks and the 240
registered gold answers agreed with independent reference algorithms. These
are data/evaluator checks, not independent model-performance trials.

The one-GPU launchers enforce a shared exclusive experiment lock, unique output
folders, explicit parent identity, token/time/disk bounds, and independent
source/evidence-bound approvals. CHECK_ONLY, shell syntax, multi-GPU rejection,
concurrent-lock rejection, compilation and git whitespace checks passed for the
new entry point `run_qwen4b_projected.sh`. No GitHub Actions are required.

Model weights, adapter binaries and raw CLI transcripts remain on the owner's
machine. The repository retains reports, manifests, final reviewer verdicts,
source/parent hashes, and relevant rejected outputs. No production serving
checkpoint was activated or replaced.

## Interpretation and stopping

One successful generation is the count verified so far, not an established
maximum. These tested methods plateaued under the fixed acceptance criteria.
On a fixed distribution above 92% parent accuracy, an eight-point improvement
would become mathematically unavailable; the system must report that ceiling
rather than lower the gate, reset the comparator or relabel reproductions.

Further public-development experiments may investigate more local updates or
stronger token-level preservation, but their gains must still be verified
against the accepted parent on a fresh, unconsumed final evaluation. A new
benchmark or acceptance policy must be declared as a separate experiment.
