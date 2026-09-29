# Merged-main end-to-end retest — 29 September 2026

## Result: separate artifact reproducibility from training reproducibility

**The saved development adapter reproduces exactly. Retraining did not reproduce
its earlier development improvement. No checkpoint was promoted.**

This retest used Desktop Commander on the authorized Ubuntu .51 host, with only
physical GPU 1 visible to each model process. It ran the existing merged launcher
and three actual optimizer proposals, then a separate fresh-process cold check.
The cold process started only after the training launcher exited and held the
same exclusive one-GPU experiment lock. Other workloads were left untouched.

Tested main commit: `fe201c67c56cae04510688618b0feecfa53d8b4c`.
The executable source and tests were unchanged throughout, with bundle SHA-256
`63e891855636eba1dd73d0876d4253ab03a0ef45bd7706c527be083c8f1b70c9`.
These documentation/evidence changes do not alter training or serving behavior.

## What was executed again

- All 136 unit/regression tests passed on the merged source.
- The actual launcher requested a fresh Muse Spark 1.3 max-reasoning preflight
  and a separate source/evidence-bound approval for each of three optimizer
  proposals. All four execution approvals returned OK and were consumed once.
- The exact approved generation-1 parent was loaded and its original 15 public
  development/guard output hashes reproduced before training.
- The original BF16 model, existing rank-8 LoRA, two verified equal-weight
  repairs, LR 5e-5, full/half/quarter line search, seed and deterministic
  512-token decoding configuration were retained. The source, parent identity,
  repair dataset and first-proposal anchor identities/margins matched the prior
  experiment. Package versions and GPU identity are recorded separately.
- The two repair targets again passed their 65 relevant public cases.
- Three real backward/optimizer proposals, rejection/rollback decisions,
  checkpoint save/reload and the terminal promotion-prerequisite rejection ran.
- Compilation, shell syntax, multi-GPU rejection (64), concurrent-launch
  rejection (73), and consumed-suite rejection (65) passed.

## Fresh retraining outcome

| Proposal | Prior run retained cases | Retest retained cases | Prior scale | Retest scale |
|---|---:|---:|---:|---:|
| 1 | 372/435 | 372/435 | 0.5 | 0.0 (reverted) |
| 2 | 372/435 | 372/435 | 1.0 | 0.25 |
| 3 | 403/435 | 372/435 | 1.0 | 0.5 |

All retained states preserved the ten complete development families and the
11 separate guard cases. However, the rerun did not recover the extra 31 correct
cases. Its training result is **NOT_OK**, and the launcher exited 3 as required.
The reproducibility audit reports **REPRODUCTION_MISMATCH**, not blanket PASS.
No final confirmation outcomes were evaluated, and the registered 240-case
confirmation remains unconsumed.

The first repair loss was identical at 0.5725239962339401, but its gradient norm
changed from 4.69068717956543 to 4.699758052825928. The first protection-gradient
norm changed from 58.84942626953125 to 58.89576721191406, despite identical anchor
positions, chosen/alternative tokens and measured scoring margins. All three
retained adapter tensor hashes differed from their prior-run counterparts.
This establishes an observed numerical/training reproducibility gap. It does
not isolate the root cause or prove a specific kernel or precision change fixes
it. Do not advertise this training recipe as reliably reproducing the 403/435
checkpoint merely because its implementation tests pass.

## Fresh-process cold reproduction of fixed artifacts

One new model process loaded each fixed adapter sequentially and regenerated
all 12 public-development responses and three guard responses. It did not train,
select another checkpoint, or read the reserved confirmation suite.

| Fixed artifact | Public cases | Complete families | Guards | Recorded response hashes reproduced |
|---|---:|---:|---:|---:|
| Approved generation-1 parent | 372/435 | 10/12 | 11/11 | 15/15 |
| Previously saved development candidate | 403/435 | 11/12 | 11/11 | 15/15 |
| Newly retrained retained candidate | 372/435 | 10/12 | 11/11 | 15/15 |

**45/45 recorded output hashes and all grading results matched exactly.** Thus
the saved development gain remains an artifact-level result, although the fresh
training attempt did not reproduce it. Its sum-excluding-self result remains
33/33 and LRU remains 40/40. The saved adapter is still DEVELOPMENT_ONLY: its
original gain was 7.13 percentage points and one new family, below the unchanged
8-point/two-family promotion prerequisites. Cold reproduction is not promotion.

The saved development tensor hash remains
`e4c1e5fe16cc79013006d31a4d03f11bfdbe0df196549dbe152c982beee156e3`;
its file hash remains
`7f314b99a0414c4d0588a7f4bafa243e53fea39040ec1a55bb4b530ef2857aa1`.
The approved parent remains
`4350f1d97ab1402eae61f24d501da6d3c4f26826ff2d2cc656dfdfa9d89e7b69`.

## Lineage and unresolved scope

There is still **one verified successful generation and zero new promotions**.
Four earlier exploratory continuations contain 14 proposals. This additional
reproduction run contains three proposals, recorded separately: 17 executed
continuation/retest proposals altogether, not 17 successful generations.

The previously command-blocked two-proposal follow-up remains a separate local
draft. This request successfully executed a fresh retest of merged main; it did
not start or validate that draft and did not bypass an active tool denial.

This is known-prompt/in-domain supervised adaptation. Public-development cases
are not independent heldout tasks, and the promotion rule is an engineering
acceptance criterion, not a population-level statistical claim. The native
Transformers/PEFT research path was exercised; the legacy SGLang/PRM HTTP serving
path was not validated here. No production service, accepted adapter, dependency
installation, model cache or GitHub Actions configuration was changed.

## Evidence and reproduction

Evidence is under `validation/qwen35-4b-token-margins/main-retest-20260929/`:
the runtime/source manifest, tests, three proposal histories, four execution
review receipts, target hash, launcher contract checks, environment/base-shard
integrity, gradient discrepancy, training audit, and cold-output reproduction.
Both saved reference shards were rehashed against their previously verified
official digests. Model/adapter binaries and raw CLI streams remain local.

The existing entry point is `openclaw-rl/run_qwen4b_token_margin.sh`; CHECK_ONLY
runs its test preflight. A full retest requires explicit approved-parent,
public-counterexample, retired-public-suite, unconsumed final-suite, model and
Docker-image paths, plus exactly one numeric GPU selector. The archived
`verify_cold_public.py` reproduces these fixed public outputs from the recorded
workspace; it is an audit utility, not a training or deployment entry point.

The durable remaining issue is training reproducibility. Deterministic backward
execution and numerical sensitivity need a separate controlled investigation;
this evidence must not be relabeled as a reliably repeatable training success.
