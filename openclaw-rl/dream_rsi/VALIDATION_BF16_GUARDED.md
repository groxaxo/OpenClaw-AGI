# Verified single-GPU Qwen3.5-4B promotion

## Final result: PASS

The native BF16 + rank-8 LoRA experiment passed the original in-domain promotion
gate, the independent Muse promotion review, the final review, and a fresh-process
cold reproduction. It used only physical GPU 0 on the authorized Ubuntu .51 host.
No production inference service was changed.

| Measurement | Pinned base | Locked candidate |
|---|---:|---:|
| Confirmation checks passed | 176/240 | 204/240 |
| Case accuracy | 73.33% | 85.00% |
| Fully solved families | 8/12 | 10/12 |
| Separate regression checks | 11/11 | 11/11 |

The gain is **11.67 percentage points**, with **two complete-task wins and zero
complete-task losses**. LRU-cache handling improved from 0/20 to 20/20; closest
strictly-smaller-to-left improved from 12/20 to 20/20. The two remaining incomplete
families are window range (0/20) and sum of the other elements (4/20).

These are **known coding families and known prompts with newly generated case
inputs**, not unseen-task generalization or proof of general recursive
self-improvement. Several prompts/algorithms are intentionally trained. The
reported 240 cases are not 240 independent LLM tasks. Earlier trial scores used
different curricula and/or token budgets and are not pooled with these results.

## Model, configuration and integrity

- Model: Qwen/Qwen3.5-4B, original frozen BF16 weights, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- LoRA: rank 8, alpha 16; 10,616,832 trainable parameters.
- Same deterministic 512-token generation budget for base and candidate.
- The run generated two approved optimizer proposals. The selected checkpoint
  is epoch 1 at half the proposed parameter change. All second-proposal line
  search candidates lost fully solved public DEV tasks and were rejected.
- Selected logical adapter tensor-state SHA-256:
  `4350f1d97ab1402eae61f24d501da6d3c4f26826ff2d2cc656dfdfa9d89e7b69`.
- Actual adapter file SHA-256:
  `2f0816a757b0d2405578bb0238e8fab1459cbbc1134c2d7f80995f0a24cb31e1`.
- Reviewed/executed source bundle SHA-256:
  `957a2b99598b74389888916f7781c105b50df4b529531c9d9fb48f6e36b21fea`.
- Preregistered confirmation SHA-256:
  `1d3a63473dbe110b59b0ced3b8390068b39425ee162c696e7db0987dd9bd3eb2`.
- Verified training dataset SHA-256:
  `edd00c2a96cb8a69e4ca8de523ee55e30f3c2d79a71feb0bb3cc05be34604519`.
- Peak PyTorch allocated VRAM during the validation phase: 9.346 GiB.
  This excludes CUDA overhead and other processes, and the counter was reset
  after model loading.

The old global model cache was missing before the first startup could train.
The exact official revision was restored into a task-local directory and both
weight shards matched their official Hugging Face LFS SHA-256 values. The failed
startup is not counted as a successful training run.

## Corrections that preceded the successful run

The first BF16 attempt improved public DEV from 174/240 to 183/240 but exchanged
some fully solved families for other gains. It was interrupted before any final
confirmation outcome, and its proposals, reviewer receipts and reason are
retained under `interrupted-formatting-run/`.

A read-only Muse diagnosis and direct data audit found two concrete problems.
Eight passing self-replay answers had their code fences stripped; all passing
responses also had outer whitespace normalized. Separately, an incorrect
previous-smaller program passed only two easy examples and was mistakenly reused
as a successful target even though it failed the broader public DEV tests.

The corrected run preserves successful raw answers verbatim, including fences
and newlines. Training aliases with identical reference ASTs pool their existing
cases with exact-description public DEV cases. Both replay and repair targets
must pass this stronger public verification. This is disclosed use of DEV for
training-data validation, not an independent test set.

The fixed dataset has 48 verified exact self-replays, 10 verified reference
repairs and three verified guard replays. The public line search tests full,
half and quarter proposed adapter changes, retaining one only when named guards,
all incumbent complete DEV tasks and aggregate DEV accuracy are preserved.
Rejected proposals are reverted; learning rate is reduced and Adam state reset
where required. These combined fixes are not a single-factor causal experiment.

## Independent reviews and executed validation

Five real Muse Spark 1.3 reviews requested maximum reasoning: preflight,
optimizer proposal 1, optimizer proposal 2, checkpoint promotion and final audit.
Every required review returned OK. All action approvals were hash-bound and
consumed once; the final review was read-only. Missing/negative replies still
fail closed. The reviewer does not replace the deterministic gate.

- 100 unit/regression tests passed on the user's Ubuntu machine.
- All 58 reference programs passed 594 expanded case checks. Some cases are
  shared by aliases; these counts are not independent model-performance trials.
- Both checkpoint state roundtrips passed.
- Fresh-process model plus adapter loading reproduced every final-suite output
  hash for base and candidate, and every regression-guard output hash.
- The cold check reproduced 176/240 to 204/240, with 11/11 guard cases passing,
  without training, reselection or a new promotion attempt.
- Launcher shell syntax, rejection of a multiple-GPU selector, rejection of a
  consumed confirmation suite and its CHECK_ONLY test path passed.
- Python compilation and git diff whitespace checks passed.

The final confirmation was generated and hashed before training, remained
unconsumed through the interrupted attempt, and was first evaluated only after
the corrected run fixed its candidate using public development results. It is
now retired and published for audit; do not reuse it as a fresh holdout.

## Evidence and use

Machine-readable evidence is under
`validation/qwen35-4b-bf16-guarded/verified-run/`: the manifest, report, five final
review receipts, test report, cold-verification result, acceptance audit,
retired confirmation suite, baseline/candidate outputs and selected update
history. Prior rejected trials remain in the repository.

The launcher is `openclaw-rl/run_qwen4b_bf16_guarded.sh`. It requires one numeric
GPU selector, an existing pinned model, a digest-pinned evaluator image and a
newly preregistered, unconsumed confirmation suite. CHECK_ONLY runs tests without
training. The launcher does not activate any serving model or run periodically.

This validates the new native Transformers/PEFT BF16 research path, not the
legacy SGLang/PRM HTTP serving path. This is supervised in-domain adaptation,
not Google's official Dream-RSI implementation. No GitHub Actions were used.
