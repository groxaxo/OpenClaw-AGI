# New-task before/after experiment

This is a separately registered experiment, not another attempt on the prior
window/LRU/sum confirmation set. The exact cold-verified 403/435 development
adapter is the starting artifact, explicitly not an approved new generation.

Ten adaptation families: numeric version precedence, wildcard matching,
lexicographic topological ordering, CSV parsing, path normalization, IPv4
longest-prefix matching, JSON Pointer, token buckets, recursive merge patches,
and exact round-half-to-even formatting. Each has 24 training, 24 public
validation and 24 disjoint confirmation inputs. Six transfer-only families
(Roman numerals, maximum binary rectangles, bipartite graphs, range additions,
shell-word parsing and rational arithmetic) are withheld from adaptation and
checkpoint selection. Unknown pretraining exposure is not claimed.

Reference programs are validated against independent small-input/standard-library
oracles, then in the isolated Docker grader. Valid stdlib solutions may import
csv/decimal/fractions/ipaddress/json/posixpath/re/shlex/io. Host filesystem,
network and process modules remain prohibited. Gold answers remain outside
candidate containers. No files or network are mounted into them.

The model is original frozen BF16 Qwen3.5-4B with existing rank-8 LoRA. Only one
GPU is visible. The stored development adapter is verified and loaded by hash;
newly failed public tasks supply verified reference targets. Individual low-margin
token contrasts protect solved public responses. Actual public regression and
validation tests govern full/half/quarter update selection. Up to three updates
at initial LR 5e-5 require separate maximum-reasoning Muse approvals.

Strict deterministic PyTorch algorithms and the documented CUDA workspace are
requested, TF32 is disabled, and all switches are recorded. Unsupported
operations fail closed. This setting alone does not prove training reproducibility.

After public selection, the child is fixed and the new reserved suites are consumed
once. Approved generation 1, saved starting artifact and selected child are compared
under identical deterministic 512-token budgets. No later training or reselection
uses those final outcomes. Adaptation-input confirmation and transfer-only family
results are reported separately. The earlier chain's confirmation remains untouched.

Experimental acceptance requires >=8pp adaptation gain, >=75% final accuracy,
>=2 newly complete adaptation tasks, no lost complete tasks, no per-task transfer
or old-regression case loss, and independent review. This is an engineering gate,
not a statistical-significance claim. No production service or accepted lineage
is changed, including when this separate experiment succeeds. Cold reproduction
of the locked outputs runs in a new process after training finishes.

Executable entry points: `python -m dream_rsi.new_task_experiment` and
`python -m dream_rsi.new_task_cold`. The workspace launch script records all
paths, registry hashes and limits. Results must be attached from actual runs;
unit checks are not evidence of model improvement.

## Completed measurement — 29 September 2026

**Measurement completed; experimental acceptance NOT_OK.** The selected child
improves new-task confirmation accuracy but remains below the fixed absolute
accuracy/completed-task thresholds and loses two transfer-only cases. No model
was deployed and the original successful-generation count remains one.

Execution used Desktop Commander on `.51`, physical GPU 2 only, from the exact
saved DEVELOPMENT_ONLY checkpoint. The frozen executed source bundle is
`13f204c446402d890dc6e21645b6505d133ae4989418bda9fc8caaab2d14cb0e`.
One earlier attempt on GPU 0 stopped after preflight because free memory was
below the required 12 GiB. It loaded no model and made no optimizer update.
The subsequent GPU-2 run completed three separately approved proposals.

| Evaluation | Saved starting artifact | Selected child | Change |
|---|---:|---:|---:|
| New-task public development | 94/240 (39.17%) | 139/240 (57.92%) | +18.75 pp |
| Reserved adaptation-input confirmation | 95/240 (39.58%) | 141/240 (58.75%) | +19.17 pp |
| Fully solved confirmation families | 3/10 | 3/10 | No new complete family |
| Transfer-only cases | 92/144 (63.89%) | 90/144 (62.50%) | -1.39 pp |
| Old public development plus guards | 414/446 | 446/446 | +32 cases |

Old public regression now includes all 435 historical development cases and
11 guard cases, not 446 new heldout tasks. The outstanding historical window-
range defect improved from 0/32 to 32/32; sum and LRU remain 33/33 and 40/40.
These public improvements cannot substitute for final acceptance evidence.

### Reserved confirmation by task

Each program is generated once and graded on 24 disjoint inputs. These are ten
program-generation tasks, not 240 independent model draws.

| Task family | Start | Child |
|---|---:|---:|
| Numeric versions | 24/24 | 24/24 |
| Wildcard matching | 10/24 | 23/24 |
| Lexicographic topological ordering | 24/24 | 24/24 |
| CSV parsing | 0/24 | 6/24 |
| Path normalization | 12/24 | 21/24 |
| IPv4 routing | 1/24 | 1/24 |
| JSON Pointer | 0/24 | 0/24 |
| Token bucket | 24/24 | 24/24 |
| Merge patches | 0/24 | 18/24 |
| Exact half-even rounding | 0/24 | 0/24 |

JSON Pointer reached the fixed 512-token limit for both compared artifacts and
produced invalid code. The budget was not changed after observing the outcomes.
Most remaining failures did not hit the token limit. No additional optimizer
step, retry or checkpoint reselection occurred after opening the reserved sets.

Transfer-only rational arithmetic declined from 4/24 to 2/24. The other five
transfer-family counts were unchanged. The previously approved generation-1
reference scored 108/144 on transfer, versus 92/144 for the saved starting
artifact, so the earlier known-task improvement had not generalized uniformly
even before this new training. All three confirmation/transfer results are retained.

### Acceptance and execution evidence

The +19.17-point gain passes the gain criterion, but 58.75% is below the fixed
75% requirement, no new family is complete, and rational-arithmetic transfer
lost cases. The deterministic gate and Muse both reject experimental acceptance.
A shell exit code of zero means the measurement and its audit completed; it
must never be interpreted as an accepted model. Check the explicit acceptance
fields in report.json. The child remains an unpromoted experimental artifact.

Preflight and all three optimizer proposals received fresh Muse Spark 1.3
maximum-reasoning OK receipts. The promotion review returned NOT_OK; the final
measurement-only audit returned OK, explicitly including the negative findings.
The receipt audit checks terminal CLI verdicts, source/evidence hashes, single
consumption, parent identity, each saved epoch and the final candidate lock.

Proposal 1 improved public development to 129/240 and repaired window range.
Proposal 2 and its full/half/quarter alternatives were rejected; one alternative
would have lost LRU. Proposal 3 reached 139/240 while preserving all old public
scores. Selection was fixed at epoch 3 before reserved confirmation or transfer.

150 unit/regression tests passed again. All ten trusted reference programs had
passed 720 train/development/confirmation checks against independent oracles
and actual isolated evaluation. A separate fresh model process then reloaded
the fixed saved-start and selected-child adapters and reproduced **47/47 recorded
output hashes exactly**: both ten-family confirmation outputs, both six-family
transfer outputs, and all 15 final old-regression/guard outputs. It performed
zero training steps and zero reselection; this confirms artifact/output
repeatability, not acceptance or training determinism.

Peak PyTorch allocated GPU memory in the live run was 15.20 GiB, excluding CUDA
overhead and loading transients. No dependency installation, serving restart,
model download or GitHub Actions was involved.

Evidence is retained under `validation/qwen35-4b-new-tasks/completed-20260929/`.
Model/adapter binaries and raw CLI streams remain on the owner's machine.
The new task sets are retired after evaluation. The original chain's separate
240-case confirmation is still unconsumed and global lineage is unchanged.

### Fresh-process artifact reproduction

After the training launcher exited, a separate process on physical GPU 2 loaded
the saved start and the fixed selected child sequentially. It regenerated both
reserved evaluations for each (16 answers per artifact) and all 15 old public
regression answers for the child. **47/47 output hashes and grading results
matched exactly**, including the transfer decline. It performed zero training
steps, selected no alternative checkpoint, and advanced no lineage.

`cold-002.json` records this PASS. It proves repeatability from these saved
artifacts, not that repeating training will produce identical adapter weights,
and it does not override the rejected experimental-acceptance decision.
The selected tensor SHA-256 is
`f6c456474cf3ff8970f544c8e011cd87eab92999f90d5d97f364d69baf35038e`;
its file SHA-256 is
`91c7d7f803eb8be4fc6befbb4cbe28ed0116e3f38274a971d0a46b30f4a90f23`.
Both the original approved checkpoint and saved starting adapter remain unchanged.
