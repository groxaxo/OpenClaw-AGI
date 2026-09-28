# Qwen3.5-9B: real three-GPU validation

## Outcome

**PASS: bounded native end-to-end integration. NOT_OK: checkpoint promotion.**

This was executed on the owner's Ubuntu host through Remote Desktop Commander,
with three RTX 3090 GPUs. No existing inference service was stopped or restarted.
No GitHub Actions were used. The model was the official `Qwen/Qwen3.5-9B`, pinned
to revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.

Reviewed source-bundle SHA-256:
`7a5eabe06fc946e2dbd12a46cd45916c9297f50c41e2f9907bc67d3d7b508b1f`

## Executed checks

- 59 unit/regression tests passed, including external vetoes, stale or reused
  approvals, reference-model KL, policy-promotion vetoes and bounded preparation.
- Actual Qwen3.5-9B generation and test grading, not mocks, on all three GPUs.
- Six training tasks and six separate heldout tasks. 192 maximum generated
  tokens per attempt. Heldout evaluation was matched before/after, once each.
- Two real, synchronized LoRA optimizer updates. Each required a fresh Muse
  Spark 1.3 `max` reasoning `OK` before the update could proceed.
- Adapter hashes matched on all three ranks after each update. Both saved
  adapters were reloaded and checked for exact state equality.
- Gradient norms were 0.7395450473 and 0.4893026352, identical across ranks.
- Reference KL was zero before the first update and nonzero at the second,
  confirming that the frozen-reference term is used rather than old-policy KL.
- Peak PyTorch allocated memory during the validation phase was 7.8305,
  7.8612 and 7.8653 GiB on ranks 0/1/2 respectively. These numbers exclude
  unrelated processes, CUDA overhead, and pre-reset model-load transients.
- The final independent Muse review returned `OK` for integration validation.
- The launcher passed `bash -n` and its actual `CHECK_ONLY=1` preflight.

## Promotion was correctly rejected

The model solved 4/6 complete heldout tasks both before and after training
(14/22 individual test cases in each evaluation). There was no measured gain:
mean paired improvement 0, six ties, one-sided sign-test p=1.0. Both the
registered statistical gate and the external reviewer returned `NOT_OK` for
promotion. Candidate adapters were retained but never activated in a service.
The short generation budget truncated one heldout solution; this is a smoke
suite under a fixed budget, not a general coding-capability benchmark.

`report.json`'s `elapsed_seconds` is captured after the terminal model
comparison and **before** promotion/final-review CLI calls. Do not interpret
it as total end-to-end wall time including all reviewer latency.

## Reviewer and runtime scope

Muse was used live with `--model muse-spark-1.3 --reasoning-effort max --yolo`,
with shell execution, filesystem writes and web tools disabled for review.
Verdicts were parsed only from completed terminal events, bound to exact
source/evidence, and consumed once. Both training receipts were approved;
the promotion receipt was denied; the final integration receipt was approved.

OpenCode GLM-5.3 `max` was probed, but the installed CLI stalled during
initialization without a model verdict, including isolated-configuration
attempts. It was therefore not counted as an approval and was not a required
reviewer in this run. The optional dual-review protocol has unit coverage,
but live GLM CLI execution remains unverified on this host.

This uses the new **native Transformers + PEFT validation backend**, not the
legacy SGLang/PRM HTTP path. The latter is not end-to-end validated by this run;
its Dream-enabled training now requires external review, and serving-weight
publication remains blocked without fresh checkpoint holdout evidence.

The loss is clipped, signed-reward PPO-style optimization with a frozen
reference, not group-normalized GRPO. The controller is a replay-guided
curriculum experiment, not Google's official Dream-RSI implementation.

## Startup issues fixed during validation

The numeric hostname `12700` was interpreted by dynamic rendezvous as IPv4
`0.0.49.156`. Explicit static `--master_addr=127.0.0.1` fixed startup without
changing the machine's hostname. The first distributed model load also
exceeded its conservative process cap during PEFT's temporary float32
embedding upcast. The validated preparation retains large frozen BF16
tensors, upcasts only small parameters and uses non-reentrant checkpointing.
These failed attempts produced no training updates and are not counted as
successful runs.

## Reproduce

Use an environment containing the pinned versions in
`requirements-dream-validation.txt`, the official model snapshot, and a
locally available digest-pinned Python Docker image. Then run:

```bash
export PYTHON_BIN=/absolute/path/to/validated/environment/bin/python
export MODEL_PATH=/absolute/path/to/Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
export EVALUATOR_IMAGE=python@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
export JUDGES=muse
bash openclaw-rl/run_dream_qwen9b_3gpu.sh
```

The launcher creates a new run directory, runs tests before training, uses all
three GPUs, and imposes both internal and outer wall-time limits. `CHECK_ONLY=1`
runs only its test preflight. Nothing runs periodically or at boot.

Machine-readable evidence is in `validation/qwen35-9b-3gpu/`. Full CLI event
logs, generated solutions, model weights and adapter binaries stay on the
owner's machine and are not published to GitHub.
