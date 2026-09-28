# PR #4 pre-merge validation

Executed at 2026-09-28T11:08:14+00:00 through Remote Desktop Commander on the user's Ubuntu host `12700`, using Python 3.12.3. All checks ran on CPU in an isolated checkout. No models were loaded, training jobs launched, dependencies installed, or services restarted.

## Results

| Check | Result |
|---|---|
| Original controller suite at `28a873ad40f1c398bb5d2376b946072fedba982d` | 7/7 passed |
| Expanded controller and regression suite after hardening | 35/35 passed |
| Seeded selection property trials | 1,000/1,000 passed |
| Python compilation: controller, CLI, API proxy, trainer and tests | Passed |
| Offline replay CLI against synthetic recorded pools | Passed |
| Synthetic promotion, rejection, restart and new-window checks | Passed |
| `git diff --check` | Passed |

## Fixes included before merge

- Excluded, failed, aborted and zero-loss-mask samples cannot re-enter a selected batch. The trainer skips short valid batches rather than padding with invalid samples.
- Duplicate candidate IDs, non-finite rewards, overflowing score differences and malformed policy/configuration values fail closed.
- Holdout sessions are excluded from fitting; heldout pools must not share session IDs. A persisted cursor prevents reuse of evaluated windows, including after restart.
- Policy persistence must succeed before the in-memory policy changes.
- Trace metadata is allowlisted; prompt/response payloads are not copied from generic candidate metadata.
- Enabling the sampler requires PRM scoring. PRM next-state buffering no longer depends on enabling disk conversation logging.
- Documentation describes retrospective curriculum selection accurately, rather than claiming prefix-only search replay or demonstrated model improvement.

## Reproduce

From the repository root:

```bash
PYTHONPATH=openclaw-rl python3 -m unittest discover -s openclaw-rl/tests -p 'test_dream_rsi*.py' -v
python3 -m py_compile openclaw-rl/dream_rsi/*.py openclaw-rl/openclaw_api_server.py openclaw-rl/unsloth_qlora_trainer.py openclaw-rl/tests/test_dream_rsi*.py
git diff --check
```

The seeded property check used `random.Random(4217)`, 32 policies from `mutate_policy(PolicyConfig(), 42, 32)`, and 1,000 pools of 1–64 candidates with scores in `{-1, 0, +1}`, randomized validity and targets of 1–32. It checked deterministic output, valid-only selection, uniqueness and exact target bounds.

## Limits

This is **not** a live 3-GPU training smoke test, SGLang/PRM network integration test, full-model regression benchmark, or evidence of capability/efficiency gains. Trainer argument parsing and the proxy buffering method were tested by compiling their actual AST nodes without importing the CUDA training stack. Synthetic fixtures verify control flow, not statistical efficacy.

The sampler remains opt-in. PRM integration must be verified in a separate live run before enabling it for sustained training. No GitHub Actions dependency was introduced.
