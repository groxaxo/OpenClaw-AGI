#!/usr/bin/env bash
# New-task measurement; exactly one GPU; no existing model is deployed/changed.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
GPU=${GPU:-0}
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo 'Exactly one numeric GPU required' >&2; exit 64; }
: "${MODEL_PATH:?Pinned BF16 model required}"
: "${START_RECORD:?Cold-verified starting adapter record required}"
: "${APPROVED_RECORD:?Approved reference adapter record required}"
: "${PUBLIC_SUITE:?Registered public suite required}"
: "${RESERVED_SUITE:?Registered unconsumed reserved suite required}"
: "${REGISTRY:?Suite registry required}"
: "${EVALUATOR_IMAGE:?Digest-pinned Docker image required}"
[[ ! -e "$RESERVED_SUITE.consumed.json" ]] || { echo 'Reserved suite already consumed' >&2; exit 65; }
export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT/openclaw-rl${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HOME/.cache"
exec 9>"$HOME/.cache/dream-qwen4b-single.lock"
flock -n 9 || { echo 'Another experiment owns the GPU lock' >&2; exit 73; }
PYTHON_BIN=${PYTHON_BIN:-python3}
RUN_ROOT=${RUN_ROOT:-"$HOME/dream-qwen4b-new-task-runs"}
mkdir -p -- "$RUN_ROOT"
CASE=$(mktemp -d "$RUN_ROOT/run.XXXXXX")
"$PYTHON_BIN" - "$ROOT" "$CASE" <<'PY'
import contextlib,pathlib,sys,unittest
from dream_rsi.external_judge import ExternalJudgeGate,digest,canonical
root,case=map(pathlib.Path,sys.argv[1:])
suite=unittest.defaultTestLoader.discover(str(root/'openclaw-rl/tests'),pattern='test_*.py')
with (case/'tests.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
if not result.wasSuccessful():raise SystemExit('Tests failed; training not launched')
gate=ExternalJudgeGate(case/'attestation',root,providers=('muse',))
(case/'tests.json').write_text(canonical({'passed':True,'tests_run':result.testsRun,'code_sha256':digest(gate.sources())})+'\n')
print('TESTS_PASS',result.testsRun,'CASE',case)
PY
if [[ ${CHECK_ONLY:-0} == 1 ]]; then echo "CHECK_ONLY_PASS=$CASE"; exit 0; fi
timeout --signal=TERM --kill-after=20s 2800 "$PYTHON_BIN" -m dream_rsi.new_task_experiment \
 --model-path "$MODEL_PATH" --start-record "$START_RECORD" --approved-record "$APPROVED_RECORD" \
 --public-suite "$PUBLIC_SUITE" --reserved-suite "$RESERVED_SUITE" --registry "$REGISTRY" \
 --output "$CASE/experiment" --docker-image "$EVALUATOR_IMAGE" --test-report "$CASE/tests.json" \
 --max-epochs 3 --learning-rate 5e-5 --max-new-tokens 512 --max-wall-seconds 2700 2>&1 | tee "$CASE/live.log"
printf 'Completed measurement report: %s/experiment/report.json\n' "$CASE"
printf 'Acceptance is a separate report field, not implied by process completion.\n'
