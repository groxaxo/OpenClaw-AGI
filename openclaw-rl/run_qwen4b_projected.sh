#!/usr/bin/env bash
# Known-task protected repair, one GPU, immediate approved parent comparator.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
GPU=${GPU:-0}
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo 'Exactly one numeric GPU selector required' >&2; exit 64; }
: "${MODEL_PATH:?Pinned Qwen3.5-4B BF16 model required}"
: "${PARENT_APPROVAL:?Approved-parent acceptance record required}"
: "${RETIRED_PARENT_SUITE:?Previously published parent suite, now public development, required}"
: "${RETIRED_PARENT_SHA256:?Published parent suite digest required}"
: "${SEALED_FINAL:?Preregistered unconsumed confirmation suite required}"
: "${SEALED_SHA256:?Confirmation digest required}"
: "${EVALUATOR_IMAGE:?Digest-pinned Python Docker image required}"
[[ ! -e "$SEALED_FINAL.consumed.json" ]] || { echo 'Confirmation already consumed' >&2; exit 65; }
export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT/openclaw-rl${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HOME/.cache"
exec 9>"$HOME/.cache/dream-qwen4b-single.lock"
flock -n 9 || { echo 'Another run owns this experiment lock' >&2; exit 73; }
RUN_ROOT=${RUN_ROOT:-"$HOME/dream-qwen4b-projected-runs"}
mkdir -p -- "$RUN_ROOT"
CASE=$(mktemp -d "$RUN_ROOT/run.XXXXXX")
"$PYTHON_BIN" - "$ROOT" "$CASE" "$PARENT_APPROVAL" <<'PY'
import contextlib,pathlib,sys,unittest
from dream_rsi.external_judge import ExternalJudgeGate,canonical,digest
from dream_rsi.qwen4b_chain import validate_parent
root,case=map(pathlib.Path,sys.argv[1:3]);validate_parent(sys.argv[3])
suite=unittest.defaultTestLoader.discover(str(root/'openclaw-rl/tests'),pattern='test_*.py')
with (case/'tests.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
if not result.wasSuccessful():raise SystemExit('Tests failed; no training')
gate=ExternalJudgeGate(case/'attestation',root,providers=('muse',))
(case/'test-report.json').write_text(canonical({'passed':True,'tests_run':result.testsRun,'code_sha256':digest(gate.sources())})+'\n')
print('TESTS_PASS',result.testsRun,'CASE',case)
PY
if [[ ${CHECK_ONLY:-0} == 1 ]]; then echo "CHECK_ONLY_PASS=$CASE"; exit 0; fi
timeout --signal=TERM --kill-after=20s 2500 "$PYTHON_BIN" -m dream_rsi.qwen4b_projected_continue \
 --model-path "$MODEL_PATH" --parent-approval "$PARENT_APPROVAL" \
 --retired-parent-suite "$RETIRED_PARENT_SUITE" --retired-parent-sha256 "$RETIRED_PARENT_SHA256" \
 --sealed-final "$SEALED_FINAL" --sealed-sha256 "$SEALED_SHA256" \
 --docker-image "$EVALUATOR_IMAGE" --output "$CASE/run" --test-report "$CASE/test-report.json" \
 --max-epochs 3 --min-epochs 1 --learning-rate 5e-5 --repair-weight 8 \
 --max-new-tokens 512 --judge-timeout 300 --max-wall-seconds 2400 2>&1 | tee "$CASE/live.log"
"$PYTHON_BIN" - "$CASE/run/report.json" <<'PY'
import json,pathlib,sys
r=json.loads(pathlib.Path(sys.argv[1]).read_text())
if r.get('validation_status')!='PASS':raise SystemExit('No promotion; accepted parent preserved')
print('CANDIDATE_PASSED; requires separate locked cold reproduction before advancing lineage')
PY
