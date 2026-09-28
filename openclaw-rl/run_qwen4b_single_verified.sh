#!/usr/bin/env bash
# Exactly one GPU for this experiment; no serving-process changes or auto-deploy.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
GPU_INDEX=${GPU_INDEX:-0}
[[ "$GPU_INDEX" =~ ^[0-9]+$ ]] || { echo 'GPU_INDEX must be one integer' >&2; exit 2; }
: "${MODEL_PATH:?Provide the pinned Qwen3.5-4B snapshot path}"
: "${EVALUATOR_IMAGE:?Provide the digest-pinned Python Docker image}"
: "${SEALED_FINAL:?Provide a preregistered confirmation JSON file}"
: "${SEALED_SHA256:?Provide its preregistered SHA-256, not a recomputed acceptance value}"
[[ ! -e "$SEALED_FINAL.consumed.json" ]] || { echo 'This final suite has already been consumed' >&2; exit 2; }
CACHE=${XDG_CACHE_HOME:-"$HOME/.cache"}/openclaw-validation
mkdir -p "$CACHE"
exec 9>"$CACHE/qwen4b-single.lock"
flock -n 9 || { echo 'Another single-GPU validation owns the lock' >&2; exit 2; }
RUN_ROOT=${RUN_ROOT:-"$HOME/dream-qwen4b-runs"}
mkdir -p "$RUN_ROOT"
CASE=$(mktemp -d "$RUN_ROOT/single.XXXXXX")
export CUDA_VISIBLE_DEVICES="$GPU_INDEX" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT/openclaw-rl${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" - "$ROOT" "$CASE" "$EVALUATOR_IMAGE" "$SEALED_FINAL" "$SEALED_SHA256" <<'PY'
import contextlib, hashlib, pathlib, sys, unittest
from dream_rsi.external_judge import ExternalJudgeGate, digest, canonical
from dream_rsi.qwen4b_harder import TRAIN
from dream_rsi.coding_tasks import CodingTask, grade
from dream_rsi.qwen4b_validation import load_suite
root,case=map(pathlib.Path,sys.argv[1:3]);image,seal,expected=sys.argv[3:]
load_suite(seal,expected)
suite=unittest.defaultTestLoader.discover(str(root/'openclaw-rl/tests'),pattern='test_*.py')
with (case/'tests.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
if not result.wasSuccessful():raise SystemExit('Tests failed; no model loaded')
for example in TRAIN:
    r=grade(CodingTask(example.task_id,example.description,example.cases),example.solution,image)
    if r['passed']!=r['total']:raise SystemExit('Invalid training reference: '+example.task_id)
gate=ExternalJudgeGate(case/'attestation',root,providers=('muse',))
report={'passed':True,'tests_run':result.testsRun,'reference_solutions_passed':len(TRAIN),'code_sha256':digest(gate.sources())}
(case/'test-report.json').write_text(canonical(report)+'\n')
print(canonical(report),flush=True)
PY
if [[ ${CHECK_ONLY:-0} == 1 ]]; then
    echo "CHECK_ONLY PASS: $CASE; no GPU training started"; exit 0
fi
timeout --signal=TERM --kill-after=15s 3100 "$PYTHON_BIN" -u -m dream_rsi.qwen4b_single \
    --model-path "$MODEL_PATH" --sealed-final "$SEALED_FINAL" --sealed-sha256 "$SEALED_SHA256" \
    --docker-image "$EVALUATOR_IMAGE" --output "$CASE/run" --test-report "$CASE/test-report.json" \
    --max-epochs 6 --min-epochs 2 --learning-rate 5e-5 --max-new-tokens 256 \
    --judge-timeout 300 --max-wall-seconds 3000 2>&1 | tee "$CASE/live.log"
"$PYTHON_BIN" - "$CASE/run/report.json" <<'PY'
import json,pathlib,sys
r=json.loads(pathlib.Path(sys.argv[1]).read_text())
if r.get('validation_status')!='PASS' or r.get('promotion_approved') is not True or r.get('final_external_review_ok') is not True:
    raise SystemExit('NOT_OK: promotion and final independent review have not both passed')
print('PASS: bounded single-GPU run and experimental promotion; no production deployment')
PY
echo "REPORT=$CASE/run/report.json"
