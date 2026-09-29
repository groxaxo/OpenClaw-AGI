#!/usr/bin/env bash
# Bounded, one-GPU validation. No production activation, no automatic retries.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
GPU=${GPU:-0}
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo 'GPU must be one numeric device index'; exit 64; }
: "${MODEL_PATH:?Set the local official Qwen3.5-4B snapshot path}"
: "${SEALED_FINAL:?Set the preregistered, unconsumed confirmation JSON path}"
: "${SEALED_SHA256:?Set its preregistered SHA-256}"
: "${EVALUATOR_IMAGE:?Set digest-pinned Python Docker image}"
[[ ! -e "${SEALED_FINAL%.json}.json.consumed.json" ]] || { echo 'Confirmation suite already consumed'; exit 65; }
RUN_ROOT=${RUN_ROOT:-"$HOME/dream-qwen4b-bf16-runs"}
mkdir -p "$RUN_ROOT" "$HOME/.cache"
exec 9>"$HOME/.cache/dream-qwen4b-single.lock"
flock -n 9 || { echo 'Another Dream single-GPU run holds the lock'; exit 75; }
CASE=$(mktemp -d "$RUN_ROOT/validation.XXXXXX")
export PYTHONPATH="$ROOT/openclaw-rl${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PYTHON_BIN" - "$ROOT" "$CASE" <<'PY'
import contextlib,pathlib,sys,unittest
from dream_rsi.external_judge import ExternalJudgeGate,digest,canonical
root,case=map(pathlib.Path,sys.argv[1:])
suite=unittest.defaultTestLoader.discover(str(root/'openclaw-rl/tests'),pattern='test_*.py')
with (case/'tests.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
if not result.wasSuccessful():raise SystemExit('Tests failed; no training launched')
gate=ExternalJudgeGate(case/'attestation',root,providers=('muse',))
(case/'test-report.json').write_text(canonical({'passed':True,'tests_run':result.testsRun,'code_sha256':digest(gate.sources())})+'\n')
print('UNIT_TESTS_PASS',result.testsRun)
PY
if [[ ${CHECK_ONLY:-0} == 1 ]]; then echo "CHECK_ONLY_PASS=$CASE"; exit 0; fi
set +e
timeout --signal=TERM --kill-after=20s 3700 "$PYTHON_BIN" -m dream_rsi.qwen4b_bf16_guarded \
 --model-path "$MODEL_PATH" --docker-image "$EVALUATOR_IMAGE" \
 --sealed-final "$SEALED_FINAL" --sealed-sha256 "$SEALED_SHA256" \
 --output "$CASE/run" --test-report "$CASE/test-report.json" \
 --learning-rate 5e-5 --max-epochs 6 --min-epochs 2 --max-new-tokens 512 \
 --judge-timeout 300 --max-wall-seconds 3600 2>&1 | tee "$CASE/live.log"
STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "$STATUS" > "$CASE/exit-code.txt"
(( STATUS == 0 )) || exit "$STATUS"
"$PYTHON_BIN" - "$CASE/run/report.json" <<'PY'
import json,pathlib,sys
r=json.loads(pathlib.Path(sys.argv[1]).read_text())
if r.get('validation_status')!='PASS' or r.get('promotion_approved') is not True or r.get('final_external_review_ok') is not True:
    raise SystemExit('Final evidence did not qualify for PASS')
print('GUARDED_VALIDATION_PASS',r['promotion_review_id'])
PY
printf 'REPORT=%s/run/report.json\n' "$CASE"
