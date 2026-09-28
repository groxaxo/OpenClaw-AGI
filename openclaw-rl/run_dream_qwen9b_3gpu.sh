#!/usr/bin/env bash
# Bounded native validation. Does not deploy weights or restart existing servers.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
: "${MODEL_PATH:?Set MODEL_PATH to the pinned Qwen/Qwen3.5-9B HF snapshot directory}"
: "${EVALUATOR_IMAGE:?Set EVALUATOR_IMAGE to python@sha256:<verified Docker digest>}"
JUDGES=${JUDGES:-muse}
CYCLES=${CYCLES:-2}
RUN_ROOT=${RUN_ROOT:-"$HOME/dream-qwen9b-runs"}
mkdir -p -- "$RUN_ROOT"
CASE_DIR=$(mktemp -d "$RUN_ROOT/validation.XXXXXX")
export PYTHONPATH="$ROOT/openclaw-rl${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=3 TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PYTHON_BIN" - "$ROOT" "$CASE_DIR" <<'PY'
import contextlib,json,pathlib,sys,unittest
from dream_rsi.external_judge import ExternalJudgeGate,digest,canonical
root,case=map(pathlib.Path,sys.argv[1:])
suite=unittest.defaultTestLoader.discover(str(root/'openclaw-rl/tests'),pattern='test_*.py')
with (case/'unit-tests.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
if not result.wasSuccessful():raise SystemExit('Unit tests failed; training not launched')
gate=ExternalJudgeGate(case/'attestation',root,providers=('muse',))
report={'passed':True,'tests_run':result.testsRun,'code_sha256':digest(gate.sources())}
(case/'test-report.json').write_text(canonical(report)+'\n')
print('Tests passed:',result.testsRun,'; evidence:',case)
PY
if [[ ${CHECK_ONLY:-0} == 1 ]]; then
    printf 'CHECK_ONLY: no training launched. Evidence: %s\n' "$CASE_DIR"
    exit 0
fi
PORT=$("$PYTHON_BIN" - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(('127.0.0.1',0)); print(sock.getsockname()[1])
PY
)
# Explicit loopback avoids numeric hostnames (e.g. 12700 -> 0.0.49.156).
timeout --signal=TERM --kill-after=20s 1900 "$PYTHON_BIN" -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=3 --master_addr=127.0.0.1 --master_port="$PORT" \
    --module dream_rsi.live_validation \
    --model-path "$MODEL_PATH" --docker-image "$EVALUATOR_IMAGE" \
    --output "$CASE_DIR/run" --test-report "$CASE_DIR/test-report.json" \
    --judges "$JUDGES" --cycles "$CYCLES" --max-new-tokens 192 \
    --max-wall-seconds 1800 --judge-timeout 300 2>&1 | tee "$CASE_DIR/live.log"
"$PYTHON_BIN" - "$CASE_DIR/run/report.json" <<'PYREPORT'
import json,pathlib,sys
report=json.loads(pathlib.Path(sys.argv[1]).read_text())
if report.get('validation_status')!='PASS' or report.get('final_external_review_ok') is not True:
    raise SystemExit('Final integration review did not pass; inspect report and judge receipts')
print('Integration validation PASS; checkpoint promotion:', report.get('promotion_approved'))
PYREPORT
printf 'Report: %s/run/report.json\n' "$CASE_DIR"
