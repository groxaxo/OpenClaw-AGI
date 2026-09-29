#!/usr/bin/env bash
# One GPU, accepted-parent continuation, fresh preregistered confirmation.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
GPU=${GPU:-0}
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo 'Exactly one numeric GPU selector required' >&2; exit 64; }
: "${MODEL_PATH:?Pinned original BF16 model path required}"
: "${PARENT_APPROVAL:?Approved parent metadata required}"
: "${RETIRED_PARENT_SUITE:?Published parent confirmation, now development, required}"
: "${RETIRED_PARENT_SHA256:?Retired parent suite hash required}"
: "${SEALED_FINAL:?Fresh preregistered confirmation required}"
: "${SEALED_SHA256:?Preregistered confirmation SHA256 required}"
: "${EVALUATOR_IMAGE:?Digest-pinned Docker image required}"
[[ ! -e "$SEALED_FINAL.consumed.json" ]] || { echo 'Confirmation already consumed' >&2; exit 65; }
export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT/openclaw-rl${PYTHONPATH:+:$PYTHONPATH}"
# Shared lock used for this family's single-GPU runs; no unrelated jobs killed.
mkdir -p "$HOME/.cache"
exec 9>"$HOME/.cache/dream-qwen4b-single.lock"
flock -n 9 || { echo 'A continuation already owns this GPU lock' >&2; exit 73; }
RUN_ROOT=${RUN_ROOT:-"$HOME/dream-qwen4b-continuations"}
mkdir -p -- "$RUN_ROOT"
CASE=$(mktemp -d "$RUN_ROOT/run.XXXXXX")
"$PYTHON_BIN" - "$ROOT" "$CASE" "$PARENT_APPROVAL" <<'PY'
import contextlib,pathlib,sys,unittest
from dream_rsi.external_judge import ExternalJudgeGate,canonical,digest
from dream_rsi.qwen4b_chain import validate_parent
root,case=map(pathlib.Path,sys.argv[1:3])
validate_parent(sys.argv[3])
suite=unittest.defaultTestLoader.discover(str(root/'openclaw-rl/tests'),pattern='test_*.py')
with (case/'tests.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
if not result.wasSuccessful():raise SystemExit('unit tests failed')
gate=ExternalJudgeGate(case/'attestation',root,providers=('muse',))
record={'passed':True,'tests_run':result.testsRun,'code_sha256':digest(gate.sources())}
(case/'test-report.json').write_text(canonical(record)+'\n')
print('TESTS_PASS',result.testsRun,'CASE',case)
PY
if [[ ${CHECK_ONLY:-0} == 1 ]]; then echo "CHECK_ONLY_PASS=$CASE"; exit 0; fi
timeout --signal=TERM --kill-after=20s 3700 "$PYTHON_BIN" -m dream_rsi.qwen4b_continue \
 --model-path "$MODEL_PATH" --parent-approval "$PARENT_APPROVAL" \
 --retired-parent-suite "$RETIRED_PARENT_SUITE" --retired-parent-sha256 "$RETIRED_PARENT_SHA256" \
 --sealed-final "$SEALED_FINAL" --sealed-sha256 "$SEALED_SHA256" \
 --docker-image "$EVALUATOR_IMAGE" --output "$CASE/run" --test-report "$CASE/test-report.json" \
 --max-epochs 4 --min-epochs 2 --learning-rate 2.5e-5 --repair-weight 12 \
 --max-new-tokens 512 --judge-timeout 300 --max-wall-seconds 3600 2>&1 | tee "$CASE/live.log"
"$PYTHON_BIN" - "$CASE/run/report.json" <<'PY'
import json,pathlib,sys
r=json.loads(pathlib.Path(sys.argv[1]).read_text())
if r.get('validation_status')!='PASS':raise SystemExit('Continuation was not promoted')
print('PROMOTION_PASS_PENDING_COLD_REPRODUCTION; do not advance lineage until independently reproduced')
PY
