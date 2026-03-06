#!/bin/bash
# OpenClaw-OPD QLoRA launcher — 2x RTX 3090 (48 GB total)
#
# GPU layout:
#   GPU 0 (TRAINING_GPU) : Unsloth QLoRA student training
#   GPU 1 (ROLLOUT_GPU)  : sglang rollout inference server
#
# Usage:
#   export HF_CKPT=/path/to/Qwen3-4B
#   export SAVE_CKPT=/path/to/openclaw-qlora-opd/ckpt
#   bash run_qwen3_4b_opd_3090_2x.sh

set -euo pipefail

# ── user-configurable paths ────────────────────────────────────────────────
HF_CKPT="${HF_CKPT:-/absolute/path/to/Qwen3-4B}"
SAVE_CKPT="${SAVE_CKPT:-/absolute/path/to/openclaw-qlora-opd/ckpt}"
PRM_MODEL_PATH="${PRM_MODEL_PATH:-${HF_CKPT}}"

# ── GPU assignment ─────────────────────────────────────────────────────────
TRAINING_GPU="${TRAINING_GPU:-0}"
ROLLOUT_GPU="${ROLLOUT_GPU:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Also include the openclaw-rl directory so we can import unsloth_qlora_trainer helpers.
RL_DIR="$(cd -- "${SCRIPT_DIR}/../openclaw-rl" &>/dev/null && pwd)"

# ── API server / proxy ─────────────────────────────────────────────────────
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-30000}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b}"
export SGLANG_API_KEY="${SGLANG_API_KEY:-}"
export OPENCLAW_RECORD_ENABLED="${OPENCLAW_RECORD_ENABLED:-1}"
export OPENCLAW_RECORD_FILE="${OPENCLAW_RECORD_FILE:-${SCRIPT_DIR}/results/qwen3_4b_opd_3090_2x_record.jsonl}"
export OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY="${OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY:-3}"

mkdir -p "$(dirname "${OPENCLAW_RECORD_FILE}")"

# ── sglang rollout server ──────────────────────────────────────────────────
SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
SGLANG_PORT="${SGLANG_PORT:-30001}"
SGLANG_TP="${SGLANG_TP:-1}"
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.85}"

# ── QLoRA hyper-parameters ─────────────────────────────────────────────────
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"
CONTEXT_LEN="${CONTEXT_LEN:-32768}"

# ── OPD training hyper-parameters ─────────────────────────────────────────
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-2}"
NUM_ROLLOUT="${NUM_ROLLOUT:-100000000}"
LR="${LR:-1e-6}"
KL_COEF="${KL_COEF:-0.02}"
EPS_CLIP="${EPS_CLIP:-0.2}"
EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.28}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
UPDATE_WEIGHTS_INTERVAL="${UPDATE_WEIGHTS_INTERVAL:-1}"
DISTILL_TOPK="${DISTILL_TOPK:-0}"    # 0 = token-level OPD (default); >0 = top-K

# ── PRM / judge (optional) ────────────────────────────────────────────────
PRM_ENABLE="${PRM_ENABLE:-0}"
PRM_PORT="${PRM_PORT:-30002}"
PRM_M="${PRM_M:-3}"

echo "========================================================"
echo "  OpenClaw-OPD  |  Unsloth QLoRA  |  2x RTX 3090"
echo "  Training GPU  : ${TRAINING_GPU}"
echo "  Rollout  GPU  : ${ROLLOUT_GPU}"
echo "  Model         : ${HF_CKPT}"
echo "  Save to       : ${SAVE_CKPT}"
echo "  Proxy port    : ${PORT}  (point OpenClaw agent here)"
echo "  sglang port   : ${SGLANG_PORT}"
echo "  Top-K distill : ${DISTILL_TOPK}"
echo "========================================================"

PRM_ARGS=()
if [[ "${PRM_ENABLE}" == "1" ]]; then
    PRM_ARGS=(
        --prm-enable
        --prm-host 127.0.0.1
        --prm-port "${PRM_PORT}"
        --prm-model-path "${PRM_MODEL_PATH}"
        --prm-m "${PRM_M}"
        --prm-temperature 0.6
        --prm-max-new-tokens 8192
    )
fi

export SGLANG_CUDA_VISIBLE_DEVICES="${ROLLOUT_GPU}"

PYTHONPATH="${SCRIPT_DIR}:${RL_DIR}:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES="${TRAINING_GPU}" \
python "${SCRIPT_DIR}/unsloth_qlora_opd_trainer.py" \
    --hf-checkpoint "${HF_CKPT}" \
    --save "${SAVE_CKPT}" \
    --save-interval "${SAVE_INTERVAL}" \
    --max-seq-length "${MAX_SEQ_LEN}" \
    --load-in-4bit \
    --lora-r "${LORA_R}" \
    --lora-alpha "${LORA_ALPHA}" \
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --mini-batch-size "${MINI_BATCH_SIZE}" \
    --num-rollout "${NUM_ROLLOUT}" \
    --lr "${LR}" \
    --kl-coef "${KL_COEF}" \
    --eps-clip "${EPS_CLIP}" \
    --eps-clip-high "${EPS_CLIP_HIGH}" \
    --distill-topk "${DISTILL_TOPK}" \
    --sglang-host "${SGLANG_HOST}" \
    --sglang-port "${SGLANG_PORT}" \
    --sglang-tp "${SGLANG_TP}" \
    --sglang-mem-fraction "${SGLANG_MEM_FRACTION}" \
    --sglang-context-length "${CONTEXT_LEN}" \
    --sglang-reasoning-parser qwen3 \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --update-weights-interval "${UPDATE_WEIGHTS_INTERVAL}" \
    --proxy-host "${HOST}" \
    --proxy-port "${PORT}" \
    --teacher-lp-max-concurrency "${OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY}" \
    "${PRM_ARGS[@]}"
