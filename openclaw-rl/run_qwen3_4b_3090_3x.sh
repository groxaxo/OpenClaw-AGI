#!/bin/bash
# OpenClaw-RL Qwen 3.5 Vision launcher — 3x RTX 3090 (72 GB total)
#
# GPU layout:
#   GPUs 0,1 (TRAINING_GPUS) : Unsloth QLoRA actor — DDP via accelerate
#   GPU 2    (ROLLOUT_GPU)    : sglang rollout inference server
#
# With DDP on 2 GPUs the effective batch size doubles; adjust
# GLOBAL_BATCH_SIZE / MINI_BATCH_SIZE to taste.
#
# Usage:
#   export HF_CKPT=unsloth/Qwen3.5-4B
#   export SAVE_CKPT=/path/to/openclaw-qlora-rl/ckpt
#   bash run_qwen3_4b_3090_3x.sh

set -euo pipefail

# ── user-configurable paths ────────────────────────────────────────────────
HF_CKPT="${HF_CKPT:-unsloth/Qwen3.5-4B}"
SAVE_CKPT="${SAVE_CKPT:-/absolute/path/to/openclaw-qlora-rl/ckpt}"
PRM_MODEL_PATH="${PRM_MODEL_PATH:-${HF_CKPT}}"

# ── GPU assignment ─────────────────────────────────────────────────────────
TRAINING_GPUS="${TRAINING_GPUS:-0,1}"   # two GPUs for DDP training
ROLLOUT_GPU="${ROLLOUT_GPU:-2}"          # one GPU for sglang inference

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ── API server / proxy ─────────────────────────────────────────────────────
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-30000}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-4b}"
export SGLANG_API_KEY="${SGLANG_API_KEY:-}"
export OPENCLAW_RECORD_ENABLED="${OPENCLAW_RECORD_ENABLED:-1}"
export OPENCLAW_RECORD_FILE="${OPENCLAW_RECORD_FILE:-${SCRIPT_DIR}/results/qwen3_5_4b_vision_3090_3x_record.jsonl}"

mkdir -p "$(dirname "${OPENCLAW_RECORD_FILE}")"

# ── sglang rollout server ──────────────────────────────────────────────────
SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
SGLANG_PORT="${SGLANG_PORT:-30001}"
SGLANG_TP="${SGLANG_TP:-1}"
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.85}"
SGLANG_TOOL_CALL_PARSER="${SGLANG_TOOL_CALL_PARSER:-qwen3_coder}"

# ── QLoRA hyper-parameters ─────────────────────────────────────────────────
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"
CONTEXT_LEN="${CONTEXT_LEN:-32768}"

# ── RL training hyper-parameters ──────────────────────────────────────────
# With 2 training GPUs, double the effective throughput.
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-2}"
NUM_ROLLOUT="${NUM_ROLLOUT:-100000000}"
LR="${LR:-1e-6}"
KL_COEF="${KL_COEF:-0.02}"
EPS_CLIP="${EPS_CLIP:-0.2}"
EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.28}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
UPDATE_WEIGHTS_INTERVAL="${UPDATE_WEIGHTS_INTERVAL:-1}"

# ── PRM (optional) ────────────────────────────────────────────────────────
PRM_ENABLE="${PRM_ENABLE:-0}"
PRM_PORT="${PRM_PORT:-30002}"
PRM_M="${PRM_M:-3}"

echo "========================================================"
echo "  OpenClaw-RL  |  Unsloth Qwen 3.5 Vision  |  3x RTX 3090"
echo "  Training GPUs: ${TRAINING_GPUS}"
echo "  Rollout  GPU : ${ROLLOUT_GPU}"
echo "  Model        : ${HF_CKPT}"
echo "  Save to      : ${SAVE_CKPT}"
echo "  Proxy port   : ${PORT}  (point OpenClaw agent here)"
echo "  sglang port  : ${SGLANG_PORT}"
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
        --prm-max-new-tokens 4096
    )
fi

# ── launch with accelerate (DDP across TRAINING_GPUS) ─────────────────────
# The sglang subprocess spawned by the trainer will use ROLLOUT_GPU.
export SGLANG_CUDA_VISIBLE_DEVICES="${ROLLOUT_GPU}"

# ── count training GPUs from the comma-separated list ─────────────────────
IFS=',' read -ra _TRAINING_GPU_ARRAY <<< "${TRAINING_GPUS}"
_NUM_TRAINING_GPUS="${#_TRAINING_GPU_ARRAY[@]}"

PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES="${TRAINING_GPUS}" \
accelerate launch \
    --num_processes "${_NUM_TRAINING_GPUS}" \
    --mixed_precision bf16 \
    "${SCRIPT_DIR}/unsloth_qlora_trainer.py" \
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
        --sglang-host "${SGLANG_HOST}" \
        --sglang-port "${SGLANG_PORT}" \
        --sglang-tp "${SGLANG_TP}" \
        --sglang-mem-fraction "${SGLANG_MEM_FRACTION}" \
        --sglang-context-length "${CONTEXT_LEN}" \
        --sglang-reasoning-parser qwen3 \
        --sglang-tool-call-parser "${SGLANG_TOOL_CALL_PARSER}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --update-weights-interval "${UPDATE_WEIGHTS_INTERVAL}" \
        --proxy-host "${HOST}" \
        --proxy-port "${PORT}" \
        "${PRM_ARGS[@]}"
