# ============================================================
# OpenClaw-RL — Unsloth QLoRA Dockerfile
# ============================================================
# Designed for 2× RTX 3090 (48 GB total VRAM).
#
# Build:
#   docker build -t openclaw-rl:latest .
#
# Run (Binary RL, 2× 3090):
#   docker run --gpus '"device=0,1"' \
#     -e HF_CKPT=/models/Qwen3-4B \
#     -e SAVE_CKPT=/checkpoints/openclaw-qlora-rl \
#     -v /path/to/models:/models \
#     -v /path/to/checkpoints:/checkpoints \
#     -p 30000:30000 \
#     openclaw-rl:latest \
#     bash openclaw-rl/run_qwen3_4b_3090_2x.sh
#
# Run (OPD, 2× 3090):
#   docker run --gpus '"device=0,1"' \
#     -e HF_CKPT=/models/Qwen3-4B \
#     -e SAVE_CKPT=/checkpoints/openclaw-qlora-opd \
#     -v /path/to/models:/models \
#     -v /path/to/checkpoints:/checkpoints \
#     -p 30000:30000 \
#     openclaw-rl:latest \
#     bash openclaw-opd/run_qwen3_4b_opd_3090_2x.sh
# ============================================================

ARG CUDA_VERSION=12.4.1
ARG PYTHON_VERSION=3.11

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python3-pip \
        git \
        git-lfs \
        curl \
        wget \
        ca-certificates \
        build-essential \
        ninja-build \
        libssl-dev \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3 / pip point to the right version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
 && python -m pip install --upgrade pip setuptools wheel

# ── PyTorch (CUDA 12.4) ───────────────────────────────────────────────────────
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# ── Unsloth (CUDA 12.4 + PyTorch 2.5) ────────────────────────────────────────
# See https://github.com/unslothai/unsloth for the full install matrix.
# Override UNSLOTH_VARIANT at build time to target a different CUDA/PyTorch combo:
#   docker build --build-arg UNSLOTH_VARIANT=cu121-torch220 ...
ARG UNSLOTH_VARIANT=cu124-torch250
RUN pip install "unsloth[${UNSLOTH_VARIANT}] @ git+https://github.com/unslothai/unsloth.git"

# ── QLoRA + serving dependencies ─────────────────────────────────────────────
WORKDIR /workspace
COPY requirements-qlora.txt /workspace/requirements-qlora.txt
RUN pip install -r requirements-qlora.txt

# ── SGLang (rollout inference server) ────────────────────────────────────────
# Pin the same commit used in requirements.txt for reproducibility.
RUN pip install \
    "sglang[all] @ git+https://github.com/sgl-project/sglang.git@dce8b0606c06d3a191a24c7b8cbe8e238ab316c9#egg=sglang&subdirectory=python" \
    || pip install "sglang[all]"

# ── Copy project source ───────────────────────────────────────────────────────
COPY . /workspace
WORKDIR /workspace

# Pre-create results directories
RUN mkdir -p openclaw-rl/results openclaw-opd/results

# ── Expose the default OpenClaw proxy port ───────────────────────────────────
EXPOSE 30000

# ── Default command: show usage ───────────────────────────────────────────────
CMD ["bash", "-c", "\
echo ''; \
echo '========================================================'; \
echo '  OpenClaw-RL  |  Unsloth QLoRA  |  2x RTX 3090'; \
echo '========================================================'; \
echo ''; \
echo 'Set HF_CKPT and SAVE_CKPT, then run one of:'; \
echo ''; \
echo '  Binary RL:'; \
echo '    bash openclaw-rl/run_qwen3_4b_3090_2x.sh'; \
echo ''; \
echo '  On-Policy Distillation (OPD):'; \
echo '    bash openclaw-opd/run_qwen3_4b_opd_3090_2x.sh'; \
echo ''; \
echo 'Example (docker run):'; \
echo '  docker run --gpus '\''"device=0,1'\'' \\'; \
echo '    -e HF_CKPT=/models/Qwen3-4B \\'; \
echo '    -e SAVE_CKPT=/checkpoints/openclaw-qlora-rl \\'; \
echo '    -v /path/to/models:/models \\'; \
echo '    -v /path/to/checkpoints:/checkpoints \\'; \
echo '    -p 30000:30000 \\'; \
echo '    openclaw-rl:latest \\'; \
echo '    bash openclaw-rl/run_qwen3_4b_3090_2x.sh'; \
echo ''; \
"]
