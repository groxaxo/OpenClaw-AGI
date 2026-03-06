<div align="center">
  <h1 align="center">
    <img src="assets/spacer.png" alt="" width="23" height="40" align="absmiddle" />
    OpenClaw-RL<!--
--><sup>
    <img src="assets/clawistool.png" alt="Claw-RL logo" width="23" height="40" align="absmiddle" />
    <sup>
  </h1>

  <p><b>Empowering OpenClaw with RL — Train a personalized agent simply by talking to it.<br>Runs on as few as <u>2× RTX 3090</u> GPUs using Unsloth QLoRA.</b></p>
</div>


<p align="center">
  <img src="https://img.shields.io/badge/⚡_Fully_Async-yellow?style=for-the-badge" alt="Fully Async" />
  <img src="https://img.shields.io/badge/💰_Zero_API_Keys-blue?style=for-the-badge" alt="Zero API Keys" />
  <img src="https://img.shields.io/badge/🤖_Personalized-success?style=for-the-badge" alt="Personalized" />
  <img src="https://img.shields.io/badge/🛠️_Auto_Optimization-orange?style=for-the-badge" alt="Auto" />
  <img src="https://img.shields.io/badge/💬_Language_Feedback-purple?style=for-the-badge" alt="Language Feedback" />
  <img src="https://img.shields.io/badge/🦥_Unsloth_QLoRA-red?style=for-the-badge" alt="Unsloth QLoRA" />
  <br><br>
  <a href="https://yinjjiew.github.io/projects/openclawrl"><img src="https://img.shields.io/badge/Blog-Page-blue?style=flat-square" alt="OpenClaw-RL Blog" /></a>
  <a href="https://openclaw.ai"><img src="https://img.shields.io/badge/OpenClaw-Plugin-orange?style=flat-square" alt="OpenClaw Plugin" /></a>
  <a href="https://github.com/THUDM/slime"><img src="https://img.shields.io/badge/Slime-Based-purple?style=flat-square" alt="Slime Based" /></a>
  <a href="https://github.com/unslothai/unsloth"><img src="https://img.shields.io/badge/Unsloth-Powered-green?style=flat-square" alt="Unsloth Powered" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License MIT" /></a>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/a58aacad-3c1d-47aa-bbd1-cf8c5f36de6f" controls width="200"></video>
</p>









## 📰 News

- **[2026/3/6]** 🦥 **Unsloth QLoRA is now the default training method** — train on just 2× RTX 3090 (48 GB), no Megatron-LM or Ray cluster required. Docker and conda setup included.
- **[2026/3/3]** 🙌 Working with the authors of [SDFT](https://arxiv.org/abs/2601.19897) and [SDPO](https://arxiv.org/abs/2601.20802), we have integrated their methods into [openclaw-opd](./openclaw-opd). We welcome the integration of novel and effective methods!
- **[2026/3/3]** 📺 Check out these community tutorial videos on OpenClaw-RL: [**Video 1**](https://www.youtube.com/watch?v=5xnm1vB7G64) | [**Video 2**](https://www.youtube.com/watch?v=ZtN6Gg_bdJE)
- **[2026/2/26]** 🔥 We release **OpenClaw-RL v1** — a fully asynchronous RL framework for training personalized AI agents from natural conversation feedback. 

---

## 💡 TL;DR

> **OpenClaw-RL** is a fully asynchronous reinforcement learning framework that turns everyday conversations into training signals for personalized AI agents.

Most RL-for-LLM systems assume centralized, batch-mode training with pre-collected datasets. **OpenClaw-RL** takes a fundamentally different approach: it wraps your self-hosted model in [OpenClaw](https://openclaw.ai) as an OpenAI-compatible API, intercepts live multi-turn conversations, and continuously optimizes the policy in the background — all without interrupting your usage.


<p align="center">
  <img src="assets/rlserver.png"  alt="Overview"  width="600">
</p>

## 🌈 Features

### Fully Asynchronous 4-Component Architecture
OpenClaw-RL decouples **agent serving**, **rollout collection**, **PRM judging**, and **policy training** into independent async loops. None of them block one another — the model serves requests while training runs in the background, and PRM evaluation happens concurrently with new conversations.

### Self-Hosted & Private by Design
The entire stack (model, PRM, training) runs on **your own infrastructure**. Conversation data never leaves your system. No external API keys required.

### From Conversation to Gradient — Automatically
You don't need to manually label data. The system automatically:
- Classifies API messages into **main-line** (trainable) vs. **side** (non-trainable) turns
- Uses the next user/environment message as a natural "next state" signal
- Runs PRM evaluation asynchronously with majority voting for robust scoring
- Submits ready samples to the trainer as they become available

### Two Learning Paradigms in One Framework

**Binary RL (GRPO):** A Process Reward Model scores each turn as good/bad/neutral based on the next-state feedback. The scalar reward is used with GRPO advantage estimation and PPO-style clipped surrogate loss.

**On-Policy Distillation (OPD):** When the next state reveals useful hindsight, a judge model extracts a textual hint. This hint augments the original prompt to create an "enhanced teacher," whose token-level log-probability gap with the student becomes a directional advantage signal — richer than any scalar reward.

### Production-Ready Engineering
- **Session-aware training:** Multi-turn conversations are tracked per-session with proper turn ordering
- **Graceful weight updates:** Submission pauses during model updates, then resumes — no data corruption
- **At-least-one guarantee (Binary RL):** Every session contributes at least one effective training sample
- **Hint quality filtering (OPD):** Only the longest, most informative hint among `m` votes is selected; trivial hints are discarded
- **Teacher log-prob optimization (OPD):** Only response-suffix log-probs are computed to reduce peak memory
- **Record & debug:** All conversations and PRM evaluations are logged to JSONL for analysis

---



## 🎯 Roadmap

Our long-term goal is to **advance personalized, practically useful agents with reinforcement learning**. The roadmap has two tracks:

#### Track 1 — Personal Agent Optimization (Small-Scale but Personal)
✅ **Release v1:** Fully async OpenClaw-RL framework with Binary RL + OPD  
⬜ Broader model family support & more efficient serving  
⬜ Best recipe discovery via large-scale experiments  
⬜ Beyond the policy: extend learning to skills and memory  

#### Track 2 — General Agents Optimization (Scalable Infra)
⬜ **Next (2–3 weeks):** Scalable agentic RL infra for general agents (computer-use first)

---

## 🔧 Quick Start

> **OpenClaw-RL now uses [Unsloth](https://github.com/unslothai/unsloth) + QLoRA as the standard training method.**  
> No Megatron-LM, no Ray cluster — just 2× RTX 3090 (48 GB total VRAM).

### 1. RL Server Environment — Choose Your Setup Path

<details open>
<summary><b>🐳 Option A: Docker (Recommended)</b></summary>

```bash
# Build the image (CUDA 12.4 + PyTorch 2.5 + Unsloth)
docker build -t openclaw-rl:latest -f Dockerfile .

# Run with both 3090s exposed
docker run --gpus '"device=0,1"' \
  -e HF_CKPT=/models/Qwen3.5-4B \
  -e SAVE_CKPT=/checkpoints/openclaw-qlora-rl \
  -v /path/to/models:/models \
  -v /path/to/checkpoints:/checkpoints \
  -p 30000:30000 \
  openclaw-rl:latest \
  bash openclaw-rl/run_qwen3_4b_3090_2x.sh
```

</details>

<details>
<summary><b>🐍 Option B: Conda</b></summary>

```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate openclaw-rl

# Install Unsloth (pick the variant matching your CUDA + PyTorch):
# CUDA 12.4 + PyTorch 2.5:
pip install "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"
# CUDA 12.1 + PyTorch 2.2:
# pip install "unsloth[cu121-torch220] @ git+https://github.com/unslothai/unsloth.git"

# Install remaining QLoRA dependencies
pip install -r requirements-qlora.txt
```

See the full matrix at <https://github.com/unslothai/unsloth#installation>.

</details>

<details>
<summary><b>🔩 Option C: Manual pip install</b></summary>

```bash
# Python 3.10–3.12, CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Unsloth (adjust cu124-torch250 to match your environment)
pip install "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"

# QLoRA + serving dependencies
pip install -r requirements-qlora.txt
```

</details>

### Prerequisites

- **Hardware:** 2× RTX 3090 (24 GB each, 48 GB total) — minimum recommended
- **Software:** CUDA 12.x, Python 3.10–3.12
- **Key libraries:** [Unsloth](https://github.com/unslothai/unsloth), [SGLang](https://github.com/sgl-project/sglang), PEFT, bitsandbytes

For the deprecated legacy full-scale setup (8+ GPUs, Megatron-LM), see [`./instructions/README.md`](./instructions/README.md).







### 2. Start the RL Server

We provide two learning methods, both running via Unsloth QLoRA on 2× RTX 3090:

| Method | Signal Type | How It Works | When to Use |
|---|---|---|---|
| **[Binary RL](./openclaw-rl/)** | Scalar (+1/−1/0) | PRM judges response quality from next-state feedback via majority vote → GRPO | Abundant implicit feedback (likes, env success/failure) |
| **[On-Policy Distillation (OPD)](./openclaw-opd/)** | Token-level directional | Extract hindsight hints from next-state → construct enhanced teacher → token-level distillation | Rich textual feedback; need directional improvement |


Choose your optimization method:

<details open>
<summary><b>Option A: Binary RL — 2× RTX 3090 (Recommended)</b></summary>

```bash
export HF_CKPT=unsloth/Qwen3.5-4B
export SAVE_CKPT=/path/to/openclaw-qlora-rl/ckpt
bash openclaw-rl/run_qwen3_4b_3090_2x.sh
```

GPU layout:
```
GPU 0  →  QLoRA actor training  (Unsloth 4-bit NF4, ~8 GB)
GPU 1  →  sglang rollout server (base model BF16,   ~8 GB)
```

See [`./openclaw-rl/README.md`](./openclaw-rl/README.md) for details.
</details>

<details>
<summary><b>Option B: On-Policy Distillation (OPD) — 2× RTX 3090</b></summary>

```bash
export HF_CKPT=unsloth/Qwen3.5-4B
export SAVE_CKPT=/path/to/openclaw-qlora-opd/ckpt
bash openclaw-opd/run_qwen3_4b_opd_3090_2x.sh
```

The system extracts hindsight hints from your feedback and distills them into the policy at the token level.

See [`./openclaw-opd/README.md`](./openclaw-opd/README.md) for algorithm details.
</details>

<details>
<summary><b>Option C: Legacy research path (deprecated 8+ GPUs, Megatron-LM)</b></summary>

Deprecated research path for large multi-GPU clusters:

```bash
# Binary RL
cd slime && bash ../openclaw-rl/run_qwen3_4b_openclaw_rl.sh

# OPD
cd slime && bash ../openclaw-opd/run_qwen3_4b_openclaw_opd.sh
```

See [`./instructions/README.md`](./instructions/README.md) for full cluster setup instructions.
</details>

Once running, the model is served as an OpenAI-compatible API at:
```
http://<HOST_IP>:30000/v1
```

where `<HOST_IP>` is the **IP address** of the machine running the RL server (e.g. `115.190.98.251`). The port `30000` is the default and can be changed via the `PORT` environment variable.

**Take note of this endpoint** — you will need it when configuring OpenClaw in the next step.



### 3. OpenClaw Setup

Install OpenClaw from the version bundled in this repository (we will update it regularly):

Then configure OpenClaw to route requests to your RL server. Open your `openclaw.json` (or the equivalent settings file) and add a provider entry under `"models"` → `"providers"`:

```json
{
  "models": {
    "providers": {
      "qwen": {
        "baseUrl": "http://<HOST_IP>:30000/v1",
        "apiKey": "apiKey",
        "api": "openai-completions",
        "models": [
          {
            "id": "qwen3.5-4b",
            "name": "Qwen 3.5 4B Vision",
            "reasoning": true,
            "input": ["text", "image"],
            "cost": {
              "input": 0,
              "output": 0,
              "cacheRead": 0,
              "cacheWrite": 0
            },
            "contextWindow": 32768,
            "maxTokens": 8192
          }
        ]
      }
    }
  }
}
```

Replace `<HOST_IP>` with the IP address of your RL server machine. The `apiKey` should match the `SGLANG_API_KEY` you set when starting the server.

That's it — start chatting with your OpenClaw agent. The RL server will automatically collect conversation trajectories, compute rewards, and train the model. Your agent gets better the more you use it.


#### Configurations

Before launching, set these important environment variables as needed:

| Variable | Default | Description |
|---|---|---|
| `HF_CKPT` | `unsloth/Qwen3.5-4B` | Base HuggingFace checkpoint (default vision model) |
| `SAVE_CKPT` | *(required)* | Checkpoint output directory |
| `TRAINING_GPU` | `0` | GPU index for QLoRA training |
| `ROLLOUT_GPU` | `1` | GPU index for sglang rollout inference |
| `LORA_R` | `16` | LoRA rank |
| `LORA_ALPHA` | `16` | LoRA alpha scaling factor |
| `ROLLOUT_BATCH_SIZE` | `16` | Samples collected before each training step |
| `LR` | `1e-6` | Learning rate |
| `PRM_ENABLE` | `0` | Set to `1` to enable PRM scoring |
| `PORT` | `30000` | Port for the OpenAI-compatible proxy endpoint |
| `SGLANG_API_KEY` | — | Optional API key for the SGLang endpoint |

**VRAM breakdown for Qwen 3.5 4B Vision (default settings, 2× RTX 3090):**

| Component | VRAM |
|---|---|
| Base model weights (4-bit NF4 QLoRA) | ~2.5 GB |
| LoRA adapter (r=16) | ~0.2 GB |
| Gradients + optimizer states | ~3 GB |
| Activation checkpointing | ~2 GB |
| **Total (training GPU 0)** | **~8 GB** |
| sglang rollout server (BF16) | ~8 GB (GPU 1) |

You can check more details about configurations in [`./instructions`](./instructions).


## 📖 Citation

```
@misc{wang2026openclawrl,
  author       = {Wang, Yinjie and Wang, Mengdi and Yang, Ling},
  title        = {OpenClaw-RL},
  year         = {2026},
  organization = {GitHub},
  url          = {https://github.com/Gen-Verse/OpenClaw-RL},
}

@article{yu2025demystify,
  title={Demystifying Reinforcement Learning in Agentic Reasoning},
  author={Yu, Zhaochen and Yang, Ling and Zou, Jiaru and Yan, Shuicheng and Wang, Mengdi},
  journal={arXiv preprint arXiv:2510.11701},
  year={2025}
}

@article{wang2026rlanything,
  title={RLAnything: Forge Environment, Policy, and Reward Model in Completely Dynamic RL System},
  author={Wang, Yinjie and Xie, Tianbao and Shen, Ke and Wang, Mengdi and Yang, Ling},
  journal={arXiv preprint arXiv:2602.02488},
  year={2026}
}
```

## 🙏 Acknowledgements

This work aims to explore more effective paradigms for Agentic RL. Our implementation builds upon the excellent codebases of:

- **[Unsloth](https://github.com/unslothai/unsloth)** — by Daniel Han & Michael Han. Unsloth makes QLoRA fine-tuning 2× faster with 70% less VRAM; it is the engine that makes 2× RTX 3090 training possible. We are deeply grateful to the Unsloth team.
- **[slime](https://github.com/THUDM/slime)** — by THUDM. The scalable async RL framework underpinning the full multi-GPU path.
- **[OpenClaw](https://github.com/openclaw/openclaw)** — The AI-native browser extension that provides the conversational interface and feedback loop.
- **[Open-AgentRL](https://github.com/Gen-Verse/Open-AgentRL)** — Pioneering open-source agentic RL research that inspired this project.

We sincerely thank these projects for their valuable insights and high-quality implementations, which have greatly facilitated our research.



---



