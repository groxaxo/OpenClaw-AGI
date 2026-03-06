# Binary Reward Summarized from Next State

Online RL for agentic tool-use, using binary process reward signals from environment feedback.

## Method Overview

The policy model is deployed as an OpenAI-compatible chat proxy. External environments (e.g. OpenClaw) send multi-turn conversations through this proxy. For each **main-line turn**, the system:

1. Forwards the request to the policy model (served by SGLang) and collects the response along with per-token log-probabilities.
2. When the **next turn** arrives, its user/environment message serves as the "next state" for the previous turn.
3. A **Process Reward Model (PRM)** judges the previous response quality given the next state (could be user or env feedback). It produces `m` independent evaluations via majority vote, scoring each turn as `+1` (good), `-1` (bad), or `0` (neutral).
4. The majority-voted score becomes the scalar reward for that turn.
5. Turns that never receive a next state (i.e. the last turn in a session) are excluded from training (`loss_mask = 0`), unless they are the only turn in the session (at-least-one guarantee).

### Advantage Estimation (GRPO)

Advantages are computed using **Group Relative Policy Optimization (GRPO)**. For each sample with scalar reward `r`, the advantage is broadcast uniformly to all response tokens:

$$A_t = r, \quad \forall t \in \text{response tokens}$$

No reward normalization is applied (`--disable-rewards-normalization`).

### Policy Gradient Loss

Standard PPO-style clipped surrogate objective with asymmetric clipping:

$$\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$$

$$\mathcal{L}_{\text{pg}} = -\mathbb{E}_t\Big[\min\!\big(\rho_t A_t,\ \text{clip}(\rho_t,\, 1-\varepsilon,\, 1+\varepsilon_{\text{high}}) \cdot A_t\big)\Big]$$

where $\varepsilon = 0.2$, $\varepsilon_{\text{high}} = 0.28$.

### Total Loss

$$\mathcal{L} = \mathcal{L}_{\text{pg}} + \beta_{\text{KL}} \cdot \mathcal{L}_{\text{KL}}$$

where $\beta_{\text{KL}} = 0.02$. Entropy bonus is disabled ($\beta_{\text{ent}} = 0$).



## How to Run

### Full setup (8+ GPUs, Megatron-LM)

```bash
cd slime
bash ../openclaw-rl/run_qwen3_4b_openclaw_rl.sh
```

### Lightweight setup — Unsloth QLoRA (2x or 3x RTX 3090)

Uses [Unsloth](https://github.com/unslothai/unsloth) with 4-bit NF4 quantisation
and LoRA adapters to dramatically reduce VRAM requirements.  No Megatron-LM or
Ray cluster required.

#### Prerequisites

```bash
# 1. Install Unsloth for your CUDA version (see https://github.com/unslothai/unsloth):
pip install "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"

# 2. Install QLoRA dependencies:
pip install -r requirements-qlora.txt    # from repo root
```

#### 2x RTX 3090 (48 GB total)

```
GPU 0  →  QLoRA actor training  (Unsloth + PEFT LoRA, ~8 GB)
GPU 1  →  sglang rollout server (base model BF16,   ~8 GB)
```

```bash
export HF_CKPT=/path/to/Qwen3-4B
export SAVE_CKPT=/path/to/openclaw-qlora-rl/ckpt
bash openclaw-rl/run_qwen3_4b_3090_2x.sh
```

#### 3x RTX 3090 (72 GB total)

```
GPUs 0,1  →  QLoRA actor training  (DDP via accelerate, ~8 GB each)
GPU 2     →  sglang rollout server  (~8 GB)
```

```bash
export HF_CKPT=/path/to/Qwen3-4B
export SAVE_CKPT=/path/to/openclaw-qlora-rl/ckpt
bash openclaw-rl/run_qwen3_4b_3090_3x.sh
```

Both scripts expose the same OpenAI-compatible proxy on port `30000` (override
with `PORT=…`).  Point your OpenClaw agent to `http://<host>:30000`.

Key environment variables (both scripts):

| Variable | Default | Description |
|---|---|---|
| `HF_CKPT` | *(required)* | Path to the base model |
| `SAVE_CKPT` | *(required)* | Checkpoint output directory |
| `TRAINING_GPU` | `0` | GPU index for training (2x script) |
| `ROLLOUT_GPU` | `1` / `2` | GPU index for sglang |
| `TRAINING_GPUS` | `0,1` | GPU indices for DDP training (3x script) |
| `LORA_R` | `64` | LoRA rank |
| `LORA_ALPHA` | `128` | LoRA alpha |
| `ROLLOUT_BATCH_SIZE` | `16` / `32` | Samples per training step |
| `LR` | `1e-6` | Learning rate |
| `PRM_ENABLE` | `0` | Set to `1` to enable PRM scoring |

VRAM breakdown for Qwen3-4B with default settings:

| Component | VRAM |
|---|---|
| Base model weights (4-bit NF4) | ~2.5 GB |
| LoRA adapter (r=64) | ~0.5 GB |
| Gradient + optimizer states | ~3 GB |
| Activation checkpointing | ~2 GB |
| **Total (training GPU)** | **~8 GB** |
| sglang (BF16 base model) | ~8 GB |

## File Structure

```
openclaw-rl/
├── README.md
├── run_qwen3_4b_openclaw_rl.sh     # Full launch script (8+ GPUs, Megatron-LM)
├── run_qwen3_4b_3090_2x.sh         # QLoRA launch script for 2x RTX 3090
├── run_qwen3_4b_3090_3x.sh         # QLoRA launch script for 3x RTX 3090
├── unsloth_qlora_trainer.py         # Unsloth QLoRA GRPO trainer
├── openclaw_api_server.py           # FastAPI proxy + PRM scoring + sample submission
├── openclaw_rollout.py              # Async rollout worker (bridges API server ↔ SLIME trainer)
└── results/                         # Runtime records (auto-created)
```
