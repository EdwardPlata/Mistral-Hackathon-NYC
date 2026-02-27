# Mistral Hackathon NYC

GPU-accelerated agentic AI prototyping environment for the Mistral Hackathon NYC, integrating NVIDIA Agent Toolkit, Weights & Biases, ElevenLabs, and HuggingFace/Mistral.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/EdwardPlata/Mistral-Hackathon-NYC?quickstart=1)

---

## Codespaces Quick Setup (2–5 mins)

1. Click **Open in GitHub Codespaces** above (or go to **Code → Codespaces → Create**).
2. Select a **GPU machine type** (A100 recommended; hackathon quota may apply).
3. The devcontainer automatically installs CUDA 12, Python 3.11, W&B, and all dependencies.
4. Set your secrets in **Codespace Settings → Secrets**:

   | Secret | Description |
   |--------|-------------|
   | `NVIDIA_API_KEY` | NVIDIA API key for NeMo Agent Toolkit |
   | `WANDB_API_KEY` | Weights & Biases API key |
   | `ELEVENLABS_API_KEY` | ElevenLabs TTS API key |
   | `HF_TOKEN` | HuggingFace access token |

---

## Hackathon Tool Integration

| Tool | Setup Command | Benefit |
|------|--------------|---------|
| **NVIDIA Agent Toolkit** | `git clone https://github.com/NVIDIA/NeMo-Agent-Toolkit.git && cd NeMo-Agent-Toolkit && uv sync --all-groups` | GPU agents in minutes; swap Mistral models via YAML |
| **Weights & Biases** | `wandb login` (key auto-loaded from secret) | Auto-track experiments; share dashboards |
| **ElevenLabs Voice** | `elevenlabs text-to-speech` (key auto-loaded from secret) | Streaming TTS; <1s latency via Codespace GPU |
| **HuggingFace / Mistral** | `huggingface-cli login` then `autotrain llm --train --model mistralai/Mistral-7B-v0.1` | Fine-tune & push to Hub in 10–30 minutes |

---

## Development Workflow

1. **Prototype** – scaffold a workflow, add ElevenLabs tool, log metrics with W&B:
   ```bash
   nat scaffold workflow hackathon-agent
   wandb init --name mistral-agent
   ```

2. **Train & Upload** – fine-tune Mistral on a HuggingFace dataset and push to your org:
   ```python
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
   # ... fine-tune ...
   model.push_to_hub("your-org/mistral-hackathon")
   ```

3. **Voice & Test** – integrate TTS and run the UI:
   ```bash
   elevenlabs text-to-speech --text "Hello from Mistral Hackathon NYC"
   npm run dev   # NeMo Agent Toolkit UI (if applicable)
   ```

4. **Collaborate & Submit** – commit and share via repo link; submit on [Hackiterate](https://hackiterate.com).

---

## Local Setup (alternative)

```bash
pip install uv
uv sync
uv pip install -e '.[langchain]'
cp .env.example .env   # fill in your API keys
```

---

## Project Structure

```
.
├── .devcontainer/
│   └── devcontainer.json   # Codespaces GPU config
├── pyproject.toml           # Project dependencies
└── README.md
```