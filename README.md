# Mistral Hackathon NYC

Agentic AI prototyping environment for the Mistral Hackathon NYC, integrating NVIDIA Agent Toolkit, Weights & Biases, ElevenLabs, and HuggingFace/Mistral.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/EdwardPlata/Mistral-Hackathon-NYC?quickstart=1)

---

## Codespaces Quick Setup (2–5 mins)

1. Click **Open in GitHub Codespaces** above (or go to **Code → Codespaces → Create**).
2. Use the default **CPU-first** container for normal coding tasks (faster startup, works on standard hosts).
3. The default devcontainer installs Python 3.11 tooling only; project dependencies are installed manually when you run the setup script (so container build/deploy stays fast and reliable).
4. Install the base project dependencies when you are ready:

   ```bash
   bash .devcontainer/setup.sh

   # Optional: install heavy ML/GPU dependencies only when needed
   .devcontainer/install-ml-extras.sh
   # (equivalent to: uv sync --extra dev --extra ml-gpu --extra langchain)
   ```
5. Set your secrets in your GitHub settings (**Repository Settings → Codespaces → Repository secrets** or **User Settings → Codespaces → Secrets**) so they are available in your Codespace:

   | Secret | Description |
   |--------|-------------|
   | `NVIDIA_API_KEY` | NVIDIA API key for NeMo Agent Toolkit |
   | `WANDB_API_KEY` | Weights & Biases API key |
   | `ELEVENLABS_API_KEY` | ElevenLabs TTS API key |
   | `HF_TOKEN` | HuggingFace access token |

### CPU vs GPU Dev Container Selection

- **CPU container (default):** use `.devcontainer/devcontainer.json` for editing, prompt work, docs, tests, and most Python development.
- **GPU container (opt-in):** use `.devcontainer/devcontainer.gpu.json` only for local CUDA workloads that truly need a GPU-enabled host.

Use one of these explicit launch paths:

```bash
# Codespaces (default CPU container)
https://codespaces.new/EdwardPlata/Mistral-Hackathon-NYC?quickstart=1

# Codespaces (opt-in GPU container)
https://codespaces.new/EdwardPlata/Mistral-Hackathon-NYC?quickstart=1&devcontainer_path=.devcontainer/devcontainer.gpu.json
```

In VS Code Dev Containers (local Docker/Desktop):

1. Run **Dev Containers: Reopen in Container** for the default CPU setup.
2. Run **Dev Containers: Open Folder in Container...** and choose `.devcontainer/devcontainer.gpu.json` when you specifically need CUDA.

---

## Hackathon Tool Integration

| Tool | Setup Command | Benefit |
|------|--------------|---------|
| **NVIDIA Agent Toolkit** | `git clone https://github.com/NVIDIA/NeMo-Agent-Toolkit.git && cd NeMo-Agent-Toolkit && uv sync --all-groups` | GPU agents in minutes; swap Mistral models via YAML |
| **Weights & Biases** | `wandb login` (then paste API key from Codespaces Secret) | Auto-track experiments; share dashboards |
| **ElevenLabs Voice** | `elevenlabs text-to-speech` (key auto-loaded from secret) | Streaming TTS; <1s latency via Codespace GPU |
| **HuggingFace / Mistral** | `uv add "huggingface_hub[cli]>=0.23.0" && huggingface-cli login` then `uv add autotrain-advanced && uv run autotrain llm --train --model mistralai/Mistral-7B-v0.1` | Fine-tune & push to Hub in 10–30 minutes (requires extra dependency `autotrain-advanced`) |

---

## Development Workflow

1. **Prototype** – scaffold a workflow, add ElevenLabs tool, log metrics with W&B:
   ```bash
   # Prerequisite: install NVIDIA Agent Toolkit (see table above), then:
   cd NeMo-Agent-Toolkit && uv run nat scaffold workflow hackathon-agent
   wandb init --name mistral-agent
   ```

2. **Train & Upload** – fine-tune Mistral on a HuggingFace dataset and push to your org:
   ```python
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
   # ... fine-tune ...
   model.push_to_hub("your-org/mistral-hackathon")
   ```

3. **Voice & Test** – integrate TTS and (optionally) run the NeMo UI:
   ```bash
   elevenlabs text-to-speech --text "Hello from Mistral Hackathon NYC"
   # Optional: to run the NeMo Agent Toolkit UI, clone its separate UI repo
   # (e.g., https://github.com/NVIDIA/NeMo-Agent-Toolkit-UI), ensure Node.js/npm is installed,
   # then from that cloned UI directory:
   cd NeMo-Agent-Toolkit-UI
   npm install
   npm run dev   # Start NeMo Agent Toolkit UI
   ```

4. **Collaborate & Submit** – commit and share via repo link; submit on [Hackiterate](https://hackiterate.com).

---

## Local Setup (alternative to Codespaces)

[Install UV](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone, install dependencies, and configure secrets:

```bash
git clone https://github.com/EdwardPlata/Mistral-Hackathon-NYC.git
cd Mistral-Hackathon-NYC

# Fast default install for everyday development (creates .venv automatically)
uv sync --extra dev

# Optional: install heavyweight ML/GPU + LangChain dependencies when needed
.devcontainer/install-ml-extras.sh
# (or run: uv sync --extra dev --extra ml-gpu --extra langchain)

# Copy and fill in your API keys
cp .env.example .env
```

---

## Running Python Scripts

All scripts should be run through UV so the correct virtual environment is always used:

```bash
# Run any Python script
uv run python path/to/script.py

# Run with extra arguments
uv run python path/to/script.py --arg value

# Open an interactive Python REPL
uv run python

# Run a module directly (e.g., pytest)
uv run pytest

# Add a new dependency and update the lockfile
uv add some-package

# Remove a dependency
uv remove some-package
```

> **Tip:** Never call `python` or `pip` directly – always prefix with `uv run` / `uv add` to keep the lockfile and virtual environment in sync.

---

## Project Structure

```
.
├── .devcontainer/
│   ├── devcontainer.json      # Default CPU-first devcontainer config
│   ├── devcontainer.gpu.json  # Opt-in CUDA/GPU devcontainer config
│   ├── setup.sh               # Fast default bootstrap install
│   └── install-ml-extras.sh   # Optional heavyweight ML/GPU extras
├── pyproject.toml           # Project dependencies (UV-managed)
├── uv.lock                  # Pinned lockfile (auto-generated by uv sync)
├── .env.example             # API key template
├── .gitignore
└── README.md
```
