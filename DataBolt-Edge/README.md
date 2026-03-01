# DataBolt Edge – On-Device Data Engineering Assistant (NVIDIA Track)

DataBolt Edge is an AI‑powered assistant that runs a quantized Mistral model locally on GPU/edge devices to help data engineers debug Spark, Airflow, and SQL issues. It ingests logs and plans, identifies errors or bottlenecks, and suggests fixes—all offline.

Below is a comprehensive specification and development plan for DataBolt Edge.

---

## 📝 Executive Summary

DataBolt Edge enables local inference of a quantized Mistral model (7B or smaller) on an NVIDIA GPU, providing real‑time debugging for Spark logs, Airflow DAG failures, and SQL query plans.

### Core components

1. **Conversion pipeline** – Mistral → ONNX → TensorRT LLM engine
2. **Inference server** – TensorRT/Triton or Python API running on a laptop/edge GPU
3. **Ingestion modules** – parse Spark logs, Airflow logs (`AIRFLOW_HOME/logs`), and SQL EXPLAIN plans
4. **Frontend UI** – Streamlit or Tauri for user interaction

Users upload logs or connect to local metadata, ask questions (e.g. “Why failed?”), and the model responds using context from the parsed data.

> **Benefits:** privacy (no cloud inference), low latency, offline operation.

The MVP targets core parsing and inference; optional enhancements include an offline web UI and fine‑tuning feedback.

---

## 🔗 Key Integrations & Tools

| Integration / Library                 | Purpose                                                                 | Docs / Links                                      |
|---------------------------------------|-------------------------------------------------------------------------|---------------------------------------------------|
| Mistral Models                        | Base LLM (e.g. Mistral‑7B). Weights from Hugging Face or official repo.| [Mistral Model Hub](#)                            |
| Hugging Face Transformers / Optimum   | Load & export Mistral to ONNX                                           | [Optimum ONNX](#)                                 |
| ONNX                                  | Intermediate format for export; used by TensorRT                        | [ONNX Format](#), [NVIDIA ONNX Guide](#)         |
| NVIDIA TensorRT‑LLM                   | Optimize & quantize ONNX for inference                                  | [TensorRT‑LLM GitHub](#), [Mistral Docs](#)      |
| NVIDIA Triton Inference Server        | Serve TensorRT engine with TensorRTLLM backend                          | [Triton Docs](#)                                  |
| PyTorch                               | Optional un‑quantized model runtime                                     | [PyTorch Export](#)                               |
| Airflow Python API / CLI              | Fetch DAG status & logs                                                  | [Airflow Logs](#)                                 |
| Spark API / Logs                      | Access Spark UI or driver/executor logs                                 | [Spark UI Debugging](#)                          |
| PostgreSQL / DuckDB                   | Metadata cache for parsed entries                                       | [PostgreSQL Docs](#), [DuckDB](#)                 |
| Docker / K3s                          | Containerize backend and Triton                                         | [Docker Docs](#), [K3s](#)                       |
| Streamlit                             | Rapid browser‑based UI                                                   | [Streamlit Docs](#)                              |
| Tauri                                 | Desktop app framework (Rust + webview)                                  | [Tauri Docs](#)                                  |
| Log parsing libraries (regex)         | Custom error extraction                                                  | Python logging                                   |

*Primary sources emphasize on‑device inference (Mistral/TensorRT) and local log paths.*

---

## 🏗️ Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Ingestion
      SparkLogs[Spark Logs (driver/executor)]
      AirflowLogs[Airflow Task Logs]
      SQLPlans[SQL EXPLAIN Outputs]
      Users(Users/Data Engineers)
    end
    subgraph Backend[DataBolt Backend]
      Parser(Spark/Airflow/SQL Parsers)
      ContextDB[(Metadata Store)]
      InferenceSrv(TensorRT Inference Server)
      ResultsDB[(Results Cache)]
    end
    subgraph Frontend[UI Layer]
      UIApp(UI (Streamlit/Tauri))
      API(FastAPI endpoints)
    end

    Users --> UIApp
    UIApp --> API
    API --> Parser
    Parser --> InferenceSrv
    Parser --> ContextDB
    ContextDB --> InferenceSrv
    InferenceSrv --> ResultsDB
    ResultsDB --> API
    API --> UIApp
    SQLPlans --> Parser
    SparkLogs --> Parser
    AirflowLogs --> Parser
```

**Components**

- **Parser Modules** – Extract errors and structures from Spark, Airflow, and SQL plans.
- **Context Store** – Cache logs, metrics, and DAG/job states for context.
- **Inference Server** – Runs the quantized Mistral model via TensorRT/Triton.
- **API Layer** – FastAPI endpoints for upload, query, and explanation retrieval.
- **Frontend** – Streamlit web UI or Tauri desktop app for interactive chat.

**User query sequence**
1. Upload logs/plan & ask question
2. Parser reads files, stores structured data
3. Context + query fed to LLM engine
4. Model generates answer → returned to UI

---

## 🔄 Model Conversion & Quantization Pipeline

1. **Export to ONNX** (Optimum or PyTorch)
   ```python
   model = MistralLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1").to('cuda')
   dummy_input = torch.ones((1,512), dtype=torch.int64).cuda()
   torch.onnx.export(model, dummy_input, "mistral7b.onnx", opset_version=17,
                     input_names=["input_ids"])
   ```
2. **Quantize with TensorRT‑LLM**
   ```python
   from trtllm_python import EngineBuilder
   builder = EngineBuilder(model_path="mistral7b.onnx", dtype="INT8", optimize=True)
   builder.build_engine("mistral7b_int8.engine")
   ```
   Supports INT8, FP8, int4_awq, etc. Calibration via workload samples.

3. **Deploy** – load engine in Triton (TensorRTLLM backend) or use Python runtime.
4. **Benchmark** – compare latency FP16 vs INT8 vs FP8 on RTX 3080/4090/Jetson.

> INT8 often ~2× faster with minimal quality loss. NVIDIA’s docs and Transformer Engine tutorials are reference material.

---

## ⚙️ Hardware & Performance Trade‑offs

**Target hardware**
- RTX 3080/4090 or Jetson Orin
- CUDA 11+, 16+ GB GPU RAM (for 7B model)
- Lower GPUs may use Minist​ral‑3B or CPU/FP16

**Trade‑offs**
- **Size vs speed** – 7B better quality, 13 GB; INT8 engine ~4 GB.
- **Precision** – FP16 good; INT8/FP8 double throughput but need calibration.
- **Batching** – single query for CLI; small batches to boost utilization.
- **Inference mode** – sync response adequate; streaming token output is optional.

*Suggestion: plot latency vs parameter count for FP16/INT8.*

---

## 📥 Data Ingestion & Parsing

**Spark logs** – driver/executor stdout; regex for exceptions and error markers.

**Airflow logs** – files in `AIRFLOW_HOME/logs/…`; scan for `ERROR` or tracebacks; use Python API for DAG/task states.

**SQL EXPLAIN plans** – parse JSON (MySQL, Spark) or indent‑based text; look for full scans, joins, cost estimates.

Parsed output stored in a local DB (`spark_errors`, `airflow_failures`, `sql_plans`) and injected into prompts.

```python
# Spark parser example

def parse_spark_logs(file_path):
    errors = []
    for line in open(file_path):
        if "Exception" in line or "ERROR" in line:
            errors.append(line.strip())
    return errors
```

```python
# JSON plan traversal
plan = spark.sql("EXPLAIN FORMAT=JSON SELECT …").collect()[0][0]
plan_json = json.loads(plan)
# recurse to identify high‑cost nodes
```

---

## 📦 Local Deployment & Frontend

Package as Docker Compose or K3s stack:
- Container 1: FastAPI + TensorRT libs
- Container 2: Triton server
- NVIDIA runtime for GPU access

**UI options**
- **Streamlit** – quick browser UI, file upload, chat widget
- **Tauri** – desktop app with bundled binary

**UX flow**
1. Open UI
2. Upload log/plan or paste SQL
3. View parsed summary
4. Ask a question → model analysis
5. See highlighted log excerpts and suggestions

**Sample Streamlit**
```python
import streamlit as st
st.title("DataBolt Edge")
…
```

---

## 🔐 Security & Privacy

- Data stays local; no cloud inference
- Sanitize inputs (scrub PII, paths)
- Sandbox any code execution in containers
- Encrypt model weights on shared machines
- Keep third‑party libs up‑to‑date

---

## 🧪 CI/CD & Testing

- **CI pipeline** – GitHub Actions build Docker images, run tests
- **Unit tests** – parsers, model loading, helper utils
- **Integration tests** – spin up compose, POST sample query
- **Performance tests** – verify <100 ms latency
- Release images via Docker Hub (or GHCR)

### NVIDIA API Management Testing

The `DataBolt-Edge/nvidia_api_management` package includes dedicated scripts for fast validation.

If you are in `DataBolt-Edge/`, you can use Make shortcuts:

```bash
make test-nvidia
make lint-nvidia
make check-nvidia
make probe-nvidia-app
```

1. Run package unit tests + lint:

    ```bash
    bash DataBolt-Edge/scripts/run_nvidia_api_management_tests.sh
    ```

    Or run unit tests directly:

    ```bash
    uv run python -m unittest discover -s DataBolt-Edge/tests -p "test_nvidia_*.py"
    ```

2. Run a live NVIDIA probe (requires `NVIDIA_API_KEY`):

    ```bash
    export NVIDIA_API_KEY="<your-token>"
    uv run python DataBolt-Edge/scripts/live_nvidia_probe.py
    ```

3. Run the app-level probe script:

    ```bash
    uv run python DataBolt-Edge/testing.py
    ```

See package details in `DataBolt-Edge/nvidia_api_management/README.md`.

---

## 🏁 MVP Feature List

| Feature                                    | Category      | Priority |
|-------------------------------------------|---------------|----------|
| Quantized Mistral inference (TensorRT)    | Core LLM      | MUST     |
| Spark log error parser                    | Core Function | MUST     |
| Airflow log parser                        | Core Function | MUST     |
| SQL EXPLAIN plan analyzer                 | Core Function | MUST     |
| Frontend UI for upload & Q&A              | Interface     | MUST     |
| Dockerized local deployment (no cloud)    | DevOps        | MUST     |
| CI tests for parsers and inference        | DevOps        | MUST     |
| Performance logging (latency/cost)        | Metrics       | Should   |
| Multiple precision modes (INT8/FP16)      | Performance   | Should   |
| Streaming mode (token output)             | UX            | Should   |
| Tauri desktop app                         | UX Packaging  | Could    |
| Local file ingestion (Parquet/CSV)        | Data Ingest   | Could    |
| Continuous fine-tuning (log mistakes)     | Enhancement   | Could    |

> **MVP focus:** quantized inference + log parsers + basic web UI.

---

## 🕒 48‑Hour Hackathon Timeline

| Time (48h)     | Task                                                                 |
|----------------|----------------------------------------------------------------------|
| Day 1 9–11 am  | Kickoff, finalize scope, setup repo & Dockerfiles, install TensorRT |
| Day 1 11–1 pm  | Export model to ONNX, begin quantization                            |
| Day 1 1–3 pm   | Build TensorRT engine, run simple inference                         |
| Day 1 3–5 pm   | Develop Spark/Airflow parsers & SQL plan reader                     |
| Day 1 5–7 pm   | Backend: integrate parsers with API                                 |
| Day 1 7–9 pm   | Frontend MVP: Streamlit page                                        |
| Day 1 9–Midnight | Evening sync & end‑to‑end test                                     |
| Day 2 9–11 am  | Optimize inference (INT8 latency)                                   |
| Day 2 11–1 pm  | Add DuckDB context store                                             |
| Day 2 1–3 pm   | Polish frontend, interactivity                                      |
| Day 2 3–5 pm   | Write tests, setup CI                                               |
| Day 2 5–6 pm   | Demo prep – sample logs/queries                                      |
| Day 2 6–7 pm   | Finalize slides & practice demo                                     |

---

## 🖥️ Hardware Checklist

| Hardware                | Role                                   |
|-------------------------|----------------------------------------|
| NVIDIA GPU (RTX 3080+)  | Model inference (TensorRT‑LLM)         |
| CPU (Quad‑core+)        | Host backend & parsing                  |
| RAM (32+ GB)            | Large log file processing              |
| Disk (1 TB SSD)         | Store logs, models, Docker images      |
| Network (optional)      | Local‑only, no cloud needed            |
| (Optional) Jetson       | Edge deployment testing                |

*GPU must support CUDA compute capability ≥ 8.0 for FP8.*

---

## 🏆 Prize Alignment

| Challenge            | Relevant Aspect                                                  |
|----------------------|------------------------------------------------------------------|
| NVIDIA On‑Device     | On‑device LLM inference with TensorRT                            |
| Mistral (Global)     | Uses Mistral API/models                                          |
| Data Usage (Jump)    | Parsing Spark/Airflow logs for insights                          |
| NYC (Tilde)          | Innovative edge computing architecture                           |

> DataBolt Edge satisfies multiple hackathon tracks by combining on‑device inference, rich data integration, and edge‑first design.

---

## 💡 Code Snippets (Pseudocode)

```python
# Convert Mistral to ONNX
from transformers import MistralForCausalLM
model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B")
model.eval().cuda()
dummy_input = {"input_ids": torch.ones((1,512), dtype=torch.int64).cuda()}
torch.onnx.export(model, (dummy_input,), "mistral.onnx", opset_version=17,
                  input_names=["input_ids"], output_names=["output"])
```

```python
# Build TensorRT‑LLM Engine
from trt_llm import create_engine
engine = create_engine(model="mistral7b", builder_config={"int8": True})
engine.save("mistral7b_int8.trt")
```

```python
# Parse Airflow logs
def parse_airflow_logs(base_folder):
    errors = []
    for dirpath, dirs, files in os.walk(base_folder):
        for f in files:
            with open(os.path.join(dirpath, f)) as log:
                for line in log:
                    if "ERROR" in line or "Traceback" in line:
                        errors.append(line.strip())
    return errors
```

```python
# SQL Plan Analysis (Spark)
plan_json = spark.sql("EXPLAIN FORMAT=JSON SELECT * FROM table").collect()[0][0]
import json
plan = json.loads(plan_json)
# Extract cost estimates
cost = plan['queryPlan']['cost']
```

```python
# Minimal Streamlit UI example
import streamlit as st
st.title("DataBolt Edge")
logs = st.file_uploader("Upload Spark/Airflow logs")
query = st.text_input("Ask me anything about these logs")
if st.button("Analyze"):
    parsed = backend.parse_logs(logs)
    answer = backend.query_llm(query, context=parsed)
    st.write(answer)
```

---

## 🚀 Next Steps & Roadmap

Post‑hackathon:

1. Expand parsers (SparkListener, Airflow REST API for live monitoring).
2. Support smaller models (Mistral‑3B) for low‑end GPUs.
3. Polish UI/UX; consider migrating to Tauri.
4. Automate quantized model releases via CI.
5. Collect on‑field feedback and iterate.

*DataBolt Edge can evolve into a full edge analytics platform, deploying LLMs alongside data clusters for on‑premises debugging and optimization.*
