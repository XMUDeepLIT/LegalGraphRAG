# **LegalGraphRAG: Multi-Agent Graph Retrieval-Augmented Generation for Reliable Legal Reasoning**

> An evaluation framework for legal judgment prediction that integrates multi-agent graph retrieval and supports reproducible comparisons across multiple models and baselines.

<p align="center">
  <a href="https://www.researchgate.net/publication/403734810_LegalGraphRAG_Multi-Agent_Graph_Retrieval-Augmented_Generation_for_Reliable_Legal_Reasoning" target="_blank">
    <img src="https://img.shields.io/badge/Paper-ResearchGate-blue?style=flat-square" alt="Paper">
  </a>
  <a href="https://github.com/DEEP-PolyU/LegalGraphRAG" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-Project-181717?logo=github&style=flat-square" alt="GitHub">
  </a>
</p>

---

## 🚀 **Highlights**
- ✅ **Automated Evaluation**: Computes `Accuracy (Acc)` and `Micro-F1` automatically for legal judgment prediction tasks.
- ✅ **Multi-Model Support**: Supports Qwen, DeepSeek, GPT, InternLM, GLM, Gemma, and more.
- ✅ **Dataset Coverage**: Includes legal datasets such as CAIL and CMDL.
- ✅ **Baseline Comparison**: Enables direct comparison with `HippoRAG2`, `RAPTOR`, `LightRAG`, `LegalΔ`, and `ADAPT`.

<p align="center">
  <img src="images/method.png" width="95%" alt="Framework Overview">
</p>

---

## 🧩 **Project Structure**

```text
LegalGraphRAG/
├── core/                      # Core modules
│   ├── LegalGraphRAG.py       # Main LegalGraphRAG class
│   ├── models/                # Model implementations
│   │   ├── transformers/      # Transformers-based models (Qwen, InternLM, GLM, Gemma)
│   │   └── openai/            # OpenAI-compatible models (DeepSeek, GPT)
│   ├── graph_construct/       # Graph construction and management
│   ├── judge/                 # Legal judgment modules
│   ├── preprocess/            # Data preprocessing
│   ├── prompt/                # Prompt templates
│   └── utils/                 # Utility functions
├── run.py                     # Main evaluation script
├── env.example                # Configuration file template
└── README.md                  # Project documentation
```

---

## 🛠️ **Usage**

### 1️⃣ Environment Setup

```bash
# Copy and configure environment file
cp env.example .env
# Edit .env with model paths, API keys, and runtime settings
```

### 2️⃣ Prepare Data

- Place datasets under `datasets/` (or pass a custom dataset directory).
- Follow filename format: `crime_data_{dataset}_small.json`.
- Supported datasets:
  - `CAIL` (Chinese AI and Law Challenge)
  - `CMDL` (Chinese Multi-Domain Legal Dataset)

### 3️⃣ Run Evaluation

```bash
python run.py --model qwen3 --datasets CAIL --devices cuda:2 cuda:3
```

**Main arguments**

- `--model`: `qwen3`, `qwen2_5`, `gemma3`, `internlm3`, `glm4`, `deepseek_v3`, `gpt4o_mini`
- `--datasets`: dataset name, e.g. `CAIL`, `CMDL`
- `--dotenv_path`: path to `.env` (default: `.env`)
- `--datasets_path`: path to datasets (default: `../datasets`)
- `--devices`: GPU devices, e.g. `cuda:0 cuda:1`
- `--no-build-graph`: skip graph construction when graph already exists
- `--force-rebuild`: force graph rebuild even if artifacts already exist

### 4️⃣ Output Files

- Prediction outputs:
  - `{output_dir}/{dataset}/{model}_results_combined.json`
- Statistics:
  - `{output_dir}/{dataset}/{model}_stats.json`

Example output summary:

```json
{
  "model_name": "qwen3",
  "dataset": "CAIL",
  "total_cases": 1000,
  "correct_count": 850,
  "elapsed_time": 3600.0,
  "output_file": "./outputs/CAIL/qwen3_results_combined.json"
}
```

---

## ⚙️ **Configuration**

Configuration is managed via `.env`. Key groups include:

- **Model Configuration**: model names, devices, API keys, generation parameters
- **Data Configuration**: dataset paths and output directory
- **Graph Configuration**: graph construction and retrieval settings

See `env.example` for the full configuration list.

---

## 🎯 **Supported Models and Baselines**

**Models**

- Qwen3-8B
- Qwen2.5-7B-Instruct
- DeepSeek-V3
- GPT-4o-mini
- InternLM3
- GLM-4

**Baselines**

- HippoRAG2
- RAPTOR
- LightRAG
- LegalΔ
- ADAPT

---

## ⚡ **Multi-GPU Execution**

Run on multiple GPUs by passing several devices:

```bash
python run.py --model qwen3 --datasets CAIL --devices cuda:0 cuda:1 cuda:2 cuda:3
```

Cases are automatically distributed across the selected devices.
