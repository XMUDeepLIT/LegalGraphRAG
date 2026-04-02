# LegalGraphRAG: Multi-Agent Graph Retrieval-Augmented Generation for Reliable Legal Reasoning

Code repository for paper: LegalGraphRAG: Multi-Agent Graph Retrieval-Augmented Generation for Reliable Legal Reasoning

This repository provides an evaluation framework for **Legal Judgement Prediction tasks** using the **LegalGraphRAG** method, with comparisons to several baseline approaches.

![framework](images/method.png)

## Features

- **Automated Evaluation**
  Computes **Accuracy (Acc)** and **Micro-F1** automatically.

- **Multi-model Support**
  Works with **Qwen**, **DeepSeek**, **GPT**, **Internlm**, **GLM**, **Gemma**, and other open-source LLMs.

- **Dataset Coverage**
  Includes benchmarks such as **CAIL** and **CMDL**.

- **Multiple Baselines**
  Supports comparisons with `HippoRAG2`, `RAPTOR`, `LightRAG`, `Legal$\Delta$`, `ADAPT`, etc.

## Project Structure

```
LegalGraphRAG/
├── core/                      # Core modules
│   ├── LegalGraphRAG.py       # Main LegalGraphRAG class
│   ├── models/                # Model implementations
│   │   ├── transformers/      # Transformers-based models (Qwen, Internlm, GLM, Gemma)
│   │   └── openai/           # OpenAI-compatible models (DeepSeek, GPT)
│   ├── graph_construct/       # Graph construction and management
│   ├── judge/                # Legal judgment modules
│   ├── preprocess/           # Data preprocessing
│   ├── prompt/               # Prompt templates
│   └── utils/                # Utility functions
├── run.py                     # Main evaluation script
├── env.example               # Configuration file template
└── README.md                 # This file
```

## Key Components

- **LegalGraphRAG**
  The main class for legal case analysis. It:

  1. Builds and manages a knowledge graph from legal cases
  2. Processes legal cases with multi-agent reasoning
  3. Performs legal judgment prediction (crime classification, law article retrieval, etc.)
  4. Computes evaluation metrics (Accuracy, Micro-F1)
  5. Saves detailed results to JSON

- **Metrics**

  - `Accuracy (Acc)`: Overall classification accuracy
  - `Micro-F1`: Micro-averaged F1 score across all classes

## Supported Datasets

- **CAIL** (Chinese AI and Law Challenge)
- **CMDL** (Chinese Multi-Domain Legal Dataset)

Dataset files should be placed under the datasets directory with the naming format: `crime_data_{dataset}_small.json`

## Supported Models

- **Qwen3-8B**
- **Qwen2.5-7B-Instruct**
- **DeepSeek-V3**
- **GPT-4o-mini**
- **Internlm3**
- **GLM-4**

Model configurations are defined in `core/models/` and can be extended easily.

## Baselines

The framework supports comparison with the following baseline methods:

- **HippoRAG2**
- **RAPTOR**
- **LightRAG**
- **Legal**$\Delta$
- **ADAPT**

## Usage

1. **Setup environment**

   ```bash
   # Copy and configure environment file
   cp env.example .env
   # Edit .env with your configuration (model paths, API keys, etc.)
   ```

2. **Prepare data**

   - Place datasets under `datasets/` (or specify custom path)
   - Dataset files should follow the format: `crime_data_{dataset}_small.json`

3. **Run evaluation**

   ```bash
   python run.py --model qwen3 --datasets CAIL --devices cuda:2 cuda:3
   ```

   **Arguments:**

   - `--model`: Model to use (`qwen3`, `qwen2_5`, `gemma3`, `internlm3`, `glm4`, `deepseek_v3`, `gpt4o_mini`)
   - `--datasets`: Dataset name (e.g., `CAIL`, `CMDL`)
   - `--dotenv_path`: Path to .env file (default: `.env`)
   - `--datasets_path`: Path to datasets directory (default: `../datasets`)
   - `--devices`: GPU devices to use (e.g., `cuda:2 cuda:3`)
   - `--no-build-graph`: Skip graph construction (assume graph already exists)
   - `--force-rebuild`: Force rebuild graph even if it already exists

4. **Results**

   - Results are saved to:
     ```
     {output_dir}/{dataset}/{model}_results_combined.json
     ```
   - Statistics are saved to:
     ```
     {output_dir}/{dataset}/{model}_stats.json
     ```

   Each result file includes:

   - Case-level predictions (crime classification, law articles, etc.)
   - Overall metrics (Accuracy, Micro-F1)
   - Processing statistics

Example output structure:

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

## Configuration

Configuration is managed through the `.env` file. Key settings include:

- **Model Configuration**: Model name, device, API keys, generation parameters
- **Data Configuration**: Dataset paths, output directory
- **Graph Configuration**: Graph construction and retrieval parameters

See `env.example` for all available configuration options.

## Multi-GPU Support

The framework supports parallel processing across multiple GPUs. Simply specify multiple devices:

```bash
python run.py --model qwen3 --datasets CAIL --devices cuda:0 cuda:1 cuda:2 cuda:3
```

Cases will be automatically distributed across the specified devices for efficient processing.
