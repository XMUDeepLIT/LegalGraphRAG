# Table 2 baselines

Entry point: `python -m baselines.run --list`

Every protocol reads the first 1,024 characters of `fact` and writes a JSON list scored by `evaluation/evaluate_results.py`.

## Layout

```text
baselines/
├── prompts.py            # Prompt strings
├── catalog.py            # Direct-model names, checkpoints, and decoding
├── generators.py         # Retrieval backbones: model id, base URL, API key
├── parsing.py            # JSON object and charge-list readers
├── corpus.py             # law_corpus.txt / cases_corpus.txt text
├── records.py            # Output record: id, fact, true_charge, law_article, term, judge_res
├── cli.py                # Argument parser and dispatch
├── direct/               # One prompt per defendant
│   ├── local.py          # Chat template and generate() settings
│   ├── session.py        # structured / freeform / adapt
│   └── worker.py         # One process per GPU
└── retrieval/            # One file per retrieval system
    ├── naive.py
    ├── hipporag.py
    ├── lightrag.py
    ├── raptor.py
    └── digimon.py        # G-Retriever via DIGIMON
```

## Direct models

| Model | Protocol | Decoding | Parsed field |
| --- | --- | --- | --- |
| `qwen3` | structured JSON | greedy, thinking off; text inside `<think>` is removed before parsing | `judge_result` |
| `internlm3`, `glm4` | structured JSON | greedy, thinking off; the raw generation is parsed | `judge_result` |
| `deepseek_v3`, `gpt4o_mini` | structured JSON | temperature 0.1, `max_tokens` 4096 | `judge_result` |
| `qwen2_5`, `disc_lawllm` | free-form, ending at `输出：` | greedy, then `deepseek-chat` extracts JSON at temperature 0.1 and `max_tokens` 500 | `judge_result` |
| `legal_delta` | free-form | greedy, temperature 0.1, `max_new_tokens` 4096, then the same extractor. Weights default to `./legal_delta` or `LEGAL_DELTA_PATH` | `judge_result` |
| `adapt` | one ADAPT turn | sample at temperature 0.7, `top_p` 0.9, `repetition_penalty` 1.2, `no_repeat_ngram_size` 3, `max_new_tokens` 2048, then the same extractor | `judge_result` |

`DEEPSEEK_API_KEY` is read for `deepseek_v3` and for the extractor. `OPENAI_API_KEY` is read for `gpt4o_mini`.

```bash
python -m baselines.run --model qwen3 --dataset CAIL --devices cuda:0
```

The result file is `outputs/CAIL/structured_qwen3_results.json`.

## Retrieval

Corpus text comes from `datas/main_experiment/criminal_law_processed.json` and `datas/main_experiment/cases_with_feature_big.json`, in the `law_corpus.txt` / `cases_corpus.txt` layout. Embeddings use `bge-m3` at `http://localhost:11434/api/embed`. Indexes are stored under `outputs/indexes/`. `--reindex` rebuilds an index.

The generator comes from `baselines/generators.py`. The default is `qwen3`: model id `Qwen/Qwen3-8B` at `http://localhost:8000/v1`. `--model deepseek_v3` uses `deepseek-chat` at `https://api.deepseek.com/v1` and reads `DEEPSEEK_API_KEY`. `--model gpt4o_mini` uses `gpt-4o-mini` at `OPENAI_BASE_URL` or `https://api.openai.com/v1` and reads `OPENAI_API_KEY`. `--llm-model` replaces the model id. `--llm-base-url` replaces the base URL. The result file is `outputs/{dataset}/{protocol}_{generator}_results.json`.

| Protocol | File | Retrieval | Question | Generator call | Parsed field |
| --- | --- | --- | --- | --- | --- |
| `naive_rag` | `retrieval/naive.py` | `cl100k_base` chunks of 1000 tokens, overlap 200, top 5, context capped at 3,072 characters, `rlm/rag-prompt` | charge list | one chat completion, temperature 0.7. The index is `outputs/indexes/naive_rag` | `judge_res[].crime` |
| `hipporag2` | `retrieval/hipporag.py` | blank-line documents, `retrieval_top_k` 5, `linking_top_k` 5, `qa_top_k` 5, `max_qa_steps` 3, `graph_type` `facts_and_sim_passage_node_unidirectional`. `--extra-text` adds `cases_corpus_more.txt` | structured JSON | HippoRAG reads `OPENAI_API_KEY` and the generator base URL. The index is `outputs/indexes/hipporag2_{generator}` | `judge_result` |
| `lightrag` | `retrieval/lightrag.py` | hybrid, chunks of 1200 tokens with overlap 100, `top_k` 20, `chunk_top_k` 20 | structured JSON | LightRAG completion uses the generator key and base URL. The index is `outputs/indexes/lightrag_{generator}` | `judge_result` |
| `raptor` | `retrieval/raptor.py` | `cl100k_base` chunks of 1200 tokens with overlap 100, `num_layers` 5, `max_length_in_cluster` 3500, clustering threshold 0.1. `--extra-text` is appended to the corpus | charge list | summary `max_tokens` 150, answer temperature 0. Embeddings are `bge-m3`. The index is `outputs/indexes/raptor_{generator}` | `judge_res[].crime` |
| `gretriever` | `retrieval/digimon.py` | [DIGIMON](https://github.com/JayLZhou/GraphRAG) `Option/Method/GR.yaml`: `er_graph`, chunks of 1200 tokens with overlap 100, two-step extraction, `max_gleaning` 1, Steiner subgraph with `top_k` 3, `topk_e` 2, `cost_e` 0.5, context kept to 512 tokens | charge list, then wrapped as `Question: {question}\nAnswer: ` | `open_llm`, temperature 0.0, `max_token` 4096. System text: `You are an AI assistant that helps people find information.` The index is `outputs/indexes/gretriever_{generator}` | `judge_res[].crime` |

```bash
python -m baselines.run --protocol naive_rag --dataset CAIL
python -m baselines.run --protocol raptor --model deepseek_v3 --dataset CAIL
python -m baselines.run --protocol hipporag2 --dataset CAIL --extra-text /path/to/cases_corpus_more.txt
python -m baselines.run --protocol lightrag --model gpt4o_mini --dataset CAIL
python -m baselines.run --protocol naive_rag --llm-model qwen3-8b --dataset CAIL
DIGIMON_HOME=/path/to/GraphRAG python -m baselines.run --protocol gretriever --dataset CAIL
```

`DIGIMON_HOME` is a checkout of https://github.com/JayLZhou/GraphRAG. The G-Retriever corpus records are `cases_corpus`, then `law_corpus`.

Score a result file with:

```bash
python evaluation/evaluate_results.py \
  --results outputs/CAIL/structured_qwen3_results.json \
  --crimes-by-part datas/main_experiment/crimes_by_part.json
```
