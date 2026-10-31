# Table 2 Data and Evaluation

Data files and evaluation steps for the LegalGraphRAG results in Table 2.

## 1. Retrieval corpus

The Table 2 retrieval corpus is:

```text
datas/main_experiment/cases_with_feature_big.json
```

It contains 14,049 feature-annotated cases. `datas/cases_with_feature.json` is the small demonstration corpus.

## 2. Corpus IDs and sources

The source manifest is `datas/main_experiment/corpus_sources.json`. Case IDs are sequential strings assigned after concatenating the processed sources in this order:

| Case IDs | Count | Dataset | Source split |
| --- | ---: | --- | --- |
| 1-3512 | 3,512 | [CAIL](https://github.com/thunlp/CAIL) | `test.json` |
| 3513-9264 | 5,752 | [JuDGE](https://github.com/oneal2000/JuDGE) | `filtered_ext_all.jsonl` |
| 9265-14049 | 4,785 | [CMDL](https://github.com/littlebowlnju/CMDL) | `train_small.jsonl` |

Records with incomplete judgment information were removed during processing. The released file contains 14,049 cases.

[LeCaRDv2](https://github.com/THUIR/LeCaRDv2) was used as a reference during manual corpus review, and a small amount of its data is included in the processed corpus.

## 3. Construction scripts

`scripts/sample_source_cases.py` samples CAIL and CMDL by charge and splits CMDL into one record per defendant. The released sampling rule keeps a charge while `count < limit` (`--boundary historical`). Choose the source split with `--input` and `--role`:

```bash
# Retrieval-corpus samples: use the source training/corpus split and quota 20.
python scripts/sample_source_cases.py \
  --dataset CAIL --role corpus --input raw/CAIL/test.json \
  --max-per-charge 20 --boundary historical --output work/cail.json

python scripts/sample_source_cases.py \
  --dataset CMDL --role corpus --input raw/CMDL/train_small.jsonl \
  --max-per-charge 20 --boundary historical --output work/cmdl.json

# Evaluation samples: select the corresponding held-out source split.
python scripts/sample_source_cases.py \
  --dataset CAIL --role test --input raw/CAIL/final_test.json \
  --max-per-charge 5 --boundary historical \
  --output work/crime_data_CAIL_small.json

python scripts/sample_source_cases.py \
  --dataset CMDL --role test --input raw/CMDL/test_small.jsonl \
  --max-per-charge 10 --boundary historical \
  --output work/crime_data_CMDL_small.json
```

CAIL corpus and test records come from different official CAIL subsets. CMDL uses `train_small.jsonl` for retrieval and `test_small.jsonl` for evaluation. The released CAIL test set uses a per-charge quota of 5 and contains 568 cases. The released CMDL test set uses a per-charge quota of 10 and contains 1,374 defendant records. `--exclude-test` removes normalized exact-fact overlaps when building a new corpus.

After feature extraction, merge the three processed source files with:

```bash
python scripts/merge_retrieval_corpus.py \
  --cail work/CAIL/cases_with_feature_big.json \
  --judge work/JuDGE/cases_with_feature_big.json \
  --cmdl work/CMDL/cases_with_feature_big.json \
  --exclude-test datas/main_experiment/crime_data_CAIL_small.json \
  --exclude-test datas/main_experiment/crime_data_CMDL_small.json \
  --verify-paper-counts \
  --output work/cases_with_feature_big.json \
  --manifest-output work/corpus_sources.json
```

`--deduplicate` drops exact duplicate records. The released corpus was built with this flag off. CMDL stores one record per defendant, so the same fact can appear more than once.

## 4. Table 2 evaluation

The repository includes both processed evaluation sets:

```text
datas/main_experiment/crime_data_CAIL_small.json   # 568 cases
datas/main_experiment/crime_data_CMDL_small.json   # 1,374 defendant records
```

Evaluate a generated result file with:

```bash
python evaluation/evaluate_results.py \
  --results outputs/main_experiment/CAIL/qwen3_results_combined.json

python evaluation/evaluate_results.py \
  --results outputs/main_experiment/CMDL/qwen3_results_combined.json
```

The evaluator reads `judge_result.charge_name` and `judge_res[].crime`. Accuracy is exact match, and Micro-F1 is aggregated over all labels. Each CMDL row is one defendant, so its `judgment_count` is 1,374.
