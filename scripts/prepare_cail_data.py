import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List


def load_cail_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}: {exc}") from exc


def normalize_case(raw_case: Dict[str, Any], case_id: int) -> Dict[str, Any]:
    meta = raw_case.get("meta", {})
    accusations = [f"{x}罪" for x in meta.get("accusation", []) if x]
    laws = meta.get("relevant_articles", [])
    return {
        "id": case_id,
        "name": meta.get("criminals", []),
        "fact": raw_case.get("fact", ""),
        "crime": accusations,
        "law": laws,
        "laws": laws,
        "term_of_imprisonment": meta.get("term_of_imprisonment", {}),
    }


def sample_cases_per_charge(
    raw_cases: Iterable[Dict[str, Any]], max_per_charge: int
) -> List[Dict[str, Any]]:
    charges_counter = defaultdict(int)
    sampled_cases: List[Dict[str, Any]] = []
    current_id = 0

    for raw in raw_cases:
        accusations = raw.get("meta", {}).get("accusation", [])
        should_insert = False

        for charge in accusations:
            charges_counter[charge] += 1
            if charges_counter[charge] <= max_per_charge:
                should_insert = True

        if should_insert:
            sampled_cases.append(normalize_case(raw, current_id))
            current_id += 1

    return sampled_cases


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare LegalGraphRAG cases_with_feature file from raw CAIL JSONL."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw CAIL jsonl file (e.g., final_test.json).",
    )
    parser.add_argument(
        "--max-per-charge",
        type=int,
        default=5,
        help="Keep at most this many samples per accusation label.",
    )
    parser.add_argument(
        "--cases-output",
        type=str,
        default="./datas/cases_with_feature.json",
        help="Output path for case database used by LegalGraphRAG.",
    )
    parser.add_argument(
        "--dataset-output",
        type=str,
        default="./datasets/crime_data_CAIL_small.json",
        help="Output path for evaluation dataset file.",
    )
    args = parser.parse_args()

    raw_cases = list(load_cail_jsonl(args.input))
    sampled_cases = sample_cases_per_charge(raw_cases, args.max_per_charge)

    dataset_cases = []
    for case in sampled_cases:
        dataset_cases.append(
            {
                "id": case["id"],
                "name": case["name"],
                "fact": case["fact"],
                "crime": case["crime"],
                "laws": case["law"],
                "term_of_imprisonment": case.get("term_of_imprisonment", {}),
            }
        )

    write_json(args.cases_output, sampled_cases)
    write_json(args.dataset_output, dataset_cases)

    print(f"Loaded raw cases: {len(raw_cases)}")
    print(f"Sampled cases: {len(sampled_cases)}")
    print(f"Wrote case db file: {args.cases_output}")
    print(f"Wrote dataset file: {args.dataset_output}")


if __name__ == "__main__":
    main()
