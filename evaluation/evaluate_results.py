"""Evaluate LegalGraphRAG main experiment outputs.

The script accepts both the current result format produced by run.py and the
legacy Table 2 format, where predictions are stored in ``judge_res[].crime``.

[
  {
    "true_charge": ["..."],
    "law_article": ["264", ...],
    "term_of_imprisonment": {"imprisonment": 12, ...},
    "judge_res": [
      {
        "judge_result": {
          "charge_name": ["..."] or "...",
          "law_article": ["第264条", ...],
          "term_of_imprisonment": {"imprisonment": 12, ...}
        }
      }
    ]
  }
]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


CHINESE_NUMBERS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_charge(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    if text.endswith("罪"):
        return text
    if text == "无罪":
        return text
    return text + "罪"


def charges_from_value(value: Any) -> Set[str]:
    return {x for x in (normalize_charge(v) for v in as_list(value)) if x}


def is_subsequence(needle: str, haystack: str) -> bool:
    iterator = iter(haystack)
    return all(char in iterator for char in needle)


def fuzzy_map_charges(true_labels: Set[str], pred_labels: Set[str]) -> Set[str]:
    """Match the one-way subsequence rule used by the paper evaluation."""
    mapped = set()
    unmatched = set(pred_labels)
    for true_label in true_labels:
        for pred_label in pred_labels:
            if is_subsequence(true_label, pred_label):
                mapped.add(true_label)
                unmatched.discard(pred_label)
    return mapped | unmatched


def chinese_number_to_int(text: str) -> Optional[int]:
    if not text:
        return None
    total = 0
    section = 0
    number = 0
    units = {"十": 10, "百": 100, "千": 1000}
    for char in text:
        if char in CHINESE_NUMBERS:
            number = CHINESE_NUMBERS[char]
        elif char in units:
            unit = units[char]
            section += (number or 1) * unit
            number = 0
        else:
            return None
    total += section + number
    return total or None


def law_to_int(label: Any) -> Optional[int]:
    text = str(label).strip()
    if not text:
        return None
    digits = re.findall(r"\d+", text)
    if digits:
        return int(digits[0])
    match = re.search(r"第([零一二两三四五六七八九十百千]+)条", text)
    if match:
        return chinese_number_to_int(match.group(1))
    return None


def laws_from_value(value: Any) -> Set[int]:
    labels = set()
    for item in as_list(value):
        law = law_to_int(item)
        if law is not None:
            labels.add(law)
    return labels


def get_judgments(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    judgments = item.get("judge_res", [])
    return [j for j in judgments if isinstance(j, dict)]


def iter_prediction_sets(item: Dict[str, Any]) -> Iterable[Tuple[Set[str], Set[int]]]:
    judgments = get_judgments(item)
    for judgment in judgments:
        judge_result = judgment.get("judge_result")
        if isinstance(judge_result, dict):
            pred_charges = charges_from_value(judge_result.get("charge_name"))
            pred_laws = laws_from_value(judge_result.get("law_article"))
        else:
            pred_charges = charges_from_value(judgment.get("crime"))
            pred_laws = laws_from_value(
                judgment.get("law_article", judgment.get("law"))
            )
        yield pred_charges, pred_laws


def extract_imprisonment(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        value = value.get("imprisonment")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def micro_counts(true_sets: Sequence[Set[Any]], pred_sets: Sequence[Set[Any]]) -> Tuple[int, int, int]:
    tp = fp = fn = 0
    for true_set, pred_set in zip(true_sets, pred_sets):
        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)
    return tp, fp, fn


def micro_f1(true_sets: Sequence[Set[Any]], pred_sets: Sequence[Set[Any]]) -> float:
    tp, fp, fn = micro_counts(true_sets, pred_sets)
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else 0.0


def exact_match(true_sets: Sequence[Set[Any]], pred_sets: Sequence[Set[Any]]) -> float:
    if not true_sets:
        return 0.0
    return sum(1 for true_set, pred_set in zip(true_sets, pred_sets) if true_set == pred_set) / len(true_sets)


def load_crime_parts(path: Optional[Path]) -> Dict[str, str]:
    if path is None or not path.exists():
        return {}
    part_data = load_json(path)
    mapping = {}
    for part in part_data:
        title = part.get("title", "")
        for crime in part.get("crimes", []):
            mapping[normalize_charge(crime)] = title
    return mapping


def case_parts(true_charges: Set[str], crime_to_part: Dict[str, str]) -> Set[str]:
    parts = set()
    for label in true_charges:
        for crime, part in crime_to_part.items():
            if is_subsequence(label, crime):
                parts.add(part)
    return parts


def evaluate(data: Sequence[Dict[str, Any]], crime_to_part: Dict[str, str]) -> Dict[str, Any]:
    true_charges_all: List[Set[str]] = []
    pred_charges_all: List[Set[str]] = []
    true_laws_all: List[Set[int]] = []
    pred_laws_all: List[Set[int]] = []
    term_pairs: List[Tuple[float, float]] = []
    per_part: Dict[str, Dict[str, List[Any]]] = {}

    for item in data:
        true_charges = charges_from_value(item.get("true_charge"))
        true_laws = laws_from_value(item.get("law_article"))
        has_law_labels = "law_article" in item
        parts = case_parts(true_charges, crime_to_part)
        for part in parts:
            per_part.setdefault(
                part,
                {
                    "true_charges": [],
                    "pred_charges": [],
                    "true_laws": [],
                    "pred_laws": [],
                    "term_pairs": [],
                },
            )

        for raw_pred_charges, pred_laws in iter_prediction_sets(item):
            pred_charges = fuzzy_map_charges(true_charges, raw_pred_charges)
            true_charges_all.append(true_charges)
            pred_charges_all.append(pred_charges)
            if has_law_labels:
                true_laws_all.append(true_laws)
                pred_laws_all.append(pred_laws)

            for part in parts:
                bucket = per_part[part]
                bucket["true_charges"].append(true_charges)
                bucket["pred_charges"].append(pred_charges)
                if has_law_labels:
                    bucket["true_laws"].append(true_laws)
                    bucket["pred_laws"].append(pred_laws)

        true_term = extract_imprisonment(item.get("term_of_imprisonment"))
        if true_term is not None:
            pred_term = 0.0
            judgments = get_judgments(item)
            if judgments:
                judge_result = judgments[0].get("judge_result")
                if isinstance(judge_result, dict):
                    parsed_term = extract_imprisonment(
                        judge_result.get("term_of_imprisonment")
                    )
                else:
                    parsed_term = extract_imprisonment(
                        judgments[0].get("term_of_imprisonment")
                    )
                if parsed_term is not None:
                    pred_term = parsed_term
            term_pairs.append((true_term, pred_term))
            for part in parts:
                bucket = per_part[part]
                bucket["term_pairs"].append((true_term, pred_term))

    metrics = metric_block(
        true_charges_all,
        pred_charges_all,
        true_laws_all,
        pred_laws_all,
        term_pairs,
    )
    metrics["total_cases"] = len(data)

    metrics["by_part"] = {
        part: metric_block(
            values["true_charges"],
            values["pred_charges"],
            values["true_laws"],
            values["pred_laws"],
            values["term_pairs"],
        )
        for part, values in sorted(per_part.items())
    }
    return metrics


def metric_block(
    true_charges: Sequence[Set[str]],
    pred_charges: Sequence[Set[str]],
    true_laws: Sequence[Set[int]],
    pred_laws: Sequence[Set[int]],
    term_pairs: Sequence[Tuple[float, float]],
) -> Dict[str, Any]:
    if term_pairs:
        term_em = sum(1 for true, pred in term_pairs if true == pred) / len(term_pairs)
        term_mae = sum(abs(true - pred) for true, pred in term_pairs) / len(term_pairs)
    else:
        term_em = None
        term_mae = None
    law_accuracy = exact_match(true_laws, pred_laws) if true_laws else None
    law_micro_f1 = micro_f1(true_laws, pred_laws) if true_laws else None
    return {
        "judgment_count": len(true_charges),
        "charge_accuracy": exact_match(true_charges, pred_charges),
        "charge_micro_f1": micro_f1(true_charges, pred_charges),
        "law_article_count": len(true_laws),
        "law_article_accuracy": law_accuracy,
        "law_article_micro_f1": law_micro_f1,
        "term_em": term_em,
        "term_mae": term_mae,
        "term_count": len(term_pairs),
    }


def flatten_rows(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [{"scope": "overall", **{k: v for k, v in metrics.items() if k != "by_part"}}]
    for part, values in metrics.get("by_part", {}).items():
        rows.append({"scope": part, **values})
    return rows


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "scope",
        "judgment_count",
        "charge_accuracy",
        "charge_micro_f1",
        "law_article_count",
        "law_article_accuracy",
        "law_article_micro_f1",
        "term_em",
        "term_mae",
        "term_count",
        "total_cases",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LegalGraphRAG result JSON.")
    parser.add_argument("--results", required=True, type=Path, help="Path to *_results_combined.json.")
    parser.add_argument(
        "--crimes-by-part",
        type=Path,
        default=Path("./datas/main_experiment/crimes_by_part.json"),
        help="Optional crimes_by_part.json for per-category metrics.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Metrics JSON output path.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Metrics CSV output path.")
    args = parser.parse_args()

    data = load_json(args.results)
    crime_to_part = load_crime_parts(args.crimes_by_part)
    metrics = evaluate(data, crime_to_part)

    output_json = args.output_json or args.results.with_name(args.results.stem + "_metrics.json")
    output_csv = args.output_csv or args.results.with_name(args.results.stem + "_metrics.csv")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    write_csv(flatten_rows(metrics), output_csv)

    overall = {k: v for k, v in metrics.items() if k != "by_part"}
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print(f"Metrics saved to {output_json}")
    print(f"CSV saved to {output_csv}")


if __name__ == "__main__":
    main()
