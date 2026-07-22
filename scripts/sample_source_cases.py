"""Sample CAIL or CMDL source splits using the paper's per-charge rule."""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence


def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def unique(values: Iterable[Any]) -> List[Any]:
    result = []
    seen = set()
    for value in values:
        marker = str(value)
        if marker not in seen:
            seen.add(marker)
            result.append(value)
    return result


def should_keep(
    labels: Sequence[str], counts: Dict[str, int], limit: int, boundary: str
) -> bool:
    keep = False
    for label in labels:
        counts[label] += 1
        if boundary == "historical":
            keep = keep or counts[label] < limit
        else:
            keep = keep or counts[label] <= limit
    return keep


def sample_cail(
    rows: Iterable[Dict[str, Any]], role: str, limit: int, boundary: str
) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = defaultdict(int)
    sampled = []
    for row in rows:
        meta = row.get("meta", {})
        names = meta.get("criminals", [])
        fact = row.get("fact", "")
        labels = [str(x) for x in meta.get("accusation", []) if x]
        if role == "corpus" and (len(names) != 1 or len(fact) > 1024):
            continue
        if not should_keep(labels, counts, limit, boundary):
            continue
        crimes = [x if x.endswith("罪") else x + "罪" for x in labels]
        laws = meta.get("relevant_articles", [])
        if role == "corpus":
            sampled.append(
                {"name": names[0], "fact": fact, "crime": crimes, "law": laws}
            )
        else:
            sampled.append(
                {
                    "name": names,
                    "fact": fact,
                    "crime": crimes,
                    "laws": laws,
                    "term_of_imprisonment": meta.get("term_of_imprisonment", {}),
                }
            )
    return sampled


def article_number(value: Any) -> str:
    match = re.search(r"\d+", str(value))
    return match.group(0) if match else ""


def sample_cmdl(
    rows: Iterable[Dict[str, Any]], role: str, limit: int, boundary: str
) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = defaultdict(int)
    sampled = []
    for row in rows:
        fact = row.get("fact", "")
        if len(fact) > 1024:
            continue
        for defendant in row.get("outcomes", []):
            judgments = defendant.get("judgment", [])
            labels = [str(x.get("accusation", "")) for x in judgments]
            labels = [x for x in labels if x]
            if not should_keep(labels, counts, limit, boundary):
                continue
            laws = unique(
                number
                for judgment in judgments
                for article in judgment.get("article", [])
                for number in [article_number(article)]
                if number
            )
            result = {
                "name": [defendant.get("name", "")],
                "fact": fact,
                "crime": labels,
            }
            result["law" if role == "corpus" else "laws"] = laws
            sampled.append(result)
    return sampled


def normalized_fact(row: Dict[str, Any]) -> str:
    return re.sub(r"\s+", "", str(row.get("fact", "")))


def record_key(row: Dict[str, Any]) -> str:
    payload = {key: value for key, value in row.items() if key != "id"}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def filter_records(
    rows: List[Dict[str, Any]], test_paths: Sequence[str], deduplicate: bool
) -> List[Dict[str, Any]]:
    excluded_facts = set()
    for path in test_paths:
        with open(path, "r", encoding="utf-8") as f:
            excluded_facts.update(normalized_fact(row) for row in json.load(f))

    result = []
    seen = set()
    for row in rows:
        if normalized_fact(row) in excluded_facts:
            continue
        key = record_key(row)
        if deduplicate and key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample CAIL/CMDL corpus or test records by charge frequency."
    )
    parser.add_argument("--dataset", required=True, choices=["CAIL", "CMDL"])
    parser.add_argument("--role", required=True, choices=["corpus", "test"])
    parser.add_argument("--input", required=True, help="Raw source split in JSONL.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-per-charge", required=True, type=int)
    parser.add_argument(
        "--boundary",
        choices=["historical", "inclusive"],
        default="historical",
        help="historical reproduces the original count < limit condition.",
    )
    parser.add_argument(
        "--exclude-test",
        action="append",
        default=[],
        help="Processed test JSON whose facts must be excluded; repeat as needed.",
    )
    parser.add_argument("--deduplicate", action="store_true")
    args = parser.parse_args()

    source_rows = load_jsonl(args.input)
    if args.dataset == "CAIL":
        sampled = sample_cail(
            source_rows, args.role, args.max_per_charge, args.boundary
        )
    else:
        sampled = sample_cmdl(
            source_rows, args.role, args.max_per_charge, args.boundary
        )
    filtered = filter_records(sampled, args.exclude_test, args.deduplicate)
    if args.role == "test":
        for case_id, row in enumerate(filtered):
            row["id"] = case_id

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    print(f"Sampled before exclusion/deduplication: {len(sampled)}")
    print(f"Written records: {len(filtered)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
