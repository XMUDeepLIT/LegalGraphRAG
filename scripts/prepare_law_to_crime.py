import argparse
import json
import os
from typing import Any, Dict, List, Tuple


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_law_corpus(path: str) -> Dict[int, str]:
    result: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}: {exc}") from exc
            text_id = item.get("text_id")
            text = item.get("text", "")
            if text_id is None:
                continue
            try:
                law_id = int(text_id)
            except (TypeError, ValueError):
                continue
            result[law_id] = text
    return result


def build_judge_dep_index(judge_dep_data: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    index: Dict[int, Dict[str, Any]] = {}
    for row in judge_dep_data:
        law_id = row.get("id")
        if law_id is None:
            continue
        try:
            law_id = int(law_id)
        except (TypeError, ValueError):
            continue
        index[law_id] = row
    return index


def normalize_related_laws(raw_related: Any) -> List[str]:
    if not isinstance(raw_related, list):
        return []
    values: List[str] = []
    for item in raw_related:
        if isinstance(item, dict):
            text = item.get("text")
            if text:
                values.append(str(text))
            elif item.get("id"):
                values.append(str(item["id"]))
        elif item:
            values.append(str(item))
    return values


def merge_law_data(
    base_law_to_crime: List[Dict[str, Any]],
    judge_dep_index: Dict[int, Dict[str, Any]],
    corpus_map: Dict[int, str],
) -> Tuple[List[Dict[str, Any]], int, int]:
    merged: List[Dict[str, Any]] = []
    with_judge_dep = 0
    with_corpus_fallback = 0

    for row in base_law_to_crime:
        law_id = row.get("id")
        crimes = row.get("crime", [])
        if law_id is None:
            continue
        try:
            law_id = int(law_id)
        except (TypeError, ValueError):
            continue

        judge_row = judge_dep_index.get(law_id)

        text = ""
        judge_dep: List[str] = []
        related_laws: List[str] = []
        if judge_row:
            text = str(judge_row.get("law", "")).strip()
            judge_dep = [str(x) for x in judge_row.get("judge", []) if str(x).strip()]
            related_laws = normalize_related_laws(judge_row.get("related", []))
            with_judge_dep += 1

        if not text:
            text = corpus_map.get(law_id, "")
            if text:
                with_corpus_fallback += 1

        merged.append(
            {
                "id": law_id,
                "items": [
                    {
                        "text": text,
                        "crime": crimes if isinstance(crimes, list) else [str(crimes)],
                        "judge_dep": judge_dep,
                        "related_laws": related_laws,
                    }
                ],
            }
        )

    return merged, with_judge_dep, with_corpus_fallback


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build enriched law_to_crime.json for LegalGraphRAG."
    )
    parser.add_argument(
        "--base-law-to-crime",
        type=str,
        required=True,
        help="Path to base law_to_crime json (id + crime mapping).",
    )
    parser.add_argument(
        "--law-judge-dep",
        type=str,
        required=True,
        help="Path to law_judge_dep.json (contains law text / judge / related).",
    )
    parser.add_argument(
        "--law-corpus",
        type=str,
        required=True,
        help="Path to law_corpus.jsonl (fallback law text source).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./datas/law_to_crime.json",
        help="Output path for enriched law_to_crime json.",
    )
    args = parser.parse_args()

    base_data = load_json(args.base_law_to_crime)
    judge_dep_data = load_json(args.law_judge_dep)
    corpus_map = load_law_corpus(args.law_corpus)
    judge_dep_index = build_judge_dep_index(judge_dep_data)

    merged, with_judge_dep, with_corpus_fallback = merge_law_data(
        base_data, judge_dep_index, corpus_map
    )
    write_json(args.output, merged)

    print(f"Base law rows: {len(base_data)}")
    print(f"Rows matched in law_judge_dep: {with_judge_dep}")
    print(f"Rows using law_corpus fallback text: {with_corpus_fallback}")
    print(f"Wrote enriched law mapping: {args.output}")


if __name__ == "__main__":
    main()
