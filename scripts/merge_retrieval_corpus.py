"""Merge processed CAIL, JuDGE, and CMDL records into the retrieval corpus."""

import argparse
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Sequence


SOURCE_INFO = {
    "CAIL": ("https://github.com/thunlp/CAIL", "test.json"),
    "JuDGE": ("https://github.com/oneal2000/JuDGE", "filtered_ext_all.jsonl"),
    "CMDL": ("https://github.com/littlebowlnju/CMDL", "train_small.jsonl"),
}


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def normalized_fact(row: Dict[str, Any]) -> str:
    return re.sub(r"\s+", "", str(row.get("fact", "")))


def canonical_record(row: Dict[str, Any]) -> str:
    payload = {key: value for key, value in row.items() if key != "id"}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def record_hash(row: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_record(row).encode("utf-8")).hexdigest()


def normalize_laws(row: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(row)
    laws = result.get("law", result.get("laws", []))
    normalized = [str(law).split(".")[0] for law in laws if str(law)]
    result.pop("laws", None)
    result["law"] = normalized
    return result


def load_excluded_facts(paths: Sequence[str]) -> set:
    facts = set()
    for path in paths:
        facts.update(normalized_fact(row) for row in load_json(path))
    return facts


def load_manual_exclusions(path: str) -> set:
    if not path:
        return set()
    data = load_json(path)
    hashes = set()
    for item in data:
        if isinstance(item, str):
            hashes.add(item)
        elif isinstance(item, dict) and item.get("record_sha256"):
            hashes.add(item["record_sha256"])
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge feature-annotated source files into cases_with_feature_big.json."
    )
    parser.add_argument("--cail", required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--cmdl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--exclude-test", action="append", default=[])
    parser.add_argument("--manual-exclusions", default="")
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Remove exact duplicate records; same fact with different defendants is retained.",
    )
    parser.add_argument("--verify-paper-counts", action="store_true")
    args = parser.parse_args()

    source_paths = {"CAIL": args.cail, "JuDGE": args.judge, "CMDL": args.cmdl}
    expected = {"CAIL": 3512, "JuDGE": 5752, "CMDL": 4785}
    excluded_facts = load_excluded_facts(args.exclude_test)
    manual_exclusions = load_manual_exclusions(args.manual_exclusions)
    seen = set()
    merged = []
    manifest_sources = []

    for dataset, path in source_paths.items():
        start_id = len(merged) + 1
        accepted = []
        for raw_row in load_json(path):
            row = normalize_laws(raw_row)
            row.pop("id", None)
            if normalized_fact(row) in excluded_facts:
                continue
            if record_hash(row) in manual_exclusions:
                continue
            identity = canonical_record(row)
            if args.deduplicate and identity in seen:
                continue
            seen.add(identity)
            accepted.append(row)
        if args.verify_paper_counts and len(accepted) != expected[dataset]:
            raise ValueError(
                f"{dataset}: expected {expected[dataset]} records, got {len(accepted)}"
            )
        for row in accepted:
            row["id"] = str(len(merged) + 1)
            merged.append(row)
        repository, split = SOURCE_INFO[dataset]
        source = {
            "dataset": dataset,
            "repository": repository,
            "source_split": split,
            "first_case_id": str(start_id),
            "last_case_id": str(len(merged)),
            "count": len(accepted),
        }
        if dataset == "CMDL":
            source["note"] = (
                "Multi-defendant judgments were split into one record per defendant."
            )
        manifest_sources.append(source)

    if args.verify_paper_counts and len(merged) != 14049:
        raise ValueError(f"Expected 14049 total records, got {len(merged)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    manifest = {
        "corpus": os.path.basename(args.output),
        "total_cases": len(merged),
        "id_assignment": (
            "Sequential string IDs after concatenating CAIL, JuDGE, and CMDL "
            "in that order."
        ),
        "supplementary_source_note": (
            "LeCaRDv2 was used during manual corpus review, and a small amount "
            "of its data was incorporated without separate source-ID tags: "
            "https://github.com/THUIR/LeCaRDv2"
        ),
        "sources": manifest_sources,
    }
    os.makedirs(
        os.path.dirname(os.path.abspath(args.manifest_output)), exist_ok=True
    )
    with open(args.manifest_output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Merged corpus: {len(merged)} records")
    print(f"Output: {args.output}")
    print(f"Manifest: {args.manifest_output}")


if __name__ == "__main__":
    main()
