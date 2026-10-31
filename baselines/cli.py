"""Command line for the Table 2 baselines.

    python -m baselines.run --list
    python -m baselines.run --model qwen3 --dataset CAIL --devices cuda:0
    python -m baselines.run --protocol naive_rag --dataset CAIL
"""
import argparse
import asyncio
import json
import os
from typing import Any, Dict, List

from baselines.catalog import MODELS, PROTOCOLS, get_model
from baselines.corpus import load_case_corpus, load_law_corpus, read_text
from baselines.direct.worker import run_direct
from baselines.generators import GENERATORS, ResolvedGenerator, resolve_generator
from baselines.retrieval.common import embed_host


def load_cases(dataset: str, datasets_path: str, limit: int = 0) -> List[Dict[str, Any]]:
    path = os.path.join(datasets_path, f"crime_data_{dataset}_small.json")
    with open(path, "r", encoding="utf-8") as handle:
        cases = json.load(handle)
    if limit:
        return cases[:limit]
    return cases


def write_results(path: str, results: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(path)


def load_corpora(args):
    law_text = load_law_corpus(args.law_json, args.law_text)
    case_text = load_case_corpus(args.cases_json, args.cases_text)
    extra_text = read_text(args.extra_text) if args.extra_text else ""
    return law_text, case_text, extra_text


def index_dir(args, name: str) -> str:
    path = args.index_dir or os.path.join(args.output_dir, "indexes", name)
    os.makedirs(path, exist_ok=True)
    return path


def output_path(args, name: str) -> str:
    return os.path.join(args.output_dir, args.dataset, f"{name}_results.json")


def retrieval_generator(args) -> ResolvedGenerator:
    key = args.model or "qwen3"
    if key not in GENERATORS:
        spec = get_model(key)
        known = ", ".join(GENERATORS)
        raise SystemExit(f"{key} uses protocol {spec.protocol}. Retrieval generators: {known}")
    return resolve_generator(key, args.llm_model, args.llm_base_url, args.api_key)


def run_naive(args):
    from baselines.retrieval.naive import judge_cases, load_or_build_index

    generator = retrieval_generator(args)
    cases = load_cases(args.dataset, args.datasets_path, args.limit)
    law_text, case_text, _ = load_corpora(args)
    index = load_or_build_index(
        law_text,
        case_text,
        index_dir(args, "naive_rag"),
        args.embedding_url,
        args.embedding_model,
        args.reindex,
    )
    results = judge_cases(
        cases,
        index,
        generator.base_url,
        generator.model_name,
        generator.api_key,
    )
    write_results(output_path(args, f"naive_rag_{generator.key}"), results)


def run_hipporag2(args):
    from baselines.retrieval.hipporag import judge

    generator = retrieval_generator(args)
    cases = load_cases(args.dataset, args.datasets_path, args.limit)
    law_text, case_text, extra_text = load_corpora(args)
    results = judge(
        cases,
        law_text,
        case_text,
        extra_text,
        index_dir(args, f"hipporag2_{generator.key}"),
        generator.base_url,
        generator.model_name,
        args.embedding_model,
        args.reindex,
        generator.api_key,
    )
    write_results(output_path(args, f"hipporag2_{generator.key}"), results)


def run_lightrag(args):
    from baselines.retrieval.lightrag import judge

    generator = retrieval_generator(args)
    cases = load_cases(args.dataset, args.datasets_path, args.limit)
    law_text, case_text, _ = load_corpora(args)
    results = asyncio.run(
        judge(
            cases,
            law_text,
            case_text,
            index_dir(args, f"lightrag_{generator.key}"),
            generator.base_url,
            generator.model_name,
            embed_host(args.embedding_url),
            args.embedding_model,
            args.reindex,
            generator.api_key,
        )
    )
    write_results(output_path(args, f"lightrag_{generator.key}"), results)


def run_raptor(args):
    from baselines.retrieval.raptor import judge

    generator = retrieval_generator(args)
    cases = load_cases(args.dataset, args.datasets_path, args.limit)
    law_text, case_text, extra_text = load_corpora(args)
    results = judge(
        cases,
        law_text,
        case_text,
        extra_text,
        index_dir(args, f"raptor_{generator.key}"),
        generator.base_url,
        generator.model_name,
        args.embedding_url,
        args.embedding_model,
        args.reindex,
        generator.api_key,
    )
    write_results(output_path(args, f"raptor_{generator.key}"), results)


def run_gretriever(args):
    from baselines.retrieval.digimon import judge

    if not (args.digimon_home or os.getenv("DIGIMON_HOME")):
        raise SystemExit("Set DIGIMON_HOME to a checkout of https://github.com/JayLZhou/GraphRAG")
    generator = retrieval_generator(args)
    cases = load_cases(args.dataset, args.datasets_path, args.limit)
    law_text, case_text, extra_text = load_corpora(args)
    results = judge(
        cases,
        law_text,
        case_text,
        extra_text,
        index_dir(args, f"gretriever_{generator.key}"),
        generator.base_url,
        generator.model_name,
        args.embedding_url,
        args.embedding_model,
        generator.api_key,
        args.reindex,
        args.digimon_home,
    )
    write_results(output_path(args, f"gretriever_{generator.key}"), results)


RETRIEVAL = {
    "naive_rag": run_naive,
    "hipporag2": run_hipporag2,
    "lightrag": run_lightrag,
    "raptor": run_raptor,
    "gretriever": run_gretriever,
}


def print_catalog():
    print(f"{'model':<16} {'protocol':<12} {'checkpoint'}")
    for spec in MODELS.values():
        print(f"{spec.key:<16} {spec.protocol:<12} {spec.model_name}")
        print(f"  {spec.description}")
    print(f"{'generator':<16} {'model':<28} {'base url'}")
    for spec in GENERATORS.values():
        print(f"{spec.key:<16} {spec.model_name:<28} {spec.base_url}")
        print(f"  {spec.description}")
    print("retrieval protocols: " + ", ".join(RETRIEVAL))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a LegalGraphRAG Table 2 baseline")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--model", choices=sorted(MODELS))
    parser.add_argument("--protocol", choices=PROTOCOLS)
    parser.add_argument("--dataset", default="CAIL")
    parser.add_argument("--datasets-path", default="datas/main_experiment")
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--law-json", default="datas/main_experiment/criminal_law_processed.json")
    parser.add_argument("--cases-json", default="datas/main_experiment/cases_with_feature_big.json")
    parser.add_argument("--law-text", default=None)
    parser.add_argument("--cases-text", default=None)
    parser.add_argument("--extra-text", default=None, help="cases_corpus_more.txt for HippoRAG 2 and RAPTOR")
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--embedding-url", default="http://localhost:11434/api/embed")
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--digimon-home", default=None, help="Checkout of https://github.com/JayLZhou/GraphRAG")
    return parser


def main():
    args = build_parser().parse_args()
    if args.list:
        print_catalog()
        return
    protocol = args.protocol
    spec = get_model(args.model) if args.model else None
    if protocol in RETRIEVAL:
        runner = RETRIEVAL[protocol]
        runner(args)
        return
    if spec is not None:
        if protocol and protocol != spec.protocol:
            raise SystemExit(f"{spec.key} uses protocol {spec.protocol}")
        protocol = spec.protocol
    if protocol in ("structured", "freeform", "adapt"):
        if spec is None:
            raise SystemExit("Pass --model for a direct baseline")
        cases = load_cases(args.dataset, args.datasets_path, args.limit)
        results = run_direct(args, spec, cases, output_path(args, f"{spec.protocol}_{spec.key}"))
        write_results(output_path(args, f"{spec.protocol}_{spec.key}"), results)
        return
    raise SystemExit("Pass --model for a direct baseline, or --protocol for a retrieval baseline")
