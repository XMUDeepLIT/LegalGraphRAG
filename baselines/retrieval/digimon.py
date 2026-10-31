"""G-Retriever through DIGIMON, method config Option/Method/GR.yaml.

https://github.com/JayLZhou/GraphRAG
"""
import asyncio
import os

from ..parsing import parse_charge_list
from ..prompts import digimon_question
from ..records import case_record, defendant_names
from .common import embed_host, index_needed, mark_indexed


def judge(cases, law_text, case_text, extra_text, save_dir, llm_base_url, llm_model, embedding_url, embedding_model, api_key, reindex, digimon_home):
    home = digimon_home or os.getenv("DIGIMON_HOME")
    if not home:
        raise ImportError("Set DIGIMON_HOME to a checkout of https://github.com/JayLZhou/GraphRAG")
    home = os.path.abspath(home)
    import sys

    if home not in sys.path:
        sys.path.insert(0, home)
    try:
        from Core.GraphRAG import GraphRAG
        from Option.Config2 import Config, merge_dict, parse
    except ImportError as exc:
        raise ImportError("Set DIGIMON_HOME to a checkout of https://github.com/JayLZhou/GraphRAG") from exc

    method = parse(os.path.join(home, "Option", "Method", "GR.yaml"))
    default = Config.read_yaml(os.path.join(home, "Option", "Config2.yaml"))
    final = merge_dict([method, default])
    final["llm"]["api_type"] = "open_llm"
    final["llm"]["model"] = llm_model
    final["llm"]["base_url"] = llm_base_url
    final["llm"]["api_key"] = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
    embedding = final.setdefault("embedding", {})
    if "cache_dir" in embedding and "cache_folder" not in embedding:
        embedding["cache_folder"] = embedding.pop("cache_dir")
    embedding["api_type"] = "ollama"
    embedding["model"] = embedding_model
    embedding["dimensions"] = 1024
    embedding["base_url"] = embed_host(embedding_url)
    final["dataset_name"] = "legal"
    final["exp_name"] = "gretriever"
    final["working_dir"] = save_dir
    index_settings = {"llm_model": llm_model, "llm_base_url": llm_base_url}
    final["graph"]["force"] = index_needed(save_dir, reindex, index_settings)
    digimon = GraphRAG(config=Config(**final))

    corpus = [
        {"title": "cases_corpus", "content": case_text, "doc_id": 0},
        {"title": "law_corpus", "content": law_text, "doc_id": 1},
    ]
    if extra_text:
        corpus.append({"title": "cases_corpus_more", "content": extra_text, "doc_id": 2})

    async def run():
        await digimon.insert(corpus)
        mark_indexed(save_dir, index_settings)
        results = []
        for case in cases:
            fact = case.get("fact", "")[:1024]
            judge_res = []
            for name in defendant_names(case):
                raw = await digimon.query(digimon_question(name, fact))
                if not isinstance(raw, str):
                    raw = str(raw)
                judge_res.append({"name": name, "crime": parse_charge_list(raw), "raw_output": raw})
            results.append(case_record(case, fact, judge_res))
        return results

    return asyncio.run(run())
