"""LightRAG: hybrid query, top_k 20, chunk_top_k 20, chunks of 1200 tokens with overlap 100."""
import json
import os
import shutil

from ..parsing import parse_judgment
from ..prompts import structured_judgment_prompt
from ..records import case_record, defendant_names
from .common import index_needed, mark_indexed, paragraph_docs


SETTINGS = {
    "embedding_model": "bge-m3",
    "query_type": "hybrid",
    "chunk_token_size": 1200,
    "chunk_overlap_token_size": 100,
    "retrieval_topk": 20,
    "chunk_top_k": 20,
}


def _answer_text(query_result):
    if isinstance(query_result, str):
        try:
            payload = json.loads(query_result)
        except json.JSONDecodeError:
            return query_result.strip(), ""
    else:
        payload = query_result
    if isinstance(payload, dict) and "content" in payload:
        content = payload.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        return content.strip(), str(payload.get("raw_data", ""))
    return str(payload).strip(), ""


async def judge(cases, law_text, case_text, working_dir, llm_base_url, llm_model, embed_host, embed_model, reindex, api_key):
    try:
        from lightrag import LightRAG, QueryParam
        from lightrag.kg.shared_storage import initialize_pipeline_status
        from lightrag.llm.openai import openai_complete_if_cache
        from lightrag.llm.ollama import ollama_embed
        from lightrag.utils import EmbeddingFunc
    except ImportError as exc:
        raise ImportError("LightRAG uses the lightrag package") from exc

    async def llm_model_func(prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kwargs):
        return await openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "",
            base_url=llm_base_url,
            **kwargs,
        )

    settings = dict(SETTINGS)
    settings["llm_model"] = llm_model
    settings["llm_base_url"] = llm_base_url
    rebuild = index_needed(working_dir, reindex, settings)
    if rebuild and os.path.isdir(working_dir):
        shutil.rmtree(working_dir)
    os.makedirs(working_dir, exist_ok=True)
    rag = LightRAG(
        working_dir=working_dir,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            func=lambda texts: ollama_embed(texts, embed_model=embed_model, host=embed_host),
        ),
        llm_model_func=llm_model_func,
        chunk_token_size=SETTINGS["chunk_token_size"],
        chunk_overlap_token_size=SETTINGS["chunk_overlap_token_size"],
        top_k=SETTINGS["retrieval_topk"],
        chunk_top_k=SETTINGS["chunk_top_k"],
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    if rebuild:
        await rag.ainsert(paragraph_docs(law_text, case_text))
        mark_indexed(working_dir, settings)

    try:
        results = []
        for case in cases:
            fact = case.get("fact", "")[:1024]
            judge_res = []
            for name in defendant_names(case):
                prompt = structured_judgment_prompt(name, fact)
                param = QueryParam(
                    mode="hybrid",
                    top_k=SETTINGS["retrieval_topk"],
                    chunk_top_k=SETTINGS["chunk_top_k"],
                )
                query_result = await rag.aquery(prompt, param=param)
                raw, retrieved = _answer_text(query_result)
                judge_res.append(
                    {
                        "name": name,
                        "judge_result": parse_judgment(raw, strip_thinking=False),
                        "retrieved_laws": retrieved,
                        "raw_output": raw,
                    }
                )
            results.append(case_record(case, fact, judge_res))
    finally:
        await rag.finalize_storages()
    return results
