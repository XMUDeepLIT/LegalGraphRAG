"""HippoRAG 2: blank-line documents, retrieval_top_k 5, structured JSON prompt."""
import os

from ..parsing import parse_judgment
from ..prompts import structured_judgment_prompt
from ..records import judge_each
from .common import index_needed, mark_indexed, paragraph_docs


SETTINGS = {
    "embedding_model": "bge-m3",
    "retrieval_top_k": 5,
    "linking_top_k": 5,
    "max_qa_steps": 3,
    "qa_top_k": 5,
    "graph_type": "facts_and_sim_passage_node_unidirectional",
}


def judge(cases, law_text, case_text, extra_text, save_dir, llm_base_url, llm_model, embedding_model, reindex, api_key):
    try:
        from hipporag import HippoRAG
        from hipporag.utils.config_utils import BaseConfig
    except ImportError as exc:
        raise ImportError("HippoRAG 2 uses the hipporag package") from exc

    settings = dict(SETTINGS)
    settings["llm_model"] = llm_model
    settings["llm_base_url"] = llm_base_url
    rebuild = index_needed(save_dir, reindex, settings)
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    hipporag = HippoRAG(
        global_config=BaseConfig(
            save_dir=save_dir,
            llm_name=llm_model,
            llm_base_url=llm_base_url,
            embedding_model_name=embedding_model,
            retrieval_top_k=SETTINGS["retrieval_top_k"],
            linking_top_k=SETTINGS["linking_top_k"],
            max_qa_steps=SETTINGS["max_qa_steps"],
            qa_top_k=SETTINGS["qa_top_k"],
            graph_type=SETTINGS["graph_type"],
            force_index_from_scratch=rebuild,
        )
    )
    if rebuild:
        hipporag.index(docs=paragraph_docs(law_text, case_text, extra_text))
        hipporag.initialize_graph()
        mark_indexed(save_dir, settings)

    def answer(name, fact):
        prompt = structured_judgment_prompt(name, fact)
        retrieval_results = hipporag.retrieve(queries=[prompt], num_to_retrieve=SETTINGS["retrieval_top_k"])
        qa_results = hipporag.rag_qa(retrieval_results)
        raw = qa_results[0][0].answer.strip()
        return {
            "name": name,
            "judge_result": parse_judgment(raw, strip_thinking=False),
            "retrieved_laws": str(retrieval_results[0].to_dict()["docs"]),
            "raw_output": raw,
        }

    return judge_each(cases, answer)
