"""RAPTOR: Qwen3-8B summary and answer, bge-m3 embeddings, 1200/100 token chunks."""
import os

from ..parsing import parse_charge_list
from ..prompts import RAPTOR_QA_SYSTEM, RAPTOR_QA_USER, RAPTOR_SUMMARY_SYSTEM, RAPTOR_SUMMARY_USER, raptor_question
from ..records import judge_each
from .common import index_needed, mark_indexed, openai_client
from .splitter import split_tokens


SETTINGS = {
    "embedding_model": "bge-m3",
    "chunk_token_size": 1200,
    "chunk_overlap_token_size": 100,
    "num_layers": 5,
    "max_length_in_cluster": 3500,
    "threshold": 0.1,
    "token_encoding": "cl100k_base",
}


def judge(cases, law_text, case_text, extra_text, save_dir, llm_base_url, llm_model, embedding_url, embedding_model, reindex, api_key):
    try:
        from raptor import BaseEmbeddingModel, BaseQAModel, BaseSummarizationModel
        from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
    except ImportError as exc:
        raise ImportError("RAPTOR uses the raptor package") from exc

    from core.graph_construct.embedding import configure_embedding, get_embedding

    configure_embedding(embedding_url, embedding_model)
    client = openai_client(llm_base_url, api_key)

    class Summarizer(BaseSummarizationModel):
        def summarize(self, context, max_tokens=150):
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": RAPTOR_SUMMARY_SYSTEM},
                    {"role": "user", "content": RAPTOR_SUMMARY_USER.format(context=context)},
                ],
                max_tokens=max_tokens,
            )
            return (response.choices[0].message.content or "").strip()

    class QAModel(BaseQAModel):
        def answer_question(self, context, question):
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": RAPTOR_QA_SYSTEM},
                    {"role": "user", "content": RAPTOR_QA_USER.format(context=context, question=question)},
                ],
                temperature=0,
            )
            return response.choices[0].message.content or ""

    class Embedder(BaseEmbeddingModel):
        def create_embedding(self, text):
            import numpy as np

            try:
                return get_embedding(text)
            except Exception:
                return np.zeros(1024).tolist()

    embedder = Embedder()
    summarizer = Summarizer()
    from raptor import ClusterTreeConfig

    config = RetrievalAugmentationConfig(
        tree_builder_config=ClusterTreeConfig(
            max_tokens=SETTINGS["chunk_token_size"],
            num_layers=SETTINGS["num_layers"],
            summarization_model=summarizer,
            embedding_models={"EMB": embedder},
            cluster_embedding_model="EMB",
            clustering_params={
                "max_length_in_cluster": SETTINGS["max_length_in_cluster"],
                "threshold": SETTINGS["threshold"],
            },
        ),
        summarization_model=summarizer,
        qa_model=QAModel(),
        embedding_model=embedder,
    )
    tree_path = os.path.join(save_dir, "tree")
    settings = dict(SETTINGS)
    settings["llm_model"] = llm_model
    settings["llm_base_url"] = llm_base_url
    if index_needed(save_dir, reindex, settings) or not os.path.exists(tree_path):
        combined = law_text
        if case_text:
            combined += "\n" + case_text
        if extra_text:
            combined += "\n" + extra_text
        ra = RetrievalAugmentation(config=config)
        import raptor.tree_builder as tree_builder

        original_split = tree_builder.split_text

        def paper_split(text, tokenizer, max_tokens, overlap=0):
            return split_tokens(
                text,
                SETTINGS["chunk_token_size"],
                SETTINGS["chunk_overlap_token_size"],
                SETTINGS["token_encoding"],
            )

        tree_builder.split_text = paper_split
        try:
            ra.add_documents(combined)
        finally:
            tree_builder.split_text = original_split
        os.makedirs(save_dir, exist_ok=True)
        ra.save(tree_path)
        mark_indexed(save_dir, settings)
    else:
        ra = RetrievalAugmentation(config=config, tree=tree_path)

    def answer(name, fact):
        raw = ra.answer_question(question=raptor_question(name, fact))
        return {"name": name, "crime": parse_charge_list(raw), "raw_output": raw}

    return judge_each(cases, answer)
