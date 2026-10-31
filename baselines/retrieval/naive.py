"""Naive RAG: two corpus files, cl100k_base chunks of 1000 tokens with overlap 200, bge-m3, top 5."""
import hashlib
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np

from core.graph_construct.embedding import configure_embedding, get_embedding

from ..parsing import parse_charge_list
from ..prompts import naive_rag_prompt, naive_rag_question
from ..records import judge_each
from .splitter import split_tokens


CHUNK_TOKENS = 1000
CHUNK_OVERLAP = 200
TOKEN_ENCODING = "cl100k_base"
TOP_K = 5
CONTEXT_CHAR_LIMIT = 3072
LLM_TEMPERATURE = 0.7


def chunk_corpora(law_text: str, case_text: str) -> List[str]:
    return split_tokens(law_text, CHUNK_TOKENS, CHUNK_OVERLAP, TOKEN_ENCODING) + split_tokens(
        case_text, CHUNK_TOKENS, CHUNK_OVERLAP, TOKEN_ENCODING
    )


def _digest(law_text: str, case_text: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"{TOKEN_ENCODING}:{CHUNK_TOKENS}:{CHUNK_OVERLAP}".encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(law_text.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(case_text.encode("utf-8"))
    return hasher.hexdigest()


class NaiveIndex:
    def __init__(self, chunks: Sequence[str], embeddings: np.ndarray, digest: str):
        self.chunks = list(chunks)
        self.embeddings = embeddings.astype(np.float32)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = self.embeddings / norms
        self.digest = digest

    @classmethod
    def build(cls, law_text: str, case_text: str, embedding_url: str, embedding_model: str) -> "NaiveIndex":
        configure_embedding(embedding_url, embedding_model)
        chunks = chunk_corpora(law_text, case_text)
        vectors = [get_embedding(chunk) for chunk in chunks]
        return cls(chunks, np.asarray(vectors, dtype=np.float32), _digest(law_text, case_text))

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        with open(directory / "chunks.json", "w", encoding="utf-8") as f:
            json.dump({"digest": self.digest, "chunks": self.chunks}, f, ensure_ascii=False)

    @classmethod
    def load(cls, directory: Path, digest: str) -> "NaiveIndex":
        with open(directory / "chunks.json", "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("digest") != digest:
            raise ValueError("Stored Naive RAG index does not match this corpus and chunking")
        embeddings = np.load(directory / "embeddings.npy")
        return cls(payload["chunks"], embeddings, digest)

    def search(self, query: str, top_k: int = TOP_K) -> List[str]:
        vector = np.asarray(get_embedding(query), dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return []
        scores = self.embeddings @ (vector / norm)
        order = np.argsort(-scores)[:top_k]
        return [self.chunks[int(index)] for index in order]


def load_or_build_index(law_text, case_text, index_dir, embedding_url, embedding_model, rebuild: bool) -> NaiveIndex:
    configure_embedding(embedding_url, embedding_model)
    digest = _digest(law_text, case_text)
    index_path = Path(index_dir)
    if not rebuild and (index_path / "chunks.json").exists():
        try:
            return NaiveIndex.load(index_path, digest)
        except ValueError:
            print(f"rebuilding {index_path}")
    index = NaiveIndex.build(law_text, case_text, embedding_url, embedding_model)
    index.save(index_path)
    return index


def answer_question(question: str, context_chunks: Sequence[str], base_url: str, model_name: str, api_key: str) -> tuple:
    from openai import OpenAI

    full_context = "\n\n".join(context_chunks)
    prompt = naive_rag_prompt(question, full_context[:CONTEXT_CHAR_LIMIT])
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
    )
    answer = response.choices[0].message.content or ""
    return answer, full_context


def judge_cases(cases, index: NaiveIndex, base_url: str, model_name: str, api_key: str):
    def answer(name, fact):
        question = naive_rag_question(name, fact)
        chunks = index.search(question, TOP_K)
        raw, context = answer_question(question, chunks, base_url, model_name, api_key)
        return {"name": name, "crime": parse_charge_list(raw), "context": context, "raw_output": raw}

    return judge_each(cases, answer)
