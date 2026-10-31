"""Shared embedding backend for graph construction and retrieval."""
import requests


_embedding_api_url = "http://localhost:11434/api/embed"
_embedding_model = "bge-m3"


def configure_embedding(api_url=None, model=None):
    """Configure the embedding backend used by graph construction and retrieval."""
    global _embedding_api_url, _embedding_model
    if api_url:
        _embedding_api_url = api_url
    if model:
        _embedding_model = model


def get_embedding(text):
    data = {
        "model": _embedding_model,
        "input": text
    }

    response = requests.post(_embedding_api_url, json=data, timeout=120)
    response.raise_for_status()
    result = response.json()

    embeddings = result.get('embeddings')
    if not embeddings or not embeddings[0]:
        raise ValueError(
            f"Embedding backend returned no vectors: {_embedding_api_url} "
            f"(model={_embedding_model})"
        )
    return embeddings[0]
