"""Index files and corpus splits shared by the retrieval baselines."""
import json
import os
from typing import Any, Dict, List, Optional


def _marker(save_dir: str) -> str:
    return os.path.join(save_dir, "indexed.json")


def index_needed(save_dir: str, reindex: bool, settings: Optional[Dict[str, Any]] = None) -> bool:
    path = _marker(save_dir)
    if reindex or not os.path.exists(path):
        return True
    if settings is None:
        return False
    with open(path, "r", encoding="utf-8") as handle:
        stored = json.load(handle)
    return stored.get("settings") != settings


def mark_indexed(save_dir: str, settings: Optional[Dict[str, Any]] = None):
    os.makedirs(save_dir, exist_ok=True)
    payload: Dict[str, Any] = {} if settings is None else {"settings": settings}
    with open(_marker(save_dir), "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def paragraph_docs(*texts: str) -> List[str]:
    combined = ""
    for text in texts:
        if not text:
            continue
        if combined:
            combined += "\n"
        combined += text
    return combined.split("\n\n")


def embed_host(embedding_url: str) -> str:
    suffix = "/api/embed"
    if embedding_url.endswith(suffix):
        return embedding_url[: -len(suffix)]
    return embedding_url


def openai_client(base_url: str, api_key: str):
    from openai import OpenAI

    return OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY") or "EMPTY", base_url=base_url)
