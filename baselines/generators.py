"""Retrieval generators. Each entry is one backbone the retrieval protocols can call."""
import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class GeneratorSpec:
    key: str
    model_name: str
    base_url: str
    api_key_env: str
    description: str


GENERATORS: Dict[str, GeneratorSpec] = {
    "qwen3": GeneratorSpec(
        key="qwen3",
        model_name="Qwen/Qwen3-8B",
        base_url="http://localhost:8000/v1",
        api_key_env="OPENAI_API_KEY",
        description="Local OpenAI-compatible server. Default retrieval backbone.",
    ),
    "deepseek_v3": GeneratorSpec(
        key="deepseek_v3",
        model_name="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        description="DeepSeek API. The key is DEEPSEEK_API_KEY.",
    ),
    "gpt4o_mini": GeneratorSpec(
        key="gpt4o_mini",
        model_name="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        description="OpenAI API. OPENAI_BASE_URL replaces the base URL when --llm-base-url is omitted.",
    ),
}


@dataclass(frozen=True)
class ResolvedGenerator:
    key: str
    model_name: str
    base_url: str
    api_key: str


def get_generator(key: str) -> GeneratorSpec:
    if key not in GENERATORS:
        known = ", ".join(GENERATORS)
        raise KeyError(f"Unknown retrieval generator {key}. Known generators: {known}")
    return GENERATORS[key]


def resolve_generator(
    key: str,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ResolvedGenerator:
    spec = get_generator(key)
    url = base_url or spec.base_url
    if spec.key == "gpt4o_mini" and not base_url:
        url = os.getenv("OPENAI_BASE_URL", spec.base_url)
    key_value = api_key or os.getenv(spec.api_key_env) or ""
    if not key_value:
        if url.startswith("http://localhost") or url.startswith("http://127.0.0.1"):
            key_value = "EMPTY"
        else:
            raise SystemExit(f"Set {spec.api_key_env}")
    return ResolvedGenerator(
        key=spec.key,
        model_name=model_name or spec.model_name,
        base_url=url,
        api_key=key_value,
    )
