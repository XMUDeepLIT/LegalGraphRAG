"""Model names, checkpoints, and decoding for the direct baselines."""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ModelSpec:
    key: str
    protocol: str
    model_name: str
    backend: str
    trust_remote_code: bool = False
    enable_thinking: Optional[bool] = False
    max_new_tokens: int = 4096
    strip_think: bool = False
    generate_kwargs: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


def _spec(**kwargs) -> ModelSpec:
    return ModelSpec(**kwargs)


MODELS: Dict[str, ModelSpec] = {
    "qwen3": _spec(
        key="qwen3",
        protocol="structured",
        model_name="Qwen/Qwen3-8B",
        backend="local",
        trust_remote_code=False,
        enable_thinking=False,
        strip_think=True,
        description="Greedy generation. Think spans are removed before JSON parsing.",
    ),
    "internlm3": _spec(
        key="internlm3",
        protocol="structured",
        model_name="internlm/internlm3-8b-instruct",
        backend="local",
        trust_remote_code=True,
        enable_thinking=False,
        description="Greedy generation. The raw generation is parsed as JSON.",
    ),
    "glm4": _spec(
        key="glm4",
        protocol="structured",
        model_name="THUDM/glm-4-9b-chat",
        backend="local",
        trust_remote_code=True,
        enable_thinking=False,
        description="Greedy generation. The raw generation is parsed as JSON.",
    ),
    "deepseek_v3": _spec(
        key="deepseek_v3",
        protocol="structured",
        model_name="deepseek-chat",
        backend="deepseek",
        enable_thinking=None,
        description="API temperature 0.1, max_tokens 4096. The raw generation is parsed as JSON.",
    ),
    "gpt4o_mini": _spec(
        key="gpt4o_mini",
        protocol="structured",
        model_name="gpt-4o-mini",
        backend="openai",
        enable_thinking=None,
        description="API temperature 0.1, max_tokens 4096, using the structured judgment prompt.",
    ),
    "qwen2_5": _spec(
        key="qwen2_5",
        protocol="freeform",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        backend="local",
        trust_remote_code=True,
        enable_thinking=False,
        description="Greedy free-form generation, then DeepSeek extracts the JSON object.",
    ),
    "disc_lawllm": _spec(
        key="disc_lawllm",
        protocol="freeform",
        model_name="ShengbinYue/LawLLM-7B",
        backend="local",
        trust_remote_code=True,
        enable_thinking=False,
        description="Greedy free-form generation, then DeepSeek extracts the JSON object.",
    ),
    "legal_delta": _spec(
        key="legal_delta",
        protocol="freeform",
        model_name="./legal_delta",
        backend="local",
        trust_remote_code=True,
        enable_thinking=None,
        generate_kwargs={"do_sample": False, "temperature": 0.1},
        description="Greedy free-form generation with temperature 0.1, then DeepSeek extracts the JSON object.",
    ),
    "adapt": _spec(
        key="adapt",
        protocol="adapt",
        model_name="ChenlongDeng/ADAPT-Qwen2-7B-CAIL2018-step-8765",
        backend="local",
        trust_remote_code=True,
        enable_thinking=None,
        max_new_tokens=2048,
        generate_kwargs={
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
        },
        description="Sampled ADAPT turn, then DeepSeek extracts the JSON object.",
    ),
}


PROTOCOLS = (
    "structured",
    "freeform",
    "adapt",
    "naive_rag",
    "hipporag2",
    "lightrag",
    "raptor",
    "gretriever",
)


def get_model(key: str) -> ModelSpec:
    if key not in MODELS:
        known = ", ".join(MODELS)
        raise KeyError(f"Unknown baseline model {key}. Known models: {known}")
    return MODELS[key]
