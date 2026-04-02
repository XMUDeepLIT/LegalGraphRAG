"""Unified import interface for prompt module"""

# Preprocess prompts
from .preprocess import (
    GET_FEATURES_PROMPT,
    CASE_SEG_PROMPT,
    PRE_JUDGE_PROMPT,
)

# Judge prompts
from .judge import (
    JUDGE_LAW_PROMPT,
    JUDGE_LAW_PROMPT0,
    JUDGE_LAW_PROMPT1,
    JUDGE_CRIME_PROMPT,
    JUDGE_CRIME_ALL_PROMPT,
)

# Retrieval prompts
from .retrieval import (
    RETRIEVE_LAW_PROMPT,
)

# Graph prompts
from .graph import (
    SUMMARIZE_TEXTS_PROMPT,
    RERANK_CLUSTERS_PROMPT_TEMPLATE,
    RERANK_PROMPT_TEMPLATE,
)

__all__ = [
    # Preprocess
    "GET_FEATURES_PROMPT",
    "CASE_SEG_PROMPT",
    "PRE_JUDGE_PROMPT",
    # Judge
    "JUDGE_LAW_PROMPT",
    "JUDGE_LAW_PROMPT0",
    "JUDGE_LAW_PROMPT1",
    "JUDGE_CRIME_PROMPT",
    "JUDGE_CRIME_ALL_PROMPT",
    # Retrieval
    "RETRIEVE_LAW_PROMPT",
    # Graph
    "SUMMARIZE_TEXTS_PROMPT",
    "RERANK_CLUSTERS_PROMPT_TEMPLATE",
    "RERANK_PROMPT_TEMPLATE",
]

