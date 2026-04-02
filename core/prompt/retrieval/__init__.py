"""Retrieval related prompts"""

RETRIEVE_LAW_PROMPT = """
You are a professional criminal judge who needs to analyze up to three possible charges based on the case facts. Please follow the following requirements:

**Analysis requirements:**
1. Comprehensively analyze each behavioral aspect in the case facts
2. Consider all possible criminal charges that the behavior may violate
3. Include basic charges and special charges

**Output requirements:**
- Only output Python list format: ["charge1", "charge2", ...]
- Sort by charge possibility from high to low
- Give at most three charges
- No explanation, no numbering, no additional content

**Case information:**
Defendant: {name}
Case facts:
```
{fact}
```
Now please output:
"""

__all__ = [
    "RETRIEVE_LAW_PROMPT",
]
