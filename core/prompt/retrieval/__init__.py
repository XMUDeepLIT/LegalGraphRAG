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

RETRIEVE_LAW_PROMPT_ZH = """
你是一名专业的刑事法官，需要根据案件事实分析至多三个可能成立的罪名。请遵循以下要求：

**分析要求：**
1. 全面分析案件事实中的每个行为环节
2. 考虑行为可能触犯的所有刑法罪名
3. 包括基本罪名和特殊罪名

**输出要求：**
- 仅输出Python列表格式：["罪名1", "罪名2", ...]
- 按罪名可能性从高到低排序
- 给出至多三个罪名
- 不解释、不编号、不添加额外内容

**案件信息：**
被告：{name}
案件事实：
```
{fact}
```
现在请你输出：
"""

__all__ = [
    "RETRIEVE_LAW_PROMPT",
    "RETRIEVE_LAW_PROMPT_ZH",
]
