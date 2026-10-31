"""Judgment prompts used by the Table 2 baselines."""


def structured_judgment_prompt(name: str, fact: str) -> str:
    """JSON judgment prompt used by Qwen3, InternLM3, GLM-4, DeepSeek, LightRAG, and HippoRAG 2."""
    return f"""
你是一个专业的法律分析助手。请根据以下刑事案件事实，为被告判决罪名，并给出依据的法律条文和刑期范围。
注意，被告可能涉及多个罪名。
被告：{name}
案件事实：\n```\n{fact}\n```
- 除了你的推理过程，你的输出必须包含一个 **JSON 对象**，该 JSON 对象的结构如下：
```json
{{
    "charge_name": list(str), // 罪名名称
    "law_article": list(str), // 依据的法律条文，eg. ["第232条", "第233条"]
    "term_of_imprisonment": {{
        "death_penalty": boolean, // 是否适用死刑
        "imprisonment": integer, // 有期徒刑刑期，单位：月
        "life_imprisonment": boolean // 是否适用无期徒刑
    }} // 刑期范围
}}
```
            """


def freeform_judgment_prompt(name: str, fact: str) -> str:
    """Free-form judgment prompt used by Qwen2.5, DISC-LawLLM, and LegalΔ."""
    return f"""
你是一个专业的法律分析助手。请根据以下刑事案件事实，为被告判决罪名，并给出依据的法律条文和刑期范围。
注意，被告可能涉及多个罪名。
被告：{name}
案件事实：\n```\n{fact}\n```
输出：
            """


def extraction_prompt(response_text: str) -> str:
    """DeepSeek prompt that turns a free-form judgment into the JSON object."""
    return f"""请从以下法律分析文本中提取最终判决结果，返回一个Python JSON对象。
注意：你的任务是从文本中提取结果，而不是自己回答。

分析文本：
{response_text}

请严格按照以下格式：
```json
{{
    "charge_name": list(str), // 罪名名称
    "law_article": list(str), // 依据的法律条文，eg. ["第232条", "第233条"]
    "term_of_imprisonment": {{
        "death_penalty": boolean, // 是否适用死刑
        "imprisonment": integer, // 有期徒刑刑期，单位：月
        "life_imprisonment": boolean // 是否适用无期徒刑
    }} // 刑期范围
}}
```

提取结果："""


def adapt_prompt(name: str, fact: str) -> str:
    """Single-turn ADAPT user message."""
    return (
        f"案件描述：{fact}\n"
        f"被告人姓名：{name}\n"
        "请你采用ADAPT框架分析以上案件中该被告人可能被判处的罪名、适用法条和刑期"
    )


def naive_rag_question(name: str, fact: str) -> str:
    """Question placed in the Naive RAG prompt. The model is asked for a charge list."""
    return f"""
            你是一个专业的法律分析助手。请根据以下刑事案件事实，为被告判决罪名，你的回答中必须包含Python list(str)格式输出的罪名名称。
            注意，被告可能涉及多个罪名，请全部列出。若无罪，请仅输出["无罪"]。
            被告：{name}
            案件事实：\n```\n{fact}\n```
            """


def digimon_question(name: str, fact: str) -> str:
    """Question written for DIGIMON. DIGIMON then wraps it as ``Question: ...\\nAnswer: ``."""
    return f"""
你是一个专业的法律分析助手。请根据以下刑事案件事实，为被告判决罪名，你的回答中必须包含Python list(str)格式输出的罪名名称。
注意，被告可能涉及多个罪名，请全部列出。若无罪，请仅输出["无罪"]。
被告：{name}
案件事实：\n```\n{fact}\n```
"""


def raptor_question(name: str, fact: str) -> str:
    """Question passed to RAPTOR. The model is asked for a charge list."""
    return f"""
            你是一名法官。请根据以下刑事案件事实，为被告判决罪名，只需以Python list(str)格式输出罪名名称（以罪字结尾），不要解释。
            被告：{name}
            案件事实：\n```\n{fact}\n```
            """


# LangChain hub prompt `rlm/rag-prompt`, sent as one user message.
NAIVE_RAG_TEMPLATE = (
    "You are an assistant for question-answering tasks. Use the following pieces of "
    "retrieved context to answer the question. If you don't know the answer, just say "
    "that you don't know. Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context} \n"
    "Answer:"
)


def naive_rag_prompt(question: str, context: str) -> str:
    return NAIVE_RAG_TEMPLATE.format(question=question, context=context)


# RAPTOR QA call. The legal question is the `{question}` value.
RAPTOR_QA_SYSTEM = "You are Question Answering Portal"
RAPTOR_QA_USER = "Given Context: {context} Give the best full answer amongst the option to question {question}"
RAPTOR_SUMMARY_SYSTEM = "You are a helpful assistant."
RAPTOR_SUMMARY_USER = "Write a summary of the following, including as many key details as possible: {context}:"
