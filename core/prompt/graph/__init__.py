"""Graph construction related prompts"""

SUMMARIZE_TEXTS_PROMPT = """
You are an experienced legal expert. Your task is to abstract high-level, general crime categories from a community composed of criminal cases. Please follow these rules:

1. **Analysis and Induction**: Carefully analyze all input behaviors, identify their common essence and patterns.
2. **High-level Generalization**: The output must be a highly refined single category description, not a listing or restatement of input behaviors.
3. **Output Format**: The output must strictly follow the following format and be only one line: "Criminal Behavior: [General Category]".
4. **Prohibited Items**: Do not output any specific behavioral details, explanatory text, lists, or additional information.

    """

SUMMARIZE_TEXTS_PROMPT_ZH = """
你是一个经验丰富的法律专家，任务是从一个由刑事案件组成的社区中抽象出高层、概括性的犯罪类别。请遵循以下规则：

1. **分析与归纳**：仔细分析所有输入行为，识别它们的共同本质和模式。
2. **高度概括**：输出必须是高度精炼的单一类别描述，而不是对输入行为的罗列或复述。
3. **输出格式**：输出必须严格遵循以下格式，且仅为一行：“犯罪行为：[概括性类别]”。
4. **禁止事项**：禁止输出任何具体的行为细节、解释性文字、列表或额外信息。

    """

RERANK_CLUSTERS_PROMPT_TEMPLATE = """
You are a professional legal analysis assistant, good at mapping specific criminal case descriptions to high-level crime categories.

Processing requirements:
- Carefully analyze key behaviors, subjective intent, and legal relationships involved in the case
- Compare with feature descriptions of each category and evaluate the matching degree of each category
- Sort all categories by relevance from high to low
- Output format: e.g., "rank: [3,1,2]", no additional text allowed

Now process the following information:

Available category summaries:
{cluster_summaries}

Case description to be analyzed:
{query_text}
    """

RERANK_CLUSTERS_PROMPT_TEMPLATE_ZH = """
你是一个专业的法律分析助手，擅长将具体的刑事案件描述映射到高层次犯罪类别。

处理要求：
- 仔细分析案件中的关键行为、主观意图和所涉法律关系
- 对照各个类别的特征描述，评估每个类别的匹配度
- 对所有类别按照相关性从高到低进行排序
- 输出格式：比如："rank: [3,1,2]"，禁止任何额外文字

现在处理以下信息：

可用类别摘要：
{cluster_summaries}

待分析案件描述：
{query_text}
    """

RERANK_PROMPT_TEMPLATE = """
You are a professional legal analysis assistant. I need you to re-rank these similar cases according to their relevance to the original case, and give the three most relevant cases.

Task description:
1. Analyze the degree of association between each similar case (codeX) and the original case
2. Re-arrange these similar cases in order of relevance from high to low
3. Give the numbers of at most three most relevant cases
4. Output format must be an integer list, containing only the numeric part of case numbers

Output requirements:
- Only output one integer list, format like: [3, 1, 2]
- Numbers in the list correspond to similar case numbers (numbers after code)
- The case ranked first is the most relevant to the original case

Similar case information:
{neighbor_summaries}

Original case content:
{query_text}

Please output the re-ranked case number list:
    """

RERANK_PROMPT_TEMPLATE_ZH = """
你是一个专业的法律分析助手，我需要你根据相似案件与原始案件的相关度，对这些相似案件进行重新排序，给出最相关的三个案件。

任务描述：
1. 分析每个相似案件（codeX）与原始案件的关联程度
2. 按照相关度从高到低的顺序重新排列这些相似案件
3. 给出最多三个最相关案件的编号
4. 输出格式必须为整数列表，只包含案件编号的数字部分

输出要求：
- 只输出一个整数列表，格式如：[3, 1, 2]
- 列表中的数字对应相似案件的编号（code后面的数字）
- 排在第一位的案件是与原始案件最相关的

相似案件信息：
{neighbor_summaries}

原始案件内容：
{query_text}

请输出重新排序后的案件编号列表：
    """

__all__ = [
    "SUMMARIZE_TEXTS_PROMPT",
    "SUMMARIZE_TEXTS_PROMPT_ZH",
    "RERANK_CLUSTERS_PROMPT_TEMPLATE",
    "RERANK_CLUSTERS_PROMPT_TEMPLATE_ZH",
    "RERANK_PROMPT_TEMPLATE",
    "RERANK_PROMPT_TEMPLATE_ZH",
]
