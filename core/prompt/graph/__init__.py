"""Graph construction related prompts"""

SUMMARIZE_TEXTS_PROMPT = """
You are an experienced legal expert. Your task is to abstract high-level, general crime categories from a community composed of criminal cases. Please follow these rules:

1. **Analysis and Induction**: Carefully analyze all input behaviors, identify their common essence and patterns.
2. **High-level Generalization**: The output must be a highly refined single category description, not a listing or restatement of input behaviors.
3. **Output Format**: The output must strictly follow the following format and be only one line: "Criminal Behavior: [General Category]".
4. **Prohibited Items**: Do not output any specific behavioral details, explanatory text, lists, or additional information.

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

__all__ = [
    "SUMMARIZE_TEXTS_PROMPT",
    "RERANK_CLUSTERS_PROMPT_TEMPLATE",
    "RERANK_PROMPT_TEMPLATE",
]
