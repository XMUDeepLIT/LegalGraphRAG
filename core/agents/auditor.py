"""Auditor agent: checklist verification and pruning of inapplicable evidence."""
from core.judge.judge_law import verify_article


def prune_unsupported_cases(verified_articles, retrieved_facts):
    """
    过滤retrieved_facts，只保留law中包含至少一个retrieved_laws中条目的fact

    参数:
    verified_articles: 法律条文列表，每个元素包含id字段
    retrieved_facts: 事实列表，每个元素包含law字段（法律id列表）

    返回:
    过滤后的facts列表
    """
    # 提取所有verified_articles的id集合
    law_ids = {str(law['id']) for law in verified_articles}

    # 过滤facts，只保留law字段中至少有一个id在law_ids中的fact
    filtered_facts = [
        fact for fact in retrieved_facts
        if any(law_id in law_ids for law_id in fact.get('law', []))
    ]

    return filtered_facts


def audit(chatbot, item, retrieved_laws, retrieved_facts):
    """Verify each article and drop precedents that only cite rejected articles."""
    verified_articles = []
    for article in retrieved_laws:
        applicable, _ = verify_article(
            chatbot, f"被告人：{item['name']}，描述：{item['description']}", article)
        if applicable:
            verified_articles.append(article)
    verified_cases = prune_unsupported_cases(verified_articles, retrieved_facts)
    return verified_articles, verified_cases
