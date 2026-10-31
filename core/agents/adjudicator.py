"""Adjudicator agent: synthesize the verified articles into a judgment."""
from core.judge.judge_crime import synthesize_judgment


def adjudicate(chatbot, item, verified_articles, verified_cases):
    """Predict charge, article, and term from the verified article set."""
    return synthesize_judgment(
        chatbot,
        verified_articles,
        verified_cases,
        f"被告人：{item['name']}，描述：{item['description']}",
    )
