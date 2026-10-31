"""Researcher agent: ontology alignment and the three retrieval operators.

R(q) = R_sem(q) ∪ R_com(q) ∪ R_chg(q)
"""
from core.graph_construct.feature_graph import charge_anchored_articles, retrieve_evidence
from core.preprocess.get_features import get_features
from core.prompt import get_prompt


def ontology_query_text(description):
    """Serialize the four ontology dimensions into the text used for embedding."""
    res = ""
    res += "被告信息：" + ", ".join(description.get("defendant_info", [])) + "。"
    res += "犯罪行为：" + ", ".join(description.get("criminal_acts", [])) + "。"
    res += "犯罪对象特征：" + \
        ", ".join(description.get("victim_property_details", [])) + "。"
    res += "犯罪意图及悔罪表现：" + \
        ", ".join(description.get("intent_remorse", [])) + "。"
    return res


def charge_anchored_retrieval(chatbot, case):
    """R_chg: predict charges, then collect the articles linked to those charges."""
    fact = case["description"][:1024]
    name = case["name"]
    response = chatbot.generate_response(
        get_prompt("RETRIEVE_LAW_PROMPT").format(name=name, fact=fact), max_length=256)
    try:
        first = response.find('[')
        last = response.rfind(']') + 1
        charges = eval(response[first:last])
    except (ValueError, SyntaxError):
        return []
    articles = charge_anchored_articles(charges, top_k=1)
    return articles


def gather_evidence(chatbot, cases, law_to_crime, cases_db, retrieve_config):
    features = cases["feature"]
    evidence, retrieved_facts, retrieved_laws = retrieve_evidence(
        chatbot, ontology_query_text(features), retrieve_config)

    if not retrieved_facts:
        return {}, [], []

    anchored_articles = []
    if retrieve_config["charge_anchored"]:
        anchored_articles = charge_anchored_retrieval(chatbot, cases)
        evidence["charge_anchored"] = anchored_articles
    else:
        anchored_articles = []
    retrieved_laws = retrieved_laws + anchored_articles
    for item in retrieved_facts:
        for case in cases_db:
            if case["id"] == item["caseId"]:
                item["crime"] = case["crime"]
                item["law"] = case["law"]
                break
    final_retrieved_laws = []
    seen_law_ids = set()
    for law in retrieved_laws:
        if law["id"] in seen_law_ids:
            continue
        seen_law_ids.add(law["id"])
        law["judge_dep"] = eval(law["judge_dep"])
        law["related_laws"] = eval(law["related_laws"])
        final_retrieved_laws.append(law)

    return evidence, final_retrieved_laws, retrieved_facts


def research(chatbot, item, law_to_crime, cases_db, retrieve_config):
    """Align the defendant narrative to ontology features, then retrieve evidence."""
    item["feature"] = get_features(chatbot, item)
    return gather_evidence(chatbot, item, law_to_crime, cases_db, retrieve_config)
