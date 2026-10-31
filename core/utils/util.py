from core.agents.adjudicator import adjudicate
from core.agents.auditor import audit
from core.agents.researcher import ontology_query_text, research
from core.graph_construct.feature_graph import (
    flat_article_retrieval,
    flat_case_retrieval,
)
from core.preprocess.case_seg import segment_case_text_withname


def naive_retrieve(chatbot, cases, law_to_crime, cases_db):
    features = cases["feature"]
    retrieved_facts = flat_case_retrieval(
        chatbot, ontology_query_text(features), top_k=5)

    if not retrieved_facts:
        return None, None

    retrieved_laws = flat_article_retrieval(
        ontology_query_text(features), top_k=5)
    retrieved_laws = [str(law['entry']) for law in retrieved_laws]
    for item in retrieved_facts:
        for case in cases_db:
            if case["id"] == item["caseId"]:
                item["crime"] = case["crime"]
                item["law"] = case["law"]
                retrieved_laws.extend(case["law"])
                break
    retrieved_laws = list(set(retrieved_laws))
    final_retrieved_laws = []
    for x in retrieved_laws:
        if (int(x) < 102):
            continue
        try:
            for item in law_to_crime:
                if item["id"] == int(x):
                    for entry in item["items"]:
                        final_retrieved_laws.append(
                            {"id": item["id"], "text": entry["text"], "crime": entry["crime"], "judge_dep": entry["judge_dep"], "related_laws": entry["related_laws"]})
                    break
        except IndexError:
            continue

    return final_retrieved_laws, retrieved_facts


def locate_law(law, laws):
    for item in laws:
        if law["id"] == item["id"]:
            return item
    return law["text"]


def analyze_case(chatbot, case, law_to_crime, cases_db, retrieve_config):
    case_by_defendant = segment_case_text_withname(
        chatbot, case["fact"][:1024], case["name"])
    for item in case_by_defendant:
        evidence, retrieved_laws, retrieved_facts = research(
            chatbot, item, law_to_crime, cases_db, retrieve_config)
        if not (retrieved_laws and retrieved_facts):
            continue
        verified_articles, verified_cases = audit(
            chatbot, item, retrieved_laws, retrieved_facts)
        judge_result = adjudicate(
            chatbot, item, verified_articles, verified_cases)
        item["judge_result"] = judge_result
        item["retrieved_laws"] = retrieved_laws
        item["retrieved_facts"] = retrieved_facts
        item["original_retrieved_res"] = evidence
        item["used_laws"] = verified_articles
        item["used_facts"] = verified_cases

    return case_by_defendant
