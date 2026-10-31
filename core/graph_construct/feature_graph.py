"""HierarGraph construction and retrieval.

Fact, ontology, and rule layers write into one in-memory graph.
"""
from tqdm import tqdm

from .embedding import configure_embedding, get_embedding
from .fact_graph import (
    charge_anchored_articles,
    flat_article_retrieval,
    store_nodes_with_embeddings,
)
from .graph_db import GraphDBManager
from .ontology_graph import (
    build_communities,
    community_expansion_retrieval,
    flat_case_retrieval,
    run_knn,
    semantic_match_retrieval,
)
from .rule_graph import update_insights_in_graph


def construct_hierargraph(model, nodes_data):
    GraphDBManager.initialize()

    case_nodes_data, law_nodes_data, crime_nodes_data = nodes_data[
        'case'], nodes_data['law'], nodes_data['crime']
    # 为每个节点生成嵌入向量
    for i, node in enumerate(tqdm(case_nodes_data, desc="Generating embeddings")):
        node_embedding = get_embedding(node['description'])
        if node_embedding is not None:
            case_nodes_data[i]['embedding'] = node_embedding
        else:
            print(f"Failed to generate embedding for node {node['id']}")

    for i, node in enumerate(tqdm(law_nodes_data, desc="Generating embeddings")):
        node_embedding = get_embedding(node['description'])
        if node_embedding is not None:
            law_nodes_data[i]['embedding'] = node_embedding
        else:
            print(f"Failed to generate embedding for node {node['id']}")

    for i, node in enumerate(tqdm(crime_nodes_data, desc="Generating embeddings")):
        node_embedding = get_embedding(node['description'])
        if node_embedding is not None:
            crime_nodes_data[i]['embedding'] = node_embedding
        else:
            print(f"Failed to generate embedding for node {node['id']}")

    # 存储节点和嵌入
    store_nodes_with_embeddings(nodes_data)

    # kNN edges, then community nodes
    run_knn(top_k=3)
    build_communities(model)


def retrieve_evidence(model, query_text, retrieve_config):
    """R(q) = semantic match ∪ community expansion. Charge anchoring is added by the Researcher."""
    query_embedding = get_embedding(query_text)
    if query_embedding is None:
        return {}, [], []

    if retrieve_config["community_expansion"]:
        community_clusters, community_cases, community_articles = community_expansion_retrieval(
            model, query_embedding, query_text, top_k=retrieve_config["community_expansion_top_k"])
    else:
        community_clusters, community_cases, community_articles = [], [], []
    if retrieve_config["semantic_match"]:
        semantic_cases, semantic_articles = semantic_match_retrieval(
            model, query_embedding, query_text, top_k=retrieve_config["semantic_match_top_k"])
    else:
        semantic_cases, semantic_articles = [], []

    # 整理结果
    result_cases = []
    seen_ids_cases = set()  # 用于去重
    result_laws = []
    seen_ids_laws = set()  # 用于去重

    # Community expansion cases are merged before semantic-match cases.
    if community_cases:
        neighbors = community_cases
        for neighbor in neighbors:
            if neighbor['id'] and neighbor['id'] not in seen_ids_cases:
                result_cases.append({
                    'id': neighbor['id'],
                    'description': neighbor['description'],
                    'caseId': neighbor['caseId'],
                    'rank': neighbor['rank']
                })
                seen_ids_cases.add(neighbor['id'])

    if community_articles:
        neighbors = community_articles
        for neighbor in neighbors:
            if neighbor['id'] and neighbor['id'] not in seen_ids_laws:
                result_laws.append({
                    'id': neighbor['id'],
                    'entry': neighbor['entry'],
                    'description': neighbor['description'],
                    'crimes': neighbor['crimes'],
                    'judge_dep': neighbor['judge_dep'],
                    'related_laws': neighbor['related_laws'],
                })
                seen_ids_laws.add(neighbor['id'])

    if semantic_cases:
        neighbors = semantic_cases
        # 添加邻居节点
        for neighbor in neighbors:
            if neighbor['id'] and neighbor['id'] not in seen_ids_cases:
                result_cases.append({
                    'id': neighbor['id'],
                    'description': neighbor['description'],
                    'caseId': neighbor['caseId'],
                    'rank': neighbor['rank']
                })
                seen_ids_cases.add(neighbor['id'])

    if semantic_articles:
        neighbors = semantic_articles
        for neighbor in neighbors:
            if neighbor['id'] and neighbor['id'] not in seen_ids_laws:
                result_laws.append({
                    'id': neighbor['id'],
                    'entry': neighbor['entry'],
                    'description': neighbor['description'],
                    'crimes': neighbor['crimes'],
                    'judge_dep': neighbor['judge_dep'],
                    'related_laws': neighbor['related_laws'],
                })
                seen_ids_laws.add(neighbor['id'])
    original_retrieved_res = {
        "community_expansion": {
            "clusters": community_clusters,
            "cases": community_cases,
            "laws": community_articles
        },
        "semantic_match": {
            "cases": semantic_cases,
            "laws": semantic_articles
        },
        "charge_anchored": []
    }

    return original_retrieved_res, result_cases, result_laws


__all__ = [
    "charge_anchored_articles",
    "configure_embedding",
    "construct_hierargraph",
    "flat_article_retrieval",
    "flat_case_retrieval",
    "get_embedding",
    "retrieve_evidence",
    "update_insights_in_graph",
]
