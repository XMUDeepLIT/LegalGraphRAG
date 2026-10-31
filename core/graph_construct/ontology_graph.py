"""Ontology layer of HierarGraph: feature similarity, kNN, and communities.

Case nodes store feature-text embeddings. This layer adds SIMILAR_TO edges
and Cluster nodes. Community detection uses the full graph, including the
fact-layer edges.
"""
import re

import numpy as np
from tqdm import tqdm

from .embedding import get_embedding
from .fact_graph import laws_for_case
from .graph_db import GraphDBManager


def summarize_texts(model, text):
    from core.prompt import get_prompt
    return model.generate_response(
        get_prompt("SUMMARIZE_TEXTS_PROMPT")
        + get_prompt("SUMMARIZE_TEXTS_INPUT_PREFIX")
        + text,
        max_length=512
    ).strip()


def rerank_clusters(model, clusters, query_text):
    from core.prompt import get_prompt
    cluster_summaries = "\n".join(
        [f"code{c['code']}：{c['summary']}\n" for c in clusters])
    prompt = get_prompt("RERANK_CLUSTERS_PROMPT_TEMPLATE").format(
        cluster_summaries=cluster_summaries,
        query_text=query_text
    )
    response = model.generate_response(prompt, max_length=512)
    match = re.search(r"rank: \[([\d,]+)\]", response)
    if match:
        ranked_codes = [int(code) for code in match.group(1).split(',')]
        return ranked_codes
    return [0]


def rerank(model, query_text, neighbors):
    from core.prompt import get_prompt
    if not neighbors:
        return []

    neighbor_summaries = "\n".join(
        [f"code{n['rank']}：{n['description']}\n" for n in neighbors])
    prompt = get_prompt("RERANK_PROMPT_TEMPLATE").format(
        neighbor_summaries=neighbor_summaries,
        query_text=query_text
    )
    response = model.generate_response(prompt, max_length=512)
    try:
        first_bracket = response.find('[')
        last_bracket = response.rfind(']')
        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            list_str = response[first_bracket:last_bracket + 1]
            ranked_indices = eval(list_str)
        else:
            ranked_indices = []
    except Exception as e:
        print(f"Error parsing response: {e}")
        ranked_indices = []
    if not ranked_indices:
        return neighbors[:3]
    neighbors = [n for n in neighbors if n['rank'] in ranked_indices]
    neighbors = sorted(
        neighbors, key=lambda x: ranked_indices.index(x['rank']))
    return neighbors


def run_knn(top_k=3):
    db = GraphDBManager.get_db()

    # 获取所有Cases节点
    case_nodes = db.get_nodes_by_type('Cases')
    if len(case_nodes) < 2:
        return

    # 获取所有Cases的embeddings
    case_embeddings = []
    case_ids = []
    for node in case_nodes:
        emb = node.get('embedding')
        if emb is not None:
            case_embeddings.append(np.array(emb))
            case_ids.append(node['id'])

    if len(case_embeddings) < 2:
        return

    case_embeddings = np.array(case_embeddings)

    # 计算相似度矩阵并创建SIMILAR_TO关系
    for i in tqdm(range(len(case_ids)), desc="Running KNN"):
        similarities = []
        for j in range(len(case_ids)):
            if i != j:
                sim = db.cosine_similarity(case_embeddings[i], case_embeddings[j])
                similarities.append((j, sim))

        # 选择top_k个最相似的
        similarities.sort(key=lambda x: x[1], reverse=True)
        for j, score in similarities[:top_k]:
            db.add_edge(case_ids[i], case_ids[j], 'SIMILAR_TO', {'score': score})


def build_communities(model):
    db = GraphDBManager.get_db()

    # 运行社区检测和中心性分析
    communities = db.detect_communities()

    # 更新节点的communityId
    for node_id, comm_id in communities.items():
        db.update_node(node_id, {'communityId': comm_id})

    # 计算PageRank和度中心性
    pagerank = db.compute_pagerank()
    degrees = db.compute_degree_centrality()

    # 更新节点的pagerank和degree
    for node_id, score in pagerank.items():
        db.update_node(node_id, {'pagerank': score})
    for node_id, degree in degrees.items():
        db.update_node(node_id, {'degree': degree})

    # 获取所有唯一的社区ID（只考虑Cases节点）
    community_ids = set()
    for node_id, node_info in db.nodes_data.items():
        if node_info['type'] == 'Cases' and node_info['data'].get('communityId') is not None:
            community_ids.add(node_info['data']['communityId'])

    community_ids = sorted(list(community_ids))
    print(f"Detected {len(community_ids)} communities.")

    for community_id in tqdm(community_ids, desc="Creating clusters"):
        # 选择关键节点：综合考虑PageRank和度中心性
        important_nodes = []
        for node_id, node_info in db.nodes_data.items():
            if node_info['type'] == 'Cases' and node_info['data'].get('communityId') == community_id:
                pagerank_score = node_info['data'].get('pagerank', 0)
                degree_score = node_info['data'].get('degree', 0)
                composite_score = pagerank_score * 0.7 + degree_score * 0.3
                important_nodes.append({
                    'description': node_info['data'].get('description', ''),
                    'composite_score': composite_score
                })

        important_nodes.sort(key=lambda x: x['composite_score'], reverse=True)
        descriptions = [node['description'] for node in important_nodes[:10]]

        if not descriptions:
            continue
        descriptions = '\n'.join(descriptions)
        print(descriptions)

        # 获取该社区中连接案件最多的5个罪名
        # 统计该社区中每个crime的案件数
        crime_counts = {}
        for node_id, node_info in db.nodes_data.items():
            if node_info['type'] == 'Cases' and node_info['data'].get('communityId') == community_id:
                # 找到该case关联的laws，然后找到laws关联的crimes
                law_neighbors = db.get_neighbors(node_id, 'RELATES_TO_LAW')
                for law_id in law_neighbors:
                    crime_neighbors = db.get_neighbors(law_id, 'RELATED_CRIME')
                    for crime_id in crime_neighbors:
                        crime_data = db.get_node(crime_id)
                        if crime_data:
                            crime_name = crime_data.get('description', '')
                            crime_counts[crime_name] = crime_counts.get(crime_name, 0) + 1

        top_crimes = sorted(crime_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_crimes = [{'crime_name': name, 'case_count': count} for name, count in top_crimes]

        # 构建更聚焦的提示
        crime_context = ""
        if top_crimes:
            crime_descriptions = [
                f"{crime['crime_name']}({crime['case_count']}个案件)" for crime in top_crimes]
            crime_context = "主要涉及罪名: " + "、".join(crime_descriptions)

        enhanced_prompt = f"""
社区基本信息:
- {crime_context}

关键案例描述:
{descriptions}
"""

        summary_data = summarize_texts(model, enhanced_prompt)
        community_embedding = get_embedding(summary_data)

        # 存储top罪名信息到Cluster节点
        top_crime_names = [crime['crime_name'] for crime in top_crimes]
        top_crime_counts = [crime['case_count'] for crime in top_crimes]

        # 创建Cluster节点并存储更多元数据
        cluster_id = str(community_id)
        db.add_node(
            cluster_id,
            'Cluster',
            {
                'summary': summary_data,
                'embedding': community_embedding,
                'top_crimes': top_crime_names,
                'top_crime_counts': top_crime_counts
            }
        )

        # 连接Cluster和Node
        for node_id, node_info in db.nodes_data.items():
            if node_info['type'] == 'Cases' and node_info['data'].get('communityId') == community_id:
                db.add_edge(node_id, cluster_id, 'BELONGS_TO')


def _cases_and_laws(neighbors):
    cases = []
    laws = []
    for neighbor in neighbors:
        laws.extend(laws_for_case(neighbor['id']))
        cases.append(neighbor)
    return cases, laws


def community_expansion_retrieval(model, query_embedding, query_text, top_k=5):
    db = GraphDBManager.get_db()

    # 先找最相似的Cluster
    cluster_results = db.find_similar_nodes(query_embedding, 'Cluster', top_k=5)

    if not cluster_results:
        return [], [], []

    clusters = []
    for ids, record in enumerate(cluster_results):
        clusters.append(
            {'code': ids, 'cluster_id': record['id'], 'summary': record.get('summary', '')})

    cluster_ids = rerank_clusters(model, clusters, query_text)
    cluster_ids = [
        c for c in cluster_ids if 0 <= c and c < len(clusters)]
    if not cluster_ids:
        cluster_ids = [0]

    neighbors = []
    for cluster_id in cluster_ids[:2]:
        cluster_node_id = clusters[cluster_id]['cluster_id']

        # 在该Cluster中找最相似的Node
        # 先找到属于该cluster的所有Cases节点
        cluster_cases = []
        for node_id, node_info in db.nodes_data.items():
            if node_info['type'] == 'Cases':
                # 检查是否有BELONGS_TO关系指向该cluster
                neighbors_list = db.get_neighbors(node_id, 'BELONGS_TO')
                if cluster_node_id in neighbors_list:
                    node_data = node_info['data'].copy()
                    node_data['id'] = node_id
                    cluster_cases.append(node_data)

        # 计算相似度并排序
        case_similarities = []
        for case_data in cluster_cases:
            emb = case_data.get('embedding')
            if emb is not None:
                sim = db.cosine_similarity(query_embedding, np.array(emb))
                case_similarities.append((case_data, sim))

        case_similarities.sort(key=lambda x: x[1], reverse=True)

        for case_data, similarity in case_similarities[:top_k]:
            neighbors.append({
                'id': case_data['id'],
                'description': case_data.get('description', ''),
                'caseId': case_data.get('caseId', ''),
                'similarity': similarity
            })

    neighbors = sorted(
        neighbors, key=lambda x: x['similarity'], reverse=True)
    for ids, neighbor in enumerate(neighbors):
        neighbor['rank'] = ids + 1
    neighbors = rerank(model, query_text, neighbors)

    cases, laws = _cases_and_laws(neighbors)
    return clusters, cases, laws


def semantic_match_retrieval(model, query_embedding, query_text, top_k=5):
    db = GraphDBManager.get_db()

    # 直接在所有Cases节点中查找最相似的节点
    neighbor_results = db.find_similar_nodes(query_embedding, 'Cases', top_k=top_k)

    if not neighbor_results:
        return [], []

    neighbors = []
    for record in neighbor_results:
        neighbors.append({
            'id': record['id'],
            'description': record.get('description', ''),
            'caseId': record.get('caseId', ''),
            'similarity': record['similarity']
        })

    neighbors = sorted(
        neighbors, key=lambda x: x['similarity'], reverse=True)
    for ids, neighbor in enumerate(neighbors):
        neighbor['rank'] = ids + 1
    neighbors = rerank(model, query_text, neighbors)

    return _cases_and_laws(neighbors)


def flat_case_retrieval(model, query_text, top_k=3):
    query_embedding = get_embedding(query_text)
    if query_embedding is None:
        return []

    db = GraphDBManager.get_db()
    neighbor_results = db.find_similar_nodes(query_embedding, 'Cases', top_k=top_k)

    if not neighbor_results:
        return []

    neighbors = []
    for record in neighbor_results:
        neighbors.append({
            'id': record['id'],
            'description': record.get('description', ''),
            'caseId': record.get('caseId', ''),
            'similarity': record['similarity']
        })
    neighbors = sorted(
        neighbors, key=lambda x: x['similarity'], reverse=True)
    for ids, neighbor in enumerate(neighbors):
        neighbor['rank'] = ids + 1

    return neighbors
