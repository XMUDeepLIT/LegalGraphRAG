import numpy as np
import requests
import re
from .graph_db import GraphDBManager
from tqdm import tqdm


def get_embedding(text):
    url = "http://localhost:11434/api/embed"
    data = {
        "model": "bge-m3",
        "input": text
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        result = response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

    return result.get('embeddings', [[]])[0]


def summarize_texts(model, text):
    from core.prompt import SUMMARIZE_TEXTS_PROMPT
    return model.generate_response(SUMMARIZE_TEXTS_PROMPT + "\n**Now process the following input data**: \n" + text, max_length=512).strip()


def rerank_clusters(model, clusters, query_text):
    from core.prompt import RERANK_CLUSTERS_PROMPT_TEMPLATE
    cluster_summaries = "\n".join(
        [f"code{c['code']}：{c['summary']}\n" for c in clusters])
    prompt = RERANK_CLUSTERS_PROMPT_TEMPLATE.format(
        cluster_summaries=cluster_summaries,
        query_text=query_text
    )
    response = model.generate_response(prompt, max_length=512)
    match = re.search(r"rank: \[([\d,]+)\]", response)
    # print(f"模型回复: {response}")
    if match:
        ranked_codes = [int(code) for code in match.group(1).split(',')]
        return ranked_codes
    return [0]


def rerank(model, query_text, neighbors):
    from core.prompt import RERANK_PROMPT_TEMPLATE
    if not neighbors:
        return []

    neighbor_summaries = "\n".join(
        [f"code{n['rank']}：{n['description']}\n" for n in neighbors])
    prompt = RERANK_PROMPT_TEMPLATE.format(
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


def store_nodes_with_embeddings(nodes_data):
    """
    nodes_data: Dict with keys 'case', 'law', 'crime', each containing a list of node dicts
    """
    store_nodes(nodes_data)
    build_relationships()


def store_nodes(nodes_data):
    """
    存储节点到内存图数据库
    nodes_data: Dict with keys 'case', 'law', 'crime', each containing a list of node dicts
    """
    db = GraphDBManager.get_db()
    case_nodes_data, law_nodes_data, crime_nodes_data = nodes_data[
        'case'], nodes_data['law'], nodes_data['crime']

    # 存储Case节点
    for node in tqdm(case_nodes_data, desc="Storing case nodes"):
        db.add_node(
            node['id'], 
            'Cases',
            {
                'description': node.get('description'),
                'embedding': node.get('embedding'),
                'caseId': node.get('caseId'),
                'crime': node.get('crime'),
                'law': node.get('law')
            }
        )

    # 存储Law节点
    for node in tqdm(law_nodes_data, desc="Storing law nodes"):
        db.add_node(
            node['id'],
            'Laws',
            {
                'entry': node.get('entry'),
                'description': node.get('description'),
                'embedding': node.get('embedding'),
                'crimes': node.get('crimes'),
                'judge_dep': node.get('judge_dep'),
                'related_laws': node.get('related_laws'),
                'insights': ''
            }
        )

    # 存储Crime节点
    for node in tqdm(crime_nodes_data, desc="Storing crime nodes"):
        db.add_node(
            node['id'],
            'Crimes',
            {
                'description': node.get('description'),
                'embedding': node.get('embedding')
            }
        )


def build_relationships():
    """
    在图中建立节点之间的关系
    """
    db = GraphDBManager.get_db()
    
    # 创建Case到Law的关系（基于entry匹配）
    # 从图中检索所有Case节点及其law属性
    case_nodes = db.get_nodes_by_type('Cases')
    
    for case_node in tqdm(case_nodes, desc="Linking cases to laws"):
        case_id = case_node['id']
        law_entries = case_node.get('law')
        
        if not law_entries:
            continue

        for law_entry in law_entries:
            # 找到对应的Law节点
            law_found = False
            for node_id, node_info in db.nodes_data.items():
                if node_info['type'] == 'Laws' and node_info['data'].get('entry') == int(law_entry):
                    db.add_edge(case_id, node_id, 'RELATES_TO_LAW')
                    law_found = True
                    break
            
            if not law_found:
                print(f"警告: 未找到Law节点，entry={law_entry}")

    # 创建Law到Crime的关系（基于罪名描述匹配）
    # 从图中检索所有Law节点及其crimes属性
    law_nodes = db.get_nodes_by_type('Laws')

    for law_node in tqdm(law_nodes, desc="Linking laws to crimes"):
        law_id = law_node['id']
        crime_descriptions = law_node.get('crimes')
        
        if not crime_descriptions:
            continue

        for crime_desc in crime_descriptions:
            # 检查是否已存在关系
            existing_neighbors = db.get_neighbors(law_id, 'RELATED_CRIME')
            if existing_neighbors:
                # 检查是否已经有匹配的crime
                found = False
                for crime_id in existing_neighbors:
                    crime_data = db.get_node(crime_id)
                    if crime_data and crime_data.get('description') == crime_desc:
                        found = True
                        break
                if found:
                    continue

            # 尝试精确匹配
            crime_found = False
            for node_id, node_info in db.nodes_data.items():
                if node_info['type'] == 'Crimes' and node_info['data'].get('description') == crime_desc:
                    db.add_edge(law_id, node_id, 'RELATED_CRIME', {'match_type': 'exact'})
                    crime_found = True
                    break

            # 如果精确匹配失败，尝试模糊匹配
            if not crime_found:
                for node_id, node_info in db.nodes_data.items():
                    if node_info['type'] == 'Crimes':
                        desc = node_info['data'].get('description', '')
                        if crime_desc in desc or desc in crime_desc:
                            db.add_edge(law_id, node_id, 'RELATED_CRIME', {'match_type': 'fuzzy'})
                            break
    
    # 删除entry <= 101的Law节点
    nodes_to_delete = []
    for node_id, node_info in db.nodes_data.items():
        if node_info['type'] == 'Laws':
            entry = node_info['data'].get('entry')
            if entry is not None and int(entry) <= 101:
                nodes_to_delete.append(node_id)
    
    node_count = len(nodes_to_delete)
    print(f"将要删除 {node_count} 个Law节点及其所有关系")
    
    for node_id in nodes_to_delete:
        db.graph.remove_node(node_id)
        del db.nodes_data[node_id]
        # 从embeddings中删除
        for node_type in db.embeddings:
            if node_id in db.embeddings[node_type]:
                del db.embeddings[node_type][node_id]
                db._update_vector_index(node_type)
    
    deleted_count = len(nodes_to_delete)
    print(f"已成功删除 {deleted_count} 个Law节点及其所有关系")


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


def create_clusters(model):
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


def search_similar_nodes_top(model, query_embedding, query_text, top_k=5):
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
    
    cases = []
    laws = []
    for neighbor in neighbors:
        # 获取关联的Laws节点
        law_neighbors = db.get_neighbors(neighbor['id'], 'RELATES_TO_LAW')
        for law_id in law_neighbors:
            law_data = db.get_node(law_id)
            if law_data:
                laws.append({
                    'id': law_id,
                    'entry': law_data.get('entry'),
                    'description': law_data.get('description'),
                    'crimes': law_data.get('crimes'),
                    'judge_dep': law_data.get('judge_dep'),
                    'related_laws': law_data.get('related_laws'),
                    'insights': law_data.get('insights', '')
                })
        cases.append(neighbor)

    return clusters, cases, laws


def search_similar_nodes_direct(model, query_embedding, query_text, top_k=5):
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
    
    cases = []
    laws = []
    for neighbor in neighbors:
        # 获取关联的Laws节点
        law_neighbors = db.get_neighbors(neighbor['id'], 'RELATES_TO_LAW')
        for law_id in law_neighbors:
            law_data = db.get_node(law_id)
            if law_data:
                laws.append({
                    'id': law_id,
                    'entry': law_data.get('entry'),
                    'description': law_data.get('description'),
                    'crimes': law_data.get('crimes'),
                    'judge_dep': law_data.get('judge_dep'),
                    'related_laws': law_data.get('related_laws'),
                    'insights': law_data.get('insights', '')
                })
        cases.append(neighbor)

    return cases, laws


def query_similar_nodes_naive(model, query_text, top_k=3):
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


def query_similar_nodes(model, query_text, retrieve_config):
    query_embedding = get_embedding(query_text)
    if query_embedding is None:
        return {}, [], []

    # 调用两个查询函数
    if retrieve_config["top_retrieve"]:
        top_result_clusters, top_result_cases, top_result_laws = search_similar_nodes_top(
            model, query_embedding, query_text, top_k=retrieve_config["top_retrieve_top_k"])
    else:
        top_result_clusters, top_result_cases, top_result_laws = [], [], []
    if retrieve_config["direct_retrieve"]:
        direct_result_cases, direct_result_laws = search_similar_nodes_direct(
            model, query_embedding, query_text, top_k=retrieve_config["direct_retrieve_top_k"])
    else:
        direct_result_cases, direct_result_laws = [], []

    # 整理结果
    result_cases = []
    seen_ids_cases = set()  # 用于去重
    result_laws = []
    seen_ids_laws = set()  # 用于去重

    # 处理 search_similar_nodes_top 的结果
    if top_result_cases:  # 确保前三个元素不为None
        neighbors = top_result_cases
        for neighbor in neighbors:
            if neighbor['id'] and neighbor['id'] not in seen_ids_cases:
                result_cases.append({
                    'id': neighbor['id'],
                    'description': neighbor['description'],
                    'caseId': neighbor['caseId'],
                    'rank': neighbor['rank']
                })
                seen_ids_cases.add(neighbor['id'])

    # 处理 search_similar_nodes_top 的结果
    if top_result_laws:  # 确保前三个元素不为None
        neighbors = top_result_laws
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

    # 处理 search_similar_nodes_direct 的结果
    if direct_result_cases:  # 确保前三个元素不为None
        neighbors = direct_result_cases
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

    # 处理 search_similar_nodes_direct 的结果
    if direct_result_laws:  # 确保前三个元素不为None
        neighbors = direct_result_laws
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
        "top": {
            "clusters": top_result_clusters,
            "cases": top_result_cases,
            "laws": top_result_laws
        },
        "direct": {
            "cases": direct_result_cases,
            "laws": direct_result_laws
        },
        "augmented": []
    }

    return original_retrieved_res, result_cases, result_laws


def query_similar_laws_naive(query_text, top_k=1):
    query_embedding = get_embedding(query_text)
    if query_embedding is None:
        return []

    db = GraphDBManager.get_db()
    law_results = db.find_similar_nodes(query_embedding, 'Laws', top_k=top_k)

    result_laws = []
    seen_law_ids = set()  # For deduplication of law nodes

    for law_record in law_results:
        entry = law_record.get('entry')
        if entry is not None and entry not in seen_law_ids:
            result_laws.append({
                'entry': entry,
                'similarity': law_record['similarity']
            })
            seen_law_ids.add(entry)

    return result_laws


def query_similar_laws(crime_list, top_k=1):
    """
    Query law nodes related to the most similar crime nodes based on a list of crime descriptions.

    Args:
        crime_list (list[str]): List of crime descriptions as strings.
        top_k (int): Number of top similar crime nodes to retrieve per crime description.

    Returns:
        list[dict]: List of law nodes with their details, deduplicated.
    """
    db = GraphDBManager.get_db()
    result_laws = []
    seen_law_ids = set()  # For deduplication of law nodes

    for crime in crime_list:
        # Convert crime description to embedding
        crime_embedding = get_embedding(crime)
        if crime_embedding is None:
            continue  # Skip if embedding generation fails

        # Query the most similar crime nodes
        crime_results = db.find_similar_nodes(crime_embedding, 'Crimes', top_k=top_k)

        # Process each similar crime node
        for crime_record in crime_results:
            crime_id = crime_record['id']
            crime_similarity = crime_record['similarity']

            # Query law nodes related to this crime node
            # 找到所有指向该crime的Laws节点
            for node_id, node_info in db.nodes_data.items():
                if node_info['type'] == 'Laws':
                    neighbors = db.get_neighbors(node_id, 'RELATED_CRIME')
                    if crime_id in neighbors:
                        law_data = node_info['data']
                        law_id = node_id
                        if law_id not in seen_law_ids:
                            result_laws.append({
                                'id': law_id,
                                'entry': law_data.get('entry'),
                                'description': law_data.get('description'),
                                'crimes': law_data.get('crimes'),
                                'judge_dep': law_data.get('judge_dep'),
                                'related_laws': law_data.get('related_laws'),
                                'insights': law_data.get('insights', ''),
                                'crime_similarity': crime_similarity
                            })
                            seen_law_ids.add(law_id)

    # Sort results by crime similarity (descending) and assign ranks
    result_laws = sorted(
        result_laws, key=lambda x: x['crime_similarity'], reverse=True)
    for rank, law in enumerate(result_laws, 1):
        law['rank'] = rank

    return result_laws


def update_insights_in_graph(law_id, insights):
    db = GraphDBManager.get_db()
    db.update_node(law_id, {'insights': insights})


def construct_feature_graph(model, nodes_data):
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

    # 运行KNN和聚类
    run_knn(top_k=3)
    create_clusters(model)
