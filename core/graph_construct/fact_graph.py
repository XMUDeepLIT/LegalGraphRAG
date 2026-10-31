"""Fact layer of HierarGraph: cases, articles, and offenses.

Nodes are Cases, Laws, and Crimes. Edges are RELATES_TO_LAW and RELATED_CRIME.
"""
from tqdm import tqdm

from .embedding import get_embedding
from .graph_db import GraphDBManager
from .rule_graph import law_node_properties, retrieved_law_record


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

    # 存储Law节点。规则层字段由 rule_graph.law_node_properties 写入同一节点。
    for node in tqdm(law_nodes_data, desc="Storing law nodes"):
        db.add_node(
            node['id'],
            'Laws',
            law_node_properties(node)
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


def laws_for_case(case_id):
    """Articles cited by one case, in the original neighbor order."""
    db = GraphDBManager.get_db()
    laws = []
    law_neighbors = db.get_neighbors(case_id, 'RELATES_TO_LAW')
    for law_id in law_neighbors:
        law_data = db.get_node(law_id)
        if law_data:
            laws.append(retrieved_law_record(law_id, law_data))
    return laws


def flat_article_retrieval(query_text, top_k=1):
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


def charge_anchored_articles(crime_list, top_k=1):
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
