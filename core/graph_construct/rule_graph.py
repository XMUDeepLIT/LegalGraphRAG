"""Rule layer of HierarGraph.

Each Laws node stores its diagnostic checklist in ``judge_dep`` and its
judicial interpretations in ``related_laws``.
"""
from .graph_db import GraphDBManager


def law_node_properties(node):
    """Properties written onto a Laws node, including the rule-layer fields."""
    return {
        'entry': node.get('entry'),
        'description': node.get('description'),
        'embedding': node.get('embedding'),
        'crimes': node.get('crimes'),
        'judge_dep': node.get('judge_dep'),
        'related_laws': node.get('related_laws'),
        'insights': ''
    }


def retrieved_law_record(law_id, law_data):
    """Law record returned by case-to-article traversal."""
    return {
        'id': law_id,
        'entry': law_data.get('entry'),
        'description': law_data.get('description'),
        'crimes': law_data.get('crimes'),
        'judge_dep': law_data.get('judge_dep'),
        'related_laws': law_data.get('related_laws'),
        'insights': law_data.get('insights', '')
    }


def update_insights_in_graph(law_id, insights):
    db = GraphDBManager.get_db()
    db.update_node(law_id, {'insights': insights})
