# graph_db.py - In-memory graph database implementation based on NetworkX
import pickle
import os
import numpy as np
from scipy.spatial.distance import cosine
import networkx as nx
from typing import Dict, List, Optional, Any


class InMemoryGraphDB:
    """In-memory graph database based on NetworkX"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InMemoryGraphDB, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Use MultiDiGraph to support multiple relationship types
        self.graph = nx.MultiDiGraph()
        
        # Node data storage: {node_id: {type: 'Cases'|'Laws'|'Crimes'|'Cluster', data: {...}}}
        self.nodes_data = {}
        
        # Vector index: for fast similarity search
        # {node_type: {node_id: embedding}}
        self.embeddings = {
            'Cases': {},
            'Laws': {},
            'Crimes': {},
            'Cluster': {}
        }
        
        # Vector arrays and ID mapping (for fast search)
        self._vector_indexes = {
            'Cases': {'vectors': [], 'ids': []},
            'Laws': {'vectors': [], 'ids': []},
            'Crimes': {'vectors': [], 'ids': []},
            'Cluster': {'vectors': [], 'ids': []}
        }
        
        self._initialized = True
        print("In-memory graph database initialized")
    
    def _update_vector_index(self, node_type: str):
        """Update vector index to accelerate search"""
        embeddings_dict = self.embeddings[node_type]
        if not embeddings_dict:
            self._vector_indexes[node_type] = {'vectors': [], 'ids': []}
            return
        
        ids = []
        vectors = []
        for node_id, embedding in embeddings_dict.items():
            ids.append(node_id)
            vectors.append(embedding)
        
        self._vector_indexes[node_type] = {
            'vectors': np.array(vectors) if vectors else np.array([]),
            'ids': ids
        }
    
    def add_node(self, node_id: str, node_type: str, properties: Dict[str, Any]):
        """Add node"""
        self.graph.add_node(node_id, node_type=node_type, **properties)
        self.nodes_data[node_id] = {
            'type': node_type,
            'data': properties
        }
        
        # If node has embedding, add to vector index
        if 'embedding' in properties and properties['embedding'] is not None:
            self.embeddings[node_type][node_id] = np.array(properties['embedding'])
            self._update_vector_index(node_type)
    
    def update_node(self, node_id: str, properties: Dict[str, Any]):
        """Update node properties"""
        if node_id not in self.nodes_data:
            return
        
        node_type = self.nodes_data[node_id]['type']
        self.nodes_data[node_id]['data'].update(properties)
        
        # Update graph node properties
        for key, value in properties.items():
            self.graph.nodes[node_id][key] = value
        
        # If embedding is updated, update vector index
        if 'embedding' in properties and properties['embedding'] is not None:
            self.embeddings[node_type][node_id] = np.array(properties['embedding'])
            self._update_vector_index(node_type)
    
    def add_edge(self, source: str, target: str, relation_type: str, properties: Optional[Dict] = None):
        """Add edge (relationship)"""
        if properties is None:
            properties = {}
        self.graph.add_edge(source, target, relation_type=relation_type, **properties)
    
    def get_node(self, node_id: str) -> Optional[Dict]:
        """Get node data"""
        if node_id not in self.nodes_data:
            return None
        return self.nodes_data[node_id]['data'].copy()
    
    def get_nodes_by_type(self, node_type: str) -> List[Dict]:
        """Get all nodes of specified type"""
        results = []
        for node_id, node_info in self.nodes_data.items():
            if node_info['type'] == node_type:
                result = {'id': node_id}
                result.update(node_info['data'])
                results.append(result)
        return results
    
    def get_neighbors(self, node_id: str, relation_type: Optional[str] = None) -> List[str]:
        """Get neighbor node ID list"""
        if relation_type:
            neighbors = []
            for target in self.graph.successors(node_id):
                for edge_data in self.graph[node_id][target].values():
                    if edge_data.get('relation_type') == relation_type:
                        neighbors.append(target)
                        break
            return neighbors
        else:
            return list(self.graph.successors(node_id))
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        if vec1 is None or vec2 is None:
            return 0.0
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        if vec1.shape != vec2.shape:
            return 0.0
        # Use 1 - cosine distance to get similarity
        return 1 - cosine(vec1.flatten(), vec2.flatten())
    
    def find_similar_nodes(self, query_embedding: np.ndarray, node_type: str, top_k: int = 5) -> List[Dict]:
        """Find most similar nodes based on vector similarity"""
        if node_type not in self._vector_indexes:
            return []
        
        index_data = self._vector_indexes[node_type]
        if len(index_data['vectors']) == 0:
            return []
        
        query_vec = np.array(query_embedding).flatten()
        vectors = index_data['vectors']
        
        # Calculate all similarities
        similarities = []
        for i, vec in enumerate(vectors):
            sim = self.cosine_similarity(query_vec, vec)
            similarities.append((index_data['ids'][i], sim))
        
        # Sort and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for node_id, similarity in similarities[:top_k]:
            node_data = self.get_node(node_id)
            if node_data:
                result = {'id': node_id, 'similarity': similarity}
                result.update(node_data)
                results.append(result)
        
        return results
    
    def compute_pagerank(self) -> Dict[str, float]:
        """Compute PageRank"""
        try:
            pagerank = nx.pagerank(self.graph.to_undirected())
            return pagerank
        except:
            return {}
    
    def compute_degree_centrality(self) -> Dict[str, int]:
        """Compute degree centrality"""
        # NetworkX's degree() returns DegreeView, convert to dict then process
        degree_dict = dict(self.graph.degree())
        # Ensure return is string keys and integer values
        return {str(k): int(v) for k, v in degree_dict.items()}
    
    def detect_communities(self) -> Dict[str, int]:
        """Detect communities using Louvain algorithm"""
        try:
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.louvain_communities(self.graph.to_undirected(), seed=42)
            
            # Assign community ID
            community_map = {}
            for comm_id, comm_nodes in enumerate(communities):
                for node in comm_nodes:
                    community_map[node] = comm_id
            
            return community_map
        except Exception as e:
            print(f"Community detection failed: {e}")
            return {}
    
    def save(self, filepath: str):
        """Save graph data to file"""
        data = {
            'graph': self.graph,
            'nodes_data': self.nodes_data,
            'embeddings': {k: {nid: emb.tolist() if isinstance(emb, np.ndarray) else emb 
                              for nid, emb in v.items()} 
                          for k, v in self.embeddings.items()}
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Graph data saved to {filepath}")
    
    def load(self, filepath: str):
        """Load graph data from file"""
        if not os.path.exists(filepath):
            print(f"File does not exist: {filepath}")
            return
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.graph = data['graph']
        self.nodes_data = data['nodes_data']
        self.embeddings = {k: {nid: np.array(emb) if isinstance(emb, list) else emb 
                              for nid, emb in v.items()} 
                          for k, v in data['embeddings'].items()}
        
        # Rebuild vector index
        for node_type in self.embeddings:
            self._update_vector_index(node_type)
        
        print(f"Graph data loaded from {filepath}")


class GraphDBManager:
    """Graph database manager"""
    
    _instance = None
    _db = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GraphDBManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls):
        """Initialize in-memory graph database"""
        if cls._db is None:
            cls._db = InMemoryGraphDB()
        print("Graph database initialized")

    @classmethod
    def get_db(cls) -> InMemoryGraphDB:
        """Get underlying database instance"""
        if cls._db is None:
            cls.initialize()
        assert cls._db is not None
        return cls._db
    
    @classmethod
    def save(cls, filepath: str):
        """Save graph data"""
        if cls._db is None:
            cls.initialize()
        assert cls._db is not None
        cls._db.save(filepath)
    
    @classmethod
    def load(cls, filepath: str):
        """Load graph data"""
        cls.initialize()
        assert cls._db is not None
        cls._db.load(filepath)
