"""LegalGraphRAG main class"""
import os
import json
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
from tqdm import tqdm

from core.models import BaseModel
from core.utils.util import analyze_case
from core.graph_construct.graph_db import GraphDBManager


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "qwen3"
    device: str = "cuda:0"
    prompt_language: str = "zh"
    # OpenAI-type model configuration
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    # Generation parameters
    max_length: int = 4096
    temperature: float = 0.1
    
    def __post_init__(self):
        """Validate model name"""
        valid_models = [
            "qwen3", "qwen2_5", "gemma3", "internlm3", 
            "glm4", "deepseek_v3", "gpt4o_mini"
        ]
        if self.model_name not in valid_models:
            raise ValueError(
                f"Invalid model_name: {self.model_name}. "
                f"Must be one of {valid_models}"
            )
        valid_prompt_languages = ["en", "zh", "cn", "chinese", "english"]
        if self.prompt_language.lower() not in valid_prompt_languages:
            raise ValueError(
                f"Invalid prompt_language: {self.prompt_language}. "
                f"Must be one of {valid_prompt_languages}"
            )


@dataclass
class DataConfig:
    """Data path configuration"""
    case_db_path: str = "./datas/main_experiment/cases_with_feature_big.json"
    law_to_crime_path: str = "./datas/main_experiment/criminal_law_processed.json"
    datasets_path: Optional[str] = "./datas/main_experiment"
    output_dir: str = "./outputs/main_experiment"
    
    def __post_init__(self):
        """Create output directory"""
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class RetrieveConfig:
    """The three Researcher operators: R_com, R_sem, and R_chg."""
    community_expansion: bool = True
    semantic_match: bool = True
    charge_anchored: bool = True
    community_expansion_top_k: int = 5
    semantic_match_top_k: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "community_expansion": self.community_expansion,
            "semantic_match": self.semantic_match,
            "charge_anchored": self.charge_anchored,
            "community_expansion_top_k": self.community_expansion_top_k,
            "semantic_match_top_k": self.semantic_match_top_k
        }


@dataclass
class GraphConfig:
    """Graph database configuration"""
    graph_db_path: Optional[str] = None  # Graph database save/load path
    embedding_api_url: str = "http://localhost:11434/api/embed"
    embedding_model: str = "bge-m3"
    auto_save: bool = True  # Whether to auto-save graph database
    auto_build: bool = True  # Whether to auto-build if graph doesn't exist


@dataclass
class LegalGraphRAGConfig:
    """LegalGraphRAG complete configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    retrieve: RetrieveConfig = field(default_factory=RetrieveConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    
    @classmethod
    def from_env_file(cls, dotenv_path: str = ".env") -> "LegalGraphRAGConfig":
        """
        Load configuration from .env file
        
        Args:
            dotenv_path: Path to .env file
            
        Returns:
            LegalGraphRAGConfig instance
        """
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=dotenv_path, override=True)
        
        # Model configuration
        model_config = ModelConfig(
            model_name=os.getenv("model_name", "qwen3"),
            device=os.getenv("device", "cuda:0"),
            prompt_language=os.getenv("prompt_language", "zh"),
            api_key=os.getenv("api_key"),
            base_url=os.getenv("base_url"),
            max_length=int(os.getenv("max_length", 4096)),
            temperature=float(os.getenv("temperature", 0.1))
        )
        
        # Data configuration
        data_config = DataConfig(
            case_db_path=os.getenv("case_db_path", "./datas/main_experiment/cases_with_feature_big.json"),
            law_to_crime_path=os.getenv("law_to_crime_path", "./datas/main_experiment/criminal_law_processed.json"),
            datasets_path=os.getenv("datasets_path", "./datas/main_experiment"),
            output_dir=os.getenv("output_dir", "./outputs/main_experiment")
        )
        
        retrieve_config = RetrieveConfig(
            community_expansion=os.getenv("community_expansion", "True") == "True",
            semantic_match=os.getenv("semantic_match", "True") == "True",
            charge_anchored=os.getenv("charge_anchored", "True") == "True",
            community_expansion_top_k=int(os.getenv("community_expansion_top_k", "5")),
            semantic_match_top_k=int(os.getenv("semantic_match_top_k", "5"))
        )
        
        # Graph configuration
        graph_config = GraphConfig(
            graph_db_path=os.getenv("graph_db_path"),
            embedding_api_url=os.getenv("embedding_api_url", "http://localhost:11434/api/embed"),
            embedding_model=os.getenv("embedding_model", "bge-m3"),
            auto_save=os.getenv("auto_save", "True") == "True",
            auto_build=os.getenv("auto_build", "True") == "True"
        )
        
        return cls(
            model=model_config,
            data=data_config,
            retrieve=retrieve_config,
            graph=graph_config
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LegalGraphRAGConfig":
        """
        Create configuration from dictionary
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            LegalGraphRAGConfig instance
        """
        model_config = ModelConfig(**config_dict.get("model", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        retrieve_config = RetrieveConfig(**config_dict.get("retrieve", {}))
        graph_config = GraphConfig(**config_dict.get("graph", {}))
        
        return cls(
            model=model_config,
            data=data_config,
            retrieve=retrieve_config,
            graph=graph_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model": {
                "model_name": self.model.model_name,
                "device": self.model.device,
                "prompt_language": self.model.prompt_language,
                "api_key": self.model.api_key,
                "base_url": self.model.base_url,
                "max_length": self.model.max_length,
                "temperature": self.model.temperature
            },
            "data": {
                "case_db_path": self.data.case_db_path,
                "law_to_crime_path": self.data.law_to_crime_path,
                "datasets_path": self.data.datasets_path,
                "output_dir": self.data.output_dir
            },
            "retrieve": self.retrieve.to_dict(),
            "graph": {
                "graph_db_path": self.graph.graph_db_path,
                "embedding_api_url": self.graph.embedding_api_url,
                "embedding_model": self.graph.embedding_model,
                "auto_save": self.graph.auto_save,
                "auto_build": self.graph.auto_build
            }
        }
    
    def save(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "LegalGraphRAGConfig":
        """Load configuration from JSON file"""
        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class LegalGraphRAG:
    """LegalGraphRAG main class"""
    
    def __init__(self, config: Optional[LegalGraphRAGConfig] = None):
        """
        Initialize LegalGraphRAG
        
        Args:
            config: Configuration object, if None use default configuration
        """
        self.config = config or LegalGraphRAGConfig()
        from core.prompt import set_prompt_language
        from core.graph_construct.feature_graph import configure_embedding
        set_prompt_language(self.config.model.prompt_language)
        configure_embedding(
            api_url=self.config.graph.embedding_api_url,
            model=self.config.graph.embedding_model
        )

        # Initialize model
        self.model = self._init_model()
        
        # Load data
        self.cases_db = self._load_cases_db()
        self.law_to_crime = self._load_law_to_crime()
        
        # Initialize graph database
        if self.config.graph.graph_db_path and os.path.exists(self.config.graph.graph_db_path):
            GraphDBManager.load(self.config.graph.graph_db_path)
            print(f"Graph database loaded from {self.config.graph.graph_db_path}")
        else:
            GraphDBManager.initialize()
            # If auto-build is enabled and graph_db_path is configured, auto-build the graph
            if self.config.graph.auto_build and self.config.graph.graph_db_path:
                print("Graph database not found. Auto-building graph...")
                print("="*60)
                self.build_graph(force_rebuild=False)
                # After graph construction, reload if graph database file was created
                if os.path.exists(self.config.graph.graph_db_path):
                    GraphDBManager.load(self.config.graph.graph_db_path)
                    print(f"Graph database loaded after construction from {self.config.graph.graph_db_path}")
                print("="*60)
            elif self.config.graph.graph_db_path:
                print(f"Warning: Graph database not found at {self.config.graph.graph_db_path}")
                print("Set auto_build=True in config to automatically build the graph")
            else:
                print("Warning: graph_db_path not configured. Graph will not be persisted.")
    
    def _init_model(self) -> BaseModel:
        """Initialize model"""
        from core.models import (
            QwenChatbot, Qwen2Chatbot, GemmaChatbot, InternlmChatbot,
            GlmChatbot, DeepSeekChatbot, GPT4OMiniChatbot
        )
        
        model_map = {
            "qwen3": QwenChatbot,
            "qwen2_5": Qwen2Chatbot,
            "gemma3": GemmaChatbot,
            "internlm3": InternlmChatbot,
            "glm4": GlmChatbot,
            "deepseek_v3": DeepSeekChatbot,
            "gpt4o_mini": GPT4OMiniChatbot,
        }
        
        model_class = model_map[self.config.model.model_name]
        
        # OpenAI-type models need special handling
        if self.config.model.model_name in ["deepseek_v3", "gpt4o_mini"]:
            # OpenAI-type models need model_name, api_key, base_url
            init_kwargs = {
                "device": self.config.model.device,
            }
            if self.config.model.api_key:
                init_kwargs["api_key"] = self.config.model.api_key
            if self.config.model.base_url:
                init_kwargs["base_url"] = self.config.model.base_url
            return model_class(**init_kwargs)
        else:
            # Transformers-type models only need device
            return model_class(device=self.config.model.device)
    
    def _load_cases_db(self) -> List[Dict[str, Any]]:
        """Load case database"""
        if not os.path.exists(self.config.data.case_db_path):
            raise FileNotFoundError(
                f"Case database not found: {self.config.data.case_db_path}"
            )
        with open(self.config.data.case_db_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _load_law_to_crime(self) -> List[Dict[str, Any]]:
        """Load law to crime mapping"""
        if not os.path.exists(self.config.data.law_to_crime_path):
            raise FileNotFoundError(
                f"Law to crime mapping not found: {self.config.data.law_to_crime_path}"
            )
        with open(self.config.data.law_to_crime_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def analyze_case(self, case: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze a single case
        
        Args:
            case: Case dictionary containing "fact" and "name" fields
            
        Returns:
            List of analysis results, each element corresponds to a defendant's analysis result
        """
        retrieve_config = self.config.retrieve.to_dict()
        return analyze_case(
            self.model,
            case,
            self.law_to_crime,
            self.cases_db,
            retrieve_config
        )
    
    def analyze_cases(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch analyze cases
        
        Args:
            cases: List of cases
            
        Returns:
            List of analysis results
        """
        results = []
        for case in cases:
            case_result = self.analyze_case(case)
            results.append({
                "case_id": case.get("id"),
                "fact": case.get("fact"),
                "analysis": case_result
            })
        return results
    
    def save_graph_db(self, filepath: Optional[str] = None):
        """Save graph database"""
        save_path = filepath or self.config.graph.graph_db_path
        if not save_path:
            raise ValueError("Graph database path not specified")
        GraphDBManager.save(save_path)
        print(f"Graph database saved to {save_path}")
    
    def load_graph_db(self, filepath: Optional[str] = None):
        """Load graph database"""
        load_path = filepath or self.config.graph.graph_db_path
        if not load_path:
            raise ValueError("Graph database path not specified")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Graph database not found: {load_path}")
        GraphDBManager.load(load_path)
        print(f"Graph database loaded from {load_path}")
    
    def _ontology_feature_text(self, description: Dict[str, Any]) -> str:
        """
        Concatenate feature descriptions
        
        Args:
            description: Feature dictionary containing defendant_info, criminal_acts, 
                        victim_property_details, intent_remorse fields
            
        Returns:
            Concatenated description string
        """
        res = ""
        if description.get("defendant_info"):
            res += "Defendant Info: " + ", ".join(description.get("defendant_info", [])) + ". "
        if description.get("criminal_acts"):
            res += "Criminal Acts: " + ", ".join(description.get("criminal_acts", [])) + ". "
        if description.get("victim_property_details"):
            res += "Victim/Property Characteristics: " + ", ".join(description.get("victim_property_details", [])) + ". "
        if description.get("intent_remorse"):
            res += "Intent and Remorse: " + ", ".join(description.get("intent_remorse", [])) + ". "
        return res
    
    def _prepare_nodes_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Prepare node data
        
        Returns:
            Dictionary containing 'case', 'law', 'crime' keys
        """
        case_nodes_data = []
        law_nodes_data = []
        crime_nodes_data = []
        
        for case in tqdm(self.cases_db, desc="Preparing case nodes"):
            features = case.get("features") or {}
            if features.get("criminal_acts"):
                case_nodes_data.append({
                    'id': str(uuid.uuid4()),
                    'description': self._ontology_feature_text(features),
                    'caseId': case.get("id", ""),
                    'crime': case.get("crime", []),
                    'law': case.get("law", []),
                    'type': 'case'
                })
        
        # Process law nodes and crime nodes
        crimes = set()
        for law in tqdm(self.law_to_crime, desc="Preparing law and crime nodes"):
            text_id = law.get("id")
            
            # Process items field
            if "items" in law:
                for item in law["items"]:
                    # Collect crimes
                    if "crime" in item:
                        if isinstance(item["crime"], list):
                            crimes.update(item["crime"])
                        else:
                            crimes.add(item["crime"])
                    
                    # Create law node
                    law_nodes_data.append({
                        'id': str(uuid.uuid4()),
                        'entry': text_id,
                        'description': item.get("text", ""),
                        'crimes': item.get("crime", []),
                        "judge_dep": str(item.get("judge_dep", [])),
                        "related_laws": str(item.get("related_laws", [])),
                        'type': 'law'
                    })
            else:
                # If no items field, process law object directly
                if "crime" in law:
                    if isinstance(law["crime"], list):
                        crimes.update(law["crime"])
                    else:
                        crimes.add(law["crime"])
                
                law_nodes_data.append({
                    'id': str(uuid.uuid4()),
                    'entry': text_id,
                    'description': law.get("text", law.get("description", "")),
                    'crimes': law.get("crime", []),
                    "judge_dep": str(law.get("judge_dep", [])),
                    "related_laws": str(law.get("related_laws", [])),
                    'type': 'law'
                })
        
        # Create crime nodes
        crimes = list(crimes)
        for crime in crimes:
            if crime and crime != "":
                crime_nodes_data.append({
                    'id': str(uuid.uuid4()),
                    'description': crime,
                    'type': 'crime'
                })
        
        return {
            'case': case_nodes_data,
            'law': law_nodes_data,
            'crime': crime_nodes_data
        }
    
    def build_graph(self, force_rebuild: bool = False):
        """
        Build graph structure
        
        Args:
            force_rebuild: If True, rebuild even if graph database already exists
        """
        from core.graph_construct.feature_graph import construct_hierargraph
        
        # Check if graph database already exists
        if not force_rebuild and self.config.graph.graph_db_path and os.path.exists(self.config.graph.graph_db_path):
            print(f"Graph database already exists at {self.config.graph.graph_db_path}")
            print("Use force_rebuild=True to rebuild the graph")
            return
        
        print("Starting graph construction...")
        
        # Prepare node data
        nodes_data = self._prepare_nodes_data()
        
        print(f"Prepared {len(nodes_data['case'])} case nodes, "
              f"{len(nodes_data['law'])} law nodes, "
              f"{len(nodes_data['crime'])} crime nodes")
        
        # Build graph structure
        construct_hierargraph(self.model, nodes_data)
        
        # Save graph database
        if self.config.graph.graph_db_path:
            self.save_graph_db()
            print(f"Graph construction completed and saved to {self.config.graph.graph_db_path}")
        else:
            print("Graph construction completed (not saved, graph_db_path not specified)")
    
    def __del__(self):
        """Destructor, auto-save graph database"""
        if self.config.graph.auto_save and self.config.graph.graph_db_path:
            try:
                self.save_graph_db()
            except Exception as e:
                print(f"Failed to auto-save graph database: {e}")

