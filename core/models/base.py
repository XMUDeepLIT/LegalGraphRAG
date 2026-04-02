"""Base model interface definition"""
from abc import ABC, abstractmethod
from typing import Optional


class BaseModel(ABC):
    """Base class interface for all models"""
    
    def __init__(self, model_name: str, device: str = "cuda:0"):
        """
        Initialize model
        
        Args:
            model_name: Model name
            device: Device (may not be used for OpenAI type)
        """
        self.model_name = model_name
        self.device = device
    
    @abstractmethod
    def generate_response(self, user_input: str, max_length: int = 4096) -> str:
        """
        Generate response
        
        Args:
            user_input: User input
            max_length: Maximum generation length
            
        Returns:
            Model generated response text
        """
        pass
    
    def release_model(self):
        """Release resources occupied by model (optional implementation)"""
        pass

