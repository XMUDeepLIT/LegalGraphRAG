"""Base class for Transformers type models"""
from typing import Optional
import torch
import gc
from .base import BaseModel


class TransformersBaseModel(BaseModel):
    """Base class for models based on Transformers library"""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        trust_remote_code: bool = False,
        torch_dtype: str = "auto"
    ):
        """
        Initialize Transformers type model
        
        Args:
            model_name: Model name (HuggingFace model path)
            device: Device (e.g., "cuda:0", "cpu")
            trust_remote_code: Whether to trust remote code
            torch_dtype: torch data type
        """
        super().__init__(model_name, device)
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype
        self.model = None
        self.tokenizer = None
    
    def generate_response(self, user_input: str, max_length: int = 4096) -> str:
        """
        Generate response (needs to be implemented in subclass with specific logic)
        
        Args:
            user_input: User input
            max_length: Maximum generation length
            
        Returns:
            Model generated response text
        """
        raise NotImplementedError("Subclass must implement generate_response method")
    
    def release_model(self):
        """Release GPU memory occupied by model"""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        
        # Force garbage collection
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Model has been released from GPU memory")

