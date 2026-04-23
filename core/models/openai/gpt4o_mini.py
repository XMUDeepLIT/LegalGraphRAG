"""GPT-4o Mini model (OpenAI compatible API)"""
from typing import Optional
from ..openai_base import OpenAIBaseModel


class GPT4OMiniChatbot(OpenAIBaseModel):
    """GPT-4o Mini chatbot model"""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        device: str = "cuda:0",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1"
    ):
        """
        Initialize GPT-4o Mini model
        
        Args:
            model_name: Model name, default is "gpt-4o-mini"
            device: Device (not used for OpenAI type, kept for interface consistency)
            api_key: API key, if None then get from environment variable OPENAI_API_KEY
            base_url: API base URL, default is OpenAI API
        """
        super().__init__(
            model_name=model_name,
            device=device,
            api_key=api_key,
            base_url=base_url,
            env_api_key_name="OPENAI_API_KEY",
            default_base_url=base_url
        )

