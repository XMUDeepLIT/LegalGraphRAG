"""DeepSeek model (OpenAI compatible API)"""
from typing import Optional
from ..openai_base import OpenAIBaseModel


class DeepSeekChatbot(OpenAIBaseModel):
    """DeepSeek chatbot model"""
    
    def __init__(
        self,
        model_name: str = "deepseek-chat",
        device: str = "cuda:0",
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8000/v1"
    ):
        """
        Initialize DeepSeek model
        
        Args:
            model_name: Model name, default is "deepseek-chat"
            device: Device (not used for OpenAI type, kept for interface consistency)
            api_key: API key, if None then get from environment variable DEEPSEEK_API_KEY
            base_url: API base URL, default is local server
        """
        super().__init__(
            model_name=model_name,
            device=device,
            api_key=api_key,
            base_url=base_url,
            env_api_key_name="DEEPSEEK_API_KEY",
            default_base_url=base_url
        )

