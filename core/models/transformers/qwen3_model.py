"""Qwen3 model (Transformers)"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..transformers_base import TransformersBaseModel


class QwenChatbot(TransformersBaseModel):
    """Qwen3 chatbot model"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-8B", device: str = "cuda:0"):
        """
        Initialize Qwen3 model
        
        Args:
            model_name: Model name, default is "Qwen/Qwen3-8B"
            device: Device (e.g., "cuda:0", "cpu")
        """
        super().__init__(model_name, device, trust_remote_code=False, torch_dtype="auto")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device, torch_dtype="auto"
        )
        self.count = 0
    
    def generate_response(self, user_input: str, max_length: int = 4096) -> str:
        """
        Generate response
        
        Args:
            user_input: User input
            max_length: Maximum generation length
            
        Returns:
            Model generated response text
        """
        messages = [{"role": "user", "content": user_input}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        response_ids = self.model.generate(
            **inputs, max_new_tokens=max_length
        )[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        self.count += 1
        return response

