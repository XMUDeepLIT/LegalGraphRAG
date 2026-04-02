"""Gemma3 model (Transformers)"""
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from ..transformers_base import TransformersBaseModel


class GemmaChatbot(TransformersBaseModel):
    """Gemma3 chatbot model"""
    
    def __init__(self, model_name: str = "google/gemma-3-12b-it", device: str = "cuda:0"):
        """
        Initialize Gemma3 model
        
        Args:
            model_name: Model name, default is "google/gemma-3-12b-it"
            device: Device (e.g., "cuda:0", "cpu")
        """
        super().__init__(model_name, device, trust_remote_code=False, torch_dtype="auto")
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name, device_map=device
        )
        self.model.eval()
    
    def generate_response(self, user_input: str, max_length: int = 100) -> str:
        """
        Generate response
        
        Args:
            user_input: User input
            max_length: Maximum generation length (Gemma3 default 100)
            
        Returns:
            Model generated response text
        """
        messages = [{"role": "user", "content": [{"type": "text", "text": user_input}]}]
        
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype="auto")
        
        input_len = inputs["input_ids"].shape[-1]
        
        generation = self.model.generate(**inputs, max_new_tokens=max_length, do_sample=False)
        generation = generation[0][input_len:]
        
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        response = decoded.strip()[-10:]
        
        return response

