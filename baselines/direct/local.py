"""Apply the chat template, then call generate() with the catalog decoding settings."""
import os
from typing import Dict, List, Optional

from ..catalog import ModelSpec


class LocalCausalModel:
    def __init__(self, spec: ModelSpec, device: str, model_name: Optional[str] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        name = model_name or spec.model_name
        if spec.key == "legal_delta":
            name = model_name or os.getenv("LEGAL_DELTA_PATH", spec.model_name)
        self.spec = spec
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=spec.trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            trust_remote_code=spec.trust_remote_code,
            device_map=device,
            torch_dtype="auto",
        )
        self.model.eval()

    def generate(self, user_input: str, max_new_tokens: Optional[int] = None, history: Optional[List[Dict[str, str]]] = None) -> str:
        messages = list(history or [])
        messages.append({"role": "user", "content": user_input})
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self.spec.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.spec.enable_thinking
        text = self.tokenizer.apply_chat_template(messages, **template_kwargs)

        if self.spec.protocol == "adapt":
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(self.model.device)
        else:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        import torch

        generate_kwargs = {"max_new_tokens": max_new_tokens or self.spec.max_new_tokens}
        generate_kwargs.update(self.spec.generate_kwargs)
        if self.spec.protocol == "adapt":
            generate_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
            generate_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        with torch.no_grad():
            response_ids = self._generate(inputs, generate_kwargs)
        prompt_width = inputs["input_ids"].shape[1]
        decode_kwargs = {"skip_special_tokens": True}
        if self.spec.protocol == "adapt":
            decode_kwargs["clean_up_tokenization_spaces"] = True
        text = self.tokenizer.decode(response_ids[0][prompt_width:], **decode_kwargs)
        if self.spec.protocol == "adapt":
            return text.strip()
        return text

    def _generate(self, inputs, generate_kwargs):
        try:
            return self.model.generate(**inputs, **generate_kwargs)
        except ValueError:
            reduced = dict(generate_kwargs)
            reduced.pop("temperature", None)
            reduced.pop("top_p", None)
            if reduced == generate_kwargs:
                raise
            return self.model.generate(**inputs, **reduced)

    def release(self):
        import gc
        import torch

        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
