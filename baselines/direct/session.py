"""Load a catalog model and apply its judgment protocol."""
from typing import Any, Dict, Optional

from ..catalog import ModelSpec
from ..parsing import parse_judgment
from ..prompts import adapt_prompt, extraction_prompt, freeform_judgment_prompt, structured_judgment_prompt
from ..records import case_record, defendant_names


class JudgmentSession:
    def __init__(self, spec: ModelSpec, device: str, model_name: Optional[str] = None):
        self.spec = spec
        self.local = None
        self.api = None
        self.extractor = None
        if spec.backend == "local":
            from .local import LocalCausalModel

            self.local = LocalCausalModel(spec, device, model_name=model_name)
        elif spec.backend == "deepseek":
            from core.models import DeepSeekChatbot

            self.api = DeepSeekChatbot(model_name=model_name or spec.model_name)
        elif spec.backend == "openai":
            from core.models import GPT4OMiniChatbot

            self.api = GPT4OMiniChatbot(model_name=model_name or spec.model_name)
        else:
            raise ValueError(f"Unknown backend {spec.backend}")
        if spec.protocol in ("freeform", "adapt"):
            from core.models import DeepSeekChatbot

            self.extractor = DeepSeekChatbot()

    def complete(self, prompt: str) -> str:
        if self.local is not None:
            return self.local.generate(prompt)
        return self.api.generate_response(prompt, max_length=self.spec.max_new_tokens)

    def extract(self, response_text: str) -> Dict[str, Any]:
        extracted = self.extractor.generate_response(extraction_prompt(response_text), max_length=500)
        return parse_judgment(extracted, strip_thinking=False)

    def judge_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        fact = case.get("fact", "")[:1024]
        judge_res = []
        for name in defendant_names(case):
            if self.spec.protocol == "structured":
                raw = self.complete(structured_judgment_prompt(name, fact))
                parsed = parse_judgment(raw, strip_thinking=self.spec.strip_think)
                judge_res.append({"name": name, "judge_result": parsed, "raw_output": raw})
            elif self.spec.protocol == "freeform":
                raw = self.complete(freeform_judgment_prompt(name, fact))
                judge_res.append({"name": name, "judge_result": self.extract(raw), "raw_output": raw})
            elif self.spec.protocol == "adapt":
                raw = self.complete(adapt_prompt(name, fact))
                judge_res.append(
                    {
                        "name": name,
                        "judge_result": self.extract(raw),
                        "raw_output": raw,
                        "multi_turn_responses": {"ADAPT_Reasoning": raw},
                    }
                )
            else:
                raise ValueError(f"{self.spec.protocol} is not a direct-model protocol")
        return case_record(case, fact, judge_res)

    def close(self):
        if self.local is not None:
            self.local.release()
