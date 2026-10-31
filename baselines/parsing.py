"""Parse baseline generations into the fields read by evaluation/evaluate_results.py."""
import ast
import json
from typing import Any, Dict, List


EMPTY_JUDGMENT: Dict[str, Any] = {
    "charge_name": ["无罪"],
    "law_article": [],
    "term_of_imprisonment": {
        "death_penalty": False,
        "imprisonment": 0,
        "life_imprisonment": False,
    },
}


def strip_think(full_output: str) -> str:
    """Drop a leading think span. Qwen3 applies this before JSON parsing."""
    if "<think>" in full_output:
        think_end = full_output.find("</think>") + len("</think>")
        return full_output[think_end:].strip()
    return full_output


def parse_judgment(full_output: str, strip_thinking: bool = False) -> Dict[str, Any]:
    """Read the first JSON object. A parse failure becomes an acquittal."""
    prediction = strip_think(full_output) if strip_thinking else full_output
    try:
        first = prediction.find("{")
        last = prediction.rfind("}") + 1
        return json.loads(prediction[first:last])
    except Exception:
        return {
            "charge_name": list(EMPTY_JUDGMENT["charge_name"]),
            "law_article": [],
            "term_of_imprisonment": dict(EMPTY_JUDGMENT["term_of_imprisonment"]),
        }


def parse_charge_list(prediction: str) -> List[Any]:
    """Read the first Python list. A missing list is ["无法解析响应"]. A failed parse is ["无罪"]."""
    text = prediction.strip()
    try:
        first = text.find("[")
        last = text.rfind("]") + 1
        if first != -1 and last != 0:
            return ast.literal_eval(text[first:last])
        return ["无法解析响应"]
    except Exception:
        return ["无罪"]
