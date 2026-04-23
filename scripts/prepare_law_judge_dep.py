import argparse
import ast
import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm


JUDGE_PROMPT = """
你是一个法律AI助手，擅长解析刑法条文。你的任务是将用户提供的法条解析为一组子判断，
每个子判断以“是否”开头，用于判断该法条中的关键要素是否适用于一个刑事案件。
输出必须是一个 Python 列表形式的字符串列表（list[str]），仅包含这些子判断字符串，
不要输出任何其他解释文本。

要求：
1. 输出必须是合法的 Python 列表字符串，例如 ["是否A", "是否B"]。
2. 子判断应当是可用于案件匹配的关键要素，不要过于抽象。

输入法条：
{law_text}
""".strip()


def chinese_number_to_int(chinese_num: str) -> Optional[int]:
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    units = {"十": 10, "百": 100, "千": 1000, "万": 10000}
    total = 0
    section = 0
    number = 0
    for ch in chinese_num:
        if ch in digits:
            number = digits[ch]
        elif ch in units:
            unit = units[ch]
            if unit == 10000:
                section = (section + number) * unit
                total += section
                section = 0
                number = 0
            else:
                if number == 0:
                    number = 1
                section += number * unit
                number = 0
        else:
            return None
    return total + section + number


def extract_referenced_law_ids(text: str) -> List[int]:
    pattern = r"第([零〇一二两三四五六七八九十百千万]+)条"
    refs = []
    for match in re.findall(pattern, text):
        value = chinese_number_to_int(match)
        if value is not None:
            refs.append(value)
    return sorted(list(set(refs)))


def parse_judge_list(raw: str) -> List[str]:
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    payload = raw[start : end + 1]
    try:
        parsed = ast.literal_eval(payload)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    result = [str(x).strip() for x in parsed if str(x).strip()]
    return [x for x in result if x.startswith("是否")] or result


def load_criminal_law(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_judicial_explanations(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_law_text_index(criminal_law: List[Dict[str, Any]]) -> Dict[int, str]:
    index: Dict[int, str] = {}
    for row in criminal_law:
        law_id = row.get("id")
        if law_id is None:
            continue
        try:
            law_id = int(law_id)
        except (TypeError, ValueError):
            continue
        items = row.get("items", [])
        texts = []
        for item in items:
            text = item.get("text", "")
            if text:
                texts.append(str(text).strip())
        if texts:
            index[law_id] = "\n".join(texts)
    return index


def call_llm(client: OpenAI, model_name: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024,
        stream=False,
    )
    return (resp.choices[0].message.content or "").strip()


def build_related_laws(
    law_id: int,
    law_text: str,
    law_text_index: Dict[int, str],
    judicial_explanations: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    related: List[Dict[str, str]] = []
    seen = set()

    for ref_id in extract_referenced_law_ids(law_text):
        if ref_id == law_id:
            continue
        ref_text = law_text_index.get(ref_id, "")
        if not ref_text:
            continue
        key = f"article:{ref_id}"
        if key in seen:
            continue
        seen.add(key)
        related.append({"id": f"第{ref_id}条", "text": ref_text})

    for item in judicial_explanations:
        laws = item.get("laws", [])
        if law_id not in laws:
            continue
        explain_text = str(item.get("explain", "")).strip()
        explain_from = str(item.get("from", "")).strip()
        if not explain_text:
            continue
        key = f"exp:{explain_from}:{explain_text[:40]}"
        if key in seen:
            continue
        seen.add(key)
        related.append({"id": explain_from or "judicial_explanation", "text": explain_text})

    return related


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate law_judge_dep.json with LLM for LegalGraphRAG."
    )
    parser.add_argument(
        "--criminal-law-processed",
        type=str,
        required=True,
        help="Path to criminal_law_processed.json (contains id/items/text).",
    )
    parser.add_argument(
        "--judicial-explanations",
        type=str,
        required=True,
        help="Path to judicial_explanations.json.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./datas/law_judge_dep.json",
        help="Output path for generated law_judge_dep.json.",
    )
    parser.add_argument(
        "--dotenv-path",
        type=str,
        default=".env",
        help="Path to .env file for preprocessing model settings.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="",
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="LLM model name.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API key. If empty, reads DEEPSEEK_API_KEY or OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--min-law-id",
        type=int,
        default=-1,
        help="Only process laws with id >= min-law-id.",
    )
    args = parser.parse_args()

    load_dotenv(dotenv_path=args.dotenv_path)

    base_url = (
        args.base_url
        or os.getenv("law_judge_dep_base_url")
        or os.getenv("LAW_JUDGE_DEP_BASE_URL")
        or "https://api.deepseek.com/v1"
    )
    model_name = (
        args.model
        or os.getenv("law_judge_dep_model")
        or os.getenv("LAW_JUDGE_DEP_MODEL")
        or "deepseek-chat"
    )
    min_law_id = (
        args.min_law_id
        if args.min_law_id >= 0
        else int(
            os.getenv("law_judge_dep_min_law_id")
            or os.getenv("LAW_JUDGE_DEP_MIN_LAW_ID")
            or 102
        )
    )
    api_key = (
        args.api_key
        or os.getenv("law_judge_dep_api_key")
        or os.getenv("LAW_JUDGE_DEP_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("api_key")
    )
    if not api_key:
        raise ValueError("API key is required. Use --api-key or set DEEPSEEK_API_KEY/OPENAI_API_KEY.")

    criminal_law = load_criminal_law(args.criminal_law_processed)
    judicial_explanations = load_judicial_explanations(args.judicial_explanations)
    law_text_index = build_law_text_index(criminal_law)
    client = OpenAI(api_key=api_key, base_url=base_url)

    results: List[Dict[str, Any]] = []
    skipped = 0
    for law_id in tqdm(sorted(law_text_index.keys()), desc="Generating law judge deps"):
        if law_id < min_law_id:
            continue
        law_text = law_text_index[law_id]
        if not law_text:
            skipped += 1
            continue

        response = call_llm(client, model_name, JUDGE_PROMPT.format(law_text=law_text))
        judge_items = parse_judge_list(response)
        if not judge_items:
            skipped += 1
            continue

        related = build_related_laws(
            law_id=law_id,
            law_text=law_text,
            law_text_index=law_text_index,
            judicial_explanations=judicial_explanations,
        )
        results.append(
            {
                "id": law_id,
                "law": law_text,
                "related": related,
                "judge": judge_items,
            }
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Generated rows: {len(results)}")
    print(f"Skipped rows: {skipped}")
    print(f"LLM model: {model_name}")
    print(f"LLM base_url: {base_url}")
    print(f"Min law id: {min_law_id}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
