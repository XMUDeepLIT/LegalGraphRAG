"""Serialize law and case JSON into the baseline corpus text."""
import json
from pathlib import Path
from typing import Union


def law_corpus_text(law_db) -> str:
    """One block per statute: 法律条文编号, 条文内容, 相关罪名."""
    law_str = ""
    for law in law_db:
        law_str += f"法律条文编号: {law['id']}\n"
        for item in law["items"]:
            crimes = ", ".join(item["crime"])
            law_str += f"条文内容: {item['text']}\n"
            law_str += f"相关罪名: {crimes}\n"
            law_str += "----------------------------------------\n"
        law_str += "\n"
    return law_str


def case_corpus_text(cases) -> str:
    """One block per case: 被告人, 案件事实, 相关法律条文, 涉及罪名. Fact line endings are LF."""
    cases_str = ""
    for case in cases:
        cases_str += f"被告人: {case['name']}\n"
        fact = case["fact"].replace("\r\n", "\n").replace("\r", "\n")
        cases_str += f"案件事实: {fact}\n"
        cases_str += f"相关法律条文: {', '.join(case['law'])}\n"
        cases_str += f"涉及罪名: {', '.join(case['crime'])}\n"
        cases_str += "----------------------------------------\n\n"
    return cases_str


def load_json(path: Union[str, Path]):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: Union[str, Path]) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_law_corpus(law_json=None, law_text=None) -> str:
    if law_text:
        return read_text(law_text)
    if not law_json:
        raise ValueError("Pass law_json or law_text")
    return law_corpus_text(load_json(law_json))


def load_case_corpus(cases_json=None, cases_text=None) -> str:
    if cases_text:
        return read_text(cases_text)
    if not cases_json:
        raise ValueError("Pass cases_json or cases_text")
    return case_corpus_text(load_json(cases_json))
