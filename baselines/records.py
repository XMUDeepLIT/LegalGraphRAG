"""Shared evaluation record written by every baseline."""
from typing import Any, Callable, Dict, List


FACT_CHARS = 1024


def defendant_names(case: Dict[str, Any]) -> List[str]:
    names = case.get("name", [])
    if isinstance(names, str):
        return [names]
    return list(names)


def case_record(case: Dict[str, Any], fact: str, judge_res: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "id": case.get("id"),
        "fact": fact,
        "true_charge": case.get("crime", []),
        "law_article": case.get("laws", []),
        "term_of_imprisonment": case.get("term_of_imprisonment", {}),
        "judge_res": judge_res,
    }


def judge_each(cases, answer: Callable[[str, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run one defendant prompt per name. ``fact`` is the first 1,024 characters."""
    results = []
    for case in cases:
        fact = case.get("fact", "")[:FACT_CHARS]
        judge_res = [answer(name, fact) for name in defendant_names(case)]
        results.append(case_record(case, fact, judge_res))
    return results
