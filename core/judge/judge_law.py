from core.prompt import get_prompt


def verify_article(chatbot, case_description, article):
    """Check one article against its diagnostic checklist and judicial interpretations."""
    if type(article) == str:
        judge_res = chatbot.generate_response(
            get_prompt("JUDGE_LAW_PROMPT1").format(law=article, case=case_description), max_length=128)
        if "true" in judge_res.lower():
            return True, ""
        else:
            return False, ""
    satisfied = []
    unsatisfied = []
    diagnostic_checklist = article["judge_dep"]
    judicial_interpretations = article["related_laws"]
    for diagnostic_item in diagnostic_checklist:
        judge_res = chatbot.generate_response(get_prompt("JUDGE_LAW_PROMPT").format(
            law_item=article["description"].replace("\n", ""),
            related=judicial_interpretations,
            element=diagnostic_item,
            case=case_description
        ), max_length=128)
        if "true" in judge_res.lower():
            satisfied.append(diagnostic_item)
        elif "false" in judge_res.lower():
            unsatisfied.append(diagnostic_item)

    res = chatbot.generate_response(get_prompt("JUDGE_LAW_PROMPT0").format(
        case=case_description, law=article["description"], true_list=satisfied, false_list=unsatisfied), max_length=1024)
    if "true" in res.lower():
        return True, res
    return False, res
