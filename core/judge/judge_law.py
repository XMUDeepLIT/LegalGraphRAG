from core.prompt import JUDGE_LAW_PROMPT, JUDGE_LAW_PROMPT0, JUDGE_LAW_PROMPT1


def judge_law(chatbot, case_description, law):
    if type(law) == str:
        judge_res = chatbot.generate_response(
            JUDGE_LAW_PROMPT1.format(law=law, case=case_description), max_length=128)
        if "true" in judge_res.lower():
            return True, ""
        else:
            return False, ""
    true_list = []
    false_list = []
    for judge in law["judge_dep"]:
        judge_res = chatbot.generate_response(JUDGE_LAW_PROMPT.format(
            law_item=law["description"].replace("\n", ""),
            related=law["related_laws"],
            element=judge,
            case=case_description
        ), max_length=128)
        if "true" in judge_res.lower():
            true_list.append(judge)
        elif "false" in judge_res.lower():
            false_list.append(judge)

    res = chatbot.generate_response(JUDGE_LAW_PROMPT0.format(
        case=case_description, law=law["description"], true_list=true_list, false_list=false_list), max_length=1024)
    if "true" in res.lower():
        return True, res
    return False, res
