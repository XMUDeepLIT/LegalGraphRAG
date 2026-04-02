import json
from core.prompt import JUDGE_CRIME_PROMPT, JUDGE_CRIME_ALL_PROMPT


def format_law(law_used):
    res = ""
    for law in law_used:
        law["crimes"] = law.get("crimes", [])
        crimes = [c.replace("\n", " ") for c in law["crimes"] if c]
        res += f"触犯法条 {law['entry']}，该法条可判：{', '.join(crimes)}, 具体内容为：{law['description']}\n---\n"

    return res


def format_fact(facts):
    res = ""
    for fact in facts:
        res += f"罪名：{', '.join(fact['crime'])}。事实描述：{fact['description']}。\n"
    return res


def judge_crime(chatbot, law_used, retrieved_facts, case_description):
    response = chatbot.generate_response(
        JUDGE_CRIME_PROMPT.format(
            law=format_law(law_used),
            case=case_description
        ),
        max_length=4096
    )
    try:
        first = response.rfind('[')
        last = response.rfind(']') + 1
        response = response.replace('，', ',')
        response = list(set(eval(response[first:last])))
    except Exception as e:
        print(f"Error parsing response: {e}")
        response = ["无罪"]
    response = [str(x).strip() for x in response if str(x).strip()]
    return response

def judge_crime_all(chatbot, law_used, retrieved_facts, case_description):
    response = chatbot.generate_response(
        JUDGE_CRIME_ALL_PROMPT + ("输入：\n法条：\n-----\n" + format_law(law_used) + "\n-----\n待判决的案件：\n-----\n" + case_description + "\n-----\n输出："),
        max_length=4096
    )
    try:
        first = response.find('{')
        last = response.rfind('}') + 1
        response = response[first:last]
        response = json.loads(response)
    except Exception as e:
        print(f"Error parsing response: {e}")
        response = {"charge_name": "无罪", "law_article": "无罪", "term_of_imprisonment": {"death_penalty": False, "imprisonment": 0, "life_imprisonment": False}}
    return response