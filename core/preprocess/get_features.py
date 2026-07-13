import json
from core.prompt import get_prompt


def get_features(model, cases):
    fact = cases["description"]
    name = cases["name"]

    prompt_formatted = get_prompt("GET_FEATURES_PROMPT") + get_prompt(
        "GET_FEATURES_INPUT_TEMPLATE"
    ).format(name=name, fact=fact)
    response = model.generate_response(prompt_formatted)
    first = response.find("{")
    last = response.rfind("}") + 1
    json_data = {}
    if first != -1 and last != -1:
        json_str = response[first:last]
        try:
            json_data = json.loads(json_str)
        except json.JSONDecodeError:
            print("JSON解析错误")
    else:
        print("未找到有效的JSON字符串")
    return json_data
