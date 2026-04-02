import json
from core.prompt import GET_FEATURES_PROMPT


def get_features(model, cases):
    fact = cases["description"]
    name = cases["name"]

    prompt_formatted = GET_FEATURES_PROMPT + "\n被告人姓名：{}\n案件事实：{}".format(name, fact)
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
