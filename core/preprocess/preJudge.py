from core.prompt import PRE_JUDGE_PROMPT


def pre_judge(model, case) -> list:
    prompt_with_case = PRE_JUDGE_PROMPT.format(case_text=case)
    response = model.generate_response(prompt_with_case)
    try:
        # 尝试将响应解析为Python列表
        first_bracket = response.find('[')
        last_bracket = response.rfind(']')
        candidates = eval(response[first_bracket:last_bracket + 1])
        if isinstance(candidates, list) and all(isinstance(item, str) for item in candidates):
            return candidates[:3]
        else:
            return []
    except Exception as e:
        # 如果解析失败，返回空列表
        return []