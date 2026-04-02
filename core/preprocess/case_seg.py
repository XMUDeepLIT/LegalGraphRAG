import json
from core.prompt import CASE_SEG_PROMPT


def segment_case_text_withname(model, case_text, criminals):
    cases = []
    for name in criminals:
        prompt2_formatted = CASE_SEG_PROMPT.format(fact=case_text, name=name)
        response2 = model.generate_response(prompt2_formatted, max_length=1024)
        if response2 != "":
            cases.append({"name": name, "description": response2.strip()})
        else:
            cases.append({"name": name, "description": case_text.strip()})

    return cases
