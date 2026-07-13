"""Judgment related prompts"""

JUDGE_LAW_PROMPT = """
You are a professional legal AI assistant, good at analyzing legal provision applicability. Your task is to strictly evaluate whether the case facts meet the constitutive element specified in the legal provision based on the provided legal provision, auxiliary materials, judgment elements, and case facts.

**Input information:**
- **Legal provision (law)**: Provides legal provision text
- **Auxiliary materials (related)**: Judicial interpretations, related legal provisions, or supplementary explanations that may be related to the legal provision; if empty, ignore
- **Judgment element (element)**: Specific constitutive element in the legal provision that needs to be verified (e.g., "intent", "harmful result", etc.), you must focus on this element
- **Case (case)**: Describes specific case facts

**Analysis guidelines:**
1. Carefully read the legal provision text and understand its content and constitutive elements.
2. If auxiliary materials are not empty, use them to help explain the legal provision or elements.
3. Extract relevant information from case facts and compare with judgment elements.
4. Based on facts and logic, judge whether the case meets this element. If satisfied, output true; otherwise output false.

**Output format:**
- Only output "true" or "false", do not add any other text.

Now, analyze based on the following input:
law: {law_item}
related: {related}
element: {element}
case: {case}

Output:
"""

JUDGE_LAW_PROMPT_ZH = """
你是一个专业的法律AI助手，擅长分析法条适用性。你的任务是基于提供的法条、辅助材料、判断要素和案例事实，严格评估案例事实是否满足法条中指定的构成要件要素。

**输入信息：**
- **法条 (law)**: 提供法条文本
- **辅助材料 (related)**: 可能与法条相关的司法解释、相关法条或补充说明；如果为空，则忽略
- **判断要素 (element)**: 法条中需要验证的具体构成要件（例如，"故意"、"损害结果"等），你必须聚焦于这个要素
- **案例 (case)**: 描述具体案件事实

**分析指南：**
1. 仔细阅读法条文本，理解其内容和构成要件。
2. 如果辅助材料不为空，使用它来帮助解释法条或要素。
3. 从案例事实中提取相关信息，与判断要素进行对比。
4. 基于事实和逻辑，判断案例是否满足该要素。如果满足，输出true；否则输出false。

**输出格式：**
- 只输出“true”或“false”，不要添加任何其他文本。

现在，基于以下输入进行分析：
law: {law_item}
related: {related}
element: {element}
case: {case}

输出：
"""

JUDGE_LAW_PROMPT0 = """
You are a professional legal analysis assistant. Based on the provided legal provision and case analysis, judge whether this legal provision applies to this case (i.e., whether it constitutes a violation or crime).

**Input information:**
- case: Case description
- law: Legal provision text
- true_list: Parts of the legal provision that are found to be true for this case
- false_list: Parts of the legal provision that are found to be false for this case

**Analysis guidelines:**
1. Read the legal provision text and identify all relevant constitutive elements.
2. Note: true_list and false_list may be incomplete, you need to verify key elements based on the legal provision yourself.

**Output format:**
- Only output "true" or "false", do not add any other text, indicating whether this legal provision applies to this case.

Now please analyze the following input:
case: {case}
law: {law}
true_list: {true_list}
false_list: {false_list}

Output:
"""

JUDGE_LAW_PROMPT0_ZH = """
你是一个专业的法律分析助手。根据提供的法条和案件分析，判断该法条是否适用于该案件（即是否构成违法或犯罪）。

**输入信息：**
- case: 案件描述
- law: 法条文本
- true_list: 对于该案件，法条中查明为真的部分
- false_list: 对于该案件，法条中查明为假的部分

**分析指南：**
1. 阅读法条文本，识别其所有相关构成要件。
2. 注意：true_list和false_list可能不完整，你需要基于法条自行验证关键要件。

**输出格式：**
- 只输出“true”或“false”，不要添加任何其他文本，表示该法条是否适用于该案件。

现在请分析以下输入：
case: {case}
law: {law}
true_list: {true_list}
false_list: {false_list}

输出：
"""

JUDGE_LAW_PROMPT1 = """
You are a professional legal analysis assistant. Please directly judge whether the provided legal provision text applies to this case based on the specific case description. The legal provision may be substantive law, procedural law, or interpretative provisions.

Input information:
Legal provision
Case

Analysis requirements:
- Judge whether the case situation falls within the scope of this legal provision
- Only consider the meaning of the legal provision text itself, do not make inferences beyond the text
- Please only output true or false, indicating whether this legal provision applies to this case.

Example:
Input:
Legal provision: "If a party cannot participate in litigation due to force majeure, suspend the litigation"
Case: "The defendant cannot appear in court due to earthquake"

Output:
true

Now please analyze:
Legal provision: {law}
Case: {case}

Output:
"""

JUDGE_LAW_PROMPT1_ZH = """
你是一个专业的法律分析助手。请根据提供的法条文本和具体案件描述，直接判断该法条是否适用于此案件。法条可能是实体法、程序法或解释性规定。

输入信息：
法条
案件

分析要求：
- 判断案件情况是否落入该法条的规范范围内
- 只考虑法条文字本身的含义，不做超出文本的推断
- 请只输出 true 或 false，表示该法条是否适用于此案件。

示例：
输入：
法条: "当事人因不可抗力不能参加诉讼的，中止诉讼"
案件: "被告因地震无法出庭应诉"

输出：
true

现在请分析：
法条: {law}
案件: {case}

输出：
"""

JUDGE_CRIME_PROMPT = """
You are a professional legal analysis assistant. Please judge the charge for the defendant based on candidate charges.

Note:
- Unless necessary, do not judge multiple charges, but choose the most appropriate charge.
- Your charge selection process must follow these steps:
  1. **Behavior quantity determination**: Judge how many independent criminal behaviors exist in the case. Pay attention to distinguishing between one behavior violating multiple legal provisions (imaginative concurrence) and multiple behaviors violating different legal provisions (concurrent punishment for multiple crimes).
  2. **Final charge application**: For each independent criminal behavior, determine the final charge to be applied. When multiple legal provisions are satisfied, analyze legal provision concurrence or concurrent punishment for multiple crimes based on criminal behavior: for imaginative concurrence (i.e., one behavior violating multiple legal provisions), follow the "punish the heavier crime" principle; for multiple independent behaviors (i.e., concurrent punishment for multiple crimes), apply corresponding legal provisions separately.
- The inferred charge must have legal basis support and be closely related to case facts, not any speculation.
- Your output must be only one Python list (i.e., list(str) format), containing only the final charges reasonably derived from the above analysis process.

Input:
Legal provision:
-----
{law}
-----
Case to be judged:
-----
{case}
-----

Output:
"""

JUDGE_CRIME_PROMPT_ZH = """
你是一个专业的法律分析助手。请根据候选罪名，为被告判决罪名。

注意：
- 如非必要，不要判定多个罪名，而是选择最符合的罪名。
- 你选择罪名的过程必须遵循以下步骤：
  1. **行为数量判定**：判断案件中存在几个独立的犯罪行为。注意区分一个行为触犯多个法条（想象竞合）与多个行为触犯不同法条（数罪并罚）的情形。
  2. **罪名最终适用**：对于每个独立的犯罪行为，确定应适用的最终罪名。当多个法条被满足时，需根据犯罪行为进行法条竞合或数罪并罚分析：对于想象竞合（即一个行为触犯多个法条），遵循“从一重罪处断”原则；对于多个独立行为（即数罪并罚），则分别适用相应法条。
- 推断出的罪名必须有法律依据支持，且与案件事实紧密相关，不是任何臆断。
- 你的输出必须只有一个Python列表（即 list(str) 格式），仅包含根据上述分析过程合理推导出的最终罪名。

输入：
法条:
-----
{law}
-----
待判决的案件:
-----
{case}
-----

输出：
"""

JUDGE_CRIME_ALL_PROMPT = """
You are a professional legal analysis assistant. Please judge the charge for the defendant based on candidate charges, and predict applicable legal provisions and sentence range.

Note:
- Unless necessary, do not judge multiple charges, but choose the most appropriate charge.
- Your charge selection process must follow these steps:
  1. **Behavior quantity determination**: Judge how many independent criminal behaviors exist in the case. Pay attention to distinguishing between one behavior violating multiple legal provisions (imaginative concurrence) and multiple behaviors violating different legal provisions (concurrent punishment for multiple crimes).
  2. **Final charge application**: For each independent criminal behavior, determine the final charge to be applied. When multiple legal provisions are satisfied, analyze legal provision concurrence or concurrent punishment for multiple crimes based on criminal behavior: for imaginative concurrence (i.e., one behavior violating multiple legal provisions), follow the "punish the heavier crime" principle; for multiple independent behaviors (i.e., concurrent punishment for multiple crimes), apply corresponding legal provisions separately.
  3. **Legal provision and sentence prediction**: Clearly specify the specific legal provisions as the basis for judgment, and reasonably predict the possible sentence range based on case circumstances, legal provisions, and judicial practice.
- The inferred charge must have legal basis support and be closely related to case facts, not any speculation.
- Your output must be a **JSON object**, and only contain this JSON object. The structure of this JSON object is as follows:
```json
{{
    "charge_name": list(str), // Charge name
    "law_article": list(str), // Legal provisions as basis, e.g. ["Article 232", "Article 233"]
    "term_of_imprisonment": {{
        "death_penalty": boolean, // Whether death penalty applies
        "imprisonment": integer, // Fixed-term imprisonment sentence, unit: months
        "life_imprisonment": boolean // Whether life imprisonment applies
    }} // Sentence range
}}
```

"""

JUDGE_CRIME_ALL_PROMPT_ZH = """
你是一个专业的法律分析助手。请根据候选罪名，为被告判决罪名，并预测适用的法条及刑期范围。

注意：
- 如非必要，不要判定多个罪名，而是选择最符合的罪名。
- 你选择罪名的过程必须遵循以下步骤：
  1. **行为数量判定**：判断案件中存在几个独立的犯罪行为。注意区分一个行为触犯多个法条（想象竞合）与多个行为触犯不同法条（数罪并罚）的情形。
  2. **罪名最终适用**：对于每个独立的犯罪行为，确定应适用的最终罪名。当多个法条被满足时，需根据犯罪行为进行法条竞合或数罪并罚分析：对于想象竞合（即一个行为触犯多个法条），遵循“从一重罪处断”原则；对于多个独立行为（即数罪并罚），则分别适用相应法条。
  3. **法条与刑期预测**：明确判决依据的具体法律条文，并基于案件情节、法律规定及司法实践，合理预测可能的刑期范围。
- 推断出的罪名必须有法律依据支持，且与案件事实紧密相关，不是任何臆断。
- 你的输出必须是一个 **JSON 对象**，且只包含该 JSON 对象。该 JSON 对象的结构如下：
```json
{{
    "charge_name": list(str), // 罪名名称
    "law_article": list(str), // 依据的法律条文，eg. ["第232条", "第233条"]
    "term_of_imprisonment": {{
        "death_penalty": boolean, // 是否适用死刑
        "imprisonment": integer, // 有期徒刑刑期，单位：月
        "life_imprisonment": boolean // 是否适用无期徒刑
    }} // 刑期范围
}}
```

"""

__all__ = [
    "JUDGE_LAW_PROMPT",
    "JUDGE_LAW_PROMPT_ZH",
    "JUDGE_LAW_PROMPT0",
    "JUDGE_LAW_PROMPT0_ZH",
    "JUDGE_LAW_PROMPT1",
    "JUDGE_LAW_PROMPT1_ZH",
    "JUDGE_CRIME_PROMPT",
    "JUDGE_CRIME_PROMPT_ZH",
    "JUDGE_CRIME_ALL_PROMPT",
    "JUDGE_CRIME_ALL_PROMPT_ZH",
]
