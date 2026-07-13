"""Preprocessing related prompts"""

GET_FEATURES_PROMPT = """
As an AI model, your task is to process criminal case inputs. The input will include a description text of a criminal case and the defendant's name. Please extract keywords from the description and classify them into the following four categories: Defendant Basic Information, Criminal Acts, Victim/Property Characteristics, Intent and Remorse. The output must be a JSON object containing only JSON, without any additional text, explanations, or error messages.

Keyword explanations:
- Defendant Basic Information: Extract legal characteristics related to the defendant, such as age group, prior criminal record, occupation category, avoiding specific age numbers and specific work unit names. The defendant's name is not important.
- Criminal Acts: Extract the legal type and important methods of criminal acts, only extract methods with legal significance, avoid specific time and location details.
- Victim/Property Characteristics: Extract type characteristics of the crime target, such as property nature, location type, avoid specific place names and specific amount numbers (can be summarized as "large amount" etc.).
- Intent and Remorse: Extract legal descriptions of subjective intent and remorse performance.

JSON format requirements:
- Use double quotes to enclose keys and string values.
- Each key corresponds to a category, and the value is a string array containing keywords extracted from the description (if a category has no keywords, use an empty array `[]`).
- Key names must be: "defendant_info", "criminal_acts", "victim_property_details", "intent_remorse".

Output example (for reference only, actual output should be based on input content):
{
"defendant_info": ["adult", "prior record", "state employee"],
"criminal_acts": ["theft", "breaking and entering"],
"victim_property_details": ["private residence", "large amount"],
"intent_remorse": ["direct intent", "voluntary surrender"]
}

Please ensure only output JSON object.
Now please process the following case:
"""

GET_FEATURES_PROMPT_ZH = """
作为AI模型，你的任务是处理刑事案件的输入。输入将包括一个刑事案件的描述文本和被告人姓名。请从描述中提取关键词，并将它们分类到以下四个类别：被告基本信息、犯罪行为、犯罪对象特征、主观及悔罪表现。输出必须是一个JSON对象，且只包含JSON，没有任何额外的文本、解释或错误消息。

关键词解释：
- 被告基本信息：提取与被告人相关的法律特征，如年龄阶段、前科情况、职业类别，避免具体年龄数字、具体工作单位名称。被告人姓名不重要。
- 犯罪行为：提取犯罪行为的法律类型和重要方式，仅提取有法律意义的方式，避免具体时间、地点细节。
- 犯罪对象特征：提取犯罪对象的类型特征，如财物性质、地点类型，避免具体地名、具体金额数字（可概括为“数额较大”等）。
- 主观及悔罪表现：提取主观意图和悔罪表现的法律描述。

JSON格式要求：
- 使用双引号包围键和字符串值。
- 每个键对应一个类别，值是一个字符串数组，包含从描述中提取的关键词（如果某个类别没有关键词，使用空数组 `[]`）。
- 键名必须为： "defendant_info"、"criminal_acts"、"victim_property_details"、"intent_remorse"。

输出示例（仅供参考，实际输出应基于输入内容）：
{
"defendant_info": ["成年人", "有前科", "国家工作人员"],
"criminal_acts": ["盗窃", "入户"],
"victim_property_details": ["私人住宅", "数额较大"],
"intent_remorse": ["直接故意", "自首"]
}

请确保只输出JSON对象。
现在请你处理以下案件：
"""

CASE_SEG_PROMPT = """
You are a professional legal analysis assistant. Your task is to organize an objective factual description about the defendant based on the following case description and defendant name.

### Input format:
- Case description: {fact}
- Defendant name: {name}

### Notes:
- **Based on input content**: Only organize based on the provided case description, do not add any external information or assumptions.
- **Objectivity requirement**: The description must be strictly objective, avoid including judgment results, legal evaluations, or subjective analysis (such as motive inference or emotional coloring).
- **Completeness requirement**: Even if some behaviors are not directly initiated by the defendant, if these behaviors are related to the defendant (for example, constituting the cause and effect, background events, or directly related to the defendant's behavior), they should also be included in the factual description to ensure complete context.
- **Output format**: Directly output the organized objective factual description, the content should be concise and accurate, avoid adding irrelevant introductions, summaries, or comments.
- **Focus limitation**: The description should focus on the defendant's behavior, role, and related events, and not involve other unrelated parties or minor details, unless they have a clear connection with the defendant.

Please process the input information according to the above requirements.
"""

CASE_SEG_PROMPT_ZH = """
你是一个专业的法律分析助手。你的任务是根据以下案件描述和被告姓名，整理出关于该被告的客观事实描述。

### 输入格式：
- 案件描述: {fact}
- 被告姓名: {name}

### 注意事项：
- **基于输入内容**：只根据提供的案件描述进行整理，不添加任何外部信息或假设。
- **客观性要求**：描述必须严格客观，避免包含判决结果、法律评价、主观分析（如动机推断或情感色彩）。
- **完整性要求**：即使某些行为不是由被告直接发起，但如果这些行为与被告相关（例如，构成案件的前因后果、背景事件或与被告行为有直接关联），也应包括在事实描述中，以确保上下文完整。
- **输出格式**：直接输出整理后的客观事实描述，内容应简洁、准确，避免添加无关的引言、总结或评论。
- **焦点限制**：描述应围绕被告的行为、角色及相关事件展开，不涉及其他无关方或次要细节，除非它们与被告有明确关联。

请根据以上要求处理输入信息。
"""

PRE_JUDGE_PROMPT = """
As a criminal law analysis expert, please strictly analyze the following case based on the Criminal Law of the People's Republic of China, and output possible charges according to the following rules:
1. Only output reasonably possible charges (confidence > 30%)
2. Sort by possibility from high to low
3. If there is an obvious main charge (confidence > 70%), prioritize outputting that charge. If you are very certain that the charge is unique, only output that charge
4. If the possibility of other charges < 10%, exclude them
5. Output must be in Python list format: ['charge1', 'charge2', ...]

Ensure the output starts with "[" and only contains candidate charges that match the case details.
If the description does not match any charge, output an empty list []. Do not add any additional explanations or text.

Please analyze the following case:
{case_text}
"""

PRE_JUDGE_PROMPT_ZH = """
作为刑事法律分析专家，请严格依据《中华人民共和国刑法》分析以下案件，按以下规则输出可能的罪名：
1. 仅输出合理可能的罪名（置信度>30%）
2. 按可能性从高到低排序
3. 若存在明显主要罪名（置信度>70%），优先输出该罪名，如果你非常肯定该罪名是唯一的，则只输出该罪名
4. 若其他罪名可能性<10%则排除
5. 输出必须为Python列表格式：['罪名1', '罪名2', ...]

确保输出以"["开头，并仅包含符合案件细节的候选罪名。
如果描述不匹配任何罪名，输出空列表[]。不要添加任何额外解释或文本。

请分析以下案件：
{case_text}
"""

__all__ = [
    "GET_FEATURES_PROMPT",
    "GET_FEATURES_PROMPT_ZH",
    "CASE_SEG_PROMPT",
    "CASE_SEG_PROMPT_ZH",
    "PRE_JUDGE_PROMPT",
    "PRE_JUDGE_PROMPT_ZH",
]
