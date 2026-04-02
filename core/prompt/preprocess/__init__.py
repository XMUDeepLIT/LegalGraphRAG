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

__all__ = [
    "GET_FEATURES_PROMPT",
    "CASE_SEG_PROMPT",
    "PRE_JUDGE_PROMPT",
]
