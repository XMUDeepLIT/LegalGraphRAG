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

__all__ = [
    "JUDGE_LAW_PROMPT",
    "JUDGE_LAW_PROMPT0",
    "JUDGE_LAW_PROMPT1",
    "JUDGE_CRIME_PROMPT",
    "JUDGE_CRIME_ALL_PROMPT",
]
