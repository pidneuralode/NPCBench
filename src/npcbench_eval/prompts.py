from __future__ import annotations

from .rubrics import O1RubricItem


GENERATION_PROMPT_TEMPLATE = """
You are assisting with a nasopharyngeal carcinoma specialist consultation.
Read the de-identified case information and answer the clinical question.
Your answer should be clinically concise, evidence-grounded, and should state
important uncertainty or missing information when relevant.

# Case
{case_input}

# Question
{question}
""".strip()


NPC_O1_GRADER_TEMPLATE = """
You are evaluating an answer in a nasopharyngeal carcinoma clinical consultation benchmark.

Your job is to look at the case, the clinical question, the assistant response, and one rubric item. Judge whether the assistant response matches the rubric item according to the criterion type shown inside the rubric item.

# Case
<<case>>

# Question
<<question>>

# Assistant Response
<<assistant_response>>

# Rubric Item
<<rubric_item>>

# Instructions
Return a JSON object with exactly the following fields: "explanation" and "criteria_met".

- "criteria_met" must be a boolean.
- "explanation" must be a brief string explaining why the assistant response does or does not satisfy the rubric item. Refer to specific content from the assistant response when possible.
- Judge only the assistant response. Do not give credit for information that appears only in the case or question unless the assistant response also uses or states it.
- First identify the criterion type in the rubric item.
- For a POSITIVE REQUIREMENT, return true only when all required elements are substantially satisfied by the assistant response.
- For a NEGATIVE ERROR CRITERION, return true only if the assistant response actually makes the described mistake, unsafe recommendation, contradiction, or omission. Return false if the assistant avoids the error, gives the correct answer, or merely discusses the issue as a caveat without committing the error.
- For a NEGATIVE ERROR CRITERION, a true judgment must be supported by specific offending content in the assistant response. If you cannot identify such content, return false.
- If a negative criterion is about omission, such as "fails to mention" or "未说明", return true only when the required information is absent from the assistant response.
- If the rubric item uses phrases such as "such as", "for example", "including", "包括", "例如", or "如", the assistant response does not need to mention every example, but it must satisfy the core clinical concept.
- Accept clinically equivalent wording, synonyms, abbreviations, and reasonable Chinese-English variants.
- Do not require exact wording from the rubric item.
- Do not give extra credit for content outside the rubric item.
- For a POSITIVE REQUIREMENT, if the response is vague, generic, or only partially satisfies the item, return false.
- If the assistant response contradicts the rubric item, return false.
- If the answer states uncertainty appropriately and the rubric item allows uncertainty or missing information, count that as satisfying the item.

# Example 1
If the rubric item is "states that the adverse event is pruritus" and the assistant response says "the patient currently has skin itching", return:

{
  "explanation": "The response identifies the current adverse event as skin itching, which is clinically equivalent to pruritus.",
  "criteria_met": true
}

# Example 2
If the rubric item is "states that the severity cannot be determined from the available information" and the assistant response directly assigns CTCAE grade 2 without explaining missing grading information, return:

{
  "explanation": "The response gives a specific CTCAE grade but does not state that severity cannot be determined from the available information.",
  "criteria_met": false
}

# Example 3: negative error criterion that is present
If the criterion type is NEGATIVE ERROR CRITERION, the rubric item is "incorrectly treats anemia as the main current adverse event", and the assistant response says "the main current adverse event is anemia", return:

{
  "explanation": "The response identifies anemia as the main current adverse event, which matches the mistake described in the rubric item.",
  "criteria_met": true
}

# Example 4: negative error criterion that is absent
If the criterion type is NEGATIVE ERROR CRITERION, the rubric item is "misclassifies cT3N0M0 as T4 or N-positive disease", and the assistant response correctly says "the clinical stage is cT3N0M0 because MRI supports T3 and there is no nodal or distant metastasis", return:

{
  "explanation": "The response correctly states cT3N0M0 and does not misclassify the case as T4 or N-positive disease.",
  "criteria_met": false
}

# Final instruction
Return only the JSON object. Do not include markdown fences or any other text.
""".strip()


def render_generation_prompt(case_input: str, question: str) -> str:
    return GENERATION_PROMPT_TEMPLATE.format(case_input=case_input, question=question)


def render_generation_messages(case_input: str, question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": render_generation_prompt(case_input, question)}]


def render_rubric_item_for_judge(rubric_item: O1RubricItem) -> str:
    is_negative = rubric_item.scoring_effect in {"major_penalty", "minor_penalty"}
    if is_negative:
        lines = [
            "Criterion type: NEGATIVE ERROR CRITERION",
            "Scoring meaning: criteria_met=true means the assistant response actually makes this error and should be penalized. criteria_met=false means the response avoids this error.",
            "Criterion:",
            rubric_item.criterion,
        ]
        return "\n".join(lines)

    lines = [
        "Criterion type: POSITIVE REQUIREMENT",
        "Scoring meaning: criteria_met=true means the assistant response satisfies this required clinical content.",
        "Criterion:",
        rubric_item.criterion,
    ]
    if rubric_item.acceptable:
        lines.extend(["", "Acceptable equivalent wording:"])
        lines.extend(f"- {text}" for text in rubric_item.acceptable)
    if rubric_item.unacceptable:
        lines.extend(["", "Unacceptable or contradictory wording:"])
        lines.extend(f"- {text}" for text in rubric_item.unacceptable)
    return "\n".join(lines)


def render_judge_prompt(
    case_input: str,
    question: str,
    assistant_response: str,
    rubric_item: O1RubricItem,
) -> str:
    return (
        NPC_O1_GRADER_TEMPLATE.replace("<<case>>", case_input)
        .replace("<<question>>", question)
        .replace("<<assistant_response>>", assistant_response)
        .replace("<<rubric_item>>", render_rubric_item_for_judge(rubric_item))
    )


def render_judge_messages(
    case_input: str,
    question: str,
    assistant_response: str,
    rubric_item: O1RubricItem,
) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": render_judge_prompt(case_input, question, assistant_response, rubric_item),
        }
    ]
