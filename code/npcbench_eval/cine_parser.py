"""Captured historical cine parser; distinct from static MRI parsing."""
import re
from npcbench_eval.l3_parser import extract_boxed_content, extract

def parse_response(response):
    response = response.lower()
    if "boxed" in response:
        response = extract_boxed_content(response)
    elif "<answer>" in response:
        response = extract(response,"answer")

    # Some chat templates expose generation channels in the decoded text, e.g.
    # "analysis ... assistantfinalB". Prefer these explicit final-answer
    # markers over fuzzy matching the whole reasoning trace against options.
    tail = response[-1200:]
    explicit_letter_patterns = [
        r"assistant\s*final\s*[:：]?\s*([a-j])\b",
        r"assistantfinal\s*[:：]?\s*([a-j])\b",
        r"(?:final answer is|final answer|answer is|answer:|答案是|答案[:：]?|选项[:：]?|选择)\s*(?:option|选项)?\s*([a-j])\b",
        r"(?:therefore|thus|so|因此|所以)[^a-j]{0,30}(?:answer|答案|选项)[^a-j]{0,15}([a-j])\b",
    ]
    for pattern in explicit_letter_patterns:
        matches = re.findall(pattern, tail)
        if matches:
            return matches[-1]

    trailing_letter = re.search(r"(?:^|[\s\"'`：:])([a-j])\s*$", response.strip())
    if trailing_letter:
        return trailing_letter.group(1)

    answer_patterns = [
        "final answer is",
        "final answer",
        "**answer**:",
        "**answer**",
        "*answer*:",
        "**answer:**",
        "answer is",
        "answer:",
        "答案是",
        "答案:",
        "答案：",
    ]
    for answer_pattern in answer_patterns:
        if answer_pattern in response:
            response = response.split(answer_pattern)[-1]
    
    return response
