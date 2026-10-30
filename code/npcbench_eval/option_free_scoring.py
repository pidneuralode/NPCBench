"""Frozen answer-scoring functions; no model calls."""
from __future__ import annotations
import json
import re
from typing import Any

def normalize_text(value: str) -> str:
    value = str(value or "").strip().lower()
    value = (
        value.replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .replace("％", "%")
        .replace("＋", "+")
    )
    return re.sub(r"[\s，。；：、,.!！?？()（）\[\]【】\"'`]+", "", value)

def concept_alternatives(concept: str) -> list[str]:
    text = str(concept or "").strip()
    if not text:
        return []
    parts = re.split(r"\s*(?:或|/|\bor\b)\s*", text, flags=re.I)
    return [normalize_text(part) for part in parts if normalize_text(part)]

def contains_phrase(answer: str, phrase: str) -> bool:
    needle = normalize_text(phrase)
    return bool(needle) and needle in normalize_text(answer)

def contains_concept(answer: str, concept: str) -> bool:
    normalized = normalize_text(answer)
    alternatives = concept_alternatives(concept)
    return any(alternative in normalized for alternative in alternatives)

def extract_answer(prediction: dict[str, Any]) -> str:
    parsed = prediction.get("parsed_response")
    if isinstance(parsed, dict):
        answer = parsed.get("answer")
        if isinstance(answer, list):
            return "；".join(str(value) for value in answer)
        if isinstance(answer, dict):
            return "；".join(f"{key}: {value}" for key, value in answer.items())
        if answer is not None:
            return str(answer)
        answers = parsed.get("answers")
        if isinstance(answers, list):
            return "；".join(str(value) for value in answers)
    return ""

def align_answer_components(answer: str, expected_count: int) -> tuple[list[str], str]:
    if expected_count <= 1:
        return [answer.strip()], "single_component"

    marker_pattern = re.compile(
        r"(?<![A-Za-z0-9])"
        r"(?:\(?([1-9]\d*)\)?[\.、:：\)）]|（([1-9]\d*)）)\s*"
    )
    numbered = []
    for match in marker_pattern.finditer(answer):
        number = int(match.group(1) or match.group(2))
        if number <= expected_count:
            numbered.append((number, match))
    for start, (number, _) in enumerate(numbered):
        if number != 1:
            continue
        candidate = numbered[start : start + expected_count]
        if [value for value, _ in candidate] != list(
            range(1, expected_count + 1)
        ):
            continue
        components = []
        for index, (_, match) in enumerate(candidate):
            end = (
                candidate[index + 1][1].start()
                if index + 1 < len(candidate)
                else len(answer)
            )
            components.append(answer[match.end() : end].strip(" \n\r；;。"))
        if all(components):
            return components, "numbered_components"

    delimiters = (
        (r"[\n\r；;]+", "line_or_semicolon_components"),
        (r"\s*→\s*", "arrow_components"),
        (r"\s*、\s*", "enumeration_components"),
        (r"\s*,\s*", "comma_components"),
    )
    for pattern, status in delimiters:
        components = [
            value.strip()
            for value in re.split(pattern, answer)
            if value.strip()
        ]
        if len(components) == expected_count:
            return components, status
    return [], "component_alignment_failed"

def score_slot(answer: str, slot: dict[str, Any]) -> dict[str, Any]:
    raw_incorrect_hits = [
        value
        for value in slot["explicitly_incorrect_or_insufficient"]
        if contains_phrase(answer, value)
    ]
    aliases = [slot["minimum_sufficient_answer"], *slot["accepted_aliases"]]
    alias_hits = [value for value in aliases if contains_phrase(answer, value)]
    # A short "insufficient" phrase may be intentionally contained in a fully
    # sufficient answer (for example 放疗 inside 单纯放疗). Do not let that
    # substring shadow a physician-approved full answer. Independent negative
    # content, such as an additional contraindicated regimen, still overrides.
    incorrect_hits = [
        value
        for value in raw_incorrect_hits
        if not any(
            normalize_text(value) in normalize_text(alias)
            for alias in alias_hits
        )
    ]
    required = slot["required_concepts"]
    concept_hits = [value for value in required if contains_concept(answer, value)]
    required_complete = bool(required) and len(concept_hits) == len(required)
    insufficient_overlaps_required = bool(incorrect_hits) and all(
        any(
            normalize_text(value) in alternative
            for concept in required
            for alternative in concept_alternatives(concept)
        )
        for value in incorrect_hits
    )

    if alias_hits and not incorrect_hits:
        correct = True
        reason = "physician_alias_match"
    elif incorrect_hits and required_complete and insufficient_overlaps_required:
        correct = False
        reason = "rubric_conflict_needs_semantic_review"
    elif incorrect_hits:
        correct = False
        reason = "explicitly_incorrect_or_insufficient"
    elif required_complete:
        correct = True
        reason = "physician_required_concepts_match"
    else:
        correct = False
        reason = "unmatched"
    return {
        "correct": correct,
        "reason": reason,
        "alias_hits": alias_hits,
        "concept_hits": concept_hits,
        "incorrect_hits": incorrect_hits,
    }
