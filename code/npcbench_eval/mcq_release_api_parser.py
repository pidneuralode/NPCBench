from __future__ import annotations
import re
from difflib import SequenceMatcher
from typing import Any

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()

def resolve_candidates(
    method: str, candidates: list[tuple[str, str, Any]]
) -> tuple[str, bool, str, bool]:
    values = [candidate[1] for candidate in candidates]
    unique = sorted(set(values))
    if not values:
        return "", False, method, False
    if len(unique) == 1:
        return values[-1], True, method, False
    return "", False, method, True

def parse_answer(response: str, options: dict[str, str]) -> dict[str, Any]:
    text = response or ""
    labels = set(options.keys())

    boxed_patterns = [
        r"\\boxed\s*\{\s*([A-Ja-j])\s*\}",
        r"\\boxed\s*\{\s*\\(?:text|mathrm|mathbf|mathit|operatorname|textbf)\s*\{\s*([A-Ja-j])\s*\}\s*\}",
    ]
    boxed: list[tuple[str, str, Any]] = []
    for pattern in boxed_patterns:
        boxed.extend(
            ("boxed", match.group(1).upper(), match.group(0))
            for match in re.finditer(pattern, text)
            if match.group(1).upper() in labels
        )
    if boxed:
        pred, valid, method, conflict = resolve_candidates("boxed", boxed)
        return parse_result(pred, valid, method, boxed, conflict)

    explicit_patterns = [
        r"<answer>\s*([A-Ja-j])\s*</answer>",
        r"(?:final\s+answer|answer|答案|选择|我选|正确选项)\s*(?:is|为|是|:|：)?\s*(?:option|选项)?\s*([A-Ja-j])\b",
        r"(?:the\s+correct\s+option\s+is|the\s+best\s+answer\s+is)\s*([A-Ja-j])\b",
    ]
    explicit: list[tuple[str, str, Any]] = []
    for pattern in explicit_patterns:
        explicit.extend(
            ("explicit_marker", match.group(1).upper(), match.group(0))
            for match in re.finditer(pattern, text, flags=re.IGNORECASE)
            if match.group(1).upper() in labels
        )
    if explicit:
        pred, valid, method, conflict = resolve_candidates("explicit_marker", explicit)
        return parse_result(pred, valid, method, explicit, conflict)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        last_line = lines[-1].strip().strip("`*_# ")
        last_line = re.sub(r"^[\-\*\d\.\)\s]+", "", last_line)
        line_match = re.fullmatch(
            r"(?:option|选项)?\s*([A-Ja-j])\s*(?:[\.。!！]?)", last_line, flags=re.IGNORECASE
        )
        if line_match and line_match.group(1).upper() in labels:
            candidate = [("last_line_letter", line_match.group(1).upper(), lines[-1])]
            return parse_result(candidate[0][1], True, "last_line_letter", candidate, False)

    concise = text.strip().strip("`*_# ")
    short_labeled_match = re.fullmatch(
        r"([A-Ja-j])\s*[\.\):：、]\s*\S.{0,80}", concise, flags=re.IGNORECASE | re.DOTALL
    )
    if short_labeled_match and short_labeled_match.group(1).upper() in labels:
        candidate = [("short_labeled_option", short_labeled_match.group(1).upper(), concise)]
        return parse_result(candidate[0][1], True, "short_labeled_option", candidate, False)

    concise_match = re.fullmatch(
        r"(?:\\boxed\{)?\s*([A-Ja-j])\s*(?:\})?\s*[\.。]?", concise, flags=re.IGNORECASE
    )
    if concise_match and concise_match.group(1).upper() in labels:
        candidate = [("direct_letter", concise_match.group(1).upper(), concise)]
        return parse_result(candidate[0][1], True, "direct_letter", candidate, False)

    tail = text[-1200:]
    tail_patterns = [
        r"(?:therefore|thus|so|因此|所以|综上)\D{0,80}(?:answer|答案|选择|选项)\D{0,20}([A-Ja-j])\b",
        r"(?:recommend|choose|select|应选择|推荐选择)\D{0,20}([A-Ja-j])\b",
    ]
    tail_candidates: list[tuple[str, str, Any]] = []
    for pattern in tail_patterns:
        tail_candidates.extend(
            ("tail_decision_marker", match.group(1).upper(), match.group(0))
            for match in re.finditer(pattern, tail, flags=re.IGNORECASE)
            if match.group(1).upper() in labels
        )
    if tail_candidates:
        pred, valid, method, conflict = resolve_candidates(
            "tail_decision_marker", tail_candidates
        )
        return parse_result(pred, valid, method, tail_candidates, conflict)

    normalized_response = normalize_text(tail)
    containment = []
    for label, option_text in options.items():
        normalized_option = normalize_text(option_text)
        if len(normalized_option) >= 8 and normalized_option in normalized_response:
            containment.append(("option_text_containment", label, option_text[:120]))
    if containment:
        pred, valid, method, conflict = resolve_candidates(
            "option_text_containment", containment
        )
        return parse_result(pred, valid, method, containment, conflict)

    best_label = ""
    best_score = 0.0
    comparison_window = normalize_text(text[-2000:])
    for label, option_text in options.items():
        normalized_option = normalize_text(option_text)
        if not normalized_option:
            continue
        window = comparison_window[-max(120, len(normalized_option) * 2) :]
        score = SequenceMatcher(None, normalized_option, window).ratio()
        if score > best_score:
            best_label = label
            best_score = score
    if best_label and best_score >= 0.88:
        candidate = [("option_text_similarity", best_label, round(best_score, 4))]
        return parse_result(best_label, True, "option_text_similarity", candidate, False)

    return parse_result("", False, "unparsed", [], False)

def parse_result(
    pred_answer: str,
    valid: bool,
    method: str,
    candidates: list[tuple[str, str, Any]],
    conflict: bool,
) -> dict[str, Any]:
    return {
        "pred_answer": pred_answer,
        "parse_valid": valid,
        "parse_method": method,
        "parse_conflict": conflict,
        "parse_candidates": [
            {"method": method_name, "answer": answer, "evidence": evidence}
            for method_name, answer, evidence in candidates
        ],
    }
