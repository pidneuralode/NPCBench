"""Frozen answer-scoring functions; no model calls."""
from __future__ import annotations
import json
import re
from typing import Any

def normalize_parsed_object(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep answer parseable when a model returns a structured value."""
    normalized = dict(payload)
    if "answer" not in normalized and normalized:
        normalized["answer"] = "；".join(
            f"{key}: {value}" for key, value in normalized.items()
        )
    answer = normalized.get("answer")
    if isinstance(answer, dict):
        normalized["answer"] = "；".join(
            f"{key}: {value}" for key, value in answer.items()
        )
    elif isinstance(answer, list):
        normalized["answer"] = "；".join(str(value) for value in answer)
    elif answer is not None and not isinstance(answer, str):
        normalized["answer"] = str(answer)
    return normalized

def parse_relaxed_answer_object(text: str) -> dict[str, Any] | None:
    """Recover an explicit answer block from JSON-like model output.

    This fallback is deliberately narrow: it only accepts text containing an
    explicit ``answer`` key. It handles common invalid JSON such as unquoted
    values with clinical units, while avoiding inference from free-form prose.
    """
    answer_object = re.search(
        r'["\']?answer["\']?\s*:\s*\{(?P<body>.*?)\}',
        text,
        flags=re.I | re.S,
    )
    if answer_object:
        pairs = re.findall(
            r'["\']?([^"\'{}:,\n]+)["\']?\s*:\s*([^,}\n]+)',
            answer_object.group("body"),
        )
        if pairs:
            answer = "；".join(
                f"{key.strip()}: {value.strip().strip(chr(34) + chr(39))}"
                for key, value in pairs
                if key.strip() and value.strip()
            )
            if answer:
                return {"answer": answer}

    scalar_answer = re.search(
        r'["\']?answer["\']?\s*:\s*(?P<value>[^,}\n]+)',
        text,
        flags=re.I,
    )
    if scalar_answer:
        answer = scalar_answer.group("value").strip().strip("\"'")
        if answer:
            return {"answer": answer}
    return None

def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    cleaned = str(text or "").strip()
    without_think = re.sub(
        r"<think\b[^>]*>.*?</think>",
        "",
        cleaned,
        flags=re.I | re.S,
    ).strip()
    candidates = [without_think, cleaned]
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(
            r"```(?:json)?\s*(.*?)\s*```",
            without_think,
            flags=re.I | re.S,
        )
    )
    candidates.extend(
        candidate.replace(r"\_", "_")
        for candidate in list(candidates)
        if r"\_" in candidate
    )

    parsed_objects: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            parsed_objects.append(parsed)

        # Models may emit reasoning plus more than one JSON object. Scanning
        # every opening brace is more robust than slicing from the first brace
        # to the last, which creates invalid JSON in that case.
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", candidate):
            try:
                parsed, _ = decoder.raw_decode(candidate[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                parsed_objects.append(parsed)

    if parsed_objects:
        return normalize_parsed_object(parsed_objects[-1]), "json_scan"
    relaxed = parse_relaxed_answer_object(without_think)
    if relaxed is not None:
        return relaxed, "relaxed_answer_key"
    return None, "parse_failed" if cleaned else "empty"
