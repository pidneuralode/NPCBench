from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from o1_eval.simple_evals_main.npc_o1_schema import NEGATIVE_SECTIONS, POSITIVE_SECTIONS, O1RubricItem


def parse_judge_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)

    candidates = [stripped]
    for marker in ("assistantfinal", "<|channel|>final<|message|>", "final"):
        if marker in stripped:
            candidates.append(stripped.rsplit(marker, 1)[-1].strip())

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
        for obj in reversed(_balanced_json_objects(candidate)):
            try:
                return json.loads(obj)
            except json.JSONDecodeError as exc:
                last_error = exc
    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("No JSON object found", stripped, 0)


def _balanced_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for idx, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : idx + 1])
                start = None
    return objects


def validate_judge_output(row: dict[str, Any]) -> tuple[bool, str]:
    if "criteria_met" not in row:
        return False, "missing criteria_met"
    if not isinstance(row["criteria_met"], bool):
        return False, "criteria_met is not boolean"
    if "explanation" not in row:
        return False, "missing explanation"
    if not isinstance(row["explanation"], str):
        return False, "explanation is not string"
    return True, "ok"


def _rubric_by_item(rubric_items: list[O1RubricItem]) -> dict[str, list[O1RubricItem]]:
    grouped: dict[str, list[O1RubricItem]] = defaultdict(list)
    for rubric_item in rubric_items:
        grouped[rubric_item.item_id].append(rubric_item)
    return grouped


def _judgment_map(judgments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for row in judgments:
        rubric_id = row.get("rubric_id")
        if rubric_id:
            mapped[str(rubric_id)] = row
    return mapped


def score_single_item(
    item: dict[str, Any],
    rubric_items: list[O1RubricItem],
    judgments_by_rubric_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    item_id = item["item_id"]
    total_score = float(item["rubric_template_structured"].get("total_score", 10))

    missing = []
    invalid = []
    positive_score = 0.0
    section_scores = {section: 0.0 for section in POSITIVE_SECTIONS}
    section_possible = {section: 0.0 for section in POSITIVE_SECTIONS}
    major_hits = 0
    minor_hits = 0
    major_penalty = 0.0
    minor_penalty = 0.0

    for rubric_item in rubric_items:
        judgment = judgments_by_rubric_id.get(rubric_item.rubric_id)
        if judgment is None:
            missing.append(rubric_item.rubric_id)
            continue
        if judgment.get("judge_status", "ok") != "ok":
            invalid.append(rubric_item.rubric_id)
            continue
        ok, _ = validate_judge_output(judgment)
        if not ok:
            invalid.append(rubric_item.rubric_id)
            continue

        criteria_met = bool(judgment["criteria_met"])
        if rubric_item.scoring_effect == "positive":
            section_possible[rubric_item.section] += rubric_item.points
            if criteria_met:
                positive_score += rubric_item.points
                section_scores[rubric_item.section] += rubric_item.points
        elif rubric_item.scoring_effect == "major_penalty" and criteria_met:
            major_hits += 1
            major_penalty += rubric_item.penalty
        elif rubric_item.scoring_effect == "minor_penalty" and criteria_met:
            minor_hits += 1
            minor_penalty += rubric_item.penalty

    if missing or invalid:
        return {
            "item_id": item_id,
            "case_id": item["case_id"],
            "patient_id": item["patient_id"],
            "tag": item["tag"],
            "tag_label": item["tag_label"],
            "question_slot": item["question_slot"],
            "scoring_status": "missing_or_invalid_judgments",
            "missing_judgments": missing,
            "invalid_judgments": invalid,
            "final_score": None,
            "normalized_score": None,
        }

    final_score = max(0.0, positive_score - major_penalty - minor_penalty)
    normalized_score = final_score / total_score * 100.0 if total_score > 0 else None

    return {
        "item_id": item_id,
        "case_id": item["case_id"],
        "patient_id": item["patient_id"],
        "tag": item["tag"],
        "tag_label": item["tag_label"],
        "question_slot": item["question_slot"],
        "scoring_status": "ok",
        "positive_score": round(positive_score, 6),
        "major_penalty": round(major_penalty, 6),
        "minor_penalty": round(minor_penalty, 6),
        "major_error_hits": major_hits,
        "minor_error_hits": minor_hits,
        "final_score": round(final_score, 6),
        "normalized_score": round(normalized_score, 6) if normalized_score is not None else None,
        "component_scores": {k: round(v, 6) for k, v in section_scores.items()},
        "component_possible": {k: round(v, 6) for k, v in section_possible.items()},
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _case_bootstrap_ci(
    item_scores: list[dict[str, Any]],
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, float | None]:
    by_case: dict[str, list[float]] = defaultdict(list)
    for row in item_scores:
        if row.get("scoring_status") == "ok" and row.get("normalized_score") is not None:
            by_case[row["case_id"]].append(float(row["normalized_score"]))
    case_values = [_mean(values) for values in by_case.values()]
    case_values = [v for v in case_values if v is not None]
    if not case_values:
        return {"case_bootstrap_mean": None, "case_bootstrap_ci_low": None, "case_bootstrap_ci_high": None}

    rng = random.Random(seed)
    boot = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(case_values) for _ in case_values]
        boot.append(sum(sample) / len(sample))
    boot.sort()
    low = boot[int(0.025 * (len(boot) - 1))]
    high = boot[int(0.975 * (len(boot) - 1))]
    return {
        "case_bootstrap_mean": sum(case_values) / len(case_values),
        "case_bootstrap_ci_low": low,
        "case_bootstrap_ci_high": high,
    }


def aggregate_item_scores(item_scores: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in item_scores if row.get("scoring_status") == "ok"]
    normalized = [float(row["normalized_score"]) for row in ok_rows]
    by_case: dict[str, list[float]] = defaultdict(list)
    by_tag: dict[str, list[float]] = defaultdict(list)
    component_hits = {section: 0.0 for section in POSITIVE_SECTIONS}
    component_possible = {section: 0.0 for section in POSITIVE_SECTIONS}

    for row in ok_rows:
        by_case[row["case_id"]].append(float(row["normalized_score"]))
        by_tag[row["tag"]].append(float(row["normalized_score"]))
        for section in POSITIVE_SECTIONS:
            component_hits[section] += float(row["component_scores"][section])
            component_possible[section] += float(row["component_possible"][section])

    case_means = [_mean(values) for values in by_case.values()]
    case_means = [v for v in case_means if v is not None]
    metrics: dict[str, Any] = {
        "overall_score": _mean(normalized),
        "item_macro_score": _mean(normalized),
        "case_macro_score": _mean(case_means),
        "num_items": len(item_scores),
        "num_scored_items": len(ok_rows),
        "scoring_failure_rate": 1.0 - (len(ok_rows) / len(item_scores) if item_scores else 0.0),
        "major_error_rate": _mean([1.0 if row["major_error_hits"] > 0 else 0.0 for row in ok_rows]),
        "minor_error_rate": _mean([1.0 if row["minor_error_hits"] > 0 else 0.0 for row in ok_rows]),
        "major_error_hits_mean": _mean([float(row["major_error_hits"]) for row in ok_rows]),
        "minor_error_hits_mean": _mean([float(row["minor_error_hits"]) for row in ok_rows]),
        "per_tag": {tag: _mean(values) for tag, values in sorted(by_tag.items())},
        "components": {},
    }
    for section in POSITIVE_SECTIONS:
        possible = component_possible[section]
        metrics["components"][section] = (
            component_hits[section] / possible * 100.0 if possible > 0 else None
        )
    metrics.update(_case_bootstrap_ci(item_scores))
    return metrics


def score_dataset(
    items: list[dict[str, Any]],
    rubric_items: list[O1RubricItem],
    judgments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rubric_groups = _rubric_by_item(rubric_items)
    judgments_by_rubric_id = _judgment_map(judgments)
    item_scores = [
        score_single_item(item, rubric_groups[item["item_id"]], judgments_by_rubric_id)
        for item in items
    ]
    return item_scores, aggregate_item_scores(item_scores)


def write_scores(output_dir: str | Path, item_scores: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "item_scores.jsonl", "w", encoding="utf-8") as f:
        for row in item_scores:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, sort_keys=True)
