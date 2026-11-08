"""Read consultation questions and their scored criteria."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


POSITIVE_SECTIONS = ("core_answer", "key_evidence", "conditional_modifiers")
NEGATIVE_SECTIONS = ("major_errors", "minor_errors")


@dataclass(frozen=True)
class O1RubricItem:
    rubric_id: str
    item_id: str
    case_id: str
    patient_id: str
    tag: str
    section: str
    index: int
    criterion: str
    points: float = 0.0
    penalty: float = 0.0
    scoring_effect: str = "positive"
    acceptable: list[str] = field(default_factory=list)
    unacceptable: list[str] = field(default_factory=list)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_o1_items(path: str | Path) -> list[dict[str, Any]]:
    items = read_jsonl(path)
    required = {
        "item_id",
        "case_id",
        "patient_id",
        "tag",
        "case_input",
        "question",
        "rubric_template_structured",
    }
    for idx, item in enumerate(items):
        missing = sorted(required - set(item))
        if missing:
            raise ValueError(f"O1 item {idx} is missing fields: {missing}")
        rubric = item["rubric_template_structured"]
        if "scoring_rules" not in rubric:
            raise ValueError(f"O1 item {item['item_id']} has no scoring_rules")
    return items


def _synonym_lists(rubric: dict[str, Any]) -> tuple[list[str], list[str]]:
    synonym_rules = rubric.get("synonym_rules") or {}
    acceptable = list(synonym_rules.get("acceptable") or [])
    unacceptable = list(synonym_rules.get("unacceptable") or [])
    return acceptable, unacceptable


def flatten_item_rubrics(item: dict[str, Any]) -> list[O1RubricItem]:
    rubric = item["rubric_template_structured"]
    scoring_rules = rubric.get("scoring_rules") or {}
    item_scores = scoring_rules.get("item_scores") or {}
    acceptable, unacceptable = _synonym_lists(rubric)
    rows: list[O1RubricItem] = []

    common = dict(
        item_id=item["item_id"],
        case_id=item["case_id"],
        patient_id=item["patient_id"],
        tag=item["tag"],
        acceptable=acceptable,
        unacceptable=unacceptable,
    )

    for section in POSITIVE_SECTIONS:
        anchors = list(rubric.get(section) or [])
        scores = list(item_scores.get(section) or [])
        for idx, (criterion, points) in enumerate(zip(anchors, scores)):
            rows.append(
                O1RubricItem(
                    rubric_id=f"{item['item_id']}::{section}::{idx}",
                    section=section,
                    index=idx,
                    criterion=str(criterion),
                    points=float(points),
                    penalty=0.0,
                    scoring_effect="positive",
                    **common,
                )
            )

    major_errors = (rubric.get("negative_list") or {}).get("major_errors") or []
    for idx, criterion in enumerate(major_errors):
        rows.append(
            O1RubricItem(
                rubric_id=f"{item['item_id']}::major_errors::{idx}",
                section="major_errors",
                index=idx,
                criterion=str(criterion),
                points=0.0,
                penalty=2.0,
                scoring_effect="major_penalty",
                **common,
            )
        )

    minor_errors = (rubric.get("negative_list") or {}).get("minor_errors") or []
    for idx, criterion in enumerate(minor_errors):
        rows.append(
            O1RubricItem(
                rubric_id=f"{item['item_id']}::minor_errors::{idx}",
                section="minor_errors",
                index=idx,
                criterion=str(criterion),
                points=0.0,
                penalty=1.0,
                scoring_effect="minor_penalty",
                **common,
            )
        )

    return rows


def flatten_dataset_rubrics(items: list[dict[str, Any]]) -> list[O1RubricItem]:
    rows: list[O1RubricItem] = []
    for item in items:
        rows.extend(flatten_item_rubrics(item))
    return rows
