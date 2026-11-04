from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .choice_sets import parse_choice_set


@dataclass(frozen=True)
class MCQItem:
    item_id: str
    question: str
    options: dict[str, str]
    answer: str
    patient_id: str | None = None
    group: str | None = None
    raw: dict[str, Any] | None = None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def _clean_options(options: Any) -> dict[str, str]:
    if not isinstance(options, dict):
        raise ValueError("options must be a dictionary")
    cleaned = {
        str(key).strip().upper(): str(value).strip()
        for key, value in options.items()
        if str(key).strip() and str(value).strip()
    }
    if not cleaned:
        raise ValueError("options are empty")
    return dict(sorted(cleaned.items()))


def _item_id_from_row(row: dict[str, Any], fallback: str) -> str:
    for key in ("release_item_id", "item_uid", "item_id", "question_id", "id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return fallback


def _question_from_row(row: dict[str, Any]) -> str:
    parts = []
    for key in ("stem", "vignette", "context", "question", "prompt"):
        value = str(row.get(key) or "").strip()
        if value:
            parts.append(value)
    return "\n\n".join(parts)


def _answer_from_row(row: dict[str, Any]) -> str:
    for key in ("correct_option", "answer", "gold_letter"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip().upper()
    raise ValueError("missing correct_option/answer/gold_letter")


def load_mcq_dataset(path: str | Path) -> list[MCQItem]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = _read_jsonl(path)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("questions"), list):
            rows = data["questions"]
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError(f"Unsupported MCQ JSON schema: {path}")

    items: list[MCQItem] = []
    for idx, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{idx} is not an object")
        options = _clean_options(row.get("options"))
        answer = _answer_from_row(row)
        multiple = row.get('question_type') == 'multiple_choice'
        reference = parse_choice_set(answer, options) if multiple else None
        valid_answer = reference.valid if multiple else answer in options
        if not valid_answer:
            raise ValueError(f"{path}:{idx} answer {answer!r} is not in options")
        if multiple:
            answer = reference.prediction
        question = _question_from_row(row)
        if not question:
            raise ValueError(f"{path}:{idx} has empty question/prompt")
        patient_id = row.get("patient_id") or row.get("meta_patient_id")
        group = row.get("meta_type") or row.get("group") or row.get("tag")
        items.append(
            MCQItem(
                item_id=_item_id_from_row(row, f"row_{idx}"),
                question=question,
                options=options,
                answer=answer,
                patient_id=str(patient_id) if patient_id not in (None, "") else None,
                group=str(group) if group not in (None, "") else None,
                raw=row,
            )
        )
    return items


def parse_option_letter(text: Any, option_labels: set[str] | None = None) -> str | None:
    if text is None:
        return None
    labels = option_labels or set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    raw = str(text).strip()
    if not raw:
        return None

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        for key in ("answer", "option", "prediction", "predicted_option", "final_answer"):
            if key in obj:
                parsed = parse_option_letter(obj[key], labels)
                if parsed:
                    return parsed

    normalized = raw.replace("﹛", "{").replace("﹜", "}").replace("（", "(").replace("）", ")")
    normalized = normalized.replace("【", "[").replace("】", "]")
    normalized = normalized.upper()

    patterns = [
        r"\\?BOXED\s*\{\s*([A-Z])\s*\}",
        r"(?:答案|选项|正确选项|最终答案|ANSWER|OPTION|FINAL ANSWER)\s*[:：为是\-]*\s*[\(\[\{]?\s*([A-Z])\b",
        r"^\s*[\(\[\{]?\s*([A-Z])\s*[\)\]\}]?\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match and match.group(1) in labels:
            return match.group(1)

    standalone = re.findall(r"(?<![A-Z])([A-Z])(?![A-Z])", normalized)
    standalone = [value for value in standalone if value in labels]
    if len(standalone) == 1:
        return standalone[0]
    return None


def _response_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate response JSON key {key!r}")
        result[key] = value
    return result


def _load_responses(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = _read_jsonl(path)
    else:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_response_json_object)
        if isinstance(data, dict):
            if isinstance(data.get("results"), list):
                rows = data["results"]
            elif isinstance(data.get("responses"), list):
                rows = data["responses"]
            else:
                rows = [{"id": key, "response": value} for key, value in data.items()]
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError(f"Unsupported responses schema: {path}")

    mapped: dict[str, Any] = {}
    for idx, row in enumerate(rows, 1):
        row_id = _item_id_from_row(row, f"row_{idx}") if isinstance(row, dict) else f"row_{idx}"
        if row_id in mapped:
            raise ValueError(f"Duplicate prediction item_id {row_id!r} at record {idx} in {path.name}")
        if not isinstance(row, dict):
            mapped[row_id] = row
            continue
        for key in ("response", "output", "prediction", "model_answer", "text", "answer"):
            if key in row:
                mapped[row_id] = row[key]
                break
        else:
            mapped[row_id] = row
    return mapped


def dataset_summary(items: list[MCQItem]) -> dict[str, Any]:
    answer_counts: dict[str, int] = defaultdict(int)
    option_counts: dict[str, int] = defaultdict(int)
    group_counts: dict[str, int] = defaultdict(int)
    patients = set()
    for item in items:
        answer_counts[item.answer] += 1
        option_counts[str(len(item.options))] += 1
        if item.group:
            group_counts[item.group] += 1
        if item.patient_id:
            patients.add(item.patient_id)
    return {
        "items": len(items),
        "patients": len(patients) if patients else None,
        "answer_counts": dict(sorted(answer_counts.items())),
        "option_count_distribution": dict(sorted(option_counts.items(), key=lambda kv: int(kv[0]))),
        "group_counts": dict(sorted(group_counts.items())),
    }


def score_mcq(
    items: list[MCQItem],
    responses: dict[str, Any] | None = None,
    constant_answer: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    responses = responses or {}
    results: list[dict[str, Any]] = []
    by_patient: dict[str, list[bool]] = defaultdict(list)
    by_group: dict[str, list[bool]] = defaultdict(list)

    for item in items:
        response = constant_answer if constant_answer is not None else responses.get(item.item_id)
        status = "ok"
        if response is None:
            status = "missing_response"
        if item.raw and item.raw.get('question_type') == 'multiple_choice' and response is not None:
            parsed_set = parse_choice_set(response, item.options)
            pred = parsed_set.prediction if parsed_set.valid else None
        else:
            pred = parse_option_letter(response, set(item.options)) if response is not None else None
        if response is not None and pred is None:
            status = "parse_failed"
        correct = bool(pred == item.answer) if pred is not None else False
        if item.patient_id:
            by_patient[item.patient_id].append(correct)
        if item.group:
            by_group[item.group].append(correct)
        results.append(
            {
                "item_id": item.item_id,
                "patient_id": item.patient_id,
                "group": item.group,
                "gold": item.answer,
                "prediction": pred,
                "correct": correct,
                "status": status,
                "response": response,
            }
        )

    total = len(items)
    parsed = sum(1 for row in results if row["prediction"] is not None)
    correct_count = sum(1 for row in results if row["correct"])
    missing = sum(1 for row in results if row["status"] == "missing_response")
    parse_failed = sum(1 for row in results if row["status"] == "parse_failed")

    patient_accs = {
        patient_id: sum(values) / len(values)
        for patient_id, values in by_patient.items()
        if values
    }
    group_accs = {
        group: {"total": len(values), "right": sum(values), "acc": sum(values) / len(values)}
        for group, values in by_group.items()
        if values
    }

    metrics = {
        "total": total,
        "parsed": parsed,
        "right": correct_count,
        "missing": missing,
        "parse_failed": parse_failed,
        "acc": correct_count / total if total else None,
        "parse_rate": parsed / total if total else None,
        "patient_count": len(patient_accs) if patient_accs else None,
        "patient_macro_acc": sum(patient_accs.values()) / len(patient_accs) if patient_accs else None,
        "patient_full_success": (
            sum(1 for values in by_patient.values() if all(values)) / len(by_patient)
            if by_patient
            else None
        ),
        "group_metrics": group_accs,
    }
    return results, metrics


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NPCBench MCQ metric-only evaluator.")
    parser.add_argument("--dataset", required=True, help="MCQ JSON/JSONL dataset path.")
    parser.add_argument("--responses-file", help="Frozen response JSON/JSONL file.")
    parser.add_argument("--output-dir", help="Directory for results.json and metrics.json.")
    parser.add_argument("--validate-only", action="store_true", help="Only validate and summarize the dataset.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    items = load_mcq_dataset(args.dataset)
    summary = dataset_summary(items)
    if args.validate_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if not args.responses_file:
        raise SystemExit("Provide --responses-file or --validate-only")
    responses = _load_responses(args.responses_file)
    if not responses:
        raise SystemExit("Provide a nonempty --responses-file or --validate-only")
    results, metrics = score_mcq(items, responses)
    metrics["dataset_summary"] = summary
    if args.output_dir:
        out_dir = Path(args.output_dir)
        write_json(out_dir / "results.json", results)
        write_json(out_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
