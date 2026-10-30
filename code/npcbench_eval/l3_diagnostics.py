"""Secondary MRI evidence diagnostics under the frozen 18-structure protocol.

Missing slots in the deterministic ablation retain the original zero default.
Complete evidence requires every label; an absent label is not a correct zero.
The text-stager experiment is a separate condition and is not computed here.
"""
from __future__ import annotations
from collections import Counter
import json
from typing import Any
from .l3 import make_box
from .final_answer import json_payload, parse_t_stage

T_STAGES = ["T1", "T2", "T3", "T4"]

BASE_STRUCTURES = [
    "nasal_passage",
    "oropharynx",
    "levator_veli_palatini",
    "parapharyngeal_space",
    "carotid_sheath",
    "longus_capitis",
    "medial_pterygoid",
    "lateral_pterygoid",
    "pterygoid_base",
    "sphenoid_base",
    "clivus",
    "petrous_apex",
    "atlas",
    "axis",
    "sphenoid_sinus",
    "ethmoid_sinus",
    "maxillary_sinus",
    "cavernous_sinus",
]

T3_STRUCTURES = {
    "clivus",
    "petrous_apex",
    "pterygoid_base",
    "sphenoid_base",
    "atlas",
    "axis",
    "sphenoid_sinus",
    "ethmoid_sinus",
    "maxillary_sinus",
}

T2_STRUCTURES = {
    "parapharyngeal_space",
    "medial_pterygoid",
    "lateral_pterygoid",
    "longus_capitis",
}

def norm_t(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    text = text.replace("STAGE", "").replace(" ", "")
    if text in T_STAGES:
        return text
    for stage in T_STAGES:
        if stage in text:
            return stage
    return None

def norm_inv(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return int(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "positive", "invaded", "invasion"}:
            return 1
        if text in {"0", "false", "no", "n", "negative", "not invaded", "none"}:
            return 0
    return None

def strip_laterality(structure_id: Any) -> str | None:
    if not structure_id:
        return None
    text = str(structure_id).strip().lower().replace("-", "_").replace(" ", "_")
    for suffix in ("_left", "_right", "_bilateral"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    if text in BASE_STRUCTURES:
        return text
    return None

def derive_t_stage(invasion: dict[str, int]) -> str:
    if invasion.get("cavernous_sinus", 0):
        return "T4"
    if any(invasion.get(k, 0) for k in T3_STRUCTURES):
        return "T3"
    if any(invasion.get(k, 0) for k in T2_STRUCTURES):
        return "T2"
    return "T1"

def pct(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return round(num / den * 100.0, 1)

def patient_id(row: dict[str, Any]) -> str:
    return str(row.get("patient_case_id") or row.get("patient_id") or row.get("id"))

def get_patient_prediction(row: dict[str, Any]) -> str | None:
    parsed = row.get("parsed_prediction") or {}
    return norm_t(parsed.get("t_stage") if isinstance(parsed, dict) else None)

def summarize_patient_model(name: str, rows: list[dict[str, Any]], balanced_ids: set[str]) -> dict[str, Any]:
    counts = Counter()
    correct = Counter()
    parsed = 0
    pred_counts = Counter()
    balanced_correct = 0
    balanced_total = 0
    complete_inv = 0
    eg_t = 0
    for row in rows:
        gold = norm_t(row.get("t_stage_gt"))
        pred = get_patient_prediction(row)
        pid = patient_id(row)
        if pred:
            parsed += 1
            pred_counts[pred] += 1
        if gold:
            counts[gold] += 1
            if pred == gold:
                correct[gold] += 1
        if pid in balanced_ids and gold:
            balanced_total += 1
            if pred == gold:
                balanced_correct += 1
        if row.get("complete_structure_invasion_correct") is True:
            complete_inv += 1
        if row.get("evidence_grounded_t_stage_correct") is True:
            eg_t += 1

    total = sum(counts.values())
    recalls = {stage: pct(correct[stage], counts[stage]) for stage in T_STAGES}
    balanced_macro = round(sum(recalls.values()) / len(T_STAGES), 1)
    return {
        "model": name,
        "n": total,
        "parse": pct(parsed, len(rows)),
        "direct_t": pct(sum(correct.values()), total),
        "balanced_t": balanced_macro,
        "t_bal60": pct(balanced_correct, balanced_total),
        "complete_inv": pct(complete_inv, len(rows)),
        "eg_t": pct(eg_t, len(rows)),
        "gold_counts": dict(counts),
        "pred_counts": dict(pred_counts),
        "recall_T1": recalls["T1"],
        "recall_T2": recalls["T2"],
        "recall_T3": recalls["T3"],
        "recall_T4": recalls["T4"],
    }

def summarize_card_model(
    name: str,
    card_rows: list[dict[str, Any]],
    patient_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    patient_gold_t = {patient_id(row): norm_t(row.get("t_stage_gt")) for row in patient_rows}
    gold_inv_by_patient = {
        patient_id(row): {k: int(v) for k, v in (row.get("base_structure_invasion_gt") or {}).items()}
        for row in patient_rows
    }
    direct_t_by_patient = {patient_id(row): get_patient_prediction(row) for row in patient_rows}

    oracle_slot = {
        pid: {s: 0 for s in BASE_STRUCTURES}
        for pid in patient_gold_t
    }
    model_slot = {
        pid: {s: 0 for s in BASE_STRUCTURES}
        for pid in patient_gold_t
    }
    parsed_card = 0
    base_correct = 0
    inv_correct = 0
    inv_evaluable = 0
    joint = 0
    for row in card_rows:
        pid = patient_id(row)
        parsed = row.get("parsed_prediction") or {}
        if isinstance(parsed, dict) and not row.get("parse_error"):
            parsed_card += 1
        gold_base = strip_laterality(row.get("base_structure_id_gt") or row.get("structure_id_gt"))
        pred_base = None
        pred_inv = None
        if isinstance(parsed, dict):
            pred_base = strip_laterality(parsed.get("base_structure_id") or parsed.get("structure_id"))
            pred_inv = norm_inv(parsed.get("invasion"))
        if row.get("base_structure_correct") is True:
            base_correct += 1
        if row.get("invasion_evaluable") is True:
            inv_evaluable += 1
        if row.get("invasion_correct") is True:
            inv_correct += 1
        if row.get("joint_correct") is True:
            joint += 1
        if pred_inv is not None and gold_base in BASE_STRUCTURES and pid in oracle_slot:
            oracle_slot[pid][gold_base] = max(oracle_slot[pid][gold_base], pred_inv)
        if pred_inv is not None and pred_base in BASE_STRUCTURES and pid in model_slot:
            model_slot[pid][pred_base] = max(model_slot[pid][pred_base], pred_inv)

    n_patient = len(patient_gold_t)
    gold_rule_correct = 0
    oracle_rule_correct = 0
    model_rule_correct = 0
    direct_t_correct = 0
    for pid, gold_t in patient_gold_t.items():
        if derive_t_stage(gold_inv_by_patient.get(pid, {})) == gold_t:
            gold_rule_correct += 1
        if derive_t_stage(oracle_slot.get(pid, {})) == gold_t:
            oracle_rule_correct += 1
        if derive_t_stage(model_slot.get(pid, {})) == gold_t:
            model_rule_correct += 1
        if direct_t_by_patient.get(pid) == gold_t:
            direct_t_correct += 1

    return {
        "model": name,
        "card_n": len(card_rows),
        "card_parse": pct(parsed_card, len(card_rows)),
        "card_base_struct": pct(base_correct, len(card_rows)),
        "card_invasion": pct(inv_correct, inv_evaluable),
        "card_joint": pct(joint, len(card_rows)),
        "gold_rule_ceiling": pct(gold_rule_correct, n_patient),
        "oracle_slot_rule_t": pct(oracle_rule_correct, n_patient),
        "model_slot_rule_t": pct(model_rule_correct, n_patient),
        "direct_t": pct(direct_t_correct, n_patient),
    }

def analyze_model(model, cards, patients, card_predictions, patient_predictions, balanced_ids):
    """Join public labels and frozen raw responses by their existing item IDs."""
    card_responses = {r['id']: r['response'] for r in card_predictions}
    patient_responses = {r['id']: r['response'] for r in patient_predictions}
    if set(card_responses) != {r['id'] for r in cards} or len(card_responses) != len(card_predictions):
        raise ValueError('Card response coverage differs from labels')
    if set(patient_responses) != {r['id'] for r in patients} or len(patient_responses) != len(patient_predictions):
        raise ValueError('Patient response coverage differs from labels')
    box = make_box()
    parsed_cards, parsed_patients = [], []
    for row in cards:
        payload, _, _ = json_payload(card_responses[row['id']])
        parsed, errors = box._parse_card_prediction(json.dumps(payload) if payload is not None else '', BASE_STRUCTURES)
        base_correct = box._structure_to_base(parsed.get('structure_id')) == row['base_structure_id_gt']
        joint = base_correct and parsed.get('invasion') == row['invasion_gt']
        parsed_cards.append(dict(row, parsed_prediction=parsed, parse_error=bool(errors),
            base_structure_correct=base_correct, invasion_evaluable=base_correct, invasion_correct=joint, joint_correct=joint))
    invasion_correct = 0
    for row in patients:
        response = patient_responses[row['id']]
        payload, _, _ = json_payload(response)
        parsed, _ = box._parse_patient_prediction(json.dumps(payload) if payload is not None else '', BASE_STRUCTURES)
        stage = parse_t_stage(response); parsed['t_stage'] = stage.prediction
        vector = parsed.get('structure_invasion') or {}
        correct = sum(vector.get(k) == value for k, value in row['base_structure_invasion_gt'].items())
        invasion_correct += correct
        parsed_patients.append(dict(row, parsed_prediction=parsed, complete_structure_invasion_correct=correct == 18,
            evidence_grounded_t_stage_correct=correct == 18 and stage.prediction == row['t_stage_gt']))
    ablation = summarize_card_model(model, parsed_cards, parsed_patients)
    patient = summarize_patient_model(model, parsed_patients, set(balanced_ids))
    component = dict(model=model, base_structure_micro=100 * sum(r['base_structure_correct'] for r in parsed_cards) / len(cards),
        base_invasion_conditional=ablation['card_invasion'], base_joint_micro=ablation['card_joint'],
        patient_invasion_label_micro=100 * invasion_correct / (18 * len(patients)),
        complete_invasion=patient['complete_inv'], grounded_t=patient['eg_t'])
    return component, ablation, patient
