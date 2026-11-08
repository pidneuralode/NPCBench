"""Score MRI T and N staging from categorical predictions."""
from collections import Counter
import re

from .final_answer import parse_stage_field


def row_id(row):
    return row.get('id', row.get('sample_id', row.get('request_id')))


def score(labels, predictions, stage):
    field = stage.lower() + '_stage'
    allowed = {stage + str(i) for i in (range(1, 5) if stage == 'T' else range(4))}
    refs = {row_id(r): r.get(field, r.get('answer', r.get('gold'))) for r in labels}
    pred = {row_id(r): r for r in predictions}
    if len(refs) != len(labels) or len(pred) != len(predictions) or None in refs or None in pred:
        raise ValueError('Labels and predictions must each have unique IDs')
    if not refs or refs.keys() != pred.keys():
        raise ValueError('Score the complete, matching case set; missing requests are not model errors')
    if any(value not in allowed for value in refs.values()):
        raise ValueError('Reference stage is outside the task categories')
    confusion, details = Counter(), []
    for sid, gold in refs.items():
        answer = parse_stage_field(pred[sid].get('response'), stage)
        valid = answer.valid
        value = answer.prediction if valid else None
        confusion[(gold, value or 'invalid')] += 1
        details.append({'id': sid, 'reference': gold, 'prediction': value,
                        'valid': valid, 'correct': value == gold})
    correct = sum(r['correct'] for r in details)
    recall = {g: sum(r['correct'] for r in details if r['reference'] == g) /
              sum(r['reference'] == g for r in details) for g in sorted(allowed)
              if any(r['reference'] == g for r in details)}
    formats = {row_id(r): r.get('input_format') for r in labels}
    groups = {}
    for name in sorted({v for v in formats.values() if v}):
        members = [r for r in details if formats[r['id']] == name]
        groups[name] = {'cases': len(members), 'correct': sum(r['correct'] for r in members),
                        'accuracy': sum(r['correct'] for r in members) / len(members)}
    return {'cases': len(refs), 'correct': correct, 'accuracy': correct / len(refs),
            'class_recall': recall, 'balanced_accuracy': sum(recall.values()) / len(allowed)
                if len(recall) == len(allowed) else None, 'by_input_format': groups,
            'invalid': sum(not r['valid'] for r in details),
            'confusion': [{'reference': g, 'prediction': p, 'count': n} for (g, p), n in sorted(confusion.items())],
            'items': details}
