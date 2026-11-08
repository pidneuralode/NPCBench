"""Score L5 turns and complete patient trajectories."""
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from .final_answer import POLICY, parse_choice


def unique(rows, field="id"):
    indexed = {}
    for row in rows:
        if row[field] is None:
            raise ValueError(f'Duplicate or empty {field}')
        key = str(row[field])
        if not key.strip() or key in indexed:
            raise ValueError(f"Duplicate or empty {field}")
        indexed[key] = row
    return indexed


def display_percent(numerator, denominator):
    return str((Decimal(numerator) * 100 / Decimal(denominator)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))


def score(dataset, predictions):
    """Require exact coverage; unparsed/empty responses count as incorrect."""
    gold = unique(dataset)
    pred = unique(predictions)
    if not gold or set(gold) != set(pred):
        raise ValueError("Prediction IDs must exactly cover the nonempty dataset")
    patients = defaultdict(list)
    methods = defaultdict(int)
    parsed_rows = []
    valid = empty = 0
    for key, row in gold.items():
        if not row.get('patient_id') or row['answer'] not in row['options']:
            raise ValueError("Missing patient ID or invalid gold option")
        response = pred[key].get('response') or ''
        if not isinstance(response, str):
            raise ValueError("Response must be text or null")
        answer = parse_choice(response, row['options'])
        correct = answer.valid and answer.prediction == row['answer']
        patients[row['patient_id']].append(correct)
        methods[answer.method] += 1
        valid += answer.valid
        empty += not bool(response.strip())
        parsed_rows.append({'id': key, 'pred_answer': answer.prediction,
                            'parse_valid': answer.valid, 'parse_method': answer.method,
                            'parse_conflict': answer.status == 'conflicting_answers',
                            'correct': correct, 'answer_status': answer.status,
                            'answer_channel': answer.channel})
    count = len(gold)
    right = sum(row['correct'] for row in parsed_rows)
    full = sum(all(turns) for turns in patients.values())
    metrics = {'turn_count': count, 'patient_count': len(patients), 'correct_turns': right,
               'all_correct_patients': full, 'turn_micro_accuracy': right / count,
               'patient_macro_accuracy': sum(sum(v) / len(v) for v in patients.values()) / len(patients),
               'all_turns_correct_patient_rate': full / len(patients),
               'paper_step_percent': display_percent(right, count),
               'paper_episode_percent': display_percent(full, len(patients)),
               'parse_valid_count': valid, 'empty_response_count': empty,
               'parse_methods': dict(sorted(methods.items()))}
    metrics['answer_policy'] = POLICY
    return metrics, parsed_rows
