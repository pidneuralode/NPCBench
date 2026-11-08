"""Score L3 pathology, sequence, and structure questions."""
from collections import Counter, defaultdict
import json

from .anatomy import StructureCardParser, STRUCTURE_ROWS
from .final_answer import POLICY, parse_choice, json_payload


ENDPOINTS = ('Path', 'Seq', 'Struct')


def index(rows):
    result = {}
    for row in rows:
        key = row.get('id')
        if not isinstance(key, str) or not key.strip() or key in result:
            raise ValueError('Expected unique nonempty string IDs')
        result[key] = row
    return result


def make_box():
    box = StructureCardParser()
    box.structure_rows = STRUCTURE_ROWS
    box.structure_id_to_base = {}
    for row in STRUCTURE_ROWS:
        base = row.get('base_structure_en') or row['structure_id']
        box.structure_id_to_base[row['structure_id']] = base
        box.structure_id_to_base[base] = base
    box.structure_alias_to_id = box._load_structure_aliases()
    return box


def score(subset, dataset, predictions):
    """Reparse raw responses. Invalid main answers remain in the denominator.

    Main metrics are micro accuracy for all three tasks. Parser errors in
    secondary fields (e.g. invasion) do not invalidate a correct main field.
    """
    if subset not in ENDPOINTS:
        raise ValueError('Unknown L3 endpoint')
    gold, responses = index(dataset), index(predictions)
    if not gold or set(gold) != set(responses):
        raise ValueError('Responses must exactly cover the nonempty dataset')
    box = make_box() if subset == 'Struct' else None
    groups, errors, parsed_rows = defaultdict(list), Counter(), []
    for key, row in gold.items():
        response = responses[key].get('response')
        if response is None:
            response = ''
        if not isinstance(response, str):
            raise ValueError('Response must be text or null')
        target = row['gold']
        if subset == 'Struct':
            if target not in {box._structure_to_base(c) for c in row['structure_choices']}:
                raise ValueError('Gold is absent from structure choices')
            payload, status, channel = json_payload(response)
            answer = json.dumps(payload, ensure_ascii=False) if payload is not None else ''
            parsed, parse_errors = box._parse_card_prediction(answer, row['structure_choices'])
            pred = box._structure_to_base(parsed['structure_id'])
        else:
            options = {'0': '', '1': ''} if subset == 'Path' else {
                opt['option_id']: opt['sequence_label'] for opt in row['options']}
            if target not in options:
                raise ValueError('Gold is absent from the answer choices')
            answer = parse_choice(response, options)
            pred = answer.prediction
            status, channel = answer.status, answer.channel
            parse_errors = [] if answer.valid else ['invalid_binary' if subset == 'Path' else answer.status]
        correct = pred == target
        groups[target].append(correct)
        errors.update(parse_errors)
        parsed_rows.append({'id': key, 'prediction': pred, 'correct': correct,
                            'parser_errors': parse_errors})
        parsed_rows[-1].update(answer_status=status, answer_channel=channel)
    n = len(gold)
    right = sum(sum(values) for values in groups.values())
    metrics = {
        'count': n, 'correct': right, 'micro_percent': 100 * right / n,
        # Canonical label order makes the floating-point reduction independent
        # of each model archive's item order.
        'class_macro_percent': 100 * sum(sum(groups[k]) / len(groups[k]) for k in sorted(groups)) / len(groups),
        'class_counts': {k: len(v) for k, v in sorted(groups.items())},
        'parser_errors': dict(sorted(errors.items())),
    }
    metrics['answer_policy'] = POLICY
    return metrics, parsed_rows
