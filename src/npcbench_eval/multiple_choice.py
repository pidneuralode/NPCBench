"""Score guideline knowledge, application, and planning questions."""
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP

from . import choice_api as api_parser
from . import choice_local as local_parser
from .final_answer import POLICY, parse_choice
from .choice_sets import parse_choice_set

TASKS = {'L1': (250, 'G-Mem.', 'item_accuracy'),
         'L2': (185, 'A-App.', 'four_capability_group_macro'),
         'L4': (165, 'C-Plan', 'patient_macro_accuracy')}
PARSERS = ('api_mcq_v1', 'local_cpu_v2', POLICY)


def unique(records, field='id'):
    result = {}
    for record in records:
        value = record[field]
        key = str(value)
        if value is None or not key.strip() or key in result:
            raise ValueError(f'Duplicate or empty {field}')
        result[key] = record
    return result


def display(value):
    return str(Decimal(str(value)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))


def score(dataset, predictions, task, parser):
    if task not in TASKS or parser not in PARSERS:
        raise ValueError('Unknown task or answer parser')
    gold, raw = unique(dataset), unique(predictions)
    if not gold or set(gold) != set(raw):
        raise ValueError('Prediction IDs must exactly cover the nonempty dataset')
    groups = defaultdict(list)
    methods = Counter()
    decisions = []
    empty = 0
    for item_id, item in gold.items():
        response = raw[item_id].get('response') or ''
        multiple = item.get('question_type') == 'multiple_choice'
        reference = parse_choice_set(item['answer'], item['options']) if multiple else None
        if not isinstance(response, str) or (not reference.valid if multiple else item['answer'] not in item['options']):
            raise ValueError('Invalid response type or gold option')
        if multiple:
            if parser != POLICY:
                raise ValueError('Multiple-select questions require final_answer_v1')
            parsed = parse_choice_set(response, item['options'])
            pred, valid, method = parsed.prediction, parsed.valid, parsed.method
            correct = valid and pred == reference.prediction
        elif parser == POLICY:
            parsed = parse_choice(response, item['options'])
            pred, valid, method = parsed.prediction, parsed.valid, parsed.method
            correct = valid and pred == item['answer']
        elif parser == 'api_mcq_v1':
            parsed = api_parser.parse_answer(response, item['options'])
            correct = bool(parsed['parse_valid'] and parsed['pred_answer'] == item['answer'])
            method = parsed['parse_method']
            pred = parsed['pred_answer']
            valid = parsed['parse_valid']
        else:
            extracted = local_parser.extract(response, 'answer')
            choices = item.get('choices')
            if choices is None:
                choices = [f'{letter}. {text}' for letter, text in item['options'].items()]
            correct = bool(local_parser.judge_multi_choice(choices, item['answer'], extracted))
            method = 'frozen_local_fuzzy_judge'
            # The historical local judge does not return a parse-valid flag or a unique letter.
            pred, valid = None, None
        group = item_id if task == 'L1' else item['capability_group'] if task == 'L2' else item['patient_id']
        if group is None or not str(group).strip():
            raise ValueError('Missing primary grouping key')
        groups[group].append(correct)
        empty += not bool(response.strip())
        methods[method] += 1
        decisions.append({'id': item_id, 'correct': correct, 'prediction': pred,
                          'parse_valid': valid, 'parse_method': method})
        if parser == POLICY:
            decisions[-1].update(answer_status=parsed.status, answer_channel=parsed.channel)
    right = sum(item['correct'] for item in decisions)
    count = len(gold)
    micro = 100 * right / count
    macro = 100 * sum(sum(v) / len(v) for v in groups.values()) / len(groups)
    primary = micro if task == 'L1' else macro
    metrics = {'task': task, 'parser': parser, 'items': count, 'correct': right,
               'groups': len(groups), 'micro_percent': micro, 'group_macro_percent': macro,
               'primary_metric': TASKS[task][2], 'primary_percent': primary,
               'primary_display': display(primary), 'empty_responses': empty,
               'parse_methods': dict(sorted(methods.items())),
               'parse_invalid': sum(d['parse_valid'] is False for d in decisions) if parser != 'local_cpu_v2' else None}
    return metrics, decisions
