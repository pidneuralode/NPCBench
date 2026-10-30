"""CPU scoring of the frozen L1/L2/L4 primary release; no inference or networking."""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path

from . import mcq_release_api_parser as api_parser
from . import mcq_release_local_parser as local_parser
from .final_answer import POLICY, parse_choice

TASKS = {'L1': (250, 'G-Mem.', 'item_accuracy'),
         'L2': (185, 'A-App.', 'four_capability_group_macro'),
         'L4': (165, 'C-Plan', 'patient_macro_accuracy')}
PARSER_FILES = {'api_mcq_v1': 'mcq_release_api_parser.py',
                'local_cpu_v2': 'mcq_release_local_parser.py', POLICY: 'final_answer.py'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


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
    if task not in TASKS or parser not in PARSER_FILES:
        raise ValueError('Unknown task or frozen parser')
    gold, raw = unique(dataset), unique(predictions)
    if not gold or set(gold) != set(raw):
        raise ValueError('Prediction IDs must exactly cover the nonempty dataset')
    groups = defaultdict(list)
    methods = Counter()
    decisions = []
    empty = 0
    for item_id, item in gold.items():
        response = raw[item_id].get('response') or ''
        if not isinstance(response, str) or item['answer'] not in item['options']:
            raise ValueError('Invalid response type or gold option')
        if parser == POLICY:
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
            correct = bool(local_parser.judge_multi_choice(item['choices'], item['answer'], extracted))
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


def reproduce(bundle):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest.get('schema') != 'npcbench-text-mcq-frozen-v1':
        raise ValueError('Unsupported release schema')
    for relative, digest in manifest['files'].items():
        path = (bundle / relative).resolve()
        if bundle not in path.parents or sha(path) != digest:
            raise ValueError('File hash or path mismatch')
    for parser, digest in manifest['parsers'].items():
        if parser not in PARSER_FILES or sha(Path(__file__).with_name(PARSER_FILES[parser])) != digest:
            raise ValueError('Frozen parser hash mismatch')
    results = {}
    for task, spec in manifest['tasks'].items():
        if task not in TASKS or spec['dataset_path'] not in manifest['files']:
            raise ValueError('Unregistered task input')
        dataset = rows(bundle / spec['dataset_path'])
        if len(dataset) != TASKS[task][0] or len(spec['models']) != 30:
            raise ValueError('Incomplete primary task inventory')
        models = {}
        for entry in spec['models']:
            if entry['model'] in models or entry['predictions_path'] not in manifest['files'] or entry['requests_path'] not in manifest['files']:
                raise ValueError('Duplicate model or unregistered response/request path')
            requests = unique(rows(bundle / entry['requests_path']))
            if set(requests) != set(unique(dataset)):
                raise ValueError('Request coverage mismatch')
            metrics, _ = score(dataset, rows(bundle / entry['predictions_path']), task, entry['parser'])
            if metrics != entry['metrics']:
                raise ValueError('CPU replay differs from exported metrics')
            models[entry['model']] = metrics
        results[task] = {'models': models, 'items': len(dataset), 'model_count': len(models)}
    return {'verified': True, 'tasks': results,
            'response_count': sum(v['items'] * v['model_count'] for v in results.values()), 'model_calls': 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', type=Path)
    ap.add_argument('--dataset', type=Path)
    ap.add_argument('--predictions', type=Path)
    ap.add_argument('--task', choices=list(TASKS))
    ap.add_argument('--parser', choices=list(PARSER_FILES))
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if args.bundle and not any([args.dataset, args.predictions, args.task, args.parser]):
        result = reproduce(args.bundle)
    elif not args.bundle and all([args.dataset, args.predictions, args.task, args.parser]):
        metrics, decisions = score(rows(args.dataset), rows(args.predictions), args.task, args.parser)
        result = {'metrics': metrics, 'decisions': decisions}
    else:
        ap.error('Use --bundle OR --dataset --predictions --task --parser')
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
