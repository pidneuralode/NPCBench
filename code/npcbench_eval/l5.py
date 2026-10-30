"""Offline L5 frozen-response scoring; no model inference or API access."""
import argparse
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path

from .l5_parser import LongBenchQA
from .final_answer import POLICY


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def score(dataset, predictions, policy=None):
    """Require exact coverage; unparsed/empty responses count as incorrect."""
    gold = unique(dataset)
    pred = unique(predictions)
    if not gold or set(gold) != set(pred):
        raise ValueError("Prediction IDs must exactly cover the nonempty dataset")
    from .final_answer import POLICY, parse_choice
    if policy not in (None, POLICY):
        raise ValueError('Unknown answer policy')
    parser = LongBenchQA()
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
        if policy == POLICY:
            answer = parse_choice(response, row['options'])
            parsed = {'parse_valid': answer.valid, 'pred_answer': answer.prediction,
                      'parse_method': answer.method, 'parse_conflict': answer.status == 'conflicting_answers'}
        else:
            parsed = parser.parse_answer(response, row)
        correct = bool(parsed['parse_valid'] and parsed['pred_answer'] == row['answer'])
        patients[row['patient_id']].append(correct)
        methods[parsed['parse_method']] += 1
        valid += bool(parsed['parse_valid'])
        empty += not bool(response.strip())
        parsed_rows.append({'id': key, 'pred_answer': parsed['pred_answer'],
                            'parse_valid': parsed['parse_valid'], 'parse_method': parsed['parse_method'],
                            'parse_conflict': parsed['parse_conflict'], 'correct': correct})
        if policy == POLICY:
            parsed_rows[-1].update(answer_status=answer.status, answer_channel=answer.channel)
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
    if policy:
        metrics['answer_policy'] = policy
    return metrics, parsed_rows


def reproduce(bundle):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest.get('schema') != 'npcbench-l5-frozen-v1':
        raise ValueError('Unsupported L5 bundle schema')
    if (manifest['turn_count'], manifest['patient_count'], len(manifest['models'])) != (305, 100, 30):
        raise ValueError('Expected complete frozen L5 inventory: 305 turns / 100 patients / 30 models')
    for relative, digest in manifest['files'].items():
        path = (bundle / relative).resolve()
        if bundle not in path.parents or sha(path) != digest:
            raise ValueError('Bundle file hash or path mismatch')
    if manifest['parser_sha256'] != sha(Path(__file__).with_name('l5_parser.py')):
        raise ValueError('Parser version does not match the frozen export')
    dataset = read_rows(bundle / 'data/l5_main305.jsonl')
    if len(dataset) != manifest['turn_count'] or len({r['patient_id'] for r in dataset}) != manifest['patient_count']:
        raise ValueError('Dataset inventory mismatch')
    models = {}
    for entry in manifest['models']:
        if entry['path'] not in manifest['files']:
            raise ValueError('Prediction path is not covered by the file manifest')
        metrics, _ = score(dataset, read_rows(bundle / entry['path']))
        if metrics != entry['metrics']:
            raise ValueError('Recomputed metrics differ from exported metrics')
        if entry['model'] in models:
            raise ValueError('Duplicate model')
        models[entry['model']] = metrics
    return {'schema': manifest['schema'], 'cohort': manifest['cohort'], 'models': models,
            'model_count': len(models), 'response_count': len(models) * len(dataset), 'verified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, help='Verify and reproduce all frozen models')
    parser.add_argument('--dataset', type=Path, help='Public input JSONL for scoring new predictions')
    parser.add_argument('--predictions', type=Path, help='JSONL: id and response; exact dataset coverage required')
    parser.add_argument('--policy', choices=[POLICY], default=POLICY,
                        help='Answer policy for new predictions (default: current paper policy)')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.bundle and not (args.dataset or args.predictions):
        result = reproduce(args.bundle)
    elif args.dataset and args.predictions and not args.bundle:
        metrics, parsed = score(read_rows(args.dataset), read_rows(args.predictions), args.policy)
        result = {'metrics': metrics, 'parsed_predictions': parsed}
    else:
        parser.error('Use --bundle OR both --dataset and --predictions')
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
