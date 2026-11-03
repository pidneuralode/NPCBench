"""Prepare ordered MRI image groups and score eighth-edition T/N predictions."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from .final_answer import parse_stage_field


def uniform_indices(n, k):
    if not 2 <= k <= n:
        raise ValueError('Each sampled anatomical group needs at least two planes')
    return [(2 * i * (n - 1) + k - 1) // (2 * (k - 1)) for i in range(k)]


def select_groups(images, count):
    """Proportional uniform sampling, including both endpoints of both groups."""
    axial = sum('/axial_' in p for p in images)
    coronal = len(images) - axial
    if axial < 2 or coronal < 2 or not 4 <= count <= len(images):
        raise ValueError('Expected ordered axial and coronal groups with two or more planes each')
    if any('/axial_' not in p for p in images[:axial]) or any('/coronal_' not in p for p in images[axial:]):
        raise ValueError('Images must contain the axial group followed by the coronal group')
    na = max(2, min(axial, count - 2, (2 * count * axial + len(images)) // (2 * len(images))))
    na = max(na, count - coronal)
    return uniform_indices(axial, na) + [axial + i for i in uniform_indices(coronal, count - na)]


def sampled_request(row, indices, condition):
    images = [row['images'][i] for i in indices]
    axial = sum('/axial_' in p for p in images)
    prompt = re.sub(r'共 \d+ 张图', f'共 {len(images)} 张图', row['prompt'])
    prompt = re.sub(r'前 \d+ 张', f'前 {axial} 张', prompt)
    prompt = re.sub(r'后 \d+ 张', f'后 {len(images) - axial} 张', prompt)
    return dict(row, images=images, prompt=prompt, input_condition=condition,
                image_sampling={'source_count': len(row['images']), 'selected_count': len(images),
                                'source_indices_1based': [i + 1 for i in indices]})


def prepare(row, profile, tokenizer=None):
    if profile == 'all' or (profile == 'gpt41' and len(row['images']) <= 50):
        return dict(row)
    if profile == 'gpt41':
        return sampled_request(row, select_groups(row['images'], 50), 'GPT41_uniform50_group_bounds')
    if tokenizer is None:
        raise ValueError('InternVL preparation needs the checkpoint tokenizer')
    for count in range(len(row['images']), 3, -1):
        result = sampled_request(row, select_groups(row['images'], count), 'InternVL_native40960_dynamic4')
        # Verified for axial 512x1644 and coronal 512x1096 groups: 4 and 3 patches.
        visual = sum((4 if '/axial_' in p else 3) * 256 for p in result['images'])
        text = len(tokenizer.encode(result['prompt'])) + 16 * count + 256
        if visual + text + 8192 <= 40960:
            return result
    raise ValueError('Even four image groups exceed the reserved context budget')


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--requests', type=Path, required=True)
    p.add_argument('--profile', choices=['all', 'gpt41', 'internvl'], required=True)
    p.add_argument('--model-path', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('score')
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--stage', choices=['T', 'N'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        tokenizer = None
        if args.profile == 'internvl':
            if args.model_path is None:
                parser.error('InternVL preparation requires --model-path')
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
        results = [prepare(r, args.profile, tokenizer) for r in read_rows(args.requests)]
        text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in results)
    else:
        text = json.dumps(score(read_rows(args.labels), read_rows(args.predictions), args.stage), indent=2) + '\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)


if __name__ == '__main__':
    main()
