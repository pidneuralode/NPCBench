"""Re-score frozen archives under a narrow unfinished-think candidate gate.

No archive, gold, cohort, response, or primary parser is modified. This is a
CPU sensitivity report, not a new official leaderboard or a unified parser.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from . import l3, l5, mcq_release
from .response_channels import POLICY, gate_response, has_unfinished_think


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def gated(predictions):
    return [dict(row, response=gate_response(row.get('response'))) for row in predictions]


def compare(task, model, predictions, original, candidate, before, after, records):
    old = {row['id']: row for row in before}
    new = {row['id']: row for row in after}
    if set(old) != set(new) or len(old) != len(predictions):
        raise ValueError('Candidate decisions must preserve the original item coverage')
    affected = []
    for row in predictions:
        text = row.get('response') or ''
        if has_unfinished_think(text):
            record = {'task': task, 'model': model, 'id': row['id'],
                      'response_sha256': hashlib.sha256(text.encode()).hexdigest(),
                      'original_correct': old[row['id']]['correct'],
                      'candidate_correct': new[row['id']]['correct'],
                      'finish_reason': row.get('finish_reason')}
            affected.append(record)
    records.extend(affected)
    return {'task': task, 'model': model, 'responses': len(predictions),
            'unfinished_think': len(affected),
            'unfinished_previously_correct': sum(r['original_correct'] for r in affected),
            'original_metrics': original, 'candidate_metrics': candidate}


def audit_mcq(bundle, records):
    mcq_release.reproduce(bundle)
    manifest = load(bundle / 'manifest.json')
    cells, sensitivity = [], []
    for task, spec in manifest['tasks'].items():
        dataset = mcq_release.rows(bundle / spec['dataset_path'])
        for entry in spec['models']:
            predictions = mcq_release.rows(bundle / entry['predictions_path'])
            original, before = mcq_release.score(dataset, predictions, task, entry['parser'])
            candidate, after = mcq_release.score(dataset, gated(predictions), task, entry['parser'])
            cells.append(compare(task, entry['model'], predictions, original, candidate, before, after, records))
            if entry['parser'] == 'local_cpu_v2':
                other, decisions = mcq_release.score(dataset, predictions, task, 'api_mcq_v1')
                other_rows = {row['id']: row for row in decisions}
                sensitivity.append({'task': task, 'model': entry['model'], 'items': len(predictions),
                    'primary_original': original['primary_percent'],
                    'primary_shared_api_parser': other['primary_percent'],
                    'delta_pp': other['primary_percent'] - original['primary_percent'],
                    'frozen_only_right': sum(row['correct'] and not other_rows[row['id']]['correct'] for row in before),
                    'api_parser_only_right': sum(not row['correct'] and other_rows[row['id']]['correct'] for row in before),
                    'api_parse_invalid': other['parse_invalid']})
    return cells, sensitivity


def audit_l3(bundle, records):
    l3.reproduce(bundle)
    cells = []
    for entry in load(bundle / 'manifest.json')['entries']:
        task = entry['subset']
        dataset = l3.read_jsonl(bundle / 'data' / (task + '.jsonl'))
        predictions = l3.read_jsonl(bundle / entry['path'])
        original, before = l3.score(task, dataset, predictions)
        candidate, after = l3.score(task, dataset, gated(predictions))
        cells.append(compare('L3-' + task, entry['model'], predictions, original, candidate, before, after, records))
    return cells


def audit_l5(bundle, records):
    l5.reproduce(bundle)
    cells = []
    dataset = l5.read_rows(bundle / 'data/l5_main305.jsonl')
    for entry in load(bundle / 'manifest.json')['models']:
        predictions = l5.read_rows(bundle / entry['path'])
        original, before = l5.score(dataset, predictions)
        candidate, after = l5.score(dataset, gated(predictions))
        cells.append(compare('L5', entry['model'], predictions, original, candidate, before, after, records))
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['mcq-bundle', 'l3-bundle', 'l5-bundle']:
        ap.add_argument('--' + name, type=Path)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    bundles = {'mcq': args.mcq_bundle, 'l3': args.l3_bundle, 'l5': args.l5_bundle}
    if not any(bundles.values()):
        ap.error('Provide at least one frozen bundle')
    if args.output_dir.exists():
        ap.error('Output directory must not exist')
    cells, records, sensitivity, sources = [], [], [], {}
    for kind, bundle in bundles.items():
        if bundle is None:
            continue
        sources[kind] = {'manifest_sha256': sha(bundle / 'manifest.json')}
        if kind == 'mcq':
            new_cells, sensitivity = audit_mcq(bundle, records)
        elif kind == 'l3':
            new_cells = audit_l3(bundle, records)
        else:
            new_cells = audit_l5(bundle, records)
        cells.extend(new_cells)
    summary = {'schema': 'npcbench-channel-audit-v1', 'policy': POLICY,
        'status': 'sensitivity_candidate_not_official_scores', 'sources': sources,
        'policy_sha256': sha(Path(__file__).with_name('response_channels.py')),
        'driver_sha256': sha(Path(__file__)), 'cells': len(cells),
        'responses': sum(c['responses'] for c in cells), 'unfinished_think': len(records),
        'unfinished_previously_correct': sum(r['original_correct'] for r in records),
        'finish_reason_of_unfinished': dict(Counter(str(r['finish_reason']) for r in records)),
        'model_calls': 0, 'archives_modified': False, 'primary_policy_changed': False,
        'interpretation': 'Unfinished explicit reasoning is blanked before the same frozen parser. All other output bytes, cohorts and denominators stay fixed. This does not validate other parser decisions or prove a length stop.'}
    args.output_dir.mkdir(parents=True)
    for name, data in [('summary.json', summary), ('cells.json', cells)]:
        (args.output_dir / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    (args.output_dir / 'affected_rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    if sensitivity:
        with (args.output_dir / 'local_vs_api_parser.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(sensitivity[0]))
            writer.writeheader()
            writer.writerows(sensitivity)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
