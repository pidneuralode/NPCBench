"""CPU comparison of frozen scores and the versioned final-answer candidate.

Emits all item decisions and a deterministic format-review sample without gold
labels or correctness. No source bundle is edited and no inference is invoked.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from . import l3, l5, l6, mcq_release
from .final_answer import POLICY, parse_choice, json_payload, parse_t_stage


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_options(task, item):
    if task in ('L1', 'L2', 'L4', 'L5', 'L6-care'):
        return item['options']
    if task == 'L3-Seq':
        return {x['option_id']: x['sequence_label'] for x in item['options']}
    if task in ('L3-Path', 'L6-pathology'):
        return {'0': '', '1': ''}
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('mcq-bundle', 'l3-bundle', 'l5-bundle', 'l6-dataset', 'l6-results'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    out = args.output_dir
    if out.exists():
        ap.error('Output directory must be new')
    out.mkdir(parents=True)
    scorer_paths = {name: Path(__file__).with_name(name + '.py') for name in
                    ('answer_audit', 'final_answer', 'mcq_release', 'l3', 'l5', 'l6')}
    scorer_hashes = {name: sha(path) for name, path in scorer_paths.items()}
    sources = {name: sha(path / 'manifest.json') for name, path in
               [('mcq', args.mcq_bundle), ('l3', args.l3_bundle), ('l5', args.l5_bundle), ('l6_results', args.l6_results)]}
    sources['l6_contract'] = sha(args.l6_dataset / 'scoring_contract.json')
    checks = {'mcq': mcq_release.reproduce(args.mcq_bundle), 'l3': l3.reproduce(args.l3_bundle),
              'l5': l5.reproduce(args.l5_bundle)}
    (out / 'frozen_replay.json').write_text(json.dumps(checks, indent=2) + '\n')
    cells, statuses, samples, frontends = [], Counter(), {}, []
    decisions = (out / 'decisions.jsonl').open('w')
    response_count = 0

    def compare(task, model, dataset, predictions, original, candidate, old_rows, new_rows, metric_pairs):
        nonlocal response_count
        item_key = 'sample_id' if task == 'L6' else 'id'
        items = {r[item_key]: r for r in dataset}
        before = {r[item_key]: r for r in old_rows}
        after = {r[item_key]: r for r in new_rows}
        if set(items) != set(before) or set(items) != set(after) or len(predictions) != len(items):
            raise ValueError('Scoring changed item coverage')
        flips, invalid = 0, 0
        cell_status = Counter()
        for row in predictions:
            key = row[item_key]
            text = row.get('response') or ''
            old, new = before[key], after[key]
            scoped_task = items[key]['_task'] if task == 'L6' else task
            options = project_options(scoped_task, items[key])
            if scoped_task in ('L3-T-Stg', 'L6-mri'):
                extracted = parse_t_stage(text)
                status, channel, method = extracted.status, extracted.channel, extracted.method
            elif options is None:
                payload, status, channel = json_payload(text)
                method = 'json'
                if status == 'ok' and not new.get('prediction'):
                    status = 'invalid_task_field'
            else:
                extracted = parse_choice(text, options)
                status, channel, method = extracted.status, extracted.channel, extracted.method
            response_sha = hashlib.sha256(text.encode()).hexdigest()
            record = {'task': scoped_task, 'model': model, 'id': key, 'response_sha256': response_sha,
                      'answer_status': status, 'answer_channel': channel, 'answer_method': method,
                      'prediction': new.get('prediction', new.get('pred_answer')),
                      'original_correct': old['correct'], 'candidate_correct': new['correct'],
                      'finish_reason': row.get('finish_reason')}
            decisions.write(json.dumps(record, ensure_ascii=False) + '\n')
            statuses[status] += 1
            cell_status[status] += 1
            flips += old['correct'] != new['correct']
            invalid += status != 'ok'
            # Selection uses format, task, model and response hash only. Neither
            # gold nor old/new correctness can influence which example is read.
            group = (scoped_task, model, status, channel, method)
            sample = {'task': scoped_task, 'model': model, 'id': key, 'response_sha256': response_sha,
                      'response': text, 'options': options,
                      'expected_format': ('single choice' if options is not None else
                                          'T-stage field' if scoped_task in ('L3-T-Stg', 'L6-mri') else 'JSON object'),
                      'answer_status': status, 'answer_channel': channel, 'answer_method': method}
            if group not in samples or response_sha < samples[group]['response_sha256']:
                samples[group] = sample
        response_count += len(predictions)
        cells.append({'task': task, 'model': model, 'responses': len(predictions),
                      'correctness_changes': flips, 'invalid_answers': invalid,
                      'answer_status_counts': dict(cell_status),
                      'original_metrics': original, 'candidate_metrics': candidate,
                      'reported_metrics': [{'metric': label, 'original': original[key],
                                            'candidate': candidate[key], 'scale': scale,
                                            'delta_pp': scale * (float(candidate[key]) - float(original[key]))}
                                           for label, key, scale in metric_pairs]})

    for task, spec in load(args.mcq_bundle / 'manifest.json')['tasks'].items():
        dataset = rows(args.mcq_bundle / spec['dataset_path'])
        for entry in spec['models']:
            predictions = rows(args.mcq_bundle / entry['predictions_path'])
            original, old = mcq_release.score(dataset, predictions, task, entry['parser'])
            candidate, new = mcq_release.score(dataset, predictions, task, POLICY)
            compare(task, entry['model'], dataset, predictions, original, candidate, old, new,
                    [(original['primary_metric'], 'primary_percent', 1)])
    print('MCQ complete', flush=True)

    for entry in load(args.l3_bundle / 'manifest.json')['entries']:
        task = entry['subset']
        dataset = rows(args.l3_bundle / 'data' / (task + '.jsonl'))
        predictions = rows(args.l3_bundle / entry['path'])
        original, old = l3.score(task, dataset, predictions)
        candidate, new = l3.score(task, dataset, predictions, POLICY)
        compare('L3-' + task, entry['model'], dataset, predictions, original, candidate, old, new,
                [('micro_accuracy', 'micro_percent', 1)])
    print('L3 complete', flush=True)

    dataset = rows(args.l5_bundle / 'data/l5_main305.jsonl')
    for entry in load(args.l5_bundle / 'manifest.json')['models']:
        predictions = rows(args.l5_bundle / entry['path'])
        original, old = l5.score(dataset, predictions)
        candidate, new = l5.score(dataset, predictions, POLICY)
        compare('L5', entry['model'], dataset, predictions, original, candidate, old, new,
                [('step_accuracy', 'turn_micro_accuracy', 100), ('episode_success', 'all_turns_correct_patient_rate', 100)])
    print('L5 complete', flush=True)

    data = l6.load_dataset(args.l6_dataset)
    dataset = [dict(row, _task='L6-' + kind) for kind in ('pathology', 'mri', 'care') for row in data[kind]]
    l6_manifest = load(args.l6_results / 'manifest.json')
    if l6_manifest['scoring_contract_sha256'] != sources['l6_contract']:
        raise ValueError('L6 response bundle belongs to a different membership contract')
    expected_models = l6_manifest['complete_models']
    file_hashes = {row['path']: row['sha256'] for row in load(args.l6_results / 'file_manifest.json')}
    files = sorted((args.l6_results / 'predictions').glob('*.jsonl'))
    if (len(files) != expected_models or
            {p.relative_to(args.l6_results).as_posix() for p in files} !=
            {row['response_file'] for row in l6_manifest['models'] if row['status'] == 'complete_raw_coverage'}):
        raise ValueError('Revised L6 model inventory mismatch')
    (out / 'l6').mkdir()
    for path in files:
        if sha(path) != file_hashes.get(path.relative_to(args.l6_results).as_posix()):
            raise ValueError('L6 raw-response hash mismatch')
        predictions = rows(path)
        old, new = l6.score(data, predictions, 'frozen_v1'), l6.score(data, predictions, POLICY)
        (out / 'l6' / (path.stem + '.json')).write_text(json.dumps(new, ensure_ascii=False, indent=2) + '\n')
        compare('L6', path.stem, dataset, predictions, old['metrics'], new['metrics'],
                old['parsed_predictions'], new['parsed_predictions'],
                [('care_accuracy', 'care_micro_percent', 1), ('strict_success', 'strict_episode_percent', 1)])
        if new['frontend_interpretation_changes']:
            frontends.append({'model': path.stem, 'changes': new['frontend_interpretation_changes']})
    decisions.close()
    (out / 'cells.json').write_text(json.dumps(cells, indent=2) + '\n')
    (out / 'frontend_changes.json').write_text(json.dumps(frontends, indent=2) + '\n')
    with (out / 'metric_changes.csv').open('w', newline='') as stream:
        fields = ['task', 'model', 'responses', 'metric', 'original', 'candidate', 'delta_pp', 'invalid_answers', 'correctness_changes']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            for metric in cell['reported_metrics']:
                writer.writerow({**{k: cell[k] for k in fields if k in cell},
                                 **{k: metric[k] for k in fields if k in metric}})
    (out / 'blind_format_review.jsonl').write_text(''.join(
        json.dumps(sample, ensure_ascii=False) + '\n' for _, sample in sorted(samples.items())))
    if scorer_hashes != {name: sha(path) for name, path in scorer_paths.items()}:
        raise RuntimeError('Scoring source changed during audit; output is not a verified receipt')
    summary = {'schema': 'npcbench-final-answer-audit-v1', 'policy': POLICY,
               'status': 'candidate_format_review_pending_not_official_leaderboard',
               'responses': response_count, 'cells': len(cells), 'status_counts': dict(statuses),
               'correctness_changes': sum(c['correctness_changes'] for c in cells),
               'blind_review_examples': len(samples), 'sources': sources,
               'scorer_sha256': scorer_hashes, 'scoring_source_unchanged_during_audit': True,
               'frontend_changed_models': len(frontends), 'model_calls': 0,
               'archives_modified': False, 'primary_policy_changed': False}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
