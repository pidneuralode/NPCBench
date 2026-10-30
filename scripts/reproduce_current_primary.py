#!/usr/bin/env python3
"""Recompute the current 306-cell primary table from versioned raw bundles.

Historical bundle metrics stay unchanged. This command binds their raw outputs
to final_answer_v1, retains O1's distinct judge contract, and checks the v6 table.
It performs no model calls and needs no private patient map or source checkout.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

from npcbench_eval import final_answer, l3, l5, l6, mcq_release, o1_release


def load(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def independent_aggregate(task, dataset, decisions):
    """Use public item membership and parsed answers, not scorer aggregates."""
    items = {r['id']: r for r in dataset}
    parsed = {r['id']: r for r in decisions}
    if len(items) != len(dataset) or len(parsed) != len(decisions) or set(items) != set(parsed):
        raise ValueError('Decision membership differs: ' + task)
    groups = defaultdict(list)
    correct = []
    for sid, item in items.items():
        decision = parsed[sid]
        prediction = decision.get('prediction', decision.get('pred_answer'))
        answer = item['gold'] if task.startswith('L3-') else item['answer']
        right = prediction == answer
        if right != decision['correct']:
            raise ValueError('Prediction versus gold differs: ' + task)
        group = (item['capability_group'] if task == 'L2' else
                 item['patient_id'] if task in ('L4', 'L5') else sid)
        groups[group].append(right)
        correct.append(right)
    if task == 'L5':
        return {'L-Rev. Step': 100 * sum(correct) / len(correct),
                'L-Rev. Epi.': 100 * sum(all(v) for v in groups.values()) / len(groups)}
    endpoint = {'L1': 'G-Mem.', 'L2': 'A-App.', 'L4': 'C-Plan',
                'L3-Path': 'Path.', 'L3-Seq': 'Seq.', 'L3-Struct': 'Struct.', 'L3-T-Stg': 'T-Stg.'}[task]
    value = (100 * sum(sum(v) / len(v) for v in groups.values()) / len(groups)
             if task in ('L2', 'L4') else 100 * sum(correct) / len(correct))
    return {endpoint: value}


def compare_reference(actual, reference):
    with (reference / 'cell_ledger.csv').open() as stream:
        expected_rows = list(csv.DictReader(stream))
    expected = {(r['model'], r['column']): r for r in expected_rows}
    if len(expected) != len(expected_rows) or set(actual) != set(expected):
        raise ValueError('Model/endpoint inventory differs from the reference')
    with (reference / 'main_table_candidate.csv').open() as stream:
        table = list(csv.DictReader(stream))
    displayed = {(r['Model'], column): value for r in table for column, value in r.items()
                 if column not in ('Family', 'Model') and value not in ('', '--', '-', '—')}
    if set(displayed) != set(expected):
        raise ValueError('Displayed table inventory differs from the ledger')
    for key, value in actual.items():
        if abs(value - float(expected[key]['aligned_value'])) > 1e-10:
            raise ValueError('Current score differs from reference: ' + '/'.join(key))
        if f'{value:.1f}' != displayed[key] or displayed[key] != expected[key]['aligned_display']:
            raise ValueError('Displayed precision differs: ' + '/'.join(key))
    return table


def run(args):
    if args.output.exists():
        raise ValueError('Use a new output directory; source archives are never overwritten')
    source_roots = {k: getattr(args, k) for k in ('mcq', 'l3', 'l5', 'o1', 'l6_results')}
    source_hashes = {k: sha(v / 'manifest.json') for k, v in source_roots.items()}
    source_hashes['l6_inputs'] = sha(args.l6_inputs / 'file_manifest.json')
    source_hashes['reference_ledger'] = sha(args.reference / 'cell_ledger.csv')
    full = load(args.l6_results / 'manifest.json')
    if full['answer_parser_sha256'] != sha(final_answer.__file__):
        raise ValueError('Installed parser differs from the frozen Full100 release')
    if source_hashes['l6_inputs'] != full['input_file_manifest_sha256']:
        raise ValueError('Full100 input version differs')
    for entry in load(args.l6_results / 'file_manifest.json'):
        p = (args.l6_results / entry['path']).resolve()
        if args.l6_results.resolve() not in p.parents or sha(p) != entry['sha256']:
            raise ValueError('Full100 result payload differs')

    # Retain each historical contract and verify its complete payload first.
    historical = {'mcq': mcq_release.reproduce(args.mcq), 'l3': l3.reproduce(args.l3),
                  'l5': l5.reproduce(args.l5), 'o1': o1_release.reproduce(args.o1)}
    args.output.mkdir(parents=True)
    (args.output / 'historical_replay.json').write_text(json.dumps(historical, indent=2) + '\n')
    values, cell_rows, task_counts, response_files = {}, [], defaultdict(int), []

    def add(model, endpoints, source, independent):
        for column, value in endpoints.items():
            key = model, column
            if key in values:
                raise ValueError('Duplicate model/endpoint')
            values[key] = value
            cell_rows.append(dict(model=model, column=column, value=value,
                                  independent_aggregation=independent, source=source))

    with (args.output / 'decisions.jsonl').open('w') as out:
        def record(task, model, raw, parsed, root_key, relative):
            key = 'sample_id' if task == 'L6' else 'id'
            raw_by_id = {r[key]: r for r in raw}
            if len(raw_by_id) != len(raw) or set(raw_by_id) != {r[key] for r in parsed}:
                raise ValueError('Raw-response membership differs')
            for decision in parsed:
                response = raw_by_id[decision[key]]
                row = dict(task=task, model=model, id=decision[key],
                           prediction=decision.get('prediction', decision.get('pred_answer')),
                           correct=decision['correct'], answer_status=decision.get('answer_status'),
                           response_sha256=hashlib.sha256((response.get('response') or '').encode()).hexdigest(),
                           finish_reason=response.get('finish_reason'))
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
            task_counts[task] += len(raw)
            response_files.append(dict(bundle=root_key, path=relative,
                                       sha256=sha(source_roots[root_key] / relative), responses=len(raw)))

        for task, spec in load(args.mcq / 'manifest.json')['tasks'].items():
            data = rows(args.mcq / spec['dataset_path'])
            for entry in spec['models']:
                raw = rows(args.mcq / entry['predictions_path'])
                metrics, parsed = mcq_release.score(data, raw, task, final_answer.POLICY)
                independent = independent_aggregate(task, data, parsed)
                if abs(next(iter(independent.values())) - metrics['primary_percent']) > 1e-10:
                    raise ValueError('Independent MCQ aggregate differs')
                add(entry['model'], independent, 'mcq', True)
                record(task, entry['model'], raw, parsed, 'mcq', entry['predictions_path'])
        for entry in load(args.l3 / 'manifest.json')['entries']:
            subset, model = entry['subset'], entry['model']
            data, raw = rows(args.l3 / ('data/' + subset + '.jsonl')), rows(args.l3 / entry['path'])
            metrics, parsed = l3.score(subset, data, raw, final_answer.POLICY)
            independent = independent_aggregate('L3-' + subset, data, parsed)
            if abs(next(iter(independent.values())) - metrics['micro_percent']) > 1e-10:
                raise ValueError('Independent L3 aggregate differs')
            add(model, independent, 'l3', True)
            record('L3-' + subset, model, raw, parsed, 'l3', entry['path'])
        data = rows(args.l5 / 'data/l5_main305.jsonl')
        for entry in load(args.l5 / 'manifest.json')['models']:
            raw = rows(args.l5 / entry['path'])
            metrics, parsed = l5.score(data, raw, final_answer.POLICY)
            independent = independent_aggregate('L5', data, parsed)
            if (abs(independent['L-Rev. Step'] - 100 * metrics['turn_micro_accuracy']) > 1e-10 or
                    abs(independent['L-Rev. Epi.'] - 100 * metrics['all_turns_correct_patient_rate']) > 1e-10):
                raise ValueError('Independent L5 aggregate differs')
            add(entry['model'], independent, 'l5', True)
            record('L5', entry['model'], raw, parsed, 'l5', entry['path'])
        data = l6.load_dataset(args.l6_inputs)
        for entry in full['models']:
            raw = rows(args.l6_results / entry['response_file'])
            result = l6.score(data, raw, final_answer.POLICY)
            gold = {**{r['sample_id']: str(r['answer']) for r in data['pathology']},
                    **{r['sample_id']: r['answer']['T_stage'] for r in data['mri']},
                    **{r['sample_id']: r['gold_letter'] for r in data['care']}}
            parsed = result['parsed_predictions']
            correct = {r['sample_id']: r['prediction'] == gold[r['sample_id']] for r in parsed}
            if set(correct) != set(gold) or any(correct[r['sample_id']] != r['correct'] for r in parsed):
                raise ValueError('Full100 parsed membership or correctness differs')
            strict = sum(all(correct[sid] for sid in
                         [e['pathology_sample_id'], e['mri_sample_id']] + [t['sample_id'] for t in e['care_turns']])
                         for e in data['episodes'])
            independent = {'CP-Epi. Step': 100 * sum(correct[r['sample_id']] for r in data['care']) / len(data['care']),
                           'CP-Epi. Epi.': 100 * strict / len(data['episodes'])}
            if (abs(independent['CP-Epi. Step'] - result['metrics']['care_micro_percent']) > 1e-10 or
                    abs(independent['CP-Epi. Epi.'] - result['metrics']['strict_episode_percent']) > 1e-10):
                raise ValueError('Independent Full100 aggregate differs')
            add(entry['model'], independent, 'l6_results', True)
            record('L6', entry['model'], raw, parsed, 'l6_results', entry['response_file'])
    for model, result in historical['o1']['models'].items():
        add(model, {'C-Rub.': result['metrics']['overall_score']}, 'o1', False)
    table = compare_reference(values, args.reference)
    expected_counts = {'L1': 7500, 'L2': 5550, 'L4': 4950, 'L3-Path': 1449,
                       'L3-Seq': 4200, 'L3-Struct': 56700, 'L3-T-Stg': 2100, 'L5': 9150, 'L6': 8757}
    if dict(task_counts) != expected_counts or len(values) != 306 or len(table) != 30:
        raise ValueError('Current primary release coverage differs')
    for name, content in [('current_cells.csv', cell_rows), ('main_table.csv', table)]:
        with (args.output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(content[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(content)
    summary = dict(status='all_current_primary_values_match', answer_policy=final_answer.POLICY,
                   source_hashes=source_hashes, parser_sha256=sha(final_answer.__file__),
                   verifier_sha256=sha(__file__), models=30, cells=306,
                   independent_aggregate_cells=sum(r['independent_aggregation'] for r in cell_rows),
                   o1_replay_cells=30, response_counts=dict(task_counts),
                   mcq_and_full100_responses=sum(task_counts.values()),
                   o1_generation_rows=historical['o1']['generation_rows'],
                   o1_judgment_rows=historical['o1']['judgment_rows'],
                   raw_response_files=response_files, model_calls=0,
                   scope='Complete primary point scores and displayed precision. Existing uncertainty is separate; no new bootstrap or all-checkpoint inference certification.',
                   historical_archives_modified=False)
    (args.output / 'verification.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'raw_response_files'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('mcq', 'l3', 'l5', 'o1', 'l6-inputs', 'l6-results', 'reference', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    run(parser.parse_args())
