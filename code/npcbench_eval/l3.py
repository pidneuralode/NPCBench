"""CPU-only L3 scoring from model responses, with frozen 84-cell verification."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from .l3_parser import NPCMMBenchBox, NPCMMBenchSequence, MMBench_Pathology, STRUCTURE_ROWS
from .final_answer import POLICY, parse_choice, json_payload, parse_t_stage


ENDPOINTS = {
    'Path': (69, 'Path.', 'overview_case_accuracy', 'L3_overview69'),
    'Seq': (200, 'Seq.', 'sequence_item_accuracy', 'L3_static_sequence200'),
    'Struct': (2700, 'Struct.', 'base_structure_card_micro_accuracy', 'L3_static_cards2700'),
    'T-Stg': (100, 'T-Stg.', 'T_stage_accuracy', 'L3_static100'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def index(rows):
    result = {}
    for row in rows:
        key = row.get('id')
        if not isinstance(key, str) or not key.strip() or key in result:
            raise ValueError('Expected unique nonempty string IDs')
        result[key] = row
    return result


def make_box():
    box = NPCMMBenchBox()
    box.structure_rows = STRUCTURE_ROWS
    box.structure_id_to_base = {}
    for row in STRUCTURE_ROWS:
        base = row.get('base_structure_en') or row['structure_id']
        box.structure_id_to_base[row['structure_id']] = base
        box.structure_id_to_base[base] = base
    box.structure_alias_to_id = box._load_structure_aliases()
    return box


def score(subset, dataset, predictions, policy=None):
    """Reparse raw responses. Invalid main answers remain in the denominator.

    Main metrics are micro accuracy for all four endpoints. Parser errors in
    secondary fields (e.g. invasion) do not invalidate a correct main field.
    """
    if subset not in ENDPOINTS:
        raise ValueError('Unknown L3 endpoint')
    if policy not in (None, POLICY):
        raise ValueError('Unknown answer policy')
    gold, responses = index(dataset), index(predictions)
    if not gold or set(gold) != set(responses):
        raise ValueError('Responses must exactly cover the nonempty dataset')
    box, seq, pathology = make_box(), NPCMMBenchSequence(), MMBench_Pathology()
    groups, errors, parsed_rows = defaultdict(list), Counter(), []
    for key, row in gold.items():
        response = responses[key].get('response')
        if response is None:
            response = ''
        if not isinstance(response, str):
            raise ValueError('Response must be text or null')
        target = row['gold']
        status, channel = 'ok', 'frozen'
        if policy == POLICY and subset in ('Path', 'Seq'):
            options = {'0': '', '1': ''} if subset == 'Path' else {
                opt['option_id']: opt['sequence_label'] for opt in row['options']}
            answer = parse_choice(response, options)
            status, channel = answer.status, answer.channel
        elif policy == POLICY:
            stage_answer = parse_t_stage(response) if subset == 'T-Stg' else None
            payload, status, channel = json_payload(response)
            response = json.dumps(payload, ensure_ascii=False) if payload is not None else ''
        if subset == 'Path':
            if target not in {'0', '1'}:
                raise ValueError('Invalid binary gold')
            pred = answer.prediction if policy else pathology._parse_prediction(response)
            parse_errors = [] if pred in {'0', '1'} else ['invalid_binary']
        elif subset == 'Seq':
            if target not in {opt['option_id'] for opt in row['options']}:
                raise ValueError('Gold is absent from sequence choices')
            if policy:
                pred = answer.prediction
                parse_errors = [] if answer.valid else [answer.status]
            else:
                parsed, parse_errors = seq._parse_prediction(response, row)
                pred = parsed['option_id']
        elif subset == 'Struct':
            if target not in {box._structure_to_base(c) for c in row['structure_choices']}:
                raise ValueError('Gold is absent from structure choices')
            parsed, parse_errors = box._parse_card_prediction(response, row['structure_choices'])
            pred = box._structure_to_base(parsed['structure_id'])
        else:
            if target not in {'T1', 'T2', 'T3', 'T4'}:
                raise ValueError('Invalid T-stage gold')
            parsed, parse_errors = box._parse_patient_prediction(response, row['base_structure_choices'])
            pred = stage_answer.prediction if policy else parsed['t_stage']
            if policy:
                status, channel = stage_answer.status, stage_answer.channel
        correct = pred == target
        groups[target].append(correct)
        errors.update(parse_errors)
        parsed_rows.append({'id': key, 'prediction': pred, 'correct': correct,
                            'parser_errors': parse_errors})
        if policy:
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
    if policy:
        metrics['answer_policy'] = policy
    return metrics, parsed_rows


def canonical_model(model):
    # These five Path archives alone used the redundant suffix.
    if model.startswith(('Qwen3.5-', 'Qwen3.6-')) and model.endswith('-Instruct'):
        return model[:-len('-Instruct')]
    return model


def project_rows(subset, rows):
    """Project source records to labels/choices plus opaque IDs and responses.

    Prompts, image paths, patient IDs, messages and precomputed scores are never
    copied. The opaque identifier is stable across models within an endpoint.
    """
    dataset, predictions = [], []
    fields = {'Path': 'question_id', 'Seq': 'question_id', 'Struct': 'card_id', 'T-Stg': 'patient_case_id'}
    for row in rows:
        raw_id = row.get(fields[subset], row.get('id'))
        if raw_id is None:
            raise ValueError('Missing source item identity')
        key = subset + '-' + hashlib.sha256((subset + '\0' + str(raw_id)).encode()).hexdigest()[:24]
        item = {'id': key}
        if subset == 'Path':
            item['gold'] = str(row['answer'])
        elif subset == 'Seq':
            item.update(gold=row['correct_option_id'], options=[
                {field: option[field] for field in ('option_id', 'sequence_label', 'sequence_cn')}
                for option in row['options']])
        elif subset == 'Struct':
            # Both structure parsers use choices as a set; archive ordering is
            # nonsemantic and differs for six T-stage model sources.
            item.update(gold=row['base_structure_id_gt'], structure_choices=sorted(row['structure_choices']))
        else:
            item.update(gold=row['t_stage_gt'], base_structure_choices=sorted(row['base_structure_choices']))
        dataset.append(item)
        predictions.append({'id': key, 'response': row.get('response')})
    return dataset, predictions


def _inside(root, relative):
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError('Source path is outside its root')
    return path


def reproduce_stage(stage, export=None):
    """Verify 21 x 4 source hashes, reparse responses, compare aligned_release_v5.

    Export is optional and contains no source prompts or image/patient paths.
    Raw responses still need the release's normal privacy review before publishing.
    """
    stage = Path(stage).resolve()
    lock_path = stage / 'primary_l3_source_lock.json'
    ledger_path = stage / 'aligned_release_v5/cell_ledger.csv'
    locked = json.loads(lock_path.read_text())
    with ledger_path.open(encoding='utf-8-sig', newline='') as stream:
        ledger = {(row['model'], row['column']): row for row in csv.DictReader(stream)}
    if len(locked) != 84:
        raise ValueError('Expected 84 locked L3 source files')
    entries, datasets, seen = [], {}, set()
    pending_files = {}
    for entry in locked:
        model, subset = canonical_model(entry['model']), entry['subset']
        pair = (model, subset)
        if pair in seen or subset not in ENDPOINTS:
            raise ValueError('Duplicate model/endpoint or unknown endpoint')
        seen.add(pair)
        source = _inside(stage, entry['stage_path'])
        if sha(source) != entry['sha256']:
            raise ValueError('Locked L3 source hash mismatch')
        source_rows = json.loads(source.read_text())
        if isinstance(source_rows, dict):
            source_rows = list(source_rows.values())
        dataset, predictions = project_rows(subset, source_rows)
        expected_count, column, metric_name, cohort = ENDPOINTS[subset]
        if len(dataset) != expected_count:
            raise ValueError('Unexpected L3 cohort size')
        canonical_dataset = index(dataset)
        if subset in datasets and canonical_dataset != datasets[subset]:
            raise ValueError('Models do not share identical labels and item identities')
        datasets[subset] = canonical_dataset
        metrics, _ = score(subset, dataset, predictions)
        cell = ledger[(model, column)]
        if cell['source_sha256'] != entry['sha256'] or cell['primary_metric'] != metric_name or cell['cohort'] != cohort:
            raise ValueError('Aligned ledger source or metric/cohort mismatch')
        if abs(metrics['micro_percent'] - float(cell['aligned_value'])) > 1e-10:
            raise ValueError('Reparsed primary score differs from aligned_release_v5')
        response_path = 'responses/' + model + '/' + subset + '.jsonl'
        input_fields = ('prompt', 'messages', 'question_prompt', 'options', 'choices',
                        'structure_choices', 'base_structure_choices', 'images',
                        'asset_path', 'layout_path', 'asset_abspath', 'layout_abspath',
                        'asset_paths', 'asset_abspaths', 'protocol_version')
        present_fields = sorted({field for row in source_rows for field in input_fields if field in row})
        # Preserve evidence of the archived input view and original list ordering.
        # These digests are not claims about unarchived wire requests or wrappers.
        archived_inputs = [{'id': item['id'], 'fields': {field: row[field] for field in present_fields if field in row}}
                           for item, row in zip(dataset, source_rows)]
        prompt_view = [{'id': item['id'], 'prompt': row.get('prompt')}
                       for item, row in zip(dataset, source_rows)]
        entries.append({'model': model, 'subset': subset, 'path': response_path,
                        'source_sha256': entry['sha256'], 'metrics': metrics,
                        'aligned_value': float(cell['aligned_value']), 'primary_metric': metric_name,
                        'cohort': cohort, 'archived_input_fields': present_fields,
                        'archived_input_view_sha256': content_sha(archived_inputs),
                        'archived_prompt_list_sha256': content_sha(prompt_view)})
        if export:
            pending_files[response_path] = predictions
    models = {model for model, _ in seen}
    if len(models) != 21 or seen != {(m, s) for m in models for s in ENDPOINTS}:
        raise ValueError('Expected complete 21-model by 4-endpoint coverage')
    manifest = {'schema': 'npcbench-l3-frozen-v1', 'parser_sha256': sha(Path(__file__).with_name('l3_parser.py')),
                'class_macro_reduction': 'sorted_labels_v1',
                'source_lock_sha256': sha(lock_path), 'aligned_ledger_sha256': sha(ledger_path),
                'choice_order_normalization': {
                    'endpoints': ['Struct', 'T-Stg'],
                    'purpose': 'Offline parser set-membership semantics only; original order remains locked in source/input-view digests.',
                    'historical_prompt_identity_claimed': False,
                    'input_view_scope': 'Archived input fields, not a reconstruction of model wire requests or unarchived wrapper transforms.'},
                'entries': entries, 'files': {}}
    if export:
        export = Path(export)
        if export.exists():
            raise ValueError('Export directory must not already exist')
        for subset, dataset in datasets.items():
            pending_files['data/' + subset + '.jsonl'] = list(dataset.values())
        export.mkdir(parents=True)
        for relative, content in pending_files.items():
            path = export / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in content))
            manifest['files'][relative] = sha(path)
        (export / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return {'verified': True, 'model_count': 21, 'cell_count': 84,
            'response_count': sum(e['metrics']['count'] for e in entries), 'entries': entries,
            'parser_sha256': manifest['parser_sha256'], 'source_lock_sha256': manifest['source_lock_sha256']}


def reproduce(bundle):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest.get('schema') != 'npcbench-l3-frozen-v1':
        raise ValueError('Unsupported L3 bundle schema')
    if manifest['parser_sha256'] != sha(Path(__file__).with_name('l3_parser.py')):
        raise ValueError('Parser differs from the frozen bundle')
    for relative, digest in manifest['files'].items():
        if sha(_inside(bundle, relative)) != digest:
            raise ValueError('Bundle file hash mismatch')
    datasets = {}
    for subset, (count, _, _, _) in ENDPOINTS.items():
        path = 'data/' + subset + '.jsonl'
        if path not in manifest['files']:
            raise ValueError('Dataset is not covered by the file manifest')
        datasets[subset] = read_jsonl(bundle / path)
        if len(datasets[subset]) != count:
            raise ValueError('Dataset inventory mismatch')
    seen, results = set(), []
    for entry in manifest['entries']:
        pair = (entry['model'], entry['subset'])
        if pair in seen or entry['path'] not in manifest['files']:
            raise ValueError('Duplicate entry or unmanifested response file')
        seen.add(pair)
        metrics, _ = score(entry['subset'], datasets[entry['subset']], read_jsonl(_inside(bundle, entry['path'])))
        if metrics != entry['metrics'] or abs(metrics['micro_percent'] - entry['aligned_value']) > 1e-10:
            raise ValueError('Reparsed results do not match frozen expectations')
        results.append({'model': entry['model'], 'subset': entry['subset'], 'metrics': metrics})
    models = {m for m, _ in seen}
    if len(models) != 21 or seen != {(m, s) for m in models for s in ENDPOINTS}:
        raise ValueError('Expected 21 models and all four endpoints')
    return {'verified': True, 'model_count': 21, 'cell_count': 84,
            'response_count': sum(e['metrics']['count'] for e in results), 'entries': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--stage', type=Path)
    mode.add_argument('--bundle', type=Path)
    mode.add_argument('--dataset', type=Path)
    parser.add_argument('--export', type=Path, help='Export a portable response bundle from --stage')
    parser.add_argument('--subset', choices=ENDPOINTS)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--policy', choices=[POLICY], default=POLICY,
                        help='Answer policy for new predictions (default: current paper policy)')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.export and not args.stage:
        parser.error('--export requires --stage')
    if args.stage:
        result = reproduce_stage(args.stage, args.export)
    elif args.bundle:
        result = reproduce(args.bundle)
    else:
        if not args.predictions or not args.subset:
            parser.error('--dataset requires --predictions and --subset')
        metrics, parsed = score(args.subset, read_jsonl(args.dataset), read_jsonl(args.predictions), args.policy)
        result = {'metrics': metrics, 'parsed_predictions': parsed}
    output = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    else:
        print(output, end='')


if __name__ == '__main__':
    main()
