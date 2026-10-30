"""Build a whitelisted L5 main305 frozen-response archive from an internal stage."""
import argparse
import csv
import hashlib
import hmac
import json
from pathlib import Path
import re
import secrets

from .l5 import read_rows, score, sha, unique


def export(stage, out, private_map):
    stage, out, private_map = Path(stage).resolve(), Path(out).resolve(), Path(private_map).resolve()
    if out.exists():
        raise ValueError('Output directory must be new')
    if private_map == out or out in private_map.parents:
        raise ValueError('Private mapping must be outside public output')
    registry = json.loads((stage / 'data_registry.json').read_text())
    record = next(row for row in registry if row['dataset_id'] == 'L5_main305')
    dataset_path = stage / record['stage_path']
    if sha(dataset_path) != record['sha256']:
        raise ValueError('Frozen L5 input hash mismatch')
    dataset = read_rows(dataset_path)
    source_gold = unique(dataset)
    if len(dataset) != 305 or len({r['patient_id'] for r in dataset}) != 100:
        raise ValueError('This builder requires frozen L5 main305 / 100 patients')
    previous = json.loads(private_map.read_text()) if private_map.exists() else {'salt': secrets.token_hex(32), 'ids': {}}
    salt, ids = bytes.fromhex(previous['salt']), previous['ids']
    aliases = {}

    def public_id(kind, value):
        # Task-local namespace does not imply a cross-task patient linkage.
        key = 'l5-' + kind + ':' + str(value)
        identifier = 'l5-' + kind + '-' + hmac.new(salt, key.encode(), hashlib.sha256).hexdigest()[:24]
        if key in ids and ids[key] != identifier:
            raise ValueError('Inconsistent private mapping')
        ids[key] = identifier
        return identifier

    for row in dataset:
        aliases[row['id']] = public_id('sample', row['id'])
        aliases[row['patient_id']] = public_id('patient', row['patient_id'])
        if row.get('release_item_id'):
            aliases[row['release_item_id']] = aliases[row['id']]
    replacements = sorted(aliases.items(), key=lambda pair: len(pair[0]), reverse=True)
    private_path = re.compile(r'/(?:Users|zju_\d+|home|mnt|data_private)/|[A-Za-z]:\\')

    def clean(value):
        if isinstance(value, str):
            for source, target in replacements:
                value = value.replace(source, target)
            if private_path.search(value):
                raise ValueError('Whitelisted text still contains an internal absolute path')
            return value
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        return value

    fields = ('id', 'patient_id', 'turn_id', 'prompt', 'options', 'answer',
              'gold_option_text', 'gold_action', 'decisive_condition', 'doctor_explanation')
    public_data = [clean({key: row[key] for key in fields if key in row}) for row in dataset]
    # Original order is frozen; turn_id is retained for patient-local ordering.
    unique(public_data)
    with (stage / 'aligned_release_v5/main_table_candidate.csv').open() as handle:
        expected = {row['Model']: row for row in csv.DictReader(handle)}
    runs = [r for r in json.loads((stage / 'primary_text_source_lock.json').read_text()) if r['subset'] == 'L5']
    if len(runs) != 30 or len({r['model'] for r in runs}) != 30:
        raise ValueError('Expected 30 unique frozen L5 models')
    model_entries, public_results = [], {}
    for index, run in enumerate(runs, 1):
        path = (stage / run['stage_path']).resolve()
        if stage not in path.parents or not run['exists'] or not run['historical_hash_matches'] or sha(path) != run['original_sha256']:
            raise ValueError('Frozen response source hash mismatch')
        rows = json.loads(path.read_text())
        by_id = unique(rows)
        if set(by_id) != set(source_gold):
            raise ValueError('Frozen response coverage mismatch')
        for key, row in by_id.items():
            gold = source_gold[key]
            if any(row.get(field) != gold[field] for field in ('answer', 'options', 'prompt', 'patient_id')):
                raise ValueError('Frozen response input does not match canonical input')
            if 'question' in row and row['question'] != gold['question']:
                raise ValueError('Frozen response question mismatch')
        before, parsed_before = score(dataset, rows)
        public = [{'id': aliases[row['id']], 'response': clean(row.get('response') or '')} for row in rows]
        after, parsed_after = score(public_data, public)
        if before != after or [{k: v for k, v in r.items() if k != 'id'} for r in parsed_before] != [{k: v for k, v in r.items() if k != 'id'} for r in parsed_after]:
            raise ValueError('Public identifier substitution changed parsing or scoring')
        main = expected[run['model']]
        if before['paper_step_percent'] != main['L-Rev. Step'] or before['paper_episode_percent'] != main['L-Rev. Epi.']:
            raise ValueError('CPU replay disagrees with aligned_release_v5 L5 table')
        relative = f'predictions/model-{index:02d}.jsonl'
        public_results[relative] = public
        model_entries.append({'model': run['model'], 'path': relative, 'frozen_response_sha256': run['original_sha256'], 'metrics': after})
    excluded = [r['dataset_id'] for r in registry if r['dataset_id'] != 'L5_main305']
    manifest = {
        'schema': 'npcbench-l5-frozen-v1', 'status': 'local_staging_not_published',
        'cohort': 'L5_main305_100patients', 'turn_count': 305, 'patient_count': 100,
        'source_release_version': sorted({r['release_version'] for r in dataset}),
        'source_input_sha256': record['sha256'], 'parser_sha256': sha(Path(__file__).with_name('l5_parser.py')),
        'metric_contract': {'L-Rev. Step': 'turn_micro_accuracy', 'L-Rev. Epi.': 'all_turns_correct_patient_rate',
                            'patient_macro_accuracy': 'diagnostic_only', 'display': 'percent, ROUND_HALF_UP, one decimal'},
        'provenance': 'Frozen historical responses. Only known source IDs are substituted in text; every parsed decision is checked unchanged. No new inference.',
        'patient_identity_scope': 'L5 task-local; no claim of global 906-patient or L6 linkage',
        'excluded_registered_datasets': excluded, 'images': 'L5 main305 is text-only',
        'models': model_entries, 'files': {}}
    # All source, coverage, metric, and privacy checks occur before public files exist.
    out.mkdir(parents=True)
    files = {'data/l5_main305.jsonl': public_data, **public_results}
    for relative, rows in files.items():
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        manifest['files'][relative] = sha(path)
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    private_map.parent.mkdir(parents=True, exist_ok=True)
    private_map.write_text(json.dumps({'salt': previous['salt'], 'ids': ids}, ensure_ascii=False, indent=2) + '\n')
    private_map.chmod(0o600)
    return {'output_dir': str(out), 'turn_count': 305, 'patient_count': 100, 'model_count': 30, 'response_count': 9150}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--private-map', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.stage_root, args.output_dir, args.private_map), indent=2))


if __name__ == '__main__':
    main()
