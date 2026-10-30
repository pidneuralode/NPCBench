"""Export the exact v5-selected L1/L2/L4 inputs and frozen text responses."""
import argparse
import csv
import hashlib
import hmac
import json
from pathlib import Path
import re
import secrets

from .mcq_release import TASKS, PARSER_FILES, rows, score, sha, unique


def load(path):
    return json.loads(Path(path).read_text())


def public_parameters(parameters, allowed):
    if not set(parameters) <= set(allowed):
        raise ValueError('Frozen generation settings contain an unreviewed public field')
    return {key: parameters[key] for key in allowed if key in parameters}


def export(stage, out, private_map, tasks=('L1',)):
    stage, out, private_map = Path(stage).resolve(), Path(out).resolve(), Path(private_map).resolve()
    if out.exists() or private_map == out or out in private_map.parents:
        raise ValueError('Output must be new and private map must be outside public output')
    if not tasks or len(set(tasks)) != len(tasks) or not set(tasks) <= set(TASKS):
        raise ValueError('Select distinct supported tasks')

    def source_path(value):
        path = Path(value)
        if path.parts[:1] == ('release_artifacts',):
            path = Path(*path.parts[2:])
        path = (stage / path).resolve()
        if stage not in path.parents:
            raise ValueError('Source escapes supplied stage')
        return path

    registry = {r['dataset_id']: r for r in load(stage / 'data_registry.json')}
    with (stage / 'aligned_release_v5/cell_ledger.csv').open(encoding='utf-8-sig') as handle:
        ledger = list(csv.DictReader(handle))
    with (stage / 'aligned_release_v5/main_table_candidate.csv').open(encoding='utf-8-sig') as handle:
        table = {r['Model']: r for r in csv.DictReader(handle)}
    api_sources = {r['model']: r for r in load(stage / 'l1_api_accepted_v1/source_registry.json')}
    mcq_sources = {r['sha256']: r for r in load(stage / 'mcq_run_registry.json')}
    health_sources = {(r['model'], r['subset']): r for r in load(stage / 'aligned_release_v3/healthgpt_source_registry.json')}
    group_map = load(stage / 'mcq_primary_contract_inputs.json')['group_map']
    parsers = {key: sha(Path(__file__).with_name(name)) for key, name in PARSER_FILES.items()}
    if parsers['api_mcq_v1'] != sha(stage / 'code/api_mcq.py') or parsers['local_cpu_v2'] != sha(stage / 'code/medevalkit_release_v2.py'):
        raise ValueError('Installed parser differs from selected stage parser')
    previous = load(private_map) if private_map.exists() else {'salt': secrets.token_hex(32), 'ids': {}}
    salt, ids = bytes.fromhex(previous['salt']), previous['ids']
    files, task_specs = {}, {}
    private_path = re.compile(r'/(?:Users|zju_\d+|home|mnt|data_private|root|tmp|opt)/|[A-Za-z]:\\')

    def public_id(task, kind, value):
        key = task.lower() + '-' + kind + ':' + str(value)
        identifier = task.lower() + '-' + kind + '-' + hmac.new(salt, key.encode(), hashlib.sha256).hexdigest()[:24]
        if key in ids and ids[key] != identifier:
            raise ValueError('Inconsistent private ID mapping')
        ids[key] = identifier
        return identifier

    for task in tasks:
        dataset_id = {'L1': 'L1_mcq250', 'L2': 'L2_optimized185', 'L4': 'L4_optimized165'}[task]
        record = registry[dataset_id]
        data_path = source_path(record['stage_path'])
        if sha(data_path) != record['sha256']:
            raise ValueError('Canonical dataset hash mismatch')
        source_data = load(data_path)['questions'] if task == 'L1' else rows(data_path)
        canonical, aliases = [], {}
        for original in source_data:
            key = str(original['id'] if task == 'L1' else original['item_uid'])
            item = {'id': key, 'question': original['stem'] if task == 'L1' else original['question'],
                    'options': original['options'], 'answer': original['answer'] if task == 'L1' else original['correct_option'],
                    'choices': [f'{label}. {text}' for label, text in original['options'].items()]}
            for field in ('vignette', 'explanation', 'source', 'evidence_anchors', 'answer_text', 'domain', 'source_type', 'scope_major', 'scope_minor', 'difficulty'):
                if field in original:
                    item[field] = original[field]
            if task != 'L1':
                item['scenario_type'] = original['meta_type']
                item['patient_id'] = str(original['meta_patient_id'])
                aliases[item['patient_id']] = public_id(task, 'patient', item['patient_id'])
                aliases[key] = public_id(task, 'sample', key)
                if task == 'L2':
                    prefix = str(original['question_id']).split('::', 1)[0]
                    item['capability_group'] = group_map[prefix if prefix in group_map else original['meta_type']]
            canonical.append(item)
        gold = unique(canonical)
        if len(gold) != TASKS[task][0]:
            raise ValueError('Canonical dataset count mismatch')
        replacements = sorted(aliases.items(), key=lambda pair: len(pair[0]), reverse=True)

        def clean(value):
            if isinstance(value, str):
                for source, target in replacements:
                    value = value.replace(source, target)
                if private_path.search(value):
                    raise ValueError('Whitelisted text contains an internal absolute path')
                return value
            if isinstance(value, dict):
                return {key: clean(item) for key, item in value.items()}
            if isinstance(value, list):
                return [clean(item) for item in value]
            return value

        public_data = []
        for item in canonical:
            public_item = clean(item)
            # Numeric L1 source IDs must never be replaced within clinical text.
            public_item['id'] = public_id(task, 'sample', item['id'])
            public_data.append(public_item)
        public_gold = unique(public_data)
        sample_ids = {key: public_id(task, 'sample', key) for key in gold}
        dataset_rel = f'data/{task.lower()}.jsonl'
        files[dataset_rel] = public_data
        cells = [r for r in ledger if r['column'] == TASKS[task][1]]
        if len(cells) != 30 or len({r['model'] for r in cells}) != 30 or any(r['status'] != 'aligned' for r in cells):
            raise ValueError('Expected 30 aligned primary cells per task')
        entries = []
        for index, cell in enumerate(cells, 1):
            raw_path = source_path(cell['source'])
            if sha(raw_path) != cell['source_sha256']:
                raise ValueError('Selected ledger response hash mismatch')
            raw = rows(raw_path) if raw_path.suffix == '.jsonl' else load(raw_path)
            predictions, requests = [], []
            provenance = {'source_sha256': cell['source_sha256'], 'ledger_status': cell['status'],
                          'cohort': cell['cohort'], 'generation_parameters_status': 'unavailable_in_selected_historical_source'}
            if task == 'L1' and cell['model'] in api_sources:
                selected = api_sources[cell['model']]
                if selected['raw_sha256'] != cell['source_sha256'] or selected['dataset_sha256'] != record['sha256']:
                    raise ValueError('Accepted replacement source does not match v5 selection')
                bundle = raw_path.parents[2] / 'bundle'
                cfg = load(bundle / 'config.json')
                if sha(bundle / 'config.json') != selected['config_sha256'] or sha(bundle / 'requests.jsonl') != selected['requests_sha256'] or selected['parser_sha256'] != parsers['api_mcq_v1']:
                    raise ValueError('Replacement config/request/parser hash mismatch')
                req = unique(rows(bundle / 'requests.jsonl'), 'request_id')
                keys = unique(rows(bundle / 'answer_keys.jsonl'), 'request_id')
                if sha(bundle / 'answer_keys.jsonl') != cfg['answer_keys_sha256'] or set(unique(raw, 'request_id')) != set(req) or set(req) != set(keys):
                    raise ValueError('Replacement answer-key/coverage mismatch')
                spec = next(m for m in cfg['models'] if m['model'] == cell['model'])
                for result in raw:
                    request, keyrow = req[result['request_id']], keys[result['request_id']]
                    item_id = str(result['item_id'])
                    expected_hash = hashlib.sha256(request['prompt'].encode()).hexdigest()
                    if item_id != str(keyrow['item_id']) or result['prompt_sha256'] != expected_hash or request['prompt_sha256'] != expected_hash or keyrow['gold'] != gold[item_id]['answer'] or keyrow['options'] != gold[item_id]['options']:
                        raise ValueError('Replacement prompt/gold/options join mismatch')
                    requests.append({'id': item_id, 'prompt': request['prompt']})
                    predictions.append({'id': item_id, 'response': result['response'], 'status': result['status'],
                                        'finish_reason': result.get('finish_reason'), 'transport_attempt_count': len(result.get('attempts', []))})
                parser = 'api_mcq_v1'
                provenance.update(source_identity='new_api_replacement_not_recovered_history', run_id=selected['run_id'],
                                  snapshot_identity=selected['snapshot_identity'], config_sha256=selected['config_sha256'],
                                  requests_sha256=selected['requests_sha256'], generation_parameters=public_parameters(spec['parameters'], ('temperature', 'max_tokens', 'top_p')),
                                  generation_parameters_status='frozen_config', provider_returned_models=sorted({str(r.get('provider_model') or '') for r in raw}))
            elif task != 'L1' and 'gpu_runs' in raw_path.parts:
                bundle = raw_path.parents[2] / 'bundle'
                cfg_path = bundle / (raw_path.parent.name + '.json')
                cfg = load(cfg_path)
                health_source = health_sources[(cell['model'], task)]
                if health_source['sha256'] != cell['source_sha256'] or health_source['config_sha256'] != sha(cfg_path) or health_source['dataset_sha256'] != record['sha256']:
                    raise ValueError('Optimized replacement provenance does not match selected source')
                req = unique(rows(bundle / cfg['requests_file']), 'request_id')
                keys = unique(rows(bundle / cfg['answer_keys_file']), 'request_id')
                if sha(bundle / cfg['requests_file']) != cfg['requests_sha256'] or sha(bundle / cfg['answer_keys_file']) != cfg['answer_keys_sha256'] or cfg['scorer_sha256'] != parsers['local_cpu_v2']:
                    raise ValueError('Optimized GPU replacement input/parser mismatch')
                if set(unique(raw, 'request_id')) != set(req) or set(req) != set(keys):
                    raise ValueError('Optimized GPU replacement coverage mismatch')
                for result in raw:
                    if result['slice'] != task:
                        continue
                    request, keyrow = req[result['request_id']], keys[result['request_id']]
                    item_id = str(result['item_uid'])
                    expected_hash = hashlib.sha256(request['prompt'].encode()).hexdigest()
                    expected_group = gold[item_id]['capability_group'] if task == 'L2' else gold[item_id]['patient_id']
                    if result['prompt_sha256'] != expected_hash or request['prompt_sha256'] != expected_hash or request['dataset_sha256'] != record['sha256'] or keyrow['gold'] != gold[item_id]['answer'] or keyrow['options'] != gold[item_id]['options'] or keyrow['choices'] != gold[item_id]['choices'] or keyrow['group_id'] != expected_group:
                        raise ValueError('Optimized replacement prompt/input/group join mismatch')
                    requests.append({'id': item_id, 'prompt': request['prompt']})
                    predictions.append({'id': item_id, 'response': result['response'], 'finish_reason': result.get('finish_reason')})
                parser = 'local_cpu_v2'
                provenance.update(source_identity='optimized_input_gpu_replacement', generation_parameters=public_parameters(cfg['sampling'], ('temperature', 'top_p', 'top_k', 'repetition_penalty', 'max_tokens', 'seed', 'stop_token_ids')),
                                  chat_template_parameters=public_parameters(cfg['template'], ('add_generation_prompt', 'enable_thinking')), generation_parameters_status='frozen_config',
                                  config_sha256=sha(cfg_path), requests_sha256=cfg['requests_sha256'])
            else:
                parser = ('api_mcq_v1' if all('api_attempts' in r for r in raw) else 'local_cpu_v2') if task == 'L1' else 'api_mcq_v1' if mcq_sources[cell['source_sha256']]['scorer'] == 'api_parser' else 'local_cpu_v2'
                if task != 'L1' and mcq_sources[cell['source_sha256']]['dataset_id'] != dataset_id:
                    raise ValueError('Legacy input cannot fill optimized primary cohort')
                if len(raw) != len(canonical):
                    raise ValueError('Historical input coverage mismatch')
                for position, result in enumerate(raw):
                    item_id = str(result['id']) if task == 'L1' else canonical[position]['id']
                    item = gold[item_id]
                    if result['question'] != item['question'] or result['options'] != item['options'] or result['answer'] != item['answer']:
                        raise ValueError('Historical ordered input join mismatch')
                    if result.get('image_source'):
                        raise ValueError('This text-only builder cannot omit a model image input')
                    if parser == 'local_cpu_v2' and result['choices'] != item['choices']:
                        raise ValueError('Frozen local choices differ from canonical choices')
                    prompt = result.get('prompt')
                    if prompt is None:
                        messages = result.get('messages')
                        prompt = messages.get('prompt') if isinstance(messages, dict) else None
                    if not isinstance(prompt, str) or not prompt.strip():
                        raise ValueError('Frozen historical request prompt is unavailable')
                    requests.append({'id': item_id, 'prompt': prompt})
                    predictions.append({'id': item_id, 'response': result.get('response') or '', 'finish_reason': result.get('finish_reason'),
                                        'api_error_type': result.get('api_error_type')})
                provenance['source_identity'] = 'historical_response_rescored_cpu_v2' if parser == 'local_cpu_v2' else 'historical_api_response_frozen_parser'
            before, decisions_before = score(canonical, predictions, task, parser)
            if set(unique(requests)) != set(gold):
                raise ValueError('Frozen request coverage mismatch')
            for request in requests:
                item = gold[request['id']]
                normalized_prompt = re.sub(r'\s+', ' ', request['prompt']).strip()
                required_text = [item['question'], item.get('vignette') or '', *item['options'].values()]
                if any(re.sub(r'\s+', ' ', text).strip() not in normalized_prompt for text in required_text):
                    raise ValueError('Frozen prompt omits canonical question, vignette, or option text')
            pub_predictions = [{**clean(r), 'id': sample_ids[r['id']]} for r in predictions]
            pub_requests = [{**clean(r), 'id': sample_ids[r['id']]} for r in requests]
            after, decisions_after = score(public_data, pub_predictions, task, parser)
            if before != after or [{k: v for k, v in r.items() if k != 'id'} for r in decisions_before] != [{k: v for k, v in r.items() if k != 'id'} for r in decisions_after]:
                raise ValueError('ID substitution changed frozen scoring')
            if before['primary_display'] != cell['aligned_display'] or before['primary_display'] != table[cell['model']][TASKS[task][1]] or abs(before['primary_percent'] - float(cell['aligned_value'])) > 1e-10:
                raise ValueError('CPU primary metric differs from the current v5 ledger/table')
            pred_rel, req_rel = f'predictions/{task.lower()}/model-{index:02d}.jsonl', f'requests/{task.lower()}/model-{index:02d}.jsonl'
            files[pred_rel], files[req_rel] = pub_predictions, pub_requests
            entries.append(clean({'model': cell['model'], 'parser': parser, 'predictions_path': pred_rel,
                                  'requests_path': req_rel, 'metrics': after, 'provenance': provenance}))
        if task == 'L2' and len({r['capability_group'] for r in public_data}) != 4:
            raise ValueError('Expected four fixed L2 capability groups')
        if task == 'L4' and len({r['patient_id'] for r in public_data}) != 84:
            raise ValueError('Expected 84 L4 patient groups')
        task_specs[task] = {'dataset_id': dataset_id, 'dataset_path': dataset_rel, 'source_input_sha256': record['sha256'],
                            'primary_metric': TASKS[task][2], 'models': entries,
                            'patient_identity_scope': 'task-local; cross-task/global membership not asserted'}
        if task == 'L1':
            task_specs[task]['guideline_sources'] = [{k: d[k] for k in ('title', 'short_name', 'language', 'version', 'url') if k in d} for d in load(data_path)['source_documents']]
            task_specs[task]['dataset_version'] = load(data_path)['version']
    manifest = {'schema': 'npcbench-text-mcq-frozen-v1', 'status': 'local_staging_not_published',
                'source_selection': 'aligned_release_v5 cell ledger; no historical-source substitution',
                'source_ledger_sha256': sha(stage / 'aligned_release_v5/cell_ledger.csv'),
                'parsers': parsers, 'tasks': task_specs, 'files': {},
                'response_semantics': 'Frozen response text; known source IDs remapped. No inference or answer-based retries.',
                'local_parser_note': 'CPU v2 retains its documented legacy fuzzy comparison; parse validity is not exposed by that judge.',
                'excluded_registered_datasets': [key for key in registry if key not in {spec['dataset_id'] for spec in task_specs.values()}]}
    out.mkdir(parents=True)
    for relative, records in files.items():
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
        manifest['files'][relative] = sha(path)
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    private_map.parent.mkdir(parents=True, exist_ok=True)
    private_map.write_text(json.dumps({'salt': previous['salt'], 'ids': ids}, ensure_ascii=False, indent=2) + '\n')
    private_map.chmod(0o600)
    return {'tasks': list(tasks), 'model_task_cells': len(tasks) * 30,
            'responses': sum(TASKS[t][0] * 30 for t in tasks), 'output_dir': str(out)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage-root', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--private-map', type=Path, required=True)
    ap.add_argument('--tasks', choices=list(TASKS), nargs='+', default=['L1'])
    args = ap.parse_args()
    print(json.dumps(export(args.stage_root, args.output_dir, args.private_map, args.tasks), indent=2))


if __name__ == '__main__':
    main()
