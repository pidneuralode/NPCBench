"""Whitelist the v5 O1 generation/judge archive and verify raw-response CPU scoring."""
import argparse
import ast
from collections import Counter
import csv
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import tempfile

from .o1_release import rows, score, sha, unique, scorer, schema

ITEM_FIELDS = ('item_id', 'case_id', 'patient_id', 'tag', 'tag_label', 'question_slot', 'question', 'case_input', 'rubric_release_version')
RUBRIC_FIELDS = ('total_score', 'core_answer', 'key_evidence', 'conditional_modifiers', 'negative_list', 'scoring_rules', 'synonym_rules', 'formal_scorer_policy', 'scoring_rule')
GEN_FIELDS = ('item_id', 'case_id', 'patient_id', 'tag', 'tag_label', 'question_slot', 'model_name', 'model_class', 'prompt', 'response', 'generation_status', 'generation_error', 'finish_reason', 'api_attempts', 'api_latency_sec', 'api_attempt_latencies_sec', 'usage')
JUDGE_FIELDS = ('rubric_id', 'item_id', 'case_id', 'tag', 'model_name', 'judge_model', 'judge_provider', 'judge_status', 'criteria_met', 'explanation', 'raw_judge_response', 'retry_count', 'error')
TASK_FIELDS = ('rubric_id', 'item_id', 'case_id', 'patient_id', 'tag', 'tag_label', 'question_slot', 'rubric_section', 'model_name')


def load(path):
    return json.loads(Path(path).read_text())


def scoring_functions(path):
    return {node.name: ast.dump(node) for node in ast.parse(Path(path).read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def public_config(config, role=None):
    """Export only recorded settings, retaining generation/judge role and source."""
    allowed = ('stage', 'created_at', 'data_sha256', 'generation_prompt_version', 'judge_prompt_version', 'scorer_version', 'scoring_rule', 'judge_model_name')
    params = {
        'temperature', 'max_tokens', 'max_new_tokens', 'top_p', 'top_k', 'seed',
        'repetition_penalty', 'judge_temperature', 'judge_max_tokens', 'judge_top_p',
        'judge_top_k', 'judge_reasoning_effort', 'reasoning_effort', 'do_sample',
        'tensor_parallel_size', 'gpu_memory_utilization', 'max_model_len', 'dtype',
        'enable_thinking', 'trust_remote_code', 'max_num_seqs', 'presence_penalty',
        'frequency_penalty', 'batch_size', 'max_image_num',
    }
    containers = {'command_args', 'args', 'generation', 'judge', 'scoring',
                  'sampling_params', 'generation_config', 'runtime'}
    result = {k: config[k] for k in allowed if k in config}
    by_role, sources = {}, {}

    def visit(node, phase, path=''):
        for key, value in node.items():
            source = path + '.' + key if path else key
            if key in params and isinstance(value, (str, int, float, bool, type(None))):
                owner = 'judge' if key.startswith('judge_') else phase
                values = by_role.setdefault(owner, {})
                if key in values and values[key] != value:
                    raise ValueError('Conflicting recorded O1 parameter: ' + owner + '.' + key)
                values[key] = value
                sources.setdefault(owner, {}).setdefault(key, []).append(source)
            elif key in containers and isinstance(value, dict):
                visit(value, key if key in ('generation', 'judge', 'scoring') else phase, source)

    visit(config, role or 'unspecified')
    result['parameters_by_role'] = by_role
    result['parameter_source_paths'] = sources
    result['generation_or_judge_parameters'] = by_role.get(role or 'unspecified', {})
    result['metadata_status'] = 'sanitized_metadata_available' if config else 'sanitized_metadata_empty'
    return result


def missing_runtime_parameters(model_configs):
    recorded = {'generation': set(), 'judge': set()}
    for config in model_configs.values():
        for phase in recorded:
            recorded[phase].update(config['parameters_by_role'].get(phase, {}))
    requirements = {
        'generation': {'temperature': {'temperature'}, 'max_tokens': {'max_tokens', 'max_new_tokens'}},
        'judge': {'judge_temperature': {'judge_temperature', 'temperature'},
                  'judge_max_tokens': {'judge_max_tokens', 'max_tokens', 'max_new_tokens'},
                  'judge_top_p': {'judge_top_p', 'top_p'}},
    }
    return [phase + '.' + field for phase, fields in requirements.items()
            for field, aliases in fields.items() if not recorded[phase].intersection(aliases)]


def prompt_text(prompt):
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict) and set(prompt) == {'prompt'} and isinstance(prompt['prompt'], str):
        return prompt['prompt']
    if isinstance(prompt, list) and all(isinstance(m, dict) and set(m) == {'role', 'content'} and isinstance(m['content'], str) for m in prompt):
        return '\n'.join(m['content'] for m in prompt)
    raise ValueError('Unsupported frozen prompt/message shape')


def metrics_equal(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(metrics_equal(left[k], right[k]) for k in left)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(left - right) <= 1e-10
    return left == right


def export(stage, out, private_map):
    stage, out, private_map = Path(stage).resolve(), Path(out).resolve(), Path(private_map).resolve()
    if out.exists() or private_map == out or out in private_map.parents:
        raise ValueError('Output must be new; private map must be outside public output')
    registry = next(r for r in load(stage / 'data_registry.json') if r['dataset_id'] == 'O1_main103')
    data_path = stage / registry['stage_path']
    if sha(data_path) != registry['sha256']:
        raise ValueError('Frozen O1 dataset hash mismatch')
    for name, module in [('npc_o1_scorer.py', scorer), ('npc_o1_schema.py', schema)]:
        if scoring_functions(stage / 'code' / name) != scoring_functions(module.__file__):
            raise ValueError('Installed O1 scorer/schema functions differ from the frozen source')
    items = schema.load_o1_items(data_path)
    gold = unique(items, 'item_id')
    rubrics = schema.flatten_dataset_rubrics(items)
    rubric_map = {r.rubric_id: r for r in rubrics}
    if (len(items), len({r['case_id'] for r in items}), len(rubrics)) != (103, 42, 1692):
        raise ValueError('Expected 103 items / 42 cases / 1692 criteria')
    source_plan = {r['stage_path']: r for r in load(stage / 'source_plan.json') if r['group'] == 'o1_primary'}
    if len(source_plan) != 180:
        raise ValueError('Expected 180 locked O1 archive files')
    by_hash = {r['sha256']: relative for relative, r in source_plan.items() if relative.endswith('/judge/item_scores.jsonl')}
    with (stage / 'aligned_release_v5/cell_ledger.csv').open(encoding='utf-8-sig') as handle:
        cells = [r for r in csv.DictReader(handle) if r['column'] == 'C-Rub.']
    with (stage / 'aligned_release_v5/main_table_candidate.csv').open(encoding='utf-8-sig') as handle:
        table = {r['Model']: r for r in csv.DictReader(handle)}
    if len(cells) != 30 or len({r['model'] for r in cells}) != 30:
        raise ValueError('Expected 30 selected O1 cells')
    configs = load(stage / 'o1_sanitized_run_configs.json')
    previous = load(private_map) if private_map.exists() else {'salt': secrets.token_hex(32), 'ids': {}}
    salt, ids = bytes.fromhex(previous['salt']), previous['ids']
    aliases = {}
    for item in items:
        for field, kind in [('item_id', 'item'), ('case_id', 'case'), ('patient_id', 'patient')]:
            value = str(item[field])
            key = 'o1-' + kind + ':' + value
            identifier = 'o1-' + kind + '-' + hmac.new(salt, key.encode(), hashlib.sha256).hexdigest()[:24]
            if key in ids and ids[key] != identifier:
                raise ValueError('Private mapping is inconsistent')
            if value in aliases and aliases[value] != identifier:
                raise ValueError('Ambiguous source ID namespaces require an explicit mapping')
            ids[key] = identifier
            aliases[value] = identifier
    replacement = re.compile('|'.join(re.escape(k) for k in sorted(aliases, key=len, reverse=True)))
    private_path = re.compile(r'/(?:Users|zju_\d+|home|mnt|data_private|root|tmp|opt)/|[A-Za-z]:\\|\bsk-[A-Za-z0-9_-]{15,}')

    def clean(value):
        if isinstance(value, str):
            value = replacement.sub(lambda match: aliases[match.group(0)], value)
            if private_path.search(value):
                raise ValueError('Whitelisted output contains an internal path or key-shaped string')
            return value
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    def take(row, fields):
        return clean({k: row[k] for k in fields if k in row})

    public_items = []
    for item in items:
        public = take(item, ITEM_FIELDS)
        public['rubric_template_structured'] = take(item['rubric_template_structured'], RUBRIC_FIELDS)
        public_items.append(public)
    public_rubrics = schema.flatten_dataset_rubrics(public_items)
    if [r.to_dict() for r in public_rubrics] != clean([r.to_dict() for r in rubrics]):
        raise ValueError('Whitelisting/remapping changed flattened scoring rubrics')
    manifest = {'schema': 'npcbench-o1-frozen-v1', 'status': 'local_staging_not_published',
                'items': 103, 'cases': 42, 'rubrics': 1692, 'models': [], 'files': {},
                'source_input_sha256': registry['sha256'], 'source_ledger_sha256': sha(stage / 'aligned_release_v5/cell_ledger.csv'),
                'scorer_sha256': sha(scorer.__file__), 'schema_sha256': sha(schema.__file__),
                'source_scorer_sha256': sha(stage / 'code/npc_o1_scorer.py'), 'source_schema_sha256': sha(stage / 'code/npc_o1_schema.py'),
                'scorer_functions_AST_identical': True,
                'primary_metric': 'mean_valid_item_normalized_rubric_score',
                'failure_policy': 'Preserve failed/missing judgments; items with any missing or invalid criterion are excluded from the valid-item mean and reported as scoring failures. Never refill or reinterpret them as zero scores.',
                'patient_identity_scope': 'O1 task-local; no cross-task/global cohort membership assertion',
                'prompt_policy': 'Export exact archived generation prompts and judge messages with known source IDs remapped; do not reconstruct prompts from a newer helper.',
                'omitted_input_fields': sorted(set().union(*(set(r) - set(ITEM_FIELDS) - {'rubric_template_structured'} for r in items))),
                'omitted_rubric_metadata_fields': sorted(set().union(*(set(r['rubric_template_structured']) - set(RUBRIC_FIELDS) for r in items)))}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=out.name + '.building-', dir=out.parent) as temp:
        build = Path(temp) / 'bundle'
        build.mkdir()

        def write(relative, records):
            path = build / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('w') as handle:
                for row in records:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            manifest['files'][relative] = sha(path)

        write('data/items.jsonl', public_items)
        write('data/rubrics.jsonl', [r.to_dict() for r in public_rubrics])
        for index, cell in enumerate(cells, 1):
            if cell['status'] != 'aligned' or cell['source_sha256'] not in by_hash:
                raise ValueError('v5 selected O1 item-score source is not locked')
            score_relative = by_hash[cell['source_sha256']]
            archive_name = Path(score_relative).parts[1]
            base_rel = 'o1_primary/' + archive_name
            locked = {relative: entry for relative, entry in source_plan.items() if relative.startswith(base_rel + '/')}
            if len(locked) != 6:
                raise ValueError('Incomplete O1 model archive')
            for relative, entry in locked.items():
                if sha(stage / relative) != entry['sha256']:
                    raise ValueError('O1 source archive hash mismatch')
            base = stage / base_rel
            generation = rows(base / 'generation/predictions.jsonl')
            if generation != load(base / 'generation/results.json') or set(unique(generation, 'item_id')) != set(gold):
                raise ValueError('Generation archives disagree or have incomplete coverage')
            judgments = rows(base / 'judge/judgments.jsonl')
            tasks = rows(base / 'judge/judge_tasks.jsonl')
            if set(unique(judgments, 'rubric_id')) != set(rubric_map) or set(unique(tasks, 'rubric_id')) != set(rubric_map):
                raise ValueError('Frozen criterion judgment/task coverage is incomplete')
            generation_map = unique(generation, 'item_id')
            for row in generation:
                item = gold[row['item_id']]
                prompt = prompt_text(row['prompt'])
                if row['case_id'] != item['case_id'] or row['patient_id'] != item['patient_id'] or item['case_input'] not in prompt or item['question'] not in prompt:
                    raise ValueError('Generation identity/context differs from canonical item')
            for row in tasks:
                rubric = rubric_map[row['rubric_id']]
                item = gold[rubric.item_id]
                if row['item_id'] != rubric.item_id or row['case_id'] != rubric.case_id:
                    raise ValueError('Judge task identity differs from canonical criterion')
                if not isinstance(row['messages'], list) or any(set(m) != {'role', 'content'} or not isinstance(m['content'], str) for m in row['messages']):
                    raise ValueError('Unsupported frozen judge message shape')
                prompt = '\n'.join(m['content'] for m in row['messages'])
                if any(text not in prompt for text in (item['case_input'], item['question'], generation_map[rubric.item_id]['response'], rubric.criterion)):
                    raise ValueError('Judge messages do not contain frozen case/question/generation/criterion')
            before = score(items, judgments)
            saved_scores = [{k: v for k, v in r.items() if k != 'model_name'} for r in rows(base / 'judge/item_scores.jsonl')]
            if before['item_scores'] != saved_scores:
                raise ValueError('Raw judge replay differs from frozen per-item scores')
            expected = load(base / 'judge/metrics.json')
            if not metrics_equal(before['metrics'], {key: expected[key] for key in before['metrics']}):
                raise ValueError('CPU metrics differ from frozen scoring archive')
            value = before['metrics']['overall_score']
            shown = str(Decimal(str(value)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))
            if abs(value - float(cell['aligned_value'])) > 1e-10 or shown != cell['aligned_display'] or shown != table[cell['model']]['C-Rub.']:
                raise ValueError('O1 raw-response replay differs from v5 primary values')
            pub_generation = [take(r, GEN_FIELDS) for r in generation]
            pub_judgments = [take(r, JUDGE_FIELDS) for r in judgments]
            pub_tasks = [{**take(r, TASK_FIELDS), 'messages': clean(r['messages'])} for r in tasks]
            after = score(public_items, pub_judgments)
            if after['metrics'] != before['metrics'] or after['raw_judge_audit'] != before['raw_judge_audit'] or after['item_scores'] != clean(before['item_scores']):
                raise ValueError('Public ID substitution changed O1 parsing or scoring')
            prefix = f'models/model-{index:02d}'
            generation_path, judgment_path, task_path = (prefix + '/' + name for name in ('generation.jsonl', 'judgments.jsonl', 'judge_tasks.jsonl'))
            write(generation_path, pub_generation)
            write(judgment_path, pub_judgments)
            write(task_path, pub_tasks)
            write(prefix + '/item_scores.jsonl', after['item_scores'])
            fields_missing = {phase: {field: sum(field not in r for r in records) for field in fields}
                              for phase, records, fields in [('generation', generation, ('finish_reason', 'usage', 'api_attempts')),
                                                             ('judge', judgments, ('finish_reason', 'usage', 'attempt_history', 'raw_judge_response', 'criteria_met', 'explanation'))]}
            model_configs = {name: public_config(cfg, name.split('/')[0])
                             for name, cfg in configs[archive_name].items()}
            runtime_missing = missing_runtime_parameters(model_configs)
            manifest['models'].append(clean({'model': cell['model'], 'archive_model_name': archive_name,
                'predictions_path': generation_path, 'judgments_path': judgment_path, 'judge_tasks_path': task_path,
                'judgment_rows': len(judgments), 'metrics': after['metrics'], 'raw_judge_audit': after['raw_judge_audit'],
                'generation_status_counts': dict(Counter(r.get('generation_status', 'unrecorded') for r in generation)),
                'judge_status_counts': dict(Counter(r.get('judge_status', 'unrecorded') for r in judgments)),
                'source_file_hashes': {str(Path(relative).relative_to(base_rel)): entry['sha256'] for relative, entry in locked.items()},
                'sanitized_runtime_metadata': model_configs, 'missing_runtime_parameters_in_sanitized_stage': runtime_missing,
                'unarchived_field_counts': fields_missing}))
        (build / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        private_map.parent.mkdir(parents=True, exist_ok=True)
        private_map.write_text(json.dumps({'salt': previous['salt'], 'ids': ids}, ensure_ascii=False, indent=2) + '\n')
        private_map.chmod(0o600)
        os.replace(build, out)
    return {'models': 30, 'items': 103, 'cases': 42, 'generation_rows': 3090, 'judgment_rows': 50760, 'output_dir': str(out)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage-root', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--private-map', type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(export(args.stage_root, args.output_dir, args.private_map), indent=2))


if __name__ == '__main__':
    main()
