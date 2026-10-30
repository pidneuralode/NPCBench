"""Reparse frozen O1 judge responses and score with the installed O1 scorer."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from o1_eval.simple_evals_main import npc_o1_scorer as scorer
from o1_eval.simple_evals_main import npc_o1_schema as schema


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique(records, field):
    result = {}
    for row in records:
        value = row[field]
        key = str(value)
        if value is None or not key.strip() or key in result:
            raise ValueError(f'Duplicate or empty {field}')
        result[key] = row
    return result


def reparse_judgments(judgments):
    unique(judgments, 'rubric_id')
    reparsed, counts = [], Counter()
    for row in judgments:
        raw = row.get('raw_judge_response')
        counts['raw_response_missing'] += 'raw_judge_response' not in row or raw is None
        counts['raw_response_empty'] += isinstance(raw, str) and not raw.strip()
        try:
            parsed = scorer.parse_judge_json(raw or '')
            valid = isinstance(parsed, dict) and scorer.validate_judge_output(parsed)[0]
        except (ValueError, TypeError, AttributeError):
            parsed, valid = {}, False
        criteria = parsed['criteria_met'] if valid else None
        explanation = parsed['explanation'] if valid else ''
        counts['raw_parse_invalid'] += not valid
        counts['stored_criteria_mismatches'] += criteria != row.get('criteria_met')
        counts['stored_explanation_mismatches_valid'] += valid and explanation != row.get('explanation')
        archived_status = row.get('judge_status', 'ok')
        counts['archived_failure_statuses'] += archived_status != 'ok'
        status = archived_status if archived_status != 'ok' else ('ok' if valid else 'raw_parse_failed')
        reparsed.append({**row, 'criteria_met': criteria, 'explanation': explanation, 'judge_status': status})
    return reparsed, dict(counts)


def score(items, judgments):
    gold = unique(items, 'item_id')
    rubrics = schema.flatten_dataset_rubrics(items)
    rubric_map = unique([r.to_dict() for r in rubrics], 'rubric_id')
    archived = unique(judgments, 'rubric_id')
    if not set(archived) <= set(rubric_map):
        raise ValueError('Unknown rubric judgment')
    for rubric_id, row in archived.items():
        rubric = rubric_map[rubric_id]
        if row.get('item_id') != rubric['item_id'] or ('case_id' in row and row['case_id'] != rubric['case_id']):
            raise ValueError('Judgment identity does not match its rubric')
    reparsed, audit = reparse_judgments(judgments)
    audit['missing_judgments'] = len(set(rubric_map) - set(archived))
    item_scores, metrics = scorer.score_dataset(items, rubrics, reparsed)
    return {'metrics': metrics, 'item_scores': item_scores, 'raw_judge_audit': audit}


def reproduce(bundle):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest.get('schema') != 'npcbench-o1-frozen-v1' or len(manifest['models']) != 30:
        raise ValueError('Expected complete O1 frozen schema and 30 models')
    if 'data/items.jsonl' not in manifest['files']:
        raise ValueError('O1 items file is not covered by the manifest')
    for relative, digest in manifest['files'].items():
        path = (bundle / relative).resolve()
        if bundle not in path.parents or sha(path) != digest:
            raise ValueError('Bundle file hash or path mismatch')
    if manifest['scorer_sha256'] != sha(scorer.__file__) or manifest['schema_sha256'] != sha(schema.__file__):
        raise ValueError('Installed O1 scoring/schema version differs from the frozen bundle')
    items = rows(bundle / 'data/items.jsonl')
    rubrics = schema.flatten_dataset_rubrics(items)
    if (len(items), len({r['case_id'] for r in items}), len(rubrics)) != (103, 42, 1692):
        raise ValueError('O1 input inventory mismatch')
    rubric_map = {r.rubric_id: r for r in rubrics}
    models = {}
    judgment_rows = 0
    for entry in manifest['models']:
        if entry['model'] in models or any(entry[key] not in manifest['files'] for key in ('predictions_path', 'judgments_path', 'judge_tasks_path')):
            raise ValueError('Duplicate model or unregistered artifact')
        predictions = unique(rows(bundle / entry['predictions_path']), 'item_id')
        if set(predictions) != set(unique(items, 'item_id')):
            raise ValueError('Generation coverage mismatch')
        judge_tasks = rows(bundle / entry['judge_tasks_path'])
        if any('rubric_id' not in row or 'item_id' not in row for row in judge_tasks):
            raise ValueError('Judge task lacks rubric or item identity')
        tasks = unique(judge_tasks, 'rubric_id')
        if set(tasks) != set(rubric_map) or {r['item_id'] for r in tasks.values()} != set(unique(items, 'item_id')):
            raise ValueError('Judge task rubric/item coverage mismatch')
        if any(row['item_id'] != rubric_map[rubric_id].item_id for rubric_id, row in tasks.items()):
            raise ValueError('Judge task item does not match its rubric')
        judgments = rows(bundle / entry['judgments_path'])
        if len(judgments) != entry['judgment_rows']:
            raise ValueError('Actual judgment row count differs from the registered count')
        judgment_rows += len(judgments)
        result = score(items, judgments)
        if result['metrics'] != entry['metrics'] or result['raw_judge_audit'] != entry['raw_judge_audit']:
            raise ValueError('Raw judgment CPU replay differs from the exported evidence')
        models[entry['model']] = {'metrics': result['metrics'], 'raw_judge_audit': result['raw_judge_audit']}
    return {'verified': True, 'items': 103, 'cases': 42, 'rubrics': 1692, 'models': models,
            'generation_rows': 3090, 'judgment_rows': judgment_rows, 'model_calls': 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', type=Path)
    ap.add_argument('--data', type=Path)
    ap.add_argument('--judgments', type=Path)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if args.bundle and not args.data and not args.judgments:
        result = reproduce(args.bundle)
    elif args.data and args.judgments and not args.bundle:
        result = score(schema.load_o1_items(args.data), rows(args.judgments))
    else:
        ap.error('Use --bundle OR --data and --judgments')
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
