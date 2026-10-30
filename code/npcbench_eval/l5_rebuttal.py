"""Reproduce paired L5 interventions without model calls or clinical relabeling."""
import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import random

from .final_answer import POLICY
from .l5 import read_rows, score, sha


def interval(values):
    values = sorted(values)
    def quantile(p):
        position = (len(values) - 1) * p
        lower, upper = math.floor(position), math.ceil(position)
        return values[lower] + (position - lower) * (values[upper] - values[lower])
    return [quantile(.025), quantile(.975)]


def paired_summary(left, right, repeats=10000, seed=20261026):
    """Patient-cluster bootstrap; compare exactly the same items and patients."""
    a = {r['id']: r for r in left}
    b = {r['id']: r for r in right}
    if len(a) != len(left) or len(b) != len(right) or not a or set(a) != set(b):
        raise ValueError('Paired conditions require unique, identical item membership')
    groups = defaultdict(list)
    for key in sorted(a):
        if a[key]['patient_id'] != b[key]['patient_id']:
            raise ValueError('Paired conditions disagree on patient identity')
        groups[a[key]['patient_id']].append((int(a[key]['correct']), int(b[key]['correct'])))
    left_only = sum(x['correct'] and not b[k]['correct'] for k, x in a.items())
    right_only = sum(not x['correct'] and b[k]['correct'] for k, x in a.items())
    episode_left_only = sum(all(x for x, _ in turns) and not all(y for _, y in turns) for turns in groups.values())
    episode_right_only = sum(all(y for _, y in turns) and not all(x for x, _ in turns) for turns in groups.values())
    discordant = episode_left_only + episode_right_only
    p_value = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(episode_left_only, episode_right_only) + 1)) / 2**discordant)
    # Each patient contributes its turn sum, turn count, and episode outcome.
    patient_stats = []
    for patient in sorted(groups):
        turns = groups[patient]
        delta = sum(x - y for x, y in turns)
        completion = int(all(x for x, _ in turns)) - int(all(y for _, y in turns))
        patient_stats.append((delta, len(turns), delta / len(turns), completion))
    rng = random.Random(seed)
    distributions = [[], [], []]
    for _ in range(repeats):
        sample = rng.choices(patient_stats, k=len(patient_stats))
        distributions[0].append(sum(x[0] for x in sample) / sum(x[1] for x in sample))
        distributions[1].append(sum(x[2] for x in sample) / len(sample))
        distributions[2].append(sum(x[3] for x in sample) / len(sample))
    return {'turn_count': len(a), 'patient_count': len(groups),
            'turn_accuracy_delta': (left_only - right_only) / len(a),
            'patient_macro_delta': sum(x[2] for x in patient_stats) / len(patient_stats),
            'episode_completion_delta': sum(x[3] for x in patient_stats) / len(patient_stats),
            'turn_accuracy_delta_ci95': interval(distributions[0]),
            'patient_macro_delta_ci95': interval(distributions[1]),
            'episode_completion_delta_ci95': interval(distributions[2]),
            'left_only_correct': left_only, 'right_only_correct': right_only,
            'episode_left_only_correct': episode_left_only, 'episode_right_only_correct': episode_right_only,
            'episode_mcnemar_exact_unadjusted_p': p_value, 'mcnemar_unit': 'patient_episode',
            'bootstrap_repeats': repeats, 'bootstrap_seed': seed}


def evaluate_panel(dataset, responses, cohorts=None, repeats=10000, seed=20261026):
    inputs = defaultdict(list)
    for row in dataset:
        inputs[row['condition']].append(row)
    conditions = sorted(inputs)
    if len(conditions) < 2:
        raise ValueError('An intervention panel requires at least two conditions')
    reference = {r['id']: (r['patient_id'], r['answer'], r['options']) for r in inputs[conditions[0]]}
    for condition, items in inputs.items():
        identities = {r['id']: (r['patient_id'], r['answer'], r['options']) for r in items}
        if len(identities) != len(items) or identities != reference:
            raise ValueError('Interventions must preserve items, patient identity, options, and gold')
    by_model = defaultdict(lambda: defaultdict(list))
    for row in responses:
        by_model[row['model']][row['condition']].append(row)
    metrics, decisions, contrasts = [], [], []
    if not by_model:
        raise ValueError('No model responses supplied')
    for model, results in sorted(by_model.items()):
        if set(results) != set(conditions):
            raise ValueError('Model is missing a complete intervention condition')
        parsed = {}
        for condition in conditions:
            summary, rows = score(inputs[condition], results[condition], POLICY)
            patients = {r['id']: r['patient_id'] for r in inputs[condition]}
            old = {r['id']: r for r in results[condition]}
            parsed[condition] = [{**r, 'patient_id': patients[r['id']]} for r in rows]
            for row in parsed[condition]:
                legacy = old[row['id']]
                decisions.append({**row, 'model': model, 'condition': condition,
                                  'legacy_prediction': legacy['legacy_prediction'],
                                  'legacy_correct': legacy['legacy_correct'],
                                  'correctness_changed': row['correct'] != legacy['legacy_correct']})
        for cohort, flag in (cohorts or {'all': None}).items():
            selected = {r['id'] for r in inputs[conditions[0]] if flag is None or r.get(flag)}
            if not selected:
                raise ValueError('Declared analysis cohort is empty')
            cohort_parsed = {}
            for condition in conditions:
                cohort_data = [r for r in inputs[condition] if r['id'] in selected]
                if {r['id'] for r in inputs[condition] if flag is None or r.get(flag)} != selected:
                    raise ValueError('Analysis cohort differs between conditions')
                summary, _ = score(cohort_data, [r for r in results[condition] if r['id'] in selected], POLICY)
                metrics.append({'model': model, 'cohort': cohort, 'condition': condition, **summary})
                cohort_parsed[condition] = [r for r in parsed[condition] if r['id'] in selected]
            for left, right in itertools.combinations(conditions, 2):
                contrast = paired_summary(cohort_parsed[left], cohort_parsed[right], repeats, seed)
                contrasts.append({'model': model, 'cohort': cohort, 'left': left, 'right': right, **contrast})
    return metrics, contrasts, decisions


def reproduce(bundle, output, repeats=10000, seed=20261026):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest['schema'] != 'npcbench-l5-rebuttal-v1' or manifest['answer_policy'] != POLICY:
        raise ValueError('Unsupported intervention bundle or answer policy')
    if manifest['parser_sha256'] != sha(Path(__file__).with_name('final_answer.py')):
        raise ValueError('Fixed answer parser source differs from the release manifest')
    for relative, digest in manifest['files'].items():
        p = (bundle / relative).resolve()
        if bundle not in p.parents or sha(p) != digest:
            raise ValueError('Intervention payload checksum or path mismatch')
    all_metrics, all_contrasts, all_decisions = [], [], []
    for panel in manifest['panels']:
        for field in ('dataset', 'responses'):
            if panel[field] not in manifest['files']:
                raise ValueError('Unhashed intervention input')
        data, predictions = read_rows(bundle / panel['dataset']), read_rows(bundle / panel['responses'])
        if (len(data), len(predictions)) != (panel['input_rows'], panel['response_rows']):
            raise ValueError('Intervention inventory differs from manifest')
        metrics, contrasts, decisions = evaluate_panel(data, predictions, panel['cohorts'], repeats, seed)
        if len({r['model'] for r in predictions}) != panel['model_count']:
            raise ValueError('Model inventory differs from manifest')
        for dest, source in [(all_metrics, metrics), (all_contrasts, contrasts), (all_decisions, decisions)]:
            dest.extend({'panel': panel['name'], **row} for row in source)
    if output.exists():
        raise ValueError('Use a new output directory to preserve previous analyses')
    output.mkdir(parents=True)
    for filename, rows in [('condition_metrics.jsonl', all_metrics), ('paired_contrasts.jsonl', all_contrasts), ('decisions.jsonl', all_decisions)]:
        (output / filename).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    summary = {'answer_policy': POLICY, 'parser_sha256': manifest['parser_sha256'],
               'bundle_manifest_sha256': sha(bundle / 'manifest.json'),
               'scorer_sha256': sha(Path(__file__)), 'responses': len(all_decisions),
               'condition_metrics': len(all_metrics), 'paired_contrasts': len(all_contrasts),
               'correctness_changes': sum(r['correctness_changed'] for r in all_decisions),
               'bootstrap_unit': 'patient', 'bootstrap_repeats': repeats, 'bootstrap_seed': seed,
               'p_values': 'Exact McNemar, unadjusted; exploratory contrasts, not multiplicity-controlled claims.',
               'files': {p.name: sha(p) for p in sorted(output.iterdir())}}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--bootstrap', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=20261026)
    args = ap.parse_args()
    if args.bootstrap < 1:
        ap.error('--bootstrap must be positive')
    print(json.dumps(reproduce(args.bundle, args.output_dir, args.bootstrap, args.seed), indent=2))


if __name__ == '__main__':
    main()
