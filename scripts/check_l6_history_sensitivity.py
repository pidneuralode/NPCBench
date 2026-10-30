#!/usr/bin/env python3
"""Replay every public history trajectory, paired score and response-reuse rule."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from npcbench_eval import final_answer, l6_history
from npcbench_eval.l5_rebuttal import paired_summary


def load(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_metrics(actual, expected):
    """Allow roundoff across Python versions; keep membership/counts exact."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(same_metrics(actual[k], expected[k]) for k in expected)
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(same_metrics(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)
    return actual == expected


def verify(bundle, output):
    manifest = load(bundle / 'manifest.json')
    if manifest['schema'] != 'npcbench-l6-history-sensitivity-v1' or manifest['scoring_policy'] != final_answer.POLICY:
        raise ValueError('Unsupported bundle')
    if (sha(Path(final_answer.__file__)), sha(Path(l6_history.__file__))) != (manifest['parser_sha256'], manifest['history_policy_sha256']):
        raise ValueError('Installed answer/history policies differ')
    for relative, digest in manifest['files'].items():
        if sha(bundle / relative) != digest:
            raise ValueError('Public payload checksum differs')
    episodes = rows(bundle / manifest['input'])
    if len(episodes) != 25 or sum(len(e['turns']) for e in episodes) != 146:
        raise ValueError('Public cohort differs')
    conditions = ['frozen_history_v1', 'final_answer_history_v1']
    all_decisions, results, baselines, runtimes = [], [], {}, {}
    for panel in manifest['panels']:
        raw = rows(bundle / panel['responses'])
        indexed = {r['request_id']: r for r in raw}
        if len(indexed) != 292 or len(raw) != 292:
            raise ValueError('Incomplete or duplicate response membership')
        runtime = load(bundle / panel['runtime'])
        if panel['experiment'] == 'fixed_baseline_v2':
            prior = runtimes[panel['model']]
            for field in ('checkpoint_current_weight_sha256', 'python', 'packages', 'parameters', 'batch_size', 'max_model_len'):
                if prior[field] != runtime[field]:
                    raise ValueError('Fixed-baseline runtime differs from its baseline')
        else:
            runtimes[panel['model']] = runtime
        metrics, parsed = [], {}
        for condition in conditions:
            parsed[condition], patient_scores = [], []
            for episode in episodes:
                history, correct = [], []
                for turn in episode['turns']:
                    response = indexed[condition + ':' + turn['sample_id']]
                    expected_prompt = l6_history.render_history(turn['base_prompt'], history)
                    if response['prompt'] != expected_prompt or response['prompt_sha256'] != hashlib.sha256(expected_prompt.encode()).hexdigest():
                        raise ValueError('Public history prompt does not reconstruct')
                    if response['patient_id'] != episode['patient_id'] or response['images'] != [] or response['status'] != 'ok':
                        raise ValueError('Input patient, image or generation status differs')
                    answer = final_answer.parse_choice(response['response'], turn['options'])
                    scored = answer.prediction == turn['gold_letter']
                    row = {'id': turn['sample_id'], 'patient_id': episode['bootstrap_cluster_id'],
                           'prediction': answer.prediction, 'answer_status': answer.status, 'correct': scored}
                    parsed[condition].append(row)
                    all_decisions.append({**row, 'patient_id': episode['patient_id'], 'panel': panel['id'], 'condition': condition})
                    history.append(l6_history.action_from_response(response['response'], turn['options'], condition))
                    correct.append(scored)
                patient_scores.append(correct)
            source = [r for r in raw if r['condition'] == condition]
            metrics.append({'condition': condition, 'care_nodes': 146, 'patients': 25,
                            'correct': sum(r['correct'] for r in parsed[condition]),
                            'care_accuracy': sum(r['correct'] for r in parsed[condition]) / 146,
                            'patient_macro_accuracy': sum(sum(v) / len(v) for v in patient_scores) / 25,
                            'all_care_correct_patients': sum(all(v) for v in patient_scores),
                            'answer_statuses': dict(Counter(r['answer_status'] for r in parsed[condition])),
                            'finish_reasons': dict(Counter(str(r['finish_reason']) for r in source)),
                            'mean_output_tokens': sum(r['output_tokens'] for r in source) / 146})
        contrast = paired_summary(parsed[conditions[1]], parsed[conditions[0]])
        expected = load(bundle / panel['expected'])
        if not same_metrics(metrics, expected['conditions']) or not same_metrics(contrast, expected['paired_final_minus_frozen']):
            raise ValueError('Public scores or paired bootstrap differ from verified internal evidence')
        if panel['experiment'] == 'independent_v1':
            baselines[panel['model']] = {r['sample_id']: r for r in raw if r['condition'] == conditions[0]}
        else:
            baseline = baselines[panel['model']]
            for response in raw:
                prior = baseline[response['sample_id']]
                identical = all(response[k] == prior[k] for k in ('sample_id', 'patient_id', 'prompt', 'images'))
                origin = response['response_origin']
                expected_origin = 'paired_frozen_baseline' if response['condition'] == conditions[0] else 'paired_identical_input_reuse' if identical else 'model_generation'
                if origin != expected_origin:
                    raise ValueError('Response origin violates fixed-baseline protocol')
                if expected_origin != 'model_generation':
                    fields = ('response', 'status', 'finish_reason', 'stop_reason', 'output_tokens', 'prompt_tokens')
                    if not identical or any(response[k] != prior[k] for k in fields):
                        raise ValueError('Reused response is not exactly its baseline observation')
        results.append({'panel': panel['id'], 'model': panel['model'], 'experiment': panel['experiment'],
                        'conditions': metrics, 'paired_final_minus_frozen': contrast,
                        'response_origins': dict(Counter(r['response_origin'] for r in raw))})
    if output.exists():
        raise ValueError('Use a new output directory')
    output.mkdir(parents=True)
    (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    (output / 'decisions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in all_decisions))
    summary = {'status': 'verified', 'bundle_manifest_sha256': sha(bundle / 'manifest.json'),
               'panels': len(results), 'conditions': 8, 'response_and_prompt_checks': len(all_decisions),
               'actual_new_model_calls': sum(r['response_origins'].get('model_generation', 0) for r in results),
               'same_current_weights_and_runtime_verified': True,
               'point_metrics_and_patient_bootstrap_match_source': True,
               'floating_point_absolute_tolerance': 1e-12,
               'scored_pending_quality_exclusions': 0}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.bundle.resolve(), args.output_dir.resolve()), indent=2))
