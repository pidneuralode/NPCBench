"""Recompute the paper's main table from released predictions on a CPU."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path

from npcbench_eval import l3, l5, l6, mcq_release, mri_extension
from npcbench_eval.final_answer import POLICY
from o1_eval.simple_evals_main import npc_o1_schema, npc_o1_scorer


def rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def canonical_model(name):
    return name.removesuffix('-Instruct-2507').removesuffix('-Instruct')


def reproduce(data, result_dir):
    predictions = defaultdict(list)
    for row in rows(result_dir / 'final_predictions.jsonl.gz'):
        predictions[canonical_model(row['model']), row['task']].append(row)
    judgments = defaultdict(list)
    for row in rows(result_dir / 'criterion_judgments.jsonl.gz'):
        judgments[canonical_model(row['model'])].append(row)
    with (result_dir / 'main_results.csv').open() as stream:
        reference = list(csv.DictReader(stream))
    models = {canonical_model(row['Model']) for row in reference}
    if {name for name, _ in predictions} != models or set(judgments) != models:
        raise ValueError('Prediction and judgment models differ from the results table')

    questions = {task: rows(data / task / 'questions.jsonl') for task in ('L1', 'L2', 'L4', 'L5')}
    component = {task: rows(data / 'L3' / (task + '.jsonl')) for task in ('Path', 'Seq', 'Struct')}
    stages = {task: rows(data / 'L3' / (task + '.jsonl')) for task in ('T400', 'N400')}
    carepath = l6.load_dataset(data / 'L6/full100_inputs_scored217')
    consultations = npc_o1_schema.load_o1_items(data / 'O1/questions.jsonl')
    rubrics = npc_o1_schema.flatten_dataset_rubrics(consultations)
    rubric_items = {row.rubric_id: row.item_id for row in rubrics}
    output = []
    cells = 0
    for expected in reference:
        name = canonical_model(expected['Model'])
        result = {'Family': expected['Family'], 'Model': expected['Model']}
        for task, column in [('L1', 'G-Mem.'), ('L2', 'A-App.'), ('L4', 'C-Plan')]:
            values = predictions[name, task]
            parsers = {row['parser'] for row in values}
            if len(parsers) != 1:
                raise ValueError(f'Expected one answer parser for {name}/{task}')
            metrics, _ = mcq_release.score(questions[task], values, task, parsers.pop())
            result[column] = metrics['primary_percent']
        metrics, _ = l5.score(questions['L5'], predictions[name, 'L5'], POLICY)
        result['L-Rev. Step'] = metrics['turn_micro_accuracy'] * 100
        result['L-Rev. Epi.'] = metrics['all_turns_correct_patient_rate'] * 100
        for task, column in [('Path', 'Path.'), ('Seq', 'Seq.'), ('Struct', 'Struct.')]:
            if (name, 'L3-' + task) in predictions:
                metrics, _ = l3.score(task, component[task], predictions[name, 'L3-' + task], POLICY)
                result[column] = metrics['micro_percent']
        for task, column in [('T400', 'T-Stg.'), ('N400', 'N-Stg.')]:
            if (name, task) in predictions:
                result[column] = mri_extension.score(stages[task], predictions[name, task], task[0])['accuracy'] * 100
        if (name, 'L6') in predictions:
            values = [{'sample_id': row['id'], 'response': row['response']} for row in predictions[name, 'L6']]
            metrics = l6.score(carepath, values, POLICY)['metrics']
            result['CP-Epi. Step'] = metrics['care_micro_percent']
            result['CP-Epi. Epi.'] = metrics['strict_episode_percent']
        native = judgments[name]
        if len(native) != len(rubrics) or {x['rubric_id']: x['item_id'] for x in native} != rubric_items:
            raise ValueError(f'Incomplete or mismatched rubric judgments for {name}')
        if any(x['judge_status'] != 'ok' or not isinstance(x['criteria_met'], bool) for x in native):
            raise ValueError(f'Invalid criterion judgment for {name}')
        # Explanations are omitted from the compact artifact; only Boolean
        # judgments contribute to the weighted rubric score.
        scored = [dict(row, explanation='') for row in native]
        _, metrics = npc_o1_scorer.score_dataset(consultations, rubrics, scored)
        result['C-Rub.'] = metrics['overall_score']
        for column in list(expected)[2:]:
            if not expected[column]:
                if column in result:
                    raise ValueError(f'Unexpected result for {name}/{column}')
                result[column] = ''
                continue
            # L3 component scores were published at one decimal place.
            tolerance = 0.050001 if column in ('Path.', 'Seq.', 'Struct.') else 1e-7
            if result.get(column) is None or abs(result[column] - float(expected[column])) > tolerance:
                raise ValueError(f'Score mismatch: {name}/{column}: {result.get(column)} vs {expected[column]}')
            cells += 1
        output.append(result)
    return list(reference[0]), output, cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Downloaded Hugging Face dataset directory')
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parents[1] / 'results')
    parser.add_argument('--output', type=Path, default=Path('reproduced_results.csv'))
    args = parser.parse_args()
    columns, output, cells = reproduce(args.data, args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(output)
    print(f'Reproduced {cells} scores for {len(output)} models; no model calls.')


if __name__ == '__main__':
    main()
