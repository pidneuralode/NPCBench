"""Command-line entry points for NPCBench evaluation."""

import argparse
import gzip
import json
from pathlib import Path

from . import l3, l5, l6, multiple_choice, open_response, rubrics, staging
from .final_answer import POLICY


CHOICE_TASKS = ('L1', 'L2', 'L3-Path', 'L3-Seq', 'L3-Struct', 'T400', 'N400', 'L4', 'L5', 'L6')


def rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def choice(args):
    task = args.task
    predictions = rows(args.predictions)
    if task in multiple_choice.TASKS:
        metrics, details = multiple_choice.score(rows(args.data / task / 'questions.jsonl'),
                                                 predictions, task, POLICY)
    elif task == 'L5':
        metrics, details = l5.score(rows(args.data / 'L5/questions.jsonl'), predictions)
    elif task == 'L6':
        dataset = l6.load_dataset(args.data / 'L6/full100_inputs_scored217')
        return l6.score(dataset, predictions)
    elif task in ('T400', 'N400'):
        return {'metrics': staging.score(rows(args.data / 'L3' / f'{task}.jsonl'),
                                         predictions, task[0])}
    else:
        subset = task.removeprefix('L3-')
        metrics, details = l3.score(subset, rows(args.data / 'L3' / f'{subset}.jsonl'), predictions)
    return {'metrics': metrics, 'items': details}


def open_task(args):
    if args.task == 'L1-open':
        if args.responses is None:
            raise ValueError('L1-open requires --responses')
        return open_response.score_short_answers(rows(args.data / 'supplementary/option_free/answers.jsonl'),
                                                 rows(args.responses))
    items = rubrics.load_o1_items(args.data / 'O1/questions.jsonl')
    if args.judgments:
        return open_response.score_consultations(items, rows(args.judgments))
    if not args.output:
        raise ValueError('Set --output to a JSONL path for the judge requests')
    requests = list(open_response.judge_requests(items, rows(args.responses)))
    rubrics.write_jsonl(args.output, requests)
    return {'metrics': {'judge_requests': len(requests)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    mcq = sub.add_parser('choice', help='score choices and structured MRI answers')
    mcq.add_argument('--task', choices=CHOICE_TASKS, required=True)
    mcq.add_argument('--predictions', type=Path, required=True)
    written = sub.add_parser('open', help='prepare rubric judgments or score open answers')
    written.add_argument('--task', choices=('O1', 'L1-open'), default='O1')
    inputs = written.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--responses', type=Path)
    inputs.add_argument('--judgments', type=Path)
    for command in (mcq, written):
        command.add_argument('--data', type=Path, default=Path('data'))
        command.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        result = choice(args) if args.mode == 'choice' else open_task(args)
        if args.output and 'judge_requests' not in result['metrics']:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result['metrics'], ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    return 0
