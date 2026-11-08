"""Prepare and score open-response evaluations."""

from .final_answer import json_payload
from .multiple_choice import unique
from .prompts import render_judge_messages
from .rubrics import flatten_dataset_rubrics
from .rubric_score import score_dataset
from .short_answers import align_answer_components, extract_answer, score_slot


def judge_requests(items, responses):
    questions = unique(items, 'item_id')
    answers = unique(responses, 'item_id')
    if not questions or questions.keys() != answers.keys():
        raise ValueError('Provide one response for each consultation question')
    for rubric in flatten_dataset_rubrics(items):
        item = questions[rubric.item_id]
        response = answers[rubric.item_id].get('response')
        if response is not None and not isinstance(response, str):
            raise ValueError('Response must be text or null')
        yield {
            'item_id': rubric.item_id,
            'rubric_id': rubric.rubric_id,
            'messages': render_judge_messages(item['case_input'], item['question'], response or '', rubric),
        }


def score_consultations(items, judgments):
    unique(items, 'item_id')
    rubrics = flatten_dataset_rubrics(items)
    expected = {rubric.rubric_id: rubric.item_id for rubric in rubrics}
    by_id = unique(judgments, 'rubric_id')
    if any(expected.get(rid) != row.get('item_id') or rid not in expected for rid, row in by_id.items()):
        raise ValueError('Judgment IDs must match the question and criterion')
    parsed = []
    for row in judgments:
        if 'criteria_met' in row:
            parsed.append({'explanation': '', **row})
            continue
        obj, _, _ = json_payload(row.get('response'))
        parsed.append({**row, 'criteria_met': obj.get('criteria_met') if obj else None,
                       'explanation': obj.get('explanation', '') if obj else ''})
    details, metrics = score_dataset(items, rubrics, parsed)
    return {'metrics': metrics, 'items': details}


def score_short_answers(items, responses):
    questions = unique(items, 'item_id')
    answers = unique(responses, 'item_id')
    if not questions or questions.keys() != answers.keys():
        raise ValueError('Provide one response for each short-answer question')
    details = []
    for key, item in questions.items():
        raw = answers[key].get('response')
        if raw is not None and not isinstance(raw, str):
            raise ValueError('Response must be text or null')
        parsed, _, _ = json_payload(raw or '')
        answer = extract_answer({'parsed_response': parsed})
        parts, alignment = align_answer_components(answer, item['response_count'])
        for index, slot in enumerate(item['answer_slots']):
            text = parts[index] if index < len(parts) else ''
            correct = bool(text and score_slot(text, slot)['correct'])
            details.append({'item_id': key, 'slot_id': slot['slot_id'], 'correct': correct,
                            'alignment': alignment})
    correct = sum(row['correct'] for row in details)
    metrics = {'questions': len(questions), 'slots': len(details), 'correct_slots': correct,
               'slot_accuracy': correct / len(details)}
    return {'metrics': metrics, 'slots': details}
