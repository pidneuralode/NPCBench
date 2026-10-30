"""A repeated answer must never silently replace another answer for an item."""
import json
import tempfile
import unittest
from pathlib import Path

from npcbench_eval.mcq import _load_responses
from o1_eval.simple_evals_main.npc_o1_eval import _prediction_map


class PredictionIdentityTest(unittest.TestCase):
    def test_mcq_rejects_conflicting_and_identical_duplicates_in_list_formats(self):
        for second_answer in ('A', 'B'):
            records = [{'id': 'q1', 'response': 'A'}, {'item_id': 'q1', 'response': second_answer}]
            for schema in ('jsonl', 'list', 'results', 'responses'):
                with self.subTest(answer=second_answer, schema=schema), tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / ('predictions.jsonl' if schema == 'jsonl' else 'predictions.json')
                    payload = '\n'.join(json.dumps(r) for r in records) if schema == 'jsonl' else json.dumps(records if schema == 'list' else {schema: records})
                    path.write_text(payload)
                    with self.assertRaisesRegex(ValueError, 'Duplicate prediction item_id'):
                        _load_responses(path)

    def test_mcq_preserves_unique_dictionary_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'predictions.json'
            path.write_text(json.dumps({'q1': 'A', 'q2': 'B'}))
            self.assertEqual(_load_responses(path), {'q1': 'A', 'q2': 'B'})

    def test_mcq_detects_fallback_id_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'predictions.json'
            path.write_text(json.dumps(['A', {'id': 'row_1', 'response': 'B'}]))
            with self.assertRaisesRegex(ValueError, 'Duplicate prediction item_id'):
                _load_responses(path)

    def test_mcq_rejects_repeated_ids_in_json_dictionary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'predictions.json'
            path.write_text('{"q1": "A", "q1": "B"}')
            with self.assertRaisesRegex(ValueError, 'Duplicate response JSON key'):
                _load_responses(path)

    def test_o1_rejects_duplicate_even_when_one_record_failed(self):
        for statuses in [('ok', 'ok'), ('error', 'ok'), ('ok', 'error')]:
            with self.subTest(statuses=statuses), self.assertRaisesRegex(ValueError, 'Duplicate prediction item_id'):
                _prediction_map([{'item_id': 'q1', 'generation_status': status, 'response': 'test'} for status in statuses])

    def test_o1_keeps_existing_status_filter_for_unique_ids(self):
        records = [{'item_id': 'q1', 'response': 'test'}, {'item_id': 'q2', 'generation_status': 'error'}]
        self.assertEqual(_prediction_map(records), {'q1': records[0]})


if __name__ == '__main__':
    unittest.main()
