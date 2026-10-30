import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CurrentPolicyCLITests(unittest.TestCase):
    def test_new_prediction_commands_reject_unfinished_reasoning(self):
        cases = [
            ('l3', ['--subset', 'Path'], {'id': 'q', 'gold': '1'}, '<think>answer: 1', 'correct'),
            ('l5', [], {'id': 'q', 'patient_id': 'p', 'answer': 'A',
                       'options': {'A': 'yes', 'B': 'no'}}, '<think>answer: A', 'correct_turns'),
        ]
        for module, extra, question, answer, key in cases:
            with self.subTest(module=module), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'data.jsonl').write_text(json.dumps(question) + '\n')
                (root / 'predictions.jsonl').write_text(json.dumps({'id': 'q', 'response': answer}) + '\n')
                subprocess.run([sys.executable, '-m', 'npcbench_eval.' + module, *extra,
                                '--dataset', str(root / 'data.jsonl'),
                                '--predictions', str(root / 'predictions.jsonl'),
                                '--output', str(root / 'scores.json')], check=True, capture_output=True)
                metrics = json.loads((root / 'scores.json').read_text())['metrics']
                self.assertEqual(metrics['answer_policy'], 'final_answer_v1')
                self.assertEqual(metrics[key], 0)
