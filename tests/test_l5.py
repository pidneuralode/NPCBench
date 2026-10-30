import json
from pathlib import Path
import tempfile
import unittest

from npcbench_eval.l5 import display_percent, score
from npcbench_eval.export_l5 import export


class L5ScoringTests(unittest.TestCase):
    def setUp(self):
        # Deliberately unequal episode sizes distinguish micro from patient macro.
        self.data = [{'id': str(i), 'patient_id': patient, 'answer': 'A',
                      'options': {'A': 'correct', 'B': 'incorrect'}}
                     for i, patient in enumerate(['p1', 'p1', 'p1', 'p2'])]
        self.pred = [{'id': str(i), 'response': answer}
                     for i, answer in enumerate(['A', 'A', 'A', 'B'])]

    def test_distinct_denominators(self):
        metrics, _ = score(self.data, self.pred)
        self.assertEqual(metrics['turn_micro_accuracy'], 0.75)
        self.assertEqual(metrics['patient_macro_accuracy'], 0.5)
        self.assertEqual(metrics['all_turns_correct_patient_rate'], 0.5)
        self.assertEqual(metrics['paper_step_percent'], '75.0')
        self.assertEqual(metrics['paper_episode_percent'], '50.0')

    def test_unparsed_is_in_denominator(self):
        self.pred[0]['response'] = ''
        metrics, _ = score(self.data, self.pred)
        self.assertEqual(metrics['turn_count'], 4)
        self.assertEqual(metrics['correct_turns'], 2)
        self.assertEqual(metrics['all_correct_patients'], 0)
        self.assertEqual(metrics['empty_response_count'], 1)

    def test_final_policy_does_not_drop_an_unfinished_turn_or_patient(self):
        self.pred[0]['response'] = '<think>answer: A'
        metrics, decisions = score(self.data, self.pred, 'final_answer_v1')
        self.assertEqual(metrics['turn_count'], 4)
        self.assertEqual(metrics['correct_turns'], 2)
        self.assertEqual(metrics['all_correct_patients'], 0)
        self.assertEqual(decisions[0]['answer_status'], 'unfinished_reasoning')

    def test_duplicate_prediction_fails(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            score(self.data, self.pred + [self.pred[0]])

    def test_duplicate_gold_fails(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            score(self.data + [self.data[0]], self.pred)

    def test_null_id_fails(self):
        self.pred[0]['id'] = None
        with self.assertRaisesRegex(ValueError, 'empty'):
            score(self.data, self.pred)

    def test_missing_and_extra_predictions_fail(self):
        with self.assertRaisesRegex(ValueError, 'exactly cover'):
            score(self.data, self.pred[:-1])
        with self.assertRaisesRegex(ValueError, 'exactly cover'):
            score(self.data, self.pred + [{'id': 'extra', 'response': 'A'}])

    def test_half_up_display(self):
        self.assertEqual(display_percent(1, 16), '6.3')

    def test_export_refuses_public_private_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'public'
            with self.assertRaisesRegex(ValueError, 'outside public'):
                export(tmp, out, out / 'private.json')
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
