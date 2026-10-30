import tempfile
from pathlib import Path
import unittest

from npcbench_eval.mcq_release import score, display
from npcbench_eval.export_mcq_release import export


class FrozenMCQReleaseTests(unittest.TestCase):
    def setUp(self):
        self.data = [{'id': str(i), 'answer': 'A', 'options': {'A': 'first', 'B': 'second'},
                      'choices': ['A. first', 'B. second'], 'patient_id': group, 'capability_group': group}
                     for i, group in enumerate(['g1', 'g1', 'g2', 'g3', 'g4'])]
        self.pred = [{'id': str(i), 'response': letter} for i, letter in enumerate(['A', 'A', 'B', 'B', 'B'])]

    def test_primary_micro_and_macro_are_distinct(self):
        l1, _ = score(self.data, self.pred, 'L1', 'api_mcq_v1')
        l2, _ = score(self.data, self.pred, 'L2', 'api_mcq_v1')
        l4, _ = score(self.data, self.pred, 'L4', 'api_mcq_v1')
        self.assertEqual(l1['primary_percent'], 40)
        self.assertEqual(l2['micro_percent'], 40)
        self.assertEqual(l2['primary_percent'], 25)
        self.assertEqual(l4['primary_percent'], 25)

    def test_api_conflict_and_empty_stay_in_denominator(self):
        self.pred[0]['response'] = r'\boxed{A} and \boxed{B}'
        self.pred[1]['response'] = ''
        metrics, decisions = score(self.data, self.pred, 'L1', 'api_mcq_v1')
        self.assertEqual(metrics['items'], 5)
        self.assertEqual(metrics['correct'], 0)
        self.assertEqual(metrics['parse_invalid'], 2)
        self.assertFalse(decisions[0]['correct'])

    def test_final_policy_keeps_macro_groups_with_invalid_answers(self):
        self.pred[0]['response'] = 'analysisThe answer is A.'
        metrics, decisions = score(self.data, self.pred, 'L2', 'final_answer_v1')
        self.assertEqual(metrics['items'], 5)
        self.assertEqual(metrics['primary_percent'], 12.5)
        self.assertEqual(metrics['parse_invalid'], 1)
        self.assertFalse(decisions[0]['correct'])

    def test_local_empty_wrong_and_parse_valid_not_invented(self):
        self.pred[0]['response'] = ''
        metrics, decisions = score(self.data, self.pred, 'L1', 'local_cpu_v2')
        self.assertEqual(metrics['correct'], 1)
        self.assertEqual(metrics['empty_responses'], 1)
        self.assertIsNone(metrics['parse_invalid'])
        self.assertTrue(all(r['parse_valid'] is None for r in decisions))

    def test_duplicate_fails(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            score(self.data, self.pred + [self.pred[0]], 'L1', 'api_mcq_v1')

    def test_incomplete_predictions_fail(self):
        with self.assertRaisesRegex(ValueError, 'exactly cover'):
            score(self.data, self.pred[:-1], 'L1', 'api_mcq_v1')

    def test_round_half_up(self):
        self.assertEqual(display(6.25), '6.3')

    def test_private_map_is_not_public(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'public'
            with self.assertRaisesRegex(ValueError, 'outside public'):
                export(temp, out, out / 'map.json')
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
