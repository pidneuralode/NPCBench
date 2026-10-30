import copy
import unittest

from npcbench_eval.l5_rebuttal import evaluate_panel, paired_summary


class PairedInterventionTests(unittest.TestCase):
    def setUp(self):
        self.data = [{'id': key, 'patient_id': patient, 'condition': condition,
                      'answer': 'A', 'options': {'A': 'first', 'B': 'second'}, 'core': key == 'one'}
                     for condition in ('history', 'snapshot')
                     for key, patient in [('one', 'patient-a'), ('two', 'patient-a'), ('three', 'patient-b')]]
        self.responses = [{'id': r['id'], 'condition': r['condition'], 'model': 'example',
                           'response': 'A' if r['condition'] == 'history' else 'B',
                           'legacy_prediction': 'A', 'legacy_correct': True} for r in self.data]

    def test_paired_counts_and_cohort_denominators(self):
        metrics, pairs, rows = evaluate_panel(self.data, self.responses, {'all': None, 'core': 'core'}, repeats=20)
        self.assertEqual(len(rows), 6)
        self.assertEqual([(x['cohort'], x['turn_count'], x['patient_count']) for x in pairs], [('all', 3, 2), ('core', 1, 1)])
        self.assertEqual(pairs[0]['turn_accuracy_delta'], 1)
        self.assertEqual(pairs[0]['turn_accuracy_delta_ci95'], [1, 1])
        # Two patients, not three correlated care turns, are independent pairs.
        self.assertEqual(pairs[0]['episode_mcnemar_exact_unadjusted_p'], .5)
        self.assertEqual(pairs[0]['mcnemar_unit'], 'patient_episode')
        self.assertEqual(pairs[0]['episode_left_only_correct'], 2)
        self.assertEqual(sum(r['correctness_changed'] for r in rows), 3)
        self.assertEqual(len(metrics), 4)

    def test_missing_response_is_not_silently_intersected(self):
        with self.assertRaisesRegex(ValueError, 'exactly cover'):
            evaluate_panel(self.data, self.responses[:-1], repeats=1)

    def test_changed_gold_options_or_patient_rejected(self):
        for field, value in [('answer', 'B'), ('options', {'A': 'different', 'B': 'second'}), ('patient_id', 'elsewhere')]:
            data = copy.deepcopy(self.data)
            data[-1][field] = value
            with self.assertRaisesRegex(ValueError, 'preserve'):
                evaluate_panel(data, self.responses, repeats=1)

    def test_changed_stratum_membership_rejected(self):
        data = copy.deepcopy(self.data)
        data[-1]['core'] = True
        with self.assertRaisesRegex(ValueError, 'cohort differs'):
            evaluate_panel(data, self.responses, {'core': 'core'}, repeats=1)

    def test_paired_patient_mapping_must_match(self):
        left = [{'id': 'x', 'patient_id': 'a', 'correct': True}]
        right = [{'id': 'x', 'patient_id': 'b', 'correct': True}]
        with self.assertRaisesRegex(ValueError, 'patient identity'):
            paired_summary(left, right, repeats=1)


if __name__ == '__main__':
    unittest.main()
