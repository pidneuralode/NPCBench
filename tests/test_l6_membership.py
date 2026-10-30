import copy
import unittest

from npcbench_eval.l6 import score, validate_membership


class L6MembershipTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            'episodes': [{'patient_id': p, 'pathology_sample_id': p + '-path', 'mri_sample_id': p + '-mri',
                          'care_turns': [{'sample_id': sid} for sid in ids],
                          'unscored_care_turn_ids': ['q4'] if p == 'p1' else []}
                         for p, ids in [('p1', ['q1', 'q2']), ('p2', ['q3'])]],
            'pathology': [{'sample_id': p + '-path', 'patient_id': p, 'answer': '1'} for p in ['p1', 'p2']],
            'mri': [{'sample_id': p + '-mri', 'patient_id': p,
                     'answer': {'T_stage': 'T2', 'structure_invasion': {'鼻咽': 1, '鼻腔': 1}}} for p in ['p1', 'p2']],
            'care': [{'sample_id': q, 'patient_id': p, 'gold_letter': 'A'} for p, q in [('p1', 'q1'), ('p1', 'q2'), ('p2', 'q3')]],
            'unscored': [{'sample_id': 'q4', 'patient_id': 'p1', 'status': 'unscored_pending_item_review'}],
            'hardcore': ['q1'],
            'contract': {'version': 'test-membership', 'patients': 2, 'scored_care_nodes': 3,
                         'unscored_pending_review_nodes': 1, 'hardcore_nodes': 1,
                         'scored_ids': ['q1', 'q2', 'q3'], 'excluded_ids': ['q4'],
                         'metric_denominators': {'care_micro': 3, 'patient_all_care': 2,
                            'strict_episode': 2, 'pathology': 2, 'MRI_T': 2, 'hardcore_care': 1}}
        }
        self.responses = [{'sample_id': p + suffix, 'response': answer}
                          for p in ['p1', 'p2'] for suffix, answer in [('-path', '1'), ('-mri', '{"T_stage":"T2"}')]]
        self.responses += [{'sample_id': q, 'response': value} for q, value in [('q1', 'A'), ('q2', 'B'), ('q3', 'A')]]

    def test_excluded_item_is_neither_a_failure_nor_a_success(self):
        metrics = score(self.data, self.responses, 'frozen_v1')['metrics']
        self.assertEqual((metrics['patients'], metrics['scored_care_nodes'], metrics['response_count']), (2, 3, 7))
        self.assertEqual(metrics['care_right'], 2)
        self.assertAlmostEqual(metrics['care_micro_percent'], 200 / 3)
        self.assertEqual(metrics['strict_episode_percent'], 50)

    def test_excluded_prediction_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, 'exactly cover'):
            score(self.data, self.responses + [{'sample_id': 'q4', 'response': 'A'}], 'frozen_v1')

    def test_empty_scored_episode_cannot_be_vacuously_successful(self):
        self.data['episodes'][1]['care_turns'] = []
        with self.assertRaisesRegex(ValueError, 'Empty scored'):
            validate_membership(self.data)

    def test_patient_join_is_checked(self):
        self.data['care'][2]['patient_id'] = 'p1'
        with self.assertRaisesRegex(ValueError, 'different patient'):
            validate_membership(self.data)

    def test_duplicate_or_missing_response_rejected(self):
        for responses in [self.responses + [self.responses[0]], self.responses[:-1]]:
            with self.assertRaises(ValueError):
                score(self.data, responses, 'frozen_v1')

    def test_hardcore_cannot_reintroduce_excluded_item(self):
        self.data['hardcore'] = ['q4']
        with self.assertRaisesRegex(ValueError, 'mask mismatch'):
            validate_membership(self.data)

    def test_pending_gold_not_released_as_an_answer_key(self):
        self.data['unscored'][0]['gold_letter'] = 'A'
        with self.assertRaisesRegex(ValueError, 'Unadjudicated'):
            validate_membership(self.data)

    def test_declared_denominator_cannot_remain_at_the_old_inventory(self):
        self.data['contract']['metric_denominators']['care_micro'] = 4
        with self.assertRaisesRegex(ValueError, 'denominators'):
            validate_membership(self.data)

    def test_channel_gate_changes_answers_without_changing_membership(self):
        self.responses[-1]['response'] = '<think>answer: A'
        untouched = copy.deepcopy(self.responses)
        before = score(self.data, self.responses, 'frozen_v1')['metrics']
        after = score(self.data, self.responses, 'reject_unfinished_think_v1')['metrics']
        self.assertEqual((before['care_right'], after['care_right']), (2, 1))
        self.assertEqual((before['patients'], after['patients'], after['scored_care_nodes']), (2, 2, 3))
        self.assertEqual(after['strict_episode_percent'], 0)
        self.assertEqual(untouched, self.responses)

    def test_final_policy_preserves_membership_and_flags_changed_frontends(self):
        for row in self.data['care']:
            row['options'] = {'A': 'first', 'B': 'second'}
        self.responses[1]['response'] = '{"imaging_basis":"T1 postcontrast images"}'
        self.responses[-1]['response'] = '<think>answer: A'
        before = copy.deepcopy((self.data, self.responses))
        result = score(self.data, self.responses, 'final_answer_v1')
        self.assertEqual(result['metrics']['response_count'], 7)
        self.assertEqual(result['metrics']['scored_care_nodes'], 3)
        self.assertEqual(result['metrics']['care_right'], 1)
        self.assertEqual(result['metrics']['mri_t_parse_failures'], 1)
        self.assertEqual(result['frontend_interpretation_changes'], [
            {'patient_id': 'p1', 'changed_fields': ['MRI_T'], 'care_nodes': 2}])
        self.assertEqual(before, (self.data, self.responses))


if __name__ == '__main__':
    unittest.main()
