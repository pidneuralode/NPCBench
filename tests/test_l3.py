import unittest
import json
from pathlib import Path
import tempfile
from itertools import permutations

from npcbench_eval.l3 import canonical_model, project_rows, reproduce, score, sha
from npcbench_eval import l3_parser


class L3Tests(unittest.TestCase):
    def test_metrics_are_invariant_to_item_order(self):
        groups, predictions = [], []
        for label, count, correct in [('T1', 3, 2), ('T2', 7, 4), ('T3', 11, 6), ('T4', 13, 9)]:
            group = []
            for i in range(count):
                key = label + str(i)
                group.append({'id': key, 'gold': label, 'base_structure_choices': ['clivus']})
                answer = label if i < correct else ('T2' if label == 'T1' else 'T1')
                predictions.append({'id': key, 'response': json.dumps({'t_stage': answer})})
            groups.append(group)
        expected = score('T-Stg', [r for group in groups for r in group], predictions)[0]
        for ordering in permutations(groups):
            with self.subTest(order=[group[0]['gold'] for group in ordering]):
                actual = score('T-Stg', [r for group in ordering for r in group], predictions[::-1])[0]
                self.assertEqual(actual, expected)

    def test_micro_is_primary_and_invalids_remain_in_denominator(self):
        dataset = [{'id': str(i), 'gold': label} for i, label in enumerate(['0', '0', '0', '1'])]
        predictions = [{'id': str(i), 'response': text, 'correct': True}
                       for i, text in enumerate(['0', '0', '', '0'])]
        metrics, parsed = score('Path', dataset, predictions)
        self.assertEqual(metrics['micro_percent'], 50)
        self.assertAlmostEqual(metrics['class_macro_percent'], 100 / 3)
        self.assertEqual(metrics['count'], 4)
        self.assertFalse(parsed[2]['correct'])
        self.assertFalse(parsed[3]['correct'])  # Does not trust supplied flags.

    def test_boxed_path_answer_takes_precedence(self):
        metrics, _ = score('Path', [{'id': 'p', 'gold': '1'}],
                           [{'id': 'p', 'response': 'normal reasoning, then \\boxed{1}'}])
        self.assertEqual(metrics['correct'], 1)

    def test_structure_sides_collapse_and_secondary_errors_do_not_gate(self):
        dataset = [{'id': 's', 'gold': 'cavernous_sinus', 'structure_choices': ['cavernous_sinus']}]
        metrics, parsed = score('Struct', dataset, [{'id': 's', 'response': '{"structure_id":"cavernous_sinus_left"}'}])
        self.assertEqual(metrics['correct'], 1)
        self.assertIn('invalid_invasion', parsed[0]['parser_errors'])

    def test_t_stage_does_not_require_complete_invasion_map(self):
        dataset = [{'id': 't', 'gold': 'T3', 'base_structure_choices': ['clivus']}]
        metrics, parsed = score('T-Stg', dataset, [{'id': 't', 'response': '{"t_stage":"T3"}'}])
        self.assertEqual(metrics['correct'], 1)
        self.assertIn('missing_structure_invasion:1', parsed[0]['parser_errors'])

    def test_nonobject_json_cannot_be_rescued(self):
        dataset = [{'id': 't', 'gold': 'T3', 'base_structure_choices': ['clivus']}]
        for response in ['1', 'null', '[]', '"T3"', 'true', '<answer>1</answer>', '']:
            with self.subTest(response=response):
                metrics, parsed = score('T-Stg', dataset, [{'id': 't', 'response': response}])
                self.assertEqual(metrics['correct'], 0)
                self.assertEqual(metrics['count'], 1)
                self.assertIsNone(parsed[0]['prediction'])

    def test_sequence_final_marker_without_box_is_accepted(self):
        dataset = [{'id': 'q', 'gold': 'B', 'options': [
            {'option_id': 'A', 'sequence_label': 'T1_Ax', 'sequence_cn': '轴位T1'},
            {'option_id': 'B', 'sequence_label': 'T2_Ax', 'sequence_cn': '轴位T2'}]}]
        metrics, parsed = score('Seq', dataset, [{'id': 'q', 'response': '选项 A 不正确。最终答案是 B'}])
        self.assertEqual(metrics['correct'], 1)
        self.assertIn('boxed_not_found', parsed[0]['parser_errors'])

    def test_exact_coverage_and_response_type(self):
        dataset = [{'id': 'p', 'gold': '0'}]
        for predictions in [[], [{'id': 'other', 'response': '0'}],
                            [{'id': 'p', 'response': '0'}, {'id': 'p', 'response': '0'}],
                            [{'id': 'p', 'response': 0}]]:
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                score('Path', dataset, predictions)

    def test_projection_omits_source_fields_and_flags(self):
        source = [{'question_id': 'private-id-example', 'answer': '1', 'response': '0',
                   'prompt': 'private prompt', 'patient_id': 'private patient',
                   'correct': True, 'parsed_prediction': '1', 'images': ['/private/path']}]
        dataset, predictions = project_rows('Path', source)
        self.assertEqual(set(dataset[0]), {'id', 'gold'})
        self.assertEqual(set(predictions[0]), {'id', 'response'})
        self.assertNotIn('private', dataset[0]['id'])
        self.assertEqual(score('Path', dataset, predictions)[0]['correct'], 0)

    def test_model_alias_is_bounded(self):
        self.assertEqual(canonical_model('Qwen3.5-27B-Instruct'), 'Qwen3.5-27B')
        self.assertEqual(canonical_model('Qwen3-VL-8B-Instruct'), 'Qwen3-VL-8B-Instruct')

    def test_choice_order_normalization_preserves_primary_semantics(self):
        source = {'patient_case_id': 'synthetic', 't_stage_gt': 'T3',
                  'base_structure_choices': ['clivus', 'nasal_passage'],
                  'response': '{"t_stage":"T3","invasion":{"clivus":1}}'}
        first = project_rows('T-Stg', [source])
        source['base_structure_choices'].reverse()
        second = project_rows('T-Stg', [source])
        self.assertEqual(first, second)
        self.assertEqual(score('T-Stg', *first), score('T-Stg', *second))

    def test_tampered_response_file_fails_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            response = root / 'responses.jsonl'
            response.write_text('{"id":"x","response":"1"}\n')
            digest = sha(response)
            manifest = {'schema': 'npcbench-l3-frozen-v1',
                        'parser_sha256': sha(l3_parser.__file__),
                        'files': {'responses.jsonl': digest}}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            response.write_text('{"id":"x","response":"0"}\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                reproduce(root)


if __name__ == '__main__':
    unittest.main()
