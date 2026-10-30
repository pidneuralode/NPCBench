import unittest

from npcbench_eval.l6_inputs import frontend_values, render_care


class L6InputTests(unittest.TestCase):
    def test_unknown_pathology_is_distinct_from_a_negative_prediction(self):
        unknown = frontend_values('<think>1', '{"T_stage":"T2"}')
        negative = frontend_values('0', '{"T_stage":"T2"}')
        self.assertEqual(render_care('{{PATHOLOGY_RESULT}}', unknown, {}), '病理结果未明')
        self.assertEqual(render_care('{{PATHOLOGY_RESULT}}', negative, {}), '不支持鼻咽癌诊断')
        self.assertEqual(render_care('{{PATHOLOGY_RESULT}}', unknown, {}, 'frozen_v1'), '不支持鼻咽癌诊断')

    def test_anatomy_and_sequence_mentions_cannot_produce_a_staging_input(self):
        values = frontend_values('1', '{"imaging_basis":"T1 postcontrast images", "T1 vertebra":0}')
        prompt = render_care('{{MRI_T_STAGE}} / {{TNM_STATUS}}', values, {'N': 'N1', 'M': 'M0'})
        self.assertEqual(prompt, 'T未明 / T未明N1M0')

    def test_completed_stage_survives_truncated_ancillary_output(self):
        values = frontend_values('1', '{"T_stage":"T4", "imaging_basis":"unfinished')
        self.assertEqual(render_care('{{MRI_T_STAGE}}', values, {}), 'T4')


if __name__ == '__main__':
    unittest.main()
