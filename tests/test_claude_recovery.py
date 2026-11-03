import unittest
from npcbench_eval.claude_recovery import recover


class ClaudeRecoveryTests(unittest.TestCase):
    def test_unsupported_explicit_stage_is_not_resampled(self):
        def send(*args):
            self.fail('An explicit answer must not be retried')
        result = recover({'id':'a','task':'L3-N400'}, {'response':'{"n_stage":"N3b"}'}, send)
        self.assertEqual(result['attempts'], [])

    def test_empty_case_stops_after_three_attempts(self):
        calls=[]
        def send(row, params):
            calls.append(params);return {'response':None,'finish_reason':'stop'}
        result=recover({'id':'a','task':'L3-N400'}, {'response':None}, send)
        self.assertEqual(len(calls),3)
        self.assertIsNone(result['selected']['response'])

    def test_incomplete_structure_uses_larger_budget_and_stops_at_wrong_stage(self):
        calls=[]
        def send(row, params):
            calls.append(params);return {'response':'{"t_stage":"T1"}','finish_reason':'stop'}
        result=recover({'id':'a','task':'L3-T300'},
            {'response':'{"structure_invasion": {','finish_reason':'length'}, send)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0]['max_tokens'],8192)
        self.assertEqual(result['selected']['response'],'{"t_stage":"T1"}')
