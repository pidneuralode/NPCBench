import unittest

from npcbench_eval.response_channels import gate_response, has_unfinished_think
from npcbench_eval.channel_audit import gated
from npcbench_eval.mcq_release import score


class ResponseChannelTests(unittest.TestCase):
    def test_unfinished_reasoning_cannot_supply_a_final_answer(self):
        self.assertEqual(gate_response('<think>Consider A and B. \\boxed{B}'), '')

    def test_completed_block_preserves_the_original_bytes(self):
        response = '<think>Consider A.</think>\n\\boxed{B}  '
        self.assertEqual(gate_response(response), response)

    def test_no_visible_opening_tag_is_not_a_truncation_claim(self):
        for response in ('</think>\nB', 'answer: B', '', None, 'analysis only'):
            self.assertEqual(gate_response(response), response)

    def test_nested_and_mismatched_blocks(self):
        self.assertTrue(has_unfinished_think('<think><think>x</think>'))
        self.assertTrue(has_unfinished_think('<thinking>x</think>'))
        self.assertFalse(has_unfinished_think('<think><thinking>x</thinking></think>B'))

    def test_last_block_still_open(self):
        self.assertTrue(has_unfinished_think('<think>x</think>B<think>reconsider'))

    def test_case_and_thinking_variant(self):
        self.assertTrue(has_unfinished_think('<THINKING>answer: C'))
        self.assertFalse(has_unfinished_think('<THINKING>x</thinking>C'))

    def test_escaped_or_unrelated_tags_do_not_trigger_the_gate(self):
        for response in ('&lt;think&gt; B', '<thinker>B', '<answer>B</answer>'):
            self.assertEqual(gate_response(response), response)

    def test_gate_is_idempotent(self):
        for response in ('<think>B', '<think>x</think>B', 'B'):
            self.assertEqual(gate_response(gate_response(response)), gate_response(response))

    def test_candidate_keeps_denominator_and_raw_responses(self):
        data = [{'id': str(i), 'answer': 'B', 'options': {'A': 'first', 'B': 'second'},
                 'choices': ['first', 'second']} for i in range(2)]
        raw = [{'id': '0', 'response': '<think>\\boxed{B}'},
               {'id': '1', 'response': '<think>A</think>\\boxed{B}'}]
        old, _ = score(data, raw, 'L1', 'local_cpu_v2')
        new, _ = score(data, gated(raw), 'L1', 'local_cpu_v2')
        self.assertEqual((old['correct'], new['correct'], new['items']), (2, 1, 2))
        self.assertEqual(raw[0]['response'], '<think>\\boxed{B}')


if __name__ == '__main__':
    unittest.main()
