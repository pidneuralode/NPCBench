import unittest

from npcbench_eval.l6_history import action_from_response, render_history, POLICY, UNKNOWN_ACTION


class HistoryInputTests(unittest.TestCase):
    def test_unfinished_thought_never_becomes_a_clinical_action(self):
        text = '<think>consider option A'
        options = {'A': 'first action', 'B': 'second action'}
        self.assertEqual(action_from_response(text, options, POLICY), {'letter': '', 'action_text': UNKNOWN_ACTION})
        self.assertEqual(action_from_response(text, options, 'frozen_history_v1')['action_text'], 'first action')

    def test_legacy_raw_fallback_is_not_retained_by_new_policy(self):
        text = '<think>unfinished clinical reasoning'
        options = {'A': 'first action'}
        self.assertEqual(action_from_response(text, options, 'frozen_history_v1')['action_text'], text)
        self.assertEqual(action_from_response(text, options, POLICY)['action_text'], UNKNOWN_ACTION)

    def test_explicit_final_option_supplies_exact_action(self):
        result = action_from_response('<think>B</think>\\boxed{A}', [{'letter': 'A', 'text': 'first action'}, {'letter': 'B', 'text': 'second action'}], POLICY)
        self.assertEqual(result, {'letter': 'A', 'action_text': 'first action'})

    def test_only_history_block_is_replaced(self):
        prompt = 'clinical evidence\n【历史辅助决策节点】\nold history\n\n问题：original question'
        self.assertEqual(render_history(prompt, [{'action_text': 'first action。'}]), 'clinical evidence\n【历史辅助决策节点】\n- 历史辅助决策节点1：first action。\n\n问题：original question')


if __name__ == '__main__':
    unittest.main()
