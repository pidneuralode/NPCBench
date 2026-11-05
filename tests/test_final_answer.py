import unittest

from npcbench_eval.final_answer import parse_choice, select_final, json_payload, parse_t_stage


class FinalAnswerTests(unittest.TestCase):
    options = {'A': 'first action', 'B': 'second action', 'C': 'third action', 'L': 'last action'}

    def prediction(self, value):
        return parse_choice(value, self.options)

    def test_real_decoded_harmony_format_uses_only_final(self):
        answer = self.prediction('analysisThe answer is A. assistantfinal\\boxed{B}')
        self.assertEqual(answer.prediction, 'B')
        self.assertEqual(answer.channel, 'explicit_final')

    def test_analysis_without_final_is_not_an_answer(self):
        result = self.prediction('analysisWe consider B. Therefore answer: B.')
        self.assertFalse(result.valid)
        self.assertEqual(result.status, 'unfinished_analysis_channel')

    def test_explicit_harmony_tokens(self):
        text = '<|channel|>analysis<|message|>A<|channel|>final<|message|>B<|return|>'
        self.assertEqual(self.prediction(text).prediction, 'B')

    def test_harmony_format_metadata_does_not_change_the_final_channel(self):
        for format_name in ('json', 'commentary'):
            text = ('<|channel|>analysis<|message|>The provisional stage is T4.'
                    '<|end|><|start|>assistant<|channel|>final <|constrain|>'
                    + format_name + '<|message|>{"t_stage":"T1"}')
            with self.subTest(format_name=format_name):
                self.assertEqual(parse_t_stage(text).prediction, 'T1')
        self.assertFalse(parse_t_stage('<|channel|>analysis <|constrain|>json'
                                     '<|message|>{"t_stage":"T1"}').valid)

    def test_huatuo_markdown_final_ignores_different_thinking_answer(self):
        text = '## Thinking\nThe best answer is A.\n\n## Final Response\nB. second action'
        self.assertEqual(self.prediction(text).prediction, 'B')
        self.assertFalse(self.prediction('## Thinking\nThe best answer is A.').valid)

    def test_fleming_html_final(self):
        self.assertEqual(self.prediction('<think>A</think><span style="color: #0000ff;">B</span>').prediction, 'B')
        self.assertEqual(self.prediction('<think>A</think><b>B</b>').prediction, 'B')

    def test_fleming_xml_box_final(self):
        for text in ('<think>A</think><boxed>B</boxed>',
                     '<think>A</think><BOXED>B</boxed>'):
            with self.subTest(text=text):
                answer = self.prediction(text)
                self.assertTrue(answer.valid)
                self.assertEqual(answer.prediction, 'B')

    def test_xml_box_conflicts_with_other_final_answers(self):
        for text in ('<boxed>A</boxed>\\boxed{B}',
                     '<boxed>A</boxed><answer>B</answer>'):
            with self.subTest(text=text):
                self.assertEqual(self.prediction(text).status, 'conflicting_answers')

    def test_xml_box_must_be_complete_and_valid(self):
        for text in ('<boxed>B', '<boxed>M</boxed>', '<boxed>B</answer>',
                     '<boxed>A</boxed><boxed>B', '<think><boxed>B</boxed>'):
            with self.subTest(text=text):
                self.assertFalse(self.prediction(text).valid)

    def test_first_line_and_chinese_final_prose(self):
        self.assertEqual(self.prediction('B\n\nExplanation without other final answers.').prediction, 'B')
        self.assertEqual(self.prediction('最符合该策略的是选项 B：second action。').prediction, 'B')

    def test_completed_and_implicit_reasoning_prefixes(self):
        for text in ('<think>\\boxed{A}</think>\\boxed{B}',
                     'answer: A</think>\nB', '<THINKING>A</thinking>\nB'):
            with self.subTest(text=text):
                self.assertEqual(self.prediction(text).prediction, 'B')

    def test_unfinished_reasoning_invalidates_box(self):
        for text in ('<think>\\boxed{B}', '<think><thinking>A</think>B', 'B<think>A'):
            self.assertFalse(self.prediction(text).valid)

    def test_nested_box_and_markdown(self):
        for text in ('\\boxed{\\textbf{(B)}}', '**答案：** **B**', '\\boxed{L}', 'b.'):
            with self.subTest(text=text):
                self.assertTrue(self.prediction(text).valid)

    def test_conflicts_are_not_resolved_using_last_answer(self):
        for text in ('\\boxed{A} \\boxed{B}', '<answer>A</answer>\nFinal answer: B',
                     'Final answer: A or B', 'A. first action\nB. second action'):
            with self.subTest(text=text):
                self.assertEqual(self.prediction(text).status, 'conflicting_answers')

    def test_matching_repeated_answers_are_valid(self):
        self.assertEqual(self.prediction('<answer>B</answer>\nFinal answer: B\n\\boxed{B}').prediction, 'B')

    def test_plain_exact_option_is_supported_without_fuzzy_matching(self):
        self.assertEqual(self.prediction('Second action.').prediction, 'B')
        for text in ('probably the second act', 'We discuss first action and second action.', 'A is possible but uncertain'):
            self.assertFalse(self.prediction(text).valid)

    def test_duplicate_option_text_is_ambiguous(self):
        self.assertFalse(parse_choice('same action', {'A': 'same action', 'B': 'same action'}).valid)

    def test_invalid_or_incomplete_explicit_answer_is_not_rescued(self):
        for text in ('\\boxed{M}', '\\boxed{B', '<answer>B', '\\boxed{A/B}'):
            self.assertFalse(self.prediction(text).valid)

    def test_binary_does_not_infer_from_disease_mentions_or_numbers(self):
        options = {'0': '', '1': ''}
        self.assertEqual(parse_choice('\\boxed{0}', options).prediction, '0')
        for text in ('No nasopharyngeal carcinoma.', 'There are 1 suspicious and 0 definitive findings.'):
            self.assertFalse(parse_choice(text, options).valid)

    def test_json_answer_does_not_parse_reasoning_string(self):
        self.assertEqual(self.prediction('{"reasoning":"answer: A is incorrect", "answer":"B"}\n\\boxed{B}').prediction, 'B')
        self.assertEqual(self.prediction('[{"box":"B"}]').prediction, 'B')

    def test_json_reasoning_and_answer_objects_are_separate(self):
        text = '[{"reasoning":"An earlier possibility was \\\\boxed{A}."},{"box":"B"}]'
        self.assertEqual(self.prediction(text).prediction, 'B')
        self.assertEqual(self.prediction('{"answer":"A"}\n{"answer":"B"}').status, 'conflicting_answers')
        self.assertFalse(self.prediction('{"reasoning":"answer: B"}').valid)

    def test_incidental_option_references_do_not_conflict_with_declared_final(self):
        for text in ('这属于另一组（即选项C）。\n\\boxed{B}',
                     '特别是选项A，不符合要求。\nAnswer: B',
                     '如果条件不同，应选A。\n最终选择：B\n\\boxed{B}',
                     '1. Analysis\n2. Another point\n\\boxed{B}'):
            self.assertEqual(self.prediction(text).prediction, 'B')

    def test_explicit_prose_selections_without_fuzzy_matching(self):
        for text in ('The appropriate action is Option B: second action.',
                     '最适宜的策略是选择B选项。', '方案首选B. second action.',
                     '最合理的选项是 **B**。', '应给予的医嘱是：B. second action.',
                     'The selected action follows:\n\nB. second action'):
            self.assertEqual(self.prediction(text).prediction, 'B')

    def test_json_duplicate_keys_and_conflicting_objects_rejected(self):
        self.assertEqual(json_payload('{"t_stage":"T1","t_stage":"T2"}')[1], 'invalid_json')
        self.assertEqual(json_payload('{"t_stage":"T1"}\n{"t_stage":"T2"}')[1], 'conflicting_json')

    def test_json_nested_object_and_duplicate_identical_outputs(self):
        text = '{"t_stage":"T2","structure_invasion":{"x":0}}'
        payload, status, _ = json_payload('<think>{"t_stage":"T1"}</think>' + text + '\n' + text)
        self.assertEqual(status, 'ok')
        self.assertEqual(payload['t_stage'], 'T2')
        self.assertEqual(json_payload('{"x":0,"x":0,"t_stage":"T2"}')[1], 'ok')

    def test_gold_absent_from_api_and_options_reordering(self):
        self.assertEqual(self.prediction('B'), parse_choice('B', dict(reversed(list(self.options.items())))))

    def test_original25_option_rows_preserve_explicit_labels(self):
        options = [{'letter': k, 'text': v} for k, v in reversed(list(self.options.items()))]
        self.assertEqual(parse_choice('B', options), self.prediction('B'))

    def test_multiple_final_channels_and_empty_outputs(self):
        for text in ('analysisAassistantfinalAassistantfinalB', '', None, '<think>A</think>'):
            self.assertFalse(self.prediction(text).valid)

    def test_no_invented_stop_reason(self):
        self.assertEqual(select_final('B').status, 'ok')

    def test_stage_not_sequence_or_vertebra(self):
        for text in ('T1 post-contrast images show changes.', '{"T1 vertebra":0,"T2 vertebra":0}'):
            self.assertFalse(parse_t_stage(text).valid)

    def test_stage_field_survives_incomplete_ancillary_explanation(self):
        for text in ('{"T_stage":"T4","imaging_basis":"unfinished',
                     '{"x":0}\nT_stage: T1', '{"x":0}\n"t_stage": "T1"\n}'):
            self.assertTrue(parse_t_stage(text).valid)

    def test_conflicting_stage_fields_rejected(self):
        self.assertFalse(parse_t_stage('{"T_stage":"T1","t_stage":"T4"}').valid)

    def test_stage_field_must_be_exact_and_outside_explanation_strings(self):
        for text in ('{"T_stage":"T1 weighted images"}', '{"T_stage":"T4a"}',
                     '{"T_stage":"T2", "t_stage":"unknown"}',
                     '{"imaging_basis":"Example: \\"T_stage\\":\\"T1\\""}',
                     '{"imaging_basis":"unfinished, T_stage: T1'):
            with self.subTest(text=text):
                self.assertFalse(parse_t_stage(text).valid)

    def test_ancillary_nested_stage_explanation_is_not_an_answer_field(self):
        value = '{"T_stage":"T2", "imaging_basis":{"T_stage":"Explanation of T2 staging"}}'
        self.assertEqual(parse_t_stage(value).prediction, 'T2')
        self.assertEqual(parse_t_stage('{"T stage":"T1"}').prediction, 'T1')


if __name__ == '__main__':
    unittest.main()
