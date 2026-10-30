import json
from pathlib import Path
import tempfile
import unittest

from npcbench_eval.o1_release import score
from npcbench_eval.export_o1_release import export, prompt_text, public_config, missing_runtime_parameters


def make_items():
    return [{'item_id': f'i{i}', 'case_id': case, 'patient_id': case, 'tag': 'test', 'tag_label': 'test',
             'question_slot': 'Q1', 'case_input': 'Synthetic unit-test case', 'question': 'Synthetic test question',
             'rubric_template_structured': {'total_score': 10, 'core_answer': ['Required statement'],
                 'key_evidence': [], 'conditional_modifiers': [], 'negative_list': {'major_errors': [], 'minor_errors': []},
                 'scoring_rules': {'item_scores': {'core_answer': [10], 'key_evidence': [], 'conditional_modifiers': []}}}}
            for i, case in enumerate(['c1', 'c1', 'c2'])]


def make_judgments():
    return [{'item_id': f'i{i}', 'case_id': case, 'rubric_id': f'i{i}::core_answer::0', 'judge_status': 'ok',
             'criteria_met': met, 'explanation': 'Unit-test judgment',
             'raw_judge_response': json.dumps({'criteria_met': met, 'explanation': 'Unit-test judgment'})}
            for i, (case, met) in enumerate([('c1', True), ('c1', True), ('c2', False)])]


class O1FrozenReleaseTests(unittest.TestCase):
    def test_nested_historical_configs_preserve_roles_zero_and_token_budget(self):
        # The combined judge run config embeds both phases; older exports lost
        # this nesting and treated max_new_tokens as an absent output budget.
        config = {
            'generation': {'command_args': {'max_new_tokens': 2048, 'temperature': 0.0,
                'top_p': 1.0, 'top_k': None, 'do_sample': False, 'seed': 0,
                'api_key': 'synthetic-secret', 'model_path': '/private/model'}},
            'judge': {'judge_max_tokens': 2048, 'judge_temperature': 0.0, 'judge_top_p': 1.0},
            'api_key': 'synthetic-secret',
        }
        published = public_config(config, 'judge')
        generation = published['parameters_by_role']['generation']
        self.assertEqual(generation['max_new_tokens'], 2048)
        self.assertEqual(generation['temperature'], 0.0)
        self.assertEqual(generation['seed'], 0)
        self.assertIsNone(generation['top_k'])
        self.assertIs(generation['do_sample'], False)
        self.assertNotIn('max_tokens', generation)
        self.assertNotIn('max_new_tokens', published['generation_or_judge_parameters'])
        self.assertEqual(published['parameter_source_paths']['generation']['temperature'],
                         ['generation.command_args.temperature'])
        self.assertEqual(missing_runtime_parameters({'judge/run_config.json': published}), [])
        self.assertNotIn('synthetic-secret', json.dumps(published))
        self.assertNotIn('/private/model', json.dumps(published))
        # Judge settings cannot fill a genuinely absent generation budget.
        del config['generation']['command_args']['max_new_tokens']
        self.assertEqual(missing_runtime_parameters({'judge/run_config.json': public_config(config, 'judge')}),
                         ['generation.max_tokens'])

    def test_item_mean_is_not_case_macro(self):
        result = score(make_items(), make_judgments())
        self.assertAlmostEqual(result['metrics']['overall_score'], 200 / 3)
        self.assertEqual(result['metrics']['case_macro_score'], 50)

    def test_missing_judgment_excludes_item_and_reports_failure(self):
        result = score(make_items(), make_judgments()[1:])
        self.assertEqual(result['metrics']['num_items'], 3)
        self.assertEqual(result['metrics']['num_scored_items'], 2)
        self.assertEqual(result['metrics']['overall_score'], 50)
        self.assertEqual(result['raw_judge_audit']['missing_judgments'], 1)
        self.assertIsNone(result['item_scores'][0]['normalized_score'])

    def test_raw_response_overrides_cached_boolean(self):
        judgments = make_judgments()
        judgments[0]['criteria_met'] = False
        result = score(make_items(), judgments)
        self.assertEqual(result['item_scores'][0]['normalized_score'], 100)
        self.assertEqual(result['raw_judge_audit']['stored_criteria_mismatches'], 1)

    def test_failed_status_is_not_rescued_by_valid_raw_text(self):
        judgments = make_judgments()
        judgments[0]['judge_status'] = 'parse_failed'
        result = score(make_items(), judgments)
        self.assertEqual(result['metrics']['num_scored_items'], 2)
        self.assertEqual(result['raw_judge_audit']['archived_failure_statuses'], 1)

    def test_invalid_raw_cannot_use_cached_success(self):
        judgments = make_judgments()
        judgments[0]['raw_judge_response'] = '[]'
        result = score(make_items(), judgments)
        self.assertEqual(result['metrics']['num_scored_items'], 2)
        self.assertEqual(result['raw_judge_audit']['raw_parse_invalid'], 1)

    def test_duplicate_judgment_fails(self):
        judgments = make_judgments()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            score(make_items(), judgments + [judgments[0]])

    def test_frozen_prompt_shapes(self):
        self.assertEqual(prompt_text({'prompt': 'actual local prompt'}), 'actual local prompt')
        self.assertEqual(prompt_text([{'role': 'user', 'content': 'actual API prompt'}]), 'actual API prompt')
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            prompt_text({'prompt': 'text', 'private_path': 'not allowed'})

    def test_private_map_outside_public(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'public'
            with self.assertRaisesRegex(ValueError, 'outside public'):
                export(temp, out, out / 'map.json')
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
