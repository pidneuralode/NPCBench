import argparse
import json
import os
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
from o1_eval.simple_evals_main.npc_o1_prompts import render_judge_prompt, render_rubric_item_for_judge
from o1_eval.simple_evals_main.npc_o1_schema import flatten_dataset_rubrics, flatten_item_rubrics, load_o1_items
from o1_eval.simple_evals_main.npc_o1_scorer import score_dataset
from o1_eval.simple_evals_main.npc_o1_eval import (
    API_MODEL_REGISTRY,
    _api_credentials,
    _api_request_payload,
    _public_path_ref,
    build_parser,
)


DATA_PATH = Path(os.environ.get("NPCBENCH_TEST_DATA", ROOT / "data")) / "o1_consultrubrics" / "consultrubrics_v20260504_doctor_revised.jsonl"


class TestNPCO1Eval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DATA_PATH.exists():
            raise unittest.SkipTest("Set NPCBENCH_TEST_DATA to run the full-dataset integration check")
        cls.items = load_o1_items(DATA_PATH)
        cls.rubric_items = flatten_dataset_rubrics(cls.items)

    def test_rubric_inventory(self):
        self.assertEqual(len(self.items), 103)
        self.assertEqual(len({item["case_id"] for item in self.items}), 42)
        positives = [row for row in self.rubric_items if row.scoring_effect == "positive"]
        majors = [row for row in self.rubric_items if row.scoring_effect == "major_penalty"]
        minors = [row for row in self.rubric_items if row.scoring_effect == "minor_penalty"]
        self.assertEqual(len(positives), 1035)
        self.assertEqual(len(majors), 337)
        self.assertEqual(len(minors), 320)
        self.assertAlmostEqual(sum(row.points for row in positives), 1030.0)

    def test_conditionals_are_scored_in_doctor_revised_release(self):
        first_item = self.items[0]
        flattened = flatten_item_rubrics(first_item)
        conditional = [row for row in flattened if row.section == "conditional_modifiers"]
        self.assertEqual(len(conditional), 2)
        self.assertEqual([row.points for row in conditional], [1.0, 1.0])
        all_positive = [row for row in self.rubric_items if row.scoring_effect == "positive"]
        self.assertTrue(all(row.points > 0 for row in all_positive))

    def test_major_minor_penalty_formula(self):
        item = self.items[0]
        rubric_items = flatten_item_rubrics(item)
        judgments = []
        for row in rubric_items:
            if row.scoring_effect == "positive":
                met = True
            elif row.scoring_effect == "major_penalty" and row.index == 0:
                met = True
            elif row.scoring_effect == "minor_penalty" and row.index == 0:
                met = True
            else:
                met = False
            judgments.append(
                {
                    "item_id": row.item_id,
                    "rubric_id": row.rubric_id,
                    "judge_status": "ok",
                    "criteria_met": met,
                    "explanation": "unit test",
                }
            )
        item_scores, metrics = score_dataset([item], rubric_items, judgments)
        self.assertEqual(item_scores[0]["scoring_status"], "ok")
        self.assertAlmostEqual(item_scores[0]["positive_score"], 10.0)
        self.assertAlmostEqual(item_scores[0]["major_penalty"], 2.0)
        self.assertAlmostEqual(item_scores[0]["minor_penalty"], 1.0)
        self.assertAlmostEqual(item_scores[0]["final_score"], 7.0)
        self.assertAlmostEqual(metrics["overall_score"], 70.0)

    def test_judge_prompt_hides_scoring_metadata(self):
        item = self.items[0]
        rubric_item = flatten_item_rubrics(item)[0]
        prompt = render_judge_prompt(
            item["case_input"],
            item["question"],
            "患者当前主要不良反应为皮肤瘙痒，严重程度信息不足。",
            rubric_item,
        )
        self.assertIn("nasopharyngeal carcinoma clinical consultation benchmark", prompt)
        self.assertIn("# Rubric Item", prompt)
        self.assertNotIn("major_errors", prompt)
        self.assertNotIn("minor_errors", prompt)
        self.assertNotIn("penalty", prompt)
        self.assertNotIn("points", prompt)

    def test_positive_and_negative_prompt_polarity_blocks_are_separate(self):
        rubric_items = flatten_item_rubrics(self.items[0])
        positive = next(row for row in rubric_items if row.scoring_effect == "positive")
        major = next(row for row in rubric_items if row.scoring_effect == "major_penalty")
        minor = next(row for row in rubric_items if row.scoring_effect == "minor_penalty")

        self.assertTrue(
            render_rubric_item_for_judge(positive).startswith(
                "Criterion type: POSITIVE REQUIREMENT"
            )
        )
        self.assertTrue(
            render_rubric_item_for_judge(major).startswith(
                "Criterion type: NEGATIVE ERROR CRITERION"
            )
        )
        self.assertTrue(
            render_rubric_item_for_judge(minor).startswith(
                "Criterion type: NEGATIVE ERROR CRITERION"
            )
        )

    def test_api_o1_generation_defaults_match_local_budget(self):
        parser = build_parser()
        args = parser.parse_args(["generate-api"])
        self.assertEqual(args.max_tokens, 2048)
        self.assertEqual(args.temperature, 0.0)
        self.assertEqual(args.top_p, 1.0)
        self.assertEqual(args.max_attempts, 2)
        self.assertEqual(args.dataset_name, "ConsultRubrics_v20260504")

    def test_api_payload_omits_top_p_for_claude_route(self):
        messages = [{"role": "user", "content": "test"}]
        payload = _api_request_payload(
            API_MODEL_REGISTRY["Claude-Sonnet-4.6"],
            messages,
            temperature=0.0,
            top_p=1.0,
            max_tokens=2048,
        )
        self.assertNotIn("top_p", payload)
        self.assertEqual(payload["max_tokens"], 2048)

    def test_api_payload_keeps_top_p_for_non_claude_route(self):
        messages = [{"role": "user", "content": "test"}]
        payload = _api_request_payload(
            API_MODEL_REGISTRY["GPT-4.1"],
            messages,
            temperature=0.0,
            top_p=1.0,
            max_tokens=2048,
        )
        self.assertEqual(payload["top_p"], 1.0)

    def test_api_base_url_normalizes_chat_completion_suffix(self):
        old_key = os.environ.get("NPCBENCH_API_KEY")
        os.environ["NPCBENCH_API_KEY"] = "unit-test-key"
        try:
            args = argparse.Namespace(
                api_key_env="NPCBENCH_API_KEY",
                base_url="https://api.example/v1/chat/completions",
            )
            _, base_url = _api_credentials(args)
        finally:
            if old_key is None:
                os.environ.pop("NPCBENCH_API_KEY", None)
            else:
                os.environ["NPCBENCH_API_KEY"] = old_key
        self.assertEqual(base_url, "https://api.example/v1")

    def test_public_path_ref_redacts_absolute_local_paths(self):
        self.assertEqual(_public_path_ref("/private/models/Qwen3-30B"), "Qwen3-30B")
        self.assertEqual(_public_path_ref(r"C:\models\Qwen3-30B"), "Qwen3-30B")
        self.assertEqual(_public_path_ref("Qwen/Qwen3-30B-A3B"), "Qwen/Qwen3-30B-A3B")


if __name__ == "__main__":
    unittest.main()
