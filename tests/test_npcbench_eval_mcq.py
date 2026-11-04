import json
import tempfile
import unittest
from pathlib import Path

from npcbench_eval.mcq import load_mcq_dataset, parse_option_letter, score_mcq


class TestNPCBenchMCQEval(unittest.TestCase):
    def test_parse_option_letter_common_formats(self):
        self.assertEqual(parse_option_letter(r"\boxed{C}", {"A", "B", "C"}), "C")
        self.assertEqual(parse_option_letter("答案：B", {"A", "B", "C"}), "B")
        self.assertEqual(parse_option_letter('{"answer": "A"}', {"A", "B", "C"}), "A")
        self.assertIsNone(parse_option_letter("I cannot decide between A and B", {"A", "B", "C"}))

    def test_jsonl_mcq_response_file_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = tmp_path / "toy.jsonl"
            dataset.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "item_id": "q1",
                                "patient_id": "p1",
                                "question": "Choose.",
                                "options": {"A": "alpha", "B": "beta"},
                                "correct_option": "B",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "item_id": "q2",
                                "patient_id": "p1",
                                "question": "Choose.",
                                "options": {"A": "alpha", "B": "beta"},
                                "correct_option": "A",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            items = load_mcq_dataset(dataset)
            responses = {"q1": r"\boxed{B}", "q2": "答案：B"}
            _, metrics = score_mcq(items, responses)
            self.assertEqual(metrics["right"], 1)
            self.assertAlmostEqual(metrics["acc"], 0.5)
            self.assertAlmostEqual(metrics["patient_macro_acc"], 0.5)
            self.assertAlmostEqual(metrics["patient_full_success"], 0.0)

    def test_load_and_score_multiple_select_without_partial_credit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'multiple.jsonl'
            path.write_text(json.dumps({'id': 'm1', 'question': 'Select all applicable.',
                'question_type': 'multiple_choice', 'options': {'A': 'alpha', 'B': 'beta', 'C': 'gamma'},
                'answer': 'A+C'}) + '\n')
            items = load_mcq_dataset(path)
            self.assertEqual(items[0].answer, 'A,C')
            _, complete = score_mcq(items, {'m1': r'\boxed{C,A}'})
            _, partial = score_mcq(items, {'m1': r'\boxed{A}'})
            self.assertEqual(complete['right'], 1)
            self.assertEqual(partial['right'], 0)


if __name__ == "__main__":
    unittest.main()
