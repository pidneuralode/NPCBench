"""Release regressions: preserve denominators, macro weights and missing cells."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    'current_primary', Path(__file__).resolve().parents[1] / 'scripts/reproduce_current_primary.py')
current = importlib.util.module_from_spec(spec)
spec.loader.exec_module(current)


class CurrentPrimaryTests(unittest.TestCase):
    def test_capability_groups_are_not_item_weighted(self):
        groups = ['a', 'a', 'a', 'b', 'c', 'd']
        data = [{'id': str(i), 'answer': 'A', 'capability_group': group} for i, group in enumerate(groups)]
        decisions = [{'id': str(i), 'prediction': 'A' if i < 4 else 'B', 'correct': i < 4} for i in range(6)]
        self.assertEqual(current.independent_aggregate('L2', data, decisions), {'A-App.': 50})

    def test_patient_macro_preserves_unequal_depth(self):
        data = [{'id': str(i), 'answer': 'A', 'patient_id': 'p1' if i < 3 else 'p2'} for i in range(4)]
        decisions = [{'id': str(i), 'prediction': 'A' if i < 3 else None, 'correct': i < 3} for i in range(4)]
        self.assertEqual(current.independent_aggregate('L4', data, decisions), {'C-Plan': 50})

    def test_longitudinal_invalid_answer_stays_in_episode(self):
        data = [{'id': str(i), 'answer': 'A', 'patient_id': 'p1' if i < 2 else 'p2'} for i in range(4)]
        decisions = [{'id': str(i), 'prediction': 'A' if i < 3 else None, 'correct': i < 3} for i in range(4)]
        self.assertEqual(current.independent_aggregate('L5', data, decisions),
                         {'L-Rev. Step': 75, 'L-Rev. Epi.': 50})

    def test_missing_or_duplicate_decisions_are_not_silently_removed(self):
        data = [{'id': 'a', 'answer': 'A'}, {'id': 'b', 'answer': 'B'}]
        one = {'id': 'a', 'prediction': 'A', 'correct': True}
        for decisions in ([one], [one, one]):
            with self.subTest(decisions=decisions), self.assertRaises(ValueError):
                current.independent_aggregate('L1', data, decisions)

    def test_stored_correctness_cannot_override_prediction(self):
        with self.assertRaises(ValueError):
            current.independent_aggregate('L1', [{'id': 'a', 'answer': 'A'}],
                                          [{'id': 'a', 'prediction': 'B', 'correct': True}])

    def test_reference_values_and_rendered_precision_are_both_checked(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with (root / 'cell_ledger.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['model', 'column', 'aligned_value', 'aligned_display'])
                writer.writeheader()
                writer.writerow(dict(model='model', column='G-Mem.', aligned_value=50.04, aligned_display='50.0'))
            (root / 'main_table_candidate.csv').write_text('Family,Model,G-Mem.\nfamily,model,50.0\n')
            current.compare_reference({('model', 'G-Mem.'): 50.04}, root)
            for values in ({}, {('model', 'G-Mem.'): 50.0}):
                with self.assertRaises(ValueError):
                    current.compare_reference(values, root)
            (root / 'main_table_candidate.csv').write_text('Family,Model,G-Mem.\nfamily,model,50.1\n')
            with self.assertRaises(ValueError):
                current.compare_reference({('model', 'G-Mem.'): 50.04}, root)


if __name__ == '__main__':
    unittest.main()
