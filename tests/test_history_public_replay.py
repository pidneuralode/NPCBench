"""Regression checks for observed Python 3.10/3.12 replay roundoff."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'scripts/check_l6_history_sensitivity.py'
spec = importlib.util.spec_from_file_location('history_public_replay', path)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class HistoryPublicReplayTests(unittest.TestCase):
    def test_observed_python_sum_roundoff_is_accepted(self):
        actual = {'patients': 25, 'ci': [-0.04266904761904761, 0.12333333333333331]}
        expected = {'patients': 25, 'ci': [-0.042669047619047606, 0.12333333333333334]}
        self.assertTrue(replay.same_metrics(actual, expected))

    def test_metric_changes_are_not_hidden_by_tolerance(self):
        self.assertFalse(replay.same_metrics({'score': 0.2}, {'score': 0.20000001}))
        self.assertFalse(replay.same_metrics({'score': float('nan')}, {'score': 0.2}))

    def test_denominators_and_condition_membership_remain_exact(self):
        self.assertFalse(replay.same_metrics({'patients': 25}, {'patients': 24}))
        self.assertFalse(replay.same_metrics([{'patients': 25}], []))
        self.assertFalse(replay.same_metrics({'patients': 25}, {'patient_count': 25}))
        self.assertFalse(replay.same_metrics({'patients': 25.0}, {'patients': 25}))
