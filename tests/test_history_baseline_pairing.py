import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


scripts = Path(__file__).resolve().parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('checkpoint_for_pairing', scripts / 'run_l6_checkpoint_refresh.py')
checkpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint)
with patch.dict(sys.modules, {'run_l6_checkpoint_refresh': checkpoint}):
    spec = importlib.util.spec_from_file_location('history_runner', scripts / 'run_l6_history_refresh.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)


class HistoryPairingTests(unittest.TestCase):
    def setUp(self):
        self.request = dict(sample_id='node', patient_id='patient', prompt='clinical input', images=[])

    def test_reuse_is_independent_of_answer_quality(self):
        baseline = dict(self.request, response='<think>unfinished', finish_reason='length')
        self.assertIs(runner.matching_baseline(self.request, {'node': baseline}), baseline)

    def test_each_model_input_component_must_match(self):
        baseline = dict(self.request, response='A')
        for key, value in [('sample_id', 'other'), ('patient_id', 'other'), ('prompt', 'changed'), ('images', ['changed.jpg'])]:
            request = copy.deepcopy(self.request)
            request[key] = value
            self.assertIsNone(runner.matching_baseline(request, {'node': baseline}))

    def test_absent_baseline_requires_generation(self):
        self.assertIsNone(runner.matching_baseline(self.request, {}))


if __name__ == '__main__':
    unittest.main()
