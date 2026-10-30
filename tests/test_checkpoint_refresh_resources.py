import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location('checkpoint_runner', Path(__file__).resolve().parents[1] / 'scripts/run_l6_checkpoint_refresh.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class GPUReadinessTests(unittest.TestCase):
    def test_exclusive_policy_requires_idle_device(self):
        self.assertTrue(runner.gpu_capacity_ready([[0, 1000, 0, 81920]], {}))
        self.assertFalse(runner.gpu_capacity_ready([[0, 1000, 100, 81920]], {}))
        self.assertFalse(runner.gpu_capacity_ready([[0, 1024, 0, 81920]], {}))

    def test_shared_policy_reserves_headroom_on_every_device(self):
        policy = {'mode': 'shared_capacity', 'gpu_memory_utilization': .25, 'headroom_mib': 4096}
        self.assertTrue(runner.gpu_capacity_ready([[0, 57344, 100, 81920]], policy))
        self.assertFalse(runner.gpu_capacity_ready([[0, 57344, 100, 81920], [1, 57345, 0, 81920]], policy))

    def test_invalid_budget_rejected(self):
        with self.assertRaisesRegex(ValueError, 'budget'):
            runner.gpu_capacity_ready([[0, 0, 0, 81920]], {'mode': 'shared_capacity', 'gpu_memory_utilization': 1.1, 'headroom_mib': 0})


if __name__ == '__main__':
    unittest.main()
