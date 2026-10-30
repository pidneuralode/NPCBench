import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_native_cine.py'
SPEC = importlib.util.spec_from_file_location('native_cine_contract', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NativeCineInputTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        payloads = {'clip.mp4': b'video-contract-fixture', 'frames.json': b'{"frame_count":12}'}
        for name, value in payloads.items():
            (self.root / name).write_bytes(value)
        self.row = {'sample_id': 'synthetic', 'prompt': 'q', 'videos': ['clip.mp4'],
                    'video_manifest': 'frames.json',
                    'assets': {k: hashlib.sha256(v).hexdigest() for k, v in payloads.items()}}

    def test_video_and_manifest_both_resolve_inside_asset_root(self):
        actual = MODULE.prepare(self.row, self.root)
        self.assertEqual(actual['videos'], [str(self.root / 'clip.mp4')])
        self.assertEqual(actual['video_manifest'], str(self.root / 'frames.json'))

    def test_mutated_frame_manifest_is_rejected_before_model_loading(self):
        (self.root / 'frames.json').write_text('{"frame_count":2}')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            MODULE.prepare(self.row, self.root)

    def test_image_video_protocols_are_not_silently_combined(self):
        self.row['images'] = ['unrelated.jpg']
        with self.assertRaisesRegex(ValueError, 'no image inputs'):
            MODULE.prepare(self.row, self.root)


if __name__ == '__main__':
    unittest.main()
