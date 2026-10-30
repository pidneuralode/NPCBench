import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from npcbench_eval.inference import Settings, asset_path, care_requests, prepare_request


class InferenceContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'a.jpg').write_bytes(b'asset-one')
        (self.root / 'b.jpg').write_bytes(b'asset-two')
        self.settings = Settings(adapter='qwen_vl', model_label='test', max_images=2)

    def test_asset_order_and_exact_bytes_are_preserved(self):
        row = {'id': 'x', 'prompt': 'question', 'images': ['b.jpg', 'a.jpg']}
        public, actual = prepare_request(row, self.root, self.settings)
        self.assertEqual(public['images'], ['b.jpg', 'a.jpg'])
        self.assertEqual([x['sha256'] for x in public['assets']], [hashlib.sha256(b'asset-two').hexdigest(), hashlib.sha256(b'asset-one').hexdigest()])
        self.assertEqual([Path(x).name for x in actual['images']], ['b.jpg', 'a.jpg'])

    def test_image_limit_rejects_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, 'no image will be discarded'):
            prepare_request({'id': 'x', 'prompt': 'q', 'images': ['a.jpg'] * 3}, self.root, self.settings)

    def test_image_bytes_must_match_the_declared_manifest(self):
        with self.assertRaisesRegex(ValueError, 'content differs'):
            prepare_request({'id': 'x', 'prompt': 'q', 'images': ['a.jpg']}, self.root, self.settings, {'a.jpg': 'wrong'})

    def test_relative_assets_cannot_escape_the_input_root(self):
        for path in [str(self.root / 'a.jpg'), '../elsewhere.jpg', 'missing.jpg']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                asset_path(self.root, path)

    def test_video_is_not_silently_dropped_by_an_image_runner(self):
        with self.assertRaisesRegex(ValueError, 'separate declared protocol'):
            prepare_request({'id': 'x', 'prompt': 'q', 'videos': ['cine.mp4']}, self.root, self.settings)

    def test_text_only_request_does_not_invent_media(self):
        public, actual = prepare_request({'id': 'x', 'prompt': 'q'}, self.root, self.settings)
        self.assertEqual(public['assets'], [])
        self.assertEqual(actual, {'prompt': 'q', 'images': []})

    def test_care_input_uses_generated_frontends_and_never_care_gold(self):
        dataset = {'episodes': [{'patient_id': 'p', 'pathology_sample_id': 'path', 'mri_sample_id': 'mri',
                                 'ajcc_v9_gold': {'N': 'N0', 'M': 'M0'}}],
                   'care': [{'patient_id': 'p', 'sample_id': 'care', 'prompt': '{{PATHOLOGY_RESULT}} {{MRI_T_STAGE}}', 'gold_letter': 'B'}]}
        predictions = [{'sample_id': 'path', 'response': '<think>1'}, {'sample_id': 'mri', 'response': '{"T_stage":"T2"}'}]
        initial = care_requests(dataset, predictions)
        changed = copy.deepcopy(dataset); changed['care'][0]['gold_letter'] = 'A'
        self.assertEqual(initial, care_requests(changed, predictions))
        self.assertEqual(initial[0]['prompt'], '病理结果未明 T2')
        self.assertNotIn('gold_letter', initial[0])

    def test_invalid_resource_or_sampling_configuration_is_rejected(self):
        for change in [{'max_new_tokens': 0}, {'max_images': -1}, {'temperature': -1}, {'top_p': 0}, {'adapter': 'missing'}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                Settings(**dict({'adapter': 'qwen_vl', 'model_label': 'test'}, **change))


if __name__ == '__main__':
    unittest.main()
