import base64
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import urllib.error

from npcbench_eval.inference import Settings, replace_mri_prompts
from npcbench_eval.mri_extension import prepare, score, select_groups


class MRIExtensionTests(unittest.TestCase):
    def test_exhausted_provider_budget_stops_without_retrying_or_lowering_claude_concurrency(self):
        from npcbench_eval.api_mri import run_model
        with tempfile.TemporaryDirectory() as name:
            args = SimpleNamespace(output=Path(name), data_root=Path(name), claude_fixed8=True,
                                   concurrency=16, attempts=5, timeout=1)
            row = {'request_id': 'a', 'subset': 'N400', 'images': [], 'prompt': 'q',
                   'image_profile': 'plane_groups_lossless_png_v1', 'result_field': 'n_stage'}
            error = urllib.error.HTTPError('https://example.invalid/chat/completions', 429, 'budget', {},
                                           BytesIO(b'{"error":{"type":"budget_exceeded"}}'))
            with patch('urllib.request.urlopen', side_effect=error) as call:
                result = run_model('Claude-Sonnet-4.6', [row], args,
                                   {'base_url': 'https://example.invalid', 'keys': ['dummy']}, None, None)
            self.assertEqual(call.call_count, 1)
            self.assertEqual(result['status'], 'blocked_on_budget')
            self.assertEqual(result['active_concurrency_limit'], 8)
            self.assertEqual(result['http_completed'], 0)

    def test_continuous_generation_bounds_host_queue_without_dropping_answers(self):
        from npcbench_eval.bounded_inference import vllm_batch
        class Engine:
            def __init__(self):
                self.pending = []
                self.peak = 0
                self.steps = 0
                self.request_counter = SimpleNamespace(counter=0)
                self.llm_engine = self
            def enqueue(self, inputs, **kwargs):
                ids = []
                for value in inputs:
                    rid = str(self.request_counter.counter)
                    self.request_counter.counter += 1
                    self.pending.append(rid)
                    ids.append(rid)
                self.peak = max(self.peak, len(self.pending))
                return ids
            def has_unfinished_requests(self):
                return bool(self.pending)
            def step(self):
                self.steps += 1
                if self.steps % 2:
                    return []
                rid = self.pending.pop(0)
                return [SimpleNamespace(finished=True, request_id=rid, prompt_token_ids=[1],
                        outputs=[SimpleNamespace(text=rid, finish_reason='stop', stop_reason=None, token_ids=[2])])]
        engine = Engine()
        model = SimpleNamespace(llm=engine, sampling_params=None, process_messages=lambda row: row)
        batch = [({'sample_id': str(i)}, {'prompt': 'q', 'images': []}) for i in range(120)]
        emitted = []
        results = vllm_batch(model, batch, emit=lambda row, value: emitted.append(row['sample_id']))
        self.assertEqual(engine.peak, 48)
        self.assertEqual(emitted, [str(i) for i in range(120)])
        self.assertEqual([r['response'] for r in results], emitted)

    def test_sampling_preserves_both_groups_and_endpoints_for_all_cohort_sizes(self):
        for axial in range(20, 51):
            for coronal in range(12, 30):
                images = [f'case/axial_{i}.png' for i in range(axial)] + [f'case/coronal_{i}.png' for i in range(coronal)]
                for count in [min(33, len(images)), min(50, len(images))]:
                    picked = select_groups(images, count)
                    self.assertEqual(picked, sorted(set(picked)))
                    self.assertEqual(len(picked), count)
                    self.assertTrue({0, axial - 1, axial, len(images) - 1} <= set(picked))

    def test_gpt41_sampling_changes_counts_not_task_instruction(self):
        row = {'images': [f'case/axial_{i}.png' for i in range(40)] + [f'case/coronal_{i}.png' for i in range(24)],
               'prompt': '共 64 张图，前 40 张轴位，后 24 张冠状位。请先判断 N 分期。'}
        result = prepare(row, 'gpt41')
        self.assertEqual(len(result['images']), 50)
        self.assertIn('共 50 张图', result['prompt'])
        self.assertEqual(result['prompt'].split('请先')[1], row['prompt'].split('请先')[1])
        self.assertEqual(prepare(dict(row, images=row['images'][:50]), 'gpt41')['images'], row['images'][:50])

    def test_invalid_answers_count_as_incorrect_and_missing_requests_are_distinct(self):
        labels = [{'id': 'a', 'n_stage': 'N0'}, {'id': 'b', 'n_stage': 'N2'}]
        preds = [{'id': 'a', 'response': '{"n_stage":"N0"}'}, {'id': 'b', 'response': 'Unable to determine'}]
        result = score(labels, preds, 'N')
        self.assertEqual((result['cases'], result['correct'], result['invalid'], result['accuracy']), (2, 1, 1, 0.5))
        with self.assertRaisesRegex(ValueError, 'complete, matching'):
            score(labels, preds[:1], 'N')
        with self.assertRaisesRegex(ValueError, 'unique IDs'):
            score(labels, preds + preds[:1], 'N')

    def test_schema_replacement_cannot_change_reference_or_image_membership(self):
        row = {'sample_id': 'm', 'patient_id': 'p', 'prompt': 'old', 'images': ['m.jpg'], 'answer': 'T2'}
        data = {'mri': [row]}
        result = replace_mri_prompts(data, [dict(row, prompt='new schema')])
        self.assertEqual(result['mri'][0]['prompt'], 'new schema')
        self.assertEqual(data['mri'][0]['prompt'], 'old')
        with self.assertRaisesRegex(ValueError, 'cannot change'):
            replace_mri_prompts(data, [dict(row, answer='T3')])

    def test_all_30_profiles_load_and_keep_supported_contexts(self):
        files = list((Path(__file__).parents[1] / 'configs/mri_extension').glob('*/*.json'))
        self.assertEqual(len(files), 30)
        for file in files:
            config = Settings(**json.loads(file.read_text()))
            self.assertEqual(config.max_new_tokens, 8192)
            self.assertLessEqual(config.max_model_len, 32768 if config.model_label in ['Hulu-Med-7B', 'Hulu-Med-32B'] else 40960 if config.adapter == 'internvl' else 128000)

    def test_api_keeps_lossless_image_bytes_and_order(self):
        from PIL import Image
        from npcbench_eval.api_mri import make_messages
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            Image.new('RGB', (17, 31), 'red').save(root / 'a.png')
            Image.new('RGB', (19, 29), 'blue').save(root / 'b.png')
            row = {'image_profile': 'plane_groups_lossless_png_v1', 'images': ['b.png', 'a.png'], 'prompt': 'question'}
            messages, meta = make_messages(row, root, None)
            content = messages[0]['content']
            self.assertEqual(content[-1], {'type': 'text', 'text': 'question'})
            for index, filename in enumerate(row['images']):
                encoded = content[index * 2 + 1]['image_url']['url'].split(',', 1)[1]
                self.assertEqual(base64.b64decode(encoded), (root / filename).read_bytes())
                self.assertEqual(meta[index]['original_size'], meta[index]['encoded_size'])


if __name__ == '__main__':
    unittest.main()
