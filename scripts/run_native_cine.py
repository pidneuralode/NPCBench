#!/usr/bin/env python3
"""Generate from explicit relative-path cine requests using installed adapters.

This runner preserves native-video inputs. It does not implement the separate
API frame-sampling protocol or calculate a benchmark score.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from npcbench_eval.inference import (
    ADAPTERS, Settings, asset_path, checkpoint_identity, environment,
    package_versions, rows, save, sha,
)


def prepare(row, root):
    if not row.get('sample_id') or not isinstance(row.get('prompt'), str):
        raise ValueError('Cine requests require sample_id and prompt')
    videos = row.get('videos', [])
    if not isinstance(videos, list) or not 1 <= len(videos) <= 4 or row.get('images'):
        raise ValueError('Declare one to four videos and no image inputs')
    assets = row['assets']
    references = videos + [row['video_manifest']]
    if set(assets) != set(references):
        raise ValueError('Every video and its frame manifest need a SHA-256')
    resolved = {}
    for relative in references:
        path = asset_path(root, relative)
        if sha(path) != assets[relative]:
            raise ValueError('Video or frame-manifest hash mismatch')
        resolved[relative] = str(path)
    return {'prompt': row['prompt'], 'videos': [resolved[p] for p in videos],
            'video_manifest': resolved[row['video_manifest']]}


def run(args):
    settings = Settings(**json.loads(args.config.read_text()))
    if settings.adapter not in ('qwen_vl', 'qwen25_vl'):
        raise ValueError('This native-video protocol requires a Qwen VL adapter')
    requested = rows(args.requests)
    prepared = [(r, prepare(r, args.asset_root.resolve())) for r in requested]
    if not requested or len({r['sample_id'] for r in requested}) != len(requested):
        raise ValueError('Requests must be nonempty and unique')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'fingerprinting_checkpoint', 'completed': 0,
             'expected': len(prepared), 'started_at_unix': time.time()}
    save(output / 'state.json', state)
    try:
        save(output / 'checkpoint_identity.json', checkpoint_identity(args.model_path.resolve()))
        env = environment(settings)
        env.update(NPCFULLCARE_ENABLE_VIDEO='True', NPCFULLCARE_CINE_INPUT='separate',
                   NPCFULLCARE_MAX_VIDEO_NUM='4', NPCFULLCARE_VIDEO_MIN_PIXELS=str(16 * 28 * 28),
                   NPCFULLCARE_VIDEO_MAX_PIXELS=str(256 * 28 * 28))
        os.environ.update(env)
        module_name, class_name = ADAPTERS[settings.adapter]
        module = importlib.import_module('npcbench_eval.inference_adapters.' + module_name)
        save(output / 'run.json', {
            'scope': 'explicit_native_cine_diagnostic', 'settings': asdict(settings),
            'packages': package_versions(), 'environment': env,
            'runner_sha256': sha(__file__), 'adapter_sha256': sha(module.__file__),
            'checkpoint_identity_sha256': sha(output / 'checkpoint_identity.json'),
            'requests_sha256': sha(args.requests), 'config_sha256': sha(args.config),
            'installed_adapter': 'site-packages' in Path(module.__file__).parts,
            'source_frame_rule': 'archived wrapper: requested count capped at largest even decodable count',
        })
        import torch
        torch.manual_seed(settings.seed)
        state['status'] = 'loading_model'; save(output / 'state.json', state)
        model = getattr(module, class_name)(str(args.model_path.resolve()), SimpleNamespace(
            max_image_num=settings.max_images, max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature, top_p=settings.top_p,
            repetition_penalty=settings.repetition_penalty))
        with (output / 'predictions.jsonl').open('w') as journal:
            for row, actual in prepared:
                state['status'] = 'running'; save(output / 'state.json', state)
                started = time.time()
                encoded = model.process_messages(actual)
                delivered = encoded.get('multi_modal_data', {}).get('video', [])
                if len(delivered) != len(row['videos']):
                    raise ValueError('Adapter did not deliver every declared video')
                traces = []
                for item in delivered:
                    tensor, metadata = item if isinstance(item, tuple) else (item, {})
                    total_frames = metadata.get('total_num_frames')
                    traces.append({'tensor_shape': list(tensor.shape),
                                   'frame_indices': [int(v) for v in metadata.get('frames_indices', [])],
                                   'source_total_frames': int(total_frames) if total_frames is not None else None})
                expected = row['expected_delivered_frames']
                if [t['tensor_shape'][0] for t in traces] != expected:
                    raise ValueError('Delivered frame counts differ from declared protocol')
                result, = model.llm.generate([encoded], sampling_params=model.sampling_params)
                completion = result.outputs[0]
                record = dict(row, response=completion.text, model=settings.model_label,
                              finish_reason=completion.finish_reason, stop_reason=completion.stop_reason,
                              generation_status='ok', output_tokens=len(completion.token_ids),
                              completion_token_ids=list(completion.token_ids),
                              prompt_tokens=len(result.prompt_token_ids or []), media_trace=traces,
                              rendered_prompt_sha256=hashlib.sha256(encoded['prompt'].encode()).hexdigest(),
                              elapsed_seconds=time.time() - started)
                journal.write(json.dumps(record, ensure_ascii=False) + '\n'); journal.flush()
                state['completed'] += 1; save(output / 'state.json', state)
        state.update(status='completed', predictions_sha256=sha(output / 'predictions.jsonl'))
    except BaseException as error:
        state.update(status='failed_preserved', error_type=type(error).__name__, error=str(error)[:400])
        raise
    finally:
        state['finished_at_unix'] = time.time(); save(output / 'state.json', state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'model-path', 'requests', 'asset-root', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == '__main__':
    main()
