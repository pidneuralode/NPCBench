#!/usr/bin/env python3
"""Run one pinned checkpoint through 200 frontends, 217 care and 75 controls.

Each successful generation is journaled, including empty/truncated answers.
Resuming preserves those answers. GPU waiting does not preempt other jobs.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


def load(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_capacity_ready(selected, resources):
    """Check the declared resource budget without preempting another process."""
    if resources.get('mode', 'exclusive_idle') == 'exclusive_idle':
        return all(used < 1024 and utilization < 5 for _, used, utilization, _ in selected)
    if resources['mode'] != 'shared_capacity':
        raise ValueError('Unknown GPU resource policy')
    fraction = resources['gpu_memory_utilization']
    headroom = resources['headroom_mib']
    if not 0 < fraction < 1 or headroom < 0:
        raise ValueError('Invalid GPU memory budget')
    return all(total - used >= total * fraction + headroom for _, used, _, total in selected)


def make_care(root, records):
    from npcbench_eval.l6_inputs import frontend_values, render_care
    by_id = {r['sample_id']: r for r in records if r['condition'] == 'frontend'}
    episodes = {r['patient_id']: r for r in rows(root / 'data/episodes.jsonl')}
    requests = []
    for condition, name in [('revised', 'care_templates'), ('original_control', 'control_templates')]:
        for item in rows(root / 'data' / (name + '.jsonl')):
            episode = episodes[item['patient_id']]
            values = frontend_values(by_id[episode['pathology_sample_id']]['response'],
                                     by_id[episode['mri_sample_id']]['response'])
            prompt = render_care(item['prompt'], values, episode['ajcc_v9_gold'])
            requests.append({'request_id': condition + ':' + item['sample_id'],
                             'condition': condition, 'stage': 'care', 'sample_id': item['sample_id'],
                             'patient_id': item['patient_id'], 'prompt': prompt, 'images': [],
                             'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()})
    if len(requests) != 292 or len({r['request_id'] for r in requests}) != 292:
        raise ValueError('Expected 217 care and 75 separately identified controls')
    return requests


def run(args):
    root = args.root.resolve()
    sys.path.insert(0, str(root / 'code'))
    plan = load(root / 'plan.internal.json')
    resources = plan.get('resource_policy', {})
    checkpoint = Path(plan['checkpoint'])
    for entry in load(root / 'file_manifest.json'):
        if sha(root / entry['path']) != entry['sha256']:
            raise ValueError('Run bundle file changed: ' + entry['path'])
    identity, verified = load(checkpoint.parent / 'pinned_identity.json'), load(checkpoint.parent / 'download_state.json')
    if (verified['status'] != 'verified_complete' or identity['revision'] != plan['checkpoint_revision'] or
            verified['revision'] != plan['checkpoint_revision']):
        raise ValueError('Checkpoint download not verified at the selected revision')
    for item in load(checkpoint.parent / 'pinned_tree.json')['Data']['Files']:
        receipt = verified['files'][item['Path']]
        if receipt['sha256'] != item['Sha256'] or (checkpoint / item['Path']).stat().st_size != item['Size']:
            raise ValueError('Checkpoint file differs from verification receipt')
    frontend = rows(root / 'data/frontend_requests.jsonl')
    if len(frontend) != 200 or len({r['sample_id'] for r in frontend}) != 200:
        raise ValueError('Expected exactly 200 frontend requests')
    images = {image for r in frontend for image in r['images']}
    if len(images) != 1100 or not all(Path(p).is_file() for p in images):
        raise ValueError('Frontend image inventory is incomplete')
    package_versions = {name: importlib.metadata.version(name) for name in
                        ('torch', 'vllm', 'transformers', 'qwen-vl-utils', 'Pillow')}
    config = {'plan_sha256': sha(root / 'plan.internal.json'), 'file_manifest_sha256': sha(root / 'file_manifest.json'),
              'checkpoint_identity': identity, 'checkpoint_verification_sha256': sha(checkpoint.parent / 'download_state.json'),
              'runner_sha256': sha(Path(__file__)), 'python': sys.version, 'packages': package_versions,
              'gpus': args.gpus, 'port': args.port, 'answer_policy': plan['answer_policy'], 'input_policy': plan['input_policy']}
    if args.preflight_only:
        (root / 'preflight.json').write_text(json.dumps({**config, 'images': len(images), 'status': 'passed_no_model_call'}, indent=2) + '\n')
        print(json.dumps({'status': 'passed_no_model_call', 'images': len(images), 'frontends': len(frontend)}), flush=True)
        return
    out = root / 'outputs'
    out.mkdir(exist_ok=True)
    config_path = out / 'run_config.json'
    if config_path.exists() and load(config_path) != config:
        raise ValueError('Resume runtime differs; use a new run directory')
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    journal = out / 'responses.jsonl'
    records = rows(journal) if journal.exists() else []
    cached = {r['request_id']: r for r in records}
    if len(cached) != len(records) or any(r['status'] != 'ok' for r in records):
        raise ValueError('Journal must contain unique successful generations')
    state = {'status': 'waiting_for_gpus', 'expected': 492, 'completed': len(records), 'model': plan['model'],
             'started_at_unix': time.time(), 'phase': 'preflight'}

    def save():
        p = out / 'state.tmp'
        p.write_text(json.dumps(state, indent=2) + '\n')
        p.replace(out / 'state.json')

    save()
    handle = None
    try:
        gpu_ids = {int(g) for g in args.gpus.split(',')}
        deadline = time.monotonic() + args.wait_seconds
        while True:
            probe = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,memory.total',
                                    '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=15)
            gpu_state = [[int(x.strip()) for x in line.split(',')] for line in probe.stdout.splitlines()]
            selected = [r for r in gpu_state if r[0] in gpu_ids]
            if len(selected) != len(gpu_ids):
                raise ValueError('Requested GPU is absent')
            state['gpu_state'] = selected
            save()
            if gpu_capacity_ready(selected, resources):
                break
            if time.monotonic() >= deadline:
                state['status'] = 'gpu_wait_expired_no_model_call'
                return
            time.sleep(min(60, max(0, deadline - time.monotonic())))
        env = dict(CUDA_VISIBLE_DEVICES=args.gpus, tensor_parallel_size=str(len(gpu_ids)), max_image_num='10',
                   VLLM_MAX_MODEL_LEN=str(plan['max_model_len']),
                   VLLM_GPU_MEMORY_UTILIZATION=str(resources.get('gpu_memory_utilization', .78)),
                   VLLM_ENFORCE_EAGER='True', VLLM_DISABLE_CUSTOM_ALL_REDUCE='True', VLLM_DTYPE='bfloat16',
                   TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', VLLM_WORKER_MULTIPROC_METHOD='spawn',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', NCCL_IGNORE_DISABLED_P2P='1',
                   master_port=str(args.port), MASTER_PORT=str(args.port), VLLM_PORT=str(args.port),
                   NPCFULLCARE_ENABLE_VIDEO='False')
        for field in ('max_num_seqs', 'max_num_batched_tokens', 'kv_cache_memory_bytes'):
            if field in resources:
                env['VLLM_' + field.upper()] = str(resources[field])
        os.environ.update(env)
        import torch
        torch.manual_seed(0)
        state.update(status='loading_model', phase='model_load')
        save()
        wrapper = root / 'code' / plan['wrapper_file']
        spec = importlib.util.spec_from_file_location('refresh_model_wrapper', wrapper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = getattr(module, plan['class_name'])(str(checkpoint), SimpleNamespace(max_image_num=10, **plan['parameters']))
        (out / 'runtime.json').write_text(json.dumps({'packages': package_versions, 'environment': env,
            'wrapper_sha256': sha(wrapper)}, indent=2) + '\n')
        finish_records = []
        original_generate = model.llm.generate

        def audited_generate(*a, **kw):
            outputs = original_generate(*a, **kw)
            for output in outputs:
                completion = output.outputs[0]
                finish_records.append({'finish_reason': completion.finish_reason, 'stop_reason': completion.stop_reason,
                                       'output_tokens': len(completion.token_ids), 'prompt_tokens': len(output.prompt_token_ids or [])})
            return outputs

        model.llm.generate = audited_generate
        handle = journal.open('a')

        def generate(requests, phase):
            for request in requests:
                previous = cached.get(request['request_id'])
                if previous and (previous['prompt_sha256'] != request['prompt_sha256'] or previous['images'] != request['images']):
                    raise ValueError('Cached generation has different inputs')
            pending = [r for r in requests if r['request_id'] not in cached]
            state.update(status='running', phase=phase)
            save()
            for start in range(0, len(pending), plan['batch_size']):
                batch = pending[start:start + plan['batch_size']]
                finish_records.clear()
                responses = model.generate_outputs([{'prompt': r['prompt'], 'images': r['images']} for r in batch])
                if len(responses) != len(batch) or len(finish_records) != len(batch):
                    raise ValueError('Generation count differs from request count')
                for request, response, finish in zip(batch, responses, finish_records):
                    record = {**request, 'response': response, 'status': 'ok', **finish, 'finished_at_unix': time.time()}
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                    handle.flush()
                    records.append(record)
                    cached[request['request_id']] = record
                    state['completed'] += 1
                    save()
                print(json.dumps({'phase': phase, 'completed': state['completed'], 'expected': 492}), flush=True)

        generate(frontend, 'frontend')
        care = make_care(root, records)
        requests_path = out / 'care_requests.jsonl'
        requests_text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in care)
        if requests_path.exists() and requests_path.read_text() != requests_text:
            raise ValueError('Derived care requests changed')
        requests_path.write_text(requests_text)
        generate(care, 'care_and_controls')
        if len(cached) != 492:
            raise ValueError('Unexpected final response inventory')
        state.update(status='completed', responses_sha256=sha(journal), care_requests_sha256=sha(requests_path))
    except BaseException as error:
        state.update(status='failed_preserved', error_type=type(error).__name__, error=str(error)[:300])
        raise
    finally:
        if handle is not None:
            handle.close()
        state['finished_at_unix'] = time.time()
        save()
        print(json.dumps(state), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--gpus', default='0,1,2,3')
    ap.add_argument('--port', type=int, default=59223)
    ap.add_argument('--wait-seconds', type=int, default=14400)
    ap.add_argument('--preflight-only', action='store_true')
    run(ap.parse_args())
