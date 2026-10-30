#!/usr/bin/env python3
"""Generate both complete original25 history policies under one current runtime."""
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

from run_l6_checkpoint_refresh import gpu_capacity_ready


def load(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def matching_baseline(request, baseline):
    """Pair the same clinical node only when all actual model inputs match."""
    row = baseline.get(request['sample_id'])
    return row if row and all(request[k] == row[k] for k in ('sample_id', 'patient_id', 'prompt', 'images')) else None


def run(args):
    root = args.root.resolve()
    sys.path.insert(0, str(root / 'code'))
    from npcbench_eval.l6_history import action_from_response, render_history
    plan = load(root / 'plan.internal.json')
    for entry in load(root / 'file_manifest.json'):
        if sha(root / entry['path']) != entry['sha256']:
            raise ValueError('Run bundle changed: ' + entry['path'])
    episodes = rows(root / 'data/episodes.jsonl')
    if len(episodes) != 25 or sum(len(e['turns']) for e in episodes) != 146:
        raise ValueError('Expected original25/146 input inventory')
    checkpoint = Path(plan['checkpoint'])
    historical = load(root / 'historical_manifest.internal.json')['checkpoint_identity']
    for name, digest in historical['metadata_hashes'].items():
        if sha(checkpoint / name) != digest:
            raise ValueError('Checkpoint metadata differs from historical manifest: ' + name)
    for entry in historical['weight_inventory']:
        if (checkpoint / entry['name']).stat().st_size != entry['size_bytes']:
            raise ValueError('Checkpoint size inventory differs from historical manifest')
    config = {'plan_sha256': sha(root / 'plan.internal.json'), 'bundle_sha256': sha(root / 'file_manifest.json'),
              'runner_sha256': sha(Path(__file__)), 'python': sys.version,
              'packages': {p: importlib.metadata.version(p) for p in ('torch', 'vllm', 'transformers', 'qwen-vl-utils', 'Pillow')},
              'port': args.port}
    baseline = {}
    if plan.get('paired_baseline'):
        baseline_rows = rows(root / 'data/paired_baseline.jsonl')
        baseline_config = load(root / 'data/paired_baseline_config.json')
        baseline_plan = load(root / 'data/paired_baseline_plan.json')
        baseline = {r['sample_id']: r for r in baseline_rows}
        expected_ids = {t['sample_id'] for e in episodes for t in e['turns']}
        if (len(baseline_rows) != 146 or set(baseline) != expected_ids or
                any(r['condition'] != 'frozen_history_v1' or r['status'] != 'ok' for r in baseline_rows) or
                baseline_config['packages'] != config['packages'] or baseline_plan['parameters'] != plan['parameters']):
            raise ValueError('Paired baseline membership, runtime or decoding differs')
    if args.preflight_only:
        (root / 'preflight.json').write_text(json.dumps({**config, 'status': 'passed_no_model_call', 'requests': 292}, indent=2) + '\n')
        print(json.dumps({'status': 'passed_no_model_call', 'requests': 292}), flush=True)
        return
    out = root / 'outputs'; out.mkdir(exist_ok=True)
    if (out / 'run_config.json').exists() and load(out / 'run_config.json') != config:
        raise ValueError('Resume configuration differs; use a new run directory')
    (out / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
    journal = out / 'responses.jsonl'
    if baseline and not journal.exists():
        journal.write_text(''.join(json.dumps(dict(r, response_origin='paired_frozen_baseline'), ensure_ascii=False) + '\n'
                                   for r in baseline.values()))
    records = rows(journal) if journal.exists() else []
    cached = {r['request_id']: r for r in records}
    if len(cached) != len(records) or any(r['status'] != 'ok' for r in records):
        raise ValueError('Journal contains duplicated or unsuccessful generation records')
    state = {'model': plan['model'], 'status': 'checkpoint_hashing', 'expected': 292,
             'completed': len(records), 'started_at_unix': time.time()}

    def save():
        temp = out / 'state.tmp'; temp.write_text(json.dumps(state, indent=2) + '\n'); temp.replace(out / 'state.json')

    save(); handle = None
    try:
        checkpoint_receipt = {'historical_metadata_and_sizes_match': True, 'historical_full_weight_hash_was_recorded': False,
                              'current_weight_sha256': {e['name']: sha(checkpoint / e['name']) for e in historical['weight_inventory']}}
        if baseline and checkpoint_receipt['current_weight_sha256'] != load(root / 'data/paired_baseline_checkpoint.json')['current_weight_sha256']:
            raise ValueError('Current weights differ from the paired baseline bytes')
        (out / 'checkpoint_identity.json').write_text(json.dumps(checkpoint_receipt, indent=2) + '\n')
        deadline = time.monotonic() + args.wait_seconds
        gpus = {int(x) for x in plan['gpus'].split(',')}
        while True:
            dependency = Path(plan['after_run']) / 'outputs/state.json'
            status = load(dependency)['status'] if dependency.exists() else 'not_observed'
            state.update(status='waiting_for_prior_run', prior_run_status=status); save()
            if status in ('completed', 'failed_preserved', 'gpu_wait_expired_no_model_call'):
                probe = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,memory.total', '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=15)
                selected = [[int(x.strip()) for x in line.split(',')] for line in probe.stdout.splitlines()]
                selected = [r for r in selected if r[0] in gpus]
                if len(selected) != len(gpus):
                    raise ValueError('Requested GPU absent')
                state.update(status='waiting_for_gpus', gpu_state=selected); save()
                if gpu_capacity_ready(selected, plan['resource_policy']):
                    break
            if time.monotonic() >= deadline:
                state['status'] = 'gpu_wait_expired_no_model_call'; return
            time.sleep(min(60, max(0, deadline - time.monotonic())))
        resources = plan['resource_policy']
        env = dict(CUDA_VISIBLE_DEVICES=plan['gpus'], tensor_parallel_size=str(len(gpus)), max_image_num='0',
                   VLLM_MAX_MODEL_LEN=str(plan['max_model_len']), VLLM_GPU_MEMORY_UTILIZATION=str(resources['gpu_memory_utilization']),
                   VLLM_MAX_NUM_SEQS=str(resources['max_num_seqs']), VLLM_MAX_NUM_BATCHED_TOKENS=str(resources['max_num_batched_tokens']),
                   VLLM_KV_CACHE_MEMORY_BYTES=str(resources['kv_cache_memory_bytes']), VLLM_ENFORCE_EAGER='True',
                   VLLM_DISABLE_CUSTOM_ALL_REDUCE='True', VLLM_DTYPE='bfloat16', TOKENIZERS_PARALLELISM='false',
                   OMP_NUM_THREADS='4', VLLM_WORKER_MULTIPROC_METHOD='spawn', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   NCCL_IGNORE_DISABLED_P2P='1', master_port=str(args.port), MASTER_PORT=str(args.port), VLLM_PORT=str(args.port),
                   NPCFULLCARE_ENABLE_VIDEO='False')
        os.environ.update(env)
        import torch
        torch.manual_seed(0)
        state['status'] = 'loading_model'; save()
        spec = importlib.util.spec_from_file_location('history_wrapper', root / 'code/qwen35_36_mllm_vllm.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        model = module.Qwen35_36_MLLM(str(checkpoint), SimpleNamespace(max_image_num=0, **plan['parameters']))
        (out / 'runtime.json').write_text(json.dumps({'environment': env, 'packages': config['packages']}, indent=2) + '\n')
        original_generate, finishes = model.llm.generate, []

        def audited_generate(*a, **kw):
            result = original_generate(*a, **kw)
            for r in result:
                c = r.outputs[0]
                finishes.append({'finish_reason': c.finish_reason, 'stop_reason': c.stop_reason,
                                 'output_tokens': len(c.token_ids), 'prompt_tokens': len(r.prompt_token_ids or [])})
            return result

        model.llm.generate = audited_generate
        handle = journal.open('a')
        histories = {(condition, e['patient_id']): [] for condition in plan['conditions'] for e in episodes}
        state['status'] = 'running'; save()
        for position in range(max(len(e['turns']) for e in episodes)):
            wave = []
            for condition in plan['conditions']:
                for episode in sorted(episodes, key=lambda e: e['patient_id']):
                    if position >= len(episode['turns']):
                        continue
                    turn = episode['turns'][position]
                    prompt = render_history(turn['base_prompt'], histories[condition, episode['patient_id']])
                    request = {'request_id': condition + ':' + turn['sample_id'], 'condition': condition,
                               'sample_id': turn['sample_id'], 'patient_id': episode['patient_id'], 'turn_position': position + 1,
                               'prompt': prompt, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'images': []}
                    previous = cached.get(request['request_id'])
                    if previous and previous['prompt_sha256'] != request['prompt_sha256']:
                        raise ValueError('Resumed patient history does not match recorded prompt')
                    wave.append((request, turn))
            pending = [r for r, _ in wave if r['request_id'] not in cached]
            shared = [(r, matching_baseline(r, baseline)) for r in pending] if baseline else []
            for request, source in shared:
                if source is None:
                    continue
                record = {**request, **{key: source[key] for key in ('response', 'status', 'finish_reason', 'stop_reason', 'output_tokens', 'prompt_tokens')},
                          'response_origin': 'paired_identical_input_reuse', 'shared_from_request_id': source['request_id'],
                          'finished_at_unix': time.time()}
                handle.write(json.dumps(record, ensure_ascii=False) + '\n'); handle.flush()
                cached[request['request_id']] = record; state['completed'] += 1; save()
            pending = [r for r in pending if r['request_id'] not in cached]
            for start in range(0, len(pending), plan['batch_size']):
                batch = pending[start:start + plan['batch_size']]; finishes.clear()
                responses = model.generate_outputs([{'prompt': r['prompt'], 'images': []} for r in batch])
                if len(responses) != len(batch) or len(finishes) != len(batch):
                    raise ValueError('Response inventory mismatch')
                for request, response, finish in zip(batch, responses, finishes):
                    record = {**request, 'response': response, 'status': 'ok', **finish, 'finished_at_unix': time.time(),
                              'response_origin': 'model_generation'}
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n'); handle.flush()
                    cached[record['request_id']] = record
                    state['completed'] += 1; save()
                print(json.dumps({'completed': state['completed'], 'expected': 292}), flush=True)
            for request, turn in wave:
                record = cached[request['request_id']]
                histories[request['condition'], request['patient_id']].append(action_from_response(record['response'], turn['options'], request['condition']))
        if len(cached) != 292:
            raise ValueError('Expected two complete 146-turn conditions')
        state.update(status='completed', responses_sha256=sha(journal),
                     model_generations=sum(r.get('response_origin') == 'model_generation' for r in cached.values()),
                     baseline_responses=sum(r.get('response_origin') == 'paired_frozen_baseline' for r in cached.values()),
                     shared_input_responses=sum(r.get('response_origin') == 'paired_identical_input_reuse' for r in cached.values()))
    except BaseException as error:
        state.update(status='failed_preserved', error_type=type(error).__name__, error=str(error)[:300]); raise
    finally:
        if handle is not None:
            handle.close()
        state['finished_at_unix'] = time.time(); save(); print(json.dumps(state), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--wait-seconds', type=int, default=14400)
    ap.add_argument('--preflight-only', action='store_true')
    run(ap.parse_args())
