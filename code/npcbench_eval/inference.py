"""Run archived local adapters from installed code and public relative-path inputs."""
import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from . import final_answer, l6_inputs
from .l6 import load_dataset, score, validate_membership

ADAPTERS = {
    'qwen_vl': ('qwen_vl', 'Qwen35_36_MLLM'),
    'qwen25_vl': ('qwen25_vl', 'Qwen2_5_VL'),
    'internvl': ('internvl', 'InternVL'),
    'hulumed': ('hulumed', 'HuluMed'),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


@dataclass(frozen=True)
class Settings:
    adapter: str
    model_label: str
    tensor_parallel_size: int = 1
    max_model_len: int = 50000
    max_images: int = 32
    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 0.0001
    repetition_penalty: float = 1.0
    gpu_memory_utilization: float = 0.75
    kv_cache_memory_bytes: int = 4294967296
    max_num_seqs: int = 1
    max_num_batched_tokens: int = 8192
    internvl_max_patches: int = 4
    seed: int = 0
    distributed_port: int = 29500
    batch_inference: bool = False
    mm_processor_kwargs: dict = field(default_factory=dict)
    hulu_merge_size: int = 0
    hulu_visual_budget: int = 16384

    def __post_init__(self):
        if self.adapter not in ADAPTERS or not self.model_label:
            raise ValueError('Choose a registered adapter and a model label')
        if min(self.tensor_parallel_size, self.max_model_len, self.max_new_tokens,
               self.kv_cache_memory_bytes, self.max_num_seqs, self.max_num_batched_tokens) < 1 or self.max_images < 0:
            raise ValueError('Invalid resource or output budget')
        if self.temperature < 0 or not 0 < self.top_p <= 1 or not 0 < self.gpu_memory_utilization < 1:
            raise ValueError('Invalid sampling or GPU memory setting')


def asset_path(root, relative):
    path = Path(relative)
    if path.is_absolute():
        raise ValueError('Input assets must use paths relative to --asset-root or --l6-inputs')
    resolved = (root / path).resolve()
    if root.resolve() not in resolved.parents or not resolved.is_file():
        raise ValueError('Image asset is missing or outside the supplied root')
    return resolved


def prepare_request(row, asset_root, settings, expected_hashes=None):
    sid = row.get('sample_id', row.get('id'))
    if not isinstance(sid, str) or not sid or not isinstance(row.get('prompt'), str):
        raise ValueError('Each request needs a nonempty sample_id/id and a text prompt')
    if row.get('messages') or row.get('videos') or row.get('image'):
        raise ValueError('This image/text entry point accepts prompt + images; other modalities need a separate declared protocol')
    images = row.get('images', [])
    if not isinstance(images, list) or len(images) > settings.max_images:
        raise ValueError('Image count exceeds the declared limit; no image will be discarded')
    assets, actual = [], []
    for relative in images:
        path = asset_path(asset_root, relative)
        digest = sha(path)
        if expected_hashes is not None and expected_hashes.get(relative) != digest:
            raise ValueError('Image content differs from the public input manifest')
        assets.append({'path': relative, 'sha256': digest, 'bytes': path.stat().st_size})
        actual.append(str(path))
    record = {'sample_id': sid, 'id': sid, 'patient_id': row.get('patient_id'),
            'stage': row.get('stage', row.get('task', 'generic')), 'prompt': row['prompt'],
            'prompt_sha256': hashlib.sha256(row['prompt'].encode()).hexdigest(), 'images': images,
            'assets': assets}
    record.update({key: row[key] for key in ['input_condition', 'image_sampling'] if key in row})
    return record, {'prompt': row['prompt'], 'images': actual}


def choose_l6_patients(data, limit=None):
    if limit is None:
        return data
    if not 1 <= limit <= len(data['episodes']):
        raise ValueError('Diagnostic patient limit is outside the available cohort')
    patients = {r['patient_id'] for r in sorted(data['episodes'], key=lambda x: x['patient_id'])[:limit]}
    result = {k: [r for r in data[k] if r['patient_id'] in patients] for k in ('episodes', 'pathology', 'mri', 'care', 'unscored')}
    selected = {r['sample_id'] for r in result['care']}
    hard = [sid for sid in data['hardcore'] if sid in selected]
    contract = dict(data['contract'])
    contract.update(patients=len(patients), scored_care_nodes=len(selected),
                    unscored_pending_review_nodes=len(result['unscored']), hardcore_nodes=len(hard),
                    scored_ids=sorted(selected), excluded_ids=sorted(r['sample_id'] for r in result['unscored']),
                    metric_denominators={'care_micro': len(selected), 'patient_all_care': len(patients),
                                         'strict_episode': len(patients), 'pathology': len(patients),
                                         'MRI_T': len(patients), 'hardcore_care': len(hard)})
    result.update(contract=contract, hardcore=hard)
    validate_membership(result)
    return result


def replace_mri_prompts(data, replacements):
    indexed = {r['sample_id']: r for r in replacements}
    if len(indexed) != len(replacements) or set(indexed) != {r['sample_id'] for r in data['mri']}:
        raise ValueError('MRI prompt replacement must cover the same unique patients')
    for row in data['mri']:
        other = indexed[row['sample_id']]
        if {k: v for k, v in row.items() if k != 'prompt'} != {k: v for k, v in other.items() if k != 'prompt'}:
            raise ValueError('MRI prompt replacement cannot change labels, images or cohort fields')
    return dict(data, mri=[indexed[r['sample_id']] for r in data['mri']])


def care_requests(data, predictions):
    responses = {r['sample_id']: r for r in predictions}
    episodes = {e['patient_id']: e for e in data['episodes']}
    requests = []
    for row in data['care']:
        episode = episodes[row['patient_id']]
        values = l6_inputs.frontend_values(responses[episode['pathology_sample_id']]['response'],
                                           responses[episode['mri_sample_id']]['response'])
        prompt = l6_inputs.render_care(row['prompt'], values, episode['ajcc_v9_gold'])
        requests.append({'sample_id': row['sample_id'], 'patient_id': row['patient_id'],
                         'stage': 'care', 'prompt': prompt, 'images': []})
    return requests


def environment(settings):
    return {'tensor_parallel_size': str(settings.tensor_parallel_size), 'max_image_num': str(settings.max_images),
            'VLLM_MAX_MODEL_LEN': str(settings.max_model_len), 'VLLM_GPU_MEMORY_UTILIZATION': str(settings.gpu_memory_utilization),
            'INTERNVL_GPU_MEMORY_UTILIZATION': str(settings.gpu_memory_utilization),
            'INTERNVL_MAX_MODEL_LEN': str(settings.max_model_len), 'INTERNVL_MAX_PATCHES': str(settings.internvl_max_patches),
            'INTERNVL_MIN_PATCHES': '1', 'INTERNVL_TEXT_ONLY': 'False',
            'VLLM_KV_CACHE_MEMORY_BYTES': str(settings.kv_cache_memory_bytes), 'VLLM_MAX_NUM_SEQS': str(settings.max_num_seqs),
            'VLLM_MAX_NUM_BATCHED_TOKENS': str(settings.max_num_batched_tokens), 'VLLM_DTYPE': 'bfloat16',
            'VLLM_ENFORCE_EAGER': 'True', 'VLLM_DISABLE_CUSTOM_ALL_REDUCE': 'True',
            'NPCFULLCARE_ENABLE_VIDEO': 'False', 'HULUMED_ATTENTION_IMPL': 'sdpa', 'HULUMED_DEVICE_MAP': 'auto',
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
            'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4', 'NCCL_IGNORE_DISABLED_P2P': '1',
            'master_port': str(settings.distributed_port), 'MASTER_PORT': str(settings.distributed_port),
            'VLLM_PORT': str(settings.distributed_port)}


def checkpoint_identity(root):
    files = sorted(p for p in root.iterdir() if p.is_file() and
                   (p.suffix in ('.json', '.py', '.safetensors', '.bin', '.model', '.tiktoken') or p.name == 'merges.txt'))
    return {'files': [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files],
            'historical_checkpoint_byte_equivalence_asserted': False}


def package_versions():
    versions = {}
    for name in ('torch', 'transformers', 'vllm', 'qwen-vl-utils', 'Pillow', 'accelerate',
                 'decord', 'imageio', 'ffmpeg-python', 'opencv-python'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def generate_one(model, actual, settings):
    prepared = model.process_messages(actual)
    if settings.adapter != 'hulumed':
        delivered = prepared.get('multi_modal_data', {}).get('image', [])
        if len(delivered) != len(actual['images']):
            raise ValueError('Adapter did not deliver every image to the inference engine')
        trace = {'delivered_images': len(delivered), 'image_sizes': [list(x.size) for x in delivered],
                 'rendered_prompt_sha256': hashlib.sha256(prepared['prompt'].encode()).hexdigest()}
        result, = model.llm.generate([prepared], sampling_params=model.sampling_params)
        completion = result.outputs[0]
        return {'response': completion.text, 'finish_reason': completion.finish_reason, 'stop_reason': completion.stop_reason,
                'output_tokens': len(completion.token_ids), 'prompt_tokens': len(result.prompt_token_ids or []),
                'completion_token_ids': list(completion.token_ids), 'media_trace': trace}
    import torch
    grid = prepared.get('grid_sizes', [])
    if len(grid) != len(actual['images']):
        raise ValueError('HuluMed processor did not encode every input image')
    trace = {'delivered_images': len(grid), 'tensor_shapes': {k: list(v.shape) for k, v in prepared.items() if isinstance(v, torch.Tensor)},
             **getattr(model, 'last_media_trace', {})}
    kwargs = {'repetition_penalty': settings.repetition_penalty, 'max_new_tokens': settings.max_new_tokens,
              'do_sample': settings.temperature > 0}
    if settings.temperature > 0:
        kwargs.update(temperature=settings.temperature, top_p=settings.top_p)
    with torch.inference_mode():
        generated = model.llm.generate(**prepared, **kwargs)
    input_len = prepared['input_ids'].shape[-1]
    new_tokens = generated[:, input_len:] if generated.shape[-1] > input_len else generated
    tokens = new_tokens[0].tolist()
    response = model.processor.batch_decode(new_tokens, skip_special_tokens=True, use_think=False)[0].strip()
    eos = model.llm.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    finish = 'stop' if tokens and tokens[-1] in eos else 'length' if len(tokens) >= settings.max_new_tokens else 'other'
    return {'response': response, 'finish_reason': finish, 'stop_reason': tokens[-1] if finish == 'stop' else None,
            'output_tokens': len(tokens), 'prompt_tokens': input_len, 'completion_token_ids': tokens,
            'termination_evidence': 'HF token IDs and EOS/output budget', 'media_trace': trace}


def run(args):
    settings = Settings(**json.loads(args.config.read_text()))
    root = (args.l6_inputs or args.asset_root).resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError('Use a new output directory; successful or truncated answers are never overwritten')
    expected_hashes, dataset = None, None
    if args.l6_inputs:
        dataset = load_dataset(root)
        if args.mri_prompts:
            dataset = replace_mri_prompts(dataset, rows(args.mri_prompts))
        dataset = choose_l6_patients(dataset, args.patient_limit)
        requested = [dict(r, stage=kind) for kind in ('pathology', 'mri') for r in dataset[kind]]
        expected_hashes = {r['path']: r['sha256'] for r in json.loads((root / 'file_manifest.json').read_text())}
    else:
        requested = rows(args.requests)
    prepared = [prepare_request(r, root, settings, expected_hashes) for r in requested]
    if len({r['sample_id'] for r, _ in prepared}) != len(prepared) or not prepared:
        raise ValueError('Requests must have unique sample IDs and a nonempty cohort')
    output.mkdir(parents=True)
    state = {'status': 'fingerprinting_checkpoint', 'completed': 0,
             'expected': len(prepared) + (len(dataset['care']) if dataset else 0), 'started_at_unix': time.time()}
    def state_save():
        save(output / 'state.json', state)
    state_save()
    try:
        identity = checkpoint_identity(args.model_path.resolve())
        save(output / 'checkpoint_identity.json', identity)
        env = environment(settings)
        if settings.mm_processor_kwargs:
            env['NPCBENCH_MM_PROCESSOR_KWARGS'] = json.dumps(settings.mm_processor_kwargs)
        else:
            os.environ.pop('NPCBENCH_MM_PROCESSOR_KWARGS', None)
        if settings.hulu_merge_size:
            env.update(HULUMED_MERGE_SIZE=str(settings.hulu_merge_size),
                       HULUMED_CONTEXT_LIMIT=str(settings.max_model_len),
                       HULUMED_VISUAL_BUDGET=str(settings.hulu_visual_budget))
        os.environ.update(env)
        module_name, class_name = ADAPTERS[settings.adapter]
        if settings.adapter == 'hulumed' and settings.hulu_merge_size:
            module_name = 'hulumed_budget'
        module = importlib.import_module('.inference_adapters.' + module_name, __package__)
        runtime = {'settings': asdict(settings), 'python': sys.version, 'environment': env,
                   'packages': package_versions(), 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', 'all_visible'),
                   'adapter_sha256': sha(module.__file__), 'runner_sha256': sha(__file__),
                   'answer_policy_sha256': sha(final_answer.__file__), 'input_policy_sha256': sha(l6_inputs.__file__),
                   'checkpoint_identity_sha256': sha(output / 'checkpoint_identity.json'),
                   'config_sha256': sha(args.config), 'scope': 'diagnostic_subset' if args.patient_limit else 'full100' if dataset else 'explicit_requests',
                   'source_manifest_sha256': sha(root / 'file_manifest.json') if dataset else sha(args.requests),
                   'input_policy': l6_inputs.INPUT_POLICY if dataset else None,
                   'scoring_policy': final_answer.POLICY if dataset else None,
                   'installed_package': 'site-packages' in Path(__file__).parts}
        save(output / 'run.json', runtime)
        if args.mri_prompts:
            runtime['mri_prompt_condition'] = 'Full25 output schema on all 100 MRI inputs'
            runtime['mri_prompts_sha256'] = sha(args.mri_prompts)
            save(output / 'run.json', runtime)
        import torch
        torch.manual_seed(settings.seed)
        state['status'] = 'loading_model'; state_save()
        model = getattr(module, class_name)(str(args.model_path.resolve()), SimpleNamespace(
            max_image_num=settings.max_images, max_new_tokens=settings.max_new_tokens, temperature=settings.temperature,
            top_p=settings.top_p, repetition_penalty=settings.repetition_penalty))
        predictions = []
        with (output / 'predictions.jsonl').open('w') as journal:
            def emit(record, result):
                result.update(record, model=settings.model_label, generation_status='ok')
                journal.write(json.dumps(result, ensure_ascii=False) + '\n'); journal.flush()
                predictions.append(result)
                state['completed'] += 1; state_save()

            def generate(batch):
                if settings.batch_inference and settings.adapter != 'hulumed':
                    from .bounded_inference import vllm_batch
                    state['status'] = 'running'; state_save()
                    vllm_batch(model, batch, emit=emit, progress_path=output / 'generation.json')
                    return
                for record, actual in batch:
                    state['status'] = 'running'; state_save()
                    started = time.time()
                    model.input_stage = record['stage']
                    result = generate_one(model, actual, settings)
                    result['elapsed_seconds'] = time.time() - started
                    emit(record, result)
            generate(prepared)
            if dataset:
                generate([prepare_request(r, root, settings, expected_hashes) for r in care_requests(dataset, predictions)])
        if dataset:
            result = score(dataset, predictions, final_answer.POLICY)
            result['inference_scope'] = runtime['scope']
            save(output / 'scores.json', result)
        state.update(status='completed', predictions_sha256=sha(output / 'predictions.jsonl'),
                     finish_reasons={value: sum(r['finish_reason'] == value for r in predictions) for value in sorted({r['finish_reason'] for r in predictions})})
    except BaseException as error:
        state.update(status='failed_preserved', error_type=type(error).__name__, error=str(error)[:400])
        raise
    finally:
        state['finished_at_unix'] = time.time(); state_save()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--requests', type=Path)
    source.add_argument('--l6-inputs', type=Path)
    parser.add_argument('--asset-root', type=Path)
    parser.add_argument('--patient-limit', type=int, help='Diagnostic first-N public patient IDs; never a Full100 score')
    parser.add_argument('--mri-prompts', type=Path, help='Optional Full100 MRI prompt condition; labels and assets must match')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.requests and (args.asset_root is None or args.patient_limit is not None or args.mri_prompts is not None):
        parser.error('Explicit requests require --asset-root; Full100-only options do not apply')
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__':
    main()
