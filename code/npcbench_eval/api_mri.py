"""Run MRI requests with numbered lossless images followed by the task prompt.

Transport retries never depend on answers. Responses with invalid or truncated
answers are retained, not silently regenerated. Credentials are read from NPCBENCH_API_KEY and NPCBENCH_BASE_URL.
"""
import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
from io import BytesIO
import json
from pathlib import Path
import os
import threading
import time
import urllib.error
import urllib.request

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

MODELS = {
    'GPT-5.1': 'gpt-5.1-2025-11-13',
    'GPT-4.1': 'gpt-4.1',
    'Gemini-3.1-Pro-Preview': 'gemini-3.1-pro-preview',
    'Gemini-2.5-Pro': 'gemini-2.5-pro',
    'Gemini-2.5-Flash': 'gemini-2.5-flash',
    'Claude-Sonnet-4.6': 'claude-sonnet-4-6',
}
BUDGETS = {'GPT-5.1':8192,'GPT-4.1':4096,'Claude-Sonnet-4.6':4096,'Gemini-2.5-Flash':8192,'Gemini-2.5-Pro':16384,'Gemini-3.1-Pro-Preview':16384}

PROFILES = {
    'plane_groups_lossless_png_v1': (0, 0, 512, 8192),
    'L3_API_1MB_min512_high': (1.0, 20.0, 512, 4096),
    'Full100_API_0.35MB_min384_high': (0.35, 0.35, 384, 8192),
}


def open_image(path_or_buffer):
    if Image is None:
        raise ModuleNotFoundError(
            "No module named 'PIL'. Install npcbench-eval[api] to process images."
        )
    return Image.open(path_or_buffer)


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def request_parameters(model, row):
    if model == 'GPT-5.1':
        return {'temperature': 0, 'top_p': 0.0001,
                'max_completion_tokens': 8192, 'reasoning_effort': 'none'}
    params = {'temperature': 0, 'max_tokens': BUDGETS[model]}
    if not model.startswith('Claude'):
        params['top_p'] = 0.0001
    return params


def make_messages(row, data_root, encoder):
    cap, overview_cap, min_side, _ = PROFILES[row['image_profile']]
    content, images = [], []
    for i, rel in enumerate(row['images'], 1):
        path = data_root / rel
        if encoder == 'lossless-webp':
            with open_image(path) as im:
                buf = BytesIO()
                im.save(buf, format='WEBP', lossless=True, quality=100, method=6, exact=True)
            url = 'data:image/webp;base64,'+base64.b64encode(buf.getvalue()).decode()
        else:
            url = 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()
        payload = base64.b64decode(url.split(',', 1)[1])
        with open_image(path) as im:
            original_size = list(im.size)
        with open_image(BytesIO(payload)) as im:
            encoded_size = list(im.size)
        images.append({'image': rel, 'original_size': original_size,
                       'encoded_size': encoded_size, 'encoded_bytes': len(payload)})
        content.extend([{'type': 'text', 'text': f'<image_{i}>: '},
                        {'type': 'image_url', 'image_url': {'url': url, 'detail': 'high'}}])
    if not content:
        return [{'role': 'user', 'content': row['prompt']}], images
    content.append({'type': 'text', 'text': row['prompt']})
    return [{'role': 'user', 'content': content}], images


def parse_stage(response, field, json_parser):
    from .final_answer import parse_stage_field
    answer = parse_stage_field(response, 'N' if field.lower() == 'n_stage' else 'T')
    return answer.prediction if answer.valid else None


def run_model(model, rows, args, secret, encoder, json_parser):
    route = MODELS[model]
    folder = args.output/model
    folder.mkdir(parents=True, exist_ok=True)
    journal = folder/'responses.jsonl'
    prior = read_rows(journal) if journal.exists() else []
    done = {r['request_id'] for r in prior}
    assert len(done) == len(prior), 'Duplicate responses require review before resuming'
    assert done <= {r['request_id'] for r in rows}
    lock, stop = threading.Lock(), threading.Event()
    gate = threading.Condition()
    inflight = 0
    fixed_limit = 8 if model == 'Claude-Sonnet-4.6' and args.claude_fixed8 else None
    workers = fixed_limit or args.concurrency
    limit = fixed_limit or min(8, args.concurrency)
    clean_streak = 0

    def enter():
        nonlocal inflight
        with gate:
            while inflight >= limit and not stop.is_set(): gate.wait(1)
            inflight += 1

    def leave():
        nonlocal inflight
        with gate:
            inflight -= 1
            gate.notify_all()

    def tune(success):
        nonlocal limit, clean_streak
        with gate:
            if fixed_limit is not None:
                limit = fixed_limit
            elif success:
                clean_streak += 1
                if clean_streak >= max(16,limit*2):
                    limit = min(args.concurrency, limit*2); clean_streak=0
            else:
                limit=max(2,limit//2); clean_streak=0
            gate.notify_all()
        state['active_concurrency_limit']=limit
    state = {'model': model, 'route': route, 'expected': len(rows),
             'http_completed': len(done), 'parsed': sum(r['prediction'] is not None for r in prior),
             'by_subset': dict(Counter(r['subset'] for r in prior)),
             'active': [], 'status': 'preflight', 'concurrency': workers, 'active_concurrency_limit':limit,
             'concurrency_policy': 'fixed8' if fixed_limit else 'adaptive',
             'transport_failures_this_run': 0}

    def save():
        state['updated_at_unix'] = time.time()
        write_json(folder/'state.json', state)

    def append(path, record):
        text = json.dumps(record, ensure_ascii=False)
        for key in secret['keys']:
            text = text.replace(key, '[REDACTED]')
        with path.open('a') as f:
            f.write(text+'\n')

    def call(row):
        enter()
        with lock:
            if stop.is_set() or row['request_id'] in done:
                leave(); return
            state['active'].append(row['request_id']); save()
        start = time.monotonic()
        try:
            messages, image_meta = make_messages(row, args.data_root,
                'lossless-webp' if model == 'GPT-5.1' else encoder)
            params = request_parameters(model, row)
            body = json.dumps({'model': route, 'messages': messages, **params}).encode()
            request = urllib.request.Request(
                secret['base_url'].rstrip('/')+'/chat/completions', data=body,
                headers={'Authorization': 'Bearer '+secret['keys'][0], 'Content-Type': 'application/json'})
            raw = None
            for attempt in range(1, args.attempts+1):
                try:
                    with urllib.request.urlopen(request, timeout=args.timeout) as f:
                        raw = json.load(f)
                    if not raw.get('choices'):
                        raise ValueError('Response has no choices')
                    tune(True)
                    break
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                        ConnectionError, ValueError) as exc:
                    status = getattr(exc, 'code', None)
                    detail = exc.read(6000).decode(errors='replace') if isinstance(exc, urllib.error.HTTPError) else str(exc)
                    for key in secret['keys']:
                        detail = detail.replace(key, '[REDACTED]')
                    error = {'request_id': row['request_id'], 'subset': row['subset'],
                             'attempt': attempt, 'http_status': status, 'detail': detail,
                             'seconds': time.monotonic()-start, 'at_unix': time.time()}
                    with lock:
                        append(folder/'errors.jsonl', error)
                        state['last_error'] = error; save()
                    tune(False)
                    budget_exceeded = 'budget_exceeded' in detail or 'Budget has been exceeded' in detail
                    permanent = budget_exceeded or status in (400, 401, 403, 404) or 'No available channel' in detail
                    if permanent or attempt == args.attempts:
                        with lock:
                            state['transport_failures_this_run'] += 1
                            if permanent or status == 503:
                                stop.set(); state['status'] = 'blocked_on_budget' if budget_exceeded else 'blocked_on_route'
                            save()
                        return
                    time.sleep(min(60, 3*(2**(attempt-1))))
            choice = raw['choices'][0]
            response = choice.get('message', {}).get('content')
            if isinstance(response, list):
                response = '\n'.join(x.get('text', '') for x in response if isinstance(x, dict))
            if row['result_field'] == 'choice':
                from npcbench_eval.final_answer import parse_choice
                answer = parse_choice(response or '', row['options'])
                prediction = answer.prediction if answer.valid else None
            else:
                prediction = parse_stage(response, row['result_field'], json_parser)
            record = {**row, 'model': model, 'route': route, 'parameters': params,
                      'image_encoding': image_meta, 'request_bytes': len(body),
                      'response': response, 'prediction': prediction,
                      'finish_reason': choice.get('finish_reason'), 'provider_response': raw,
                      'attempts': attempt, 'seconds': time.monotonic()-start,
                      'finished_at_unix': time.time()}
            with lock:
                append(journal, record); done.add(row['request_id'])
                state['http_completed'] = len(done)
                state['parsed'] += prediction is not None
                state['by_subset'][row['subset']] = state['by_subset'].get(row['subset'], 0)+1
                save()
        except Exception as exc:
            with lock:
                stop.set(); state['status'] = 'blocked_on_input_or_runner'
                state['runner_error'] = type(exc).__name__+': '+str(exc)
                save()
            raise
        finally:
            with lock:
                state['active'].remove(row['request_id']); save()
            leave()

    save()
    pending = [r for r in rows if r['request_id'] not in done]
    # Exercise a full case from each input family before increasing concurrency.
    verified = {r['subset'] for r in prior}
    for row in pending:
        if stop.is_set():
            break
        if row['subset'] not in verified:
            call(row)
            if row['request_id'] not in done:
                if not stop.is_set():
                    state['status'] = 'preflight_transport_failed'
                stop.set(); break
            verified.add(row['subset'])
    if not stop.is_set():
        state['status'] = 'running'; save()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(call, [r for r in pending if r['request_id'] not in done]))
        if not stop.is_set():
            state['status'] = 'completed' if len(done) == len(rows) else 'completed_with_transport_gaps'
    save()
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--routes', type=Path, help='Optional JSON mapping of model labels to provider route names')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--skip-models', nargs='*', choices=MODELS, default=[])
    parser.add_argument('--concurrency', type=int, default=16)
    parser.set_defaults(claude_fixed8=True)
    parser.add_argument('--attempts', type=int, default=5)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--l3-max-tokens', type=int, default=4096)
    args = parser.parse_args()
    cap, overview, side, _ = PROFILES['L3_API_1MB_min512_high']
    PROFILES['L3_API_1MB_min512_high'] = (cap, overview, side, args.l3_max_tokens)
    if args.routes:
        MODELS.update(json.loads(args.routes.read_text()))
    from npcbench_eval.final_answer import json_payload
    rows = read_rows(args.requests)
    for row in rows:
        task = row.get('task', row.get('subset', ''))
        stage = 'T' if task in ('T', 'L3-T300', 'T300') else 'N' if task in ('N', 'L3-N400', 'N400') else None
        if stage is None:
            parser.error('API MRI requests must identify T300 or N400')
        row.update(request_id=row.get('request_id', row.get('id', row.get('sample_id'))),
                   task=stage, subset=row.get('subset', stage + ('300' if stage == 'T' else '400')),
                   result_field=stage.lower() + '_stage', image_profile='plane_groups_lossless_png_v1')
        if not row['request_id']:
            parser.error('Every request needs a unique ID')
    first = [max((r for r in rows if r['task']==t),key=lambda r:sum((args.data_root/p).stat().st_size for p in r['images'])) for t in ('T','N') if any(r['task']==t for r in rows)]
    rows = first+[r for r in rows if r['request_id'] not in {x['request_id'] for x in first}]
    assert len({r['request_id'] for r in rows}) == len(rows)
    assert all(r['image_profile'] in PROFILES for r in rows)
    args.output.mkdir(parents=True, exist_ok=True)
    lockfile = (args.output/'run.lock').open('a')
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = {'requests_digest': hashlib.sha256(args.requests.read_bytes()).hexdigest(),
                'request_count': len(rows), 'models': {m: MODELS[m] for m in args.models},
                'profiles': PROFILES, 'output_budgets': BUDGETS,
                'encoding': 'lossless PNG; GPT-5.1 uses lossless WebP with unchanged pixels',
                'parameters': {m: request_parameters(m, {}) for m in args.models},
                'temperature': 0, 'top_p': 0.0001,
                'Claude_top_p': 'omitted', 'message_order': 'numbered images, then prompt',
                'invalid_answers': 'retained without answer-dependent retry'}
    # Normalize tuples through JSON for stable resume comparisons.
    protocol = json.loads(json.dumps(protocol))
    path = args.output/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise SystemExit('Output belongs to another request set or protocol; use a new directory.')
    write_json(path, protocol)
    secret = {'base_url': os.environ['NPCBENCH_BASE_URL'], 'keys': [os.environ['NPCBENCH_API_KEY']]}
    selected = [m for m in args.models if m not in args.skip_models]
    with ThreadPoolExecutor(max_workers=max(1, len(selected))) as pool:
        results = list(pool.map(lambda m: run_model(m, rows, args, secret,
                            None, json_payload), selected))
    write_json(args.output/'summary.json', results)


if __name__ == '__main__':
    main()
