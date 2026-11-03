"""Retry unanswered Claude MRI cases, at most three times per case."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
from pathlib import Path
import urllib.request

from .api_mri import make_messages, request_parameters
from .final_answer import json_payload, parse_stage_field, select_final


def has_stage(response, stage):
    answer = parse_stage_field(response, stage)
    if answer.valid or answer.status == 'conflicting_stage_fields':
        return True
    payload, _, _ = json_payload(response)
    keys = {stage.lower(), stage.lower() + '_stage', stage.lower() + ' stage'}
    if isinstance(payload, dict) and any(
        k.lower() in keys and isinstance(v, str) and bool(v.strip()) for k, v in payload.items()):
        return True
    channel = select_final(response)
    if channel.status != 'ok':
        return False
    # A complete stage can precede a truncated evidence string.
    text, pos, depth, decoder = channel.text, 0, 0, json.JSONDecoder()
    while pos < len(text):
        if text[pos] != '"':
            depth += (text[pos] == '{') - (text[pos] == '}')
            pos += 1
            continue
        try:
            key, pos = decoder.raw_decode(text, pos)
        except ValueError:
            break
        colon = re.match(r'\s*:\s*', text[pos:])
        if depth != 1 or key.lower() not in keys or colon is None:
            continue
        try:
            value, pos = decoder.raw_decode(text, pos + colon.end())
        except ValueError:
            break
        if isinstance(value, str) and value.strip():
            return True
    return False


def recover(row, initial, send):
    """Stop at an explicit stage, regardless of correctness or allowed category."""
    stage = 'N' if 'N400' in row.get('subset', row.get('task', '')) else 'T'
    selected, attempts = dict(initial), []
    for _ in range(3):
        if has_stage(selected.get('response'), stage):
            break
        params = request_parameters('Claude-Sonnet-4.6', row)
        if 'structure_invasion' in (selected.get('response') or '') or selected.get('finish_reason') == 'length':
            params['max_tokens'] = 8192
        result = send(row, params)
        selected = dict(result, parameters=params)
        attempts.append(selected)
    return {'id': row.get('id', row.get('request_id')), 'initial': initial,
            'attempts': attempts, 'selected': selected}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requests', type=Path, required=True)
    p.add_argument('--responses', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--route', default='claude-sonnet-4-6')
    args = p.parse_args()
    read = lambda path: [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    identity = lambda r: r.get('id', r.get('request_id', r.get('sample_id')))
    requests = {identity(r): r for r in read(args.requests)}
    raw = read(args.responses)
    answers = {identity(r): r for r in raw}
    if len(answers) != len(raw) or requests.keys() != answers.keys():
        p.error('Requests and responses must cover the same unique case IDs')
    args.output.mkdir(parents=True, exist_ok=False)
    def send(row, params):
        row = dict(row, image_profile='plane_groups_lossless_png_v1')
        messages, _ = make_messages(row, args.data_root, None)
        body = json.dumps({'model': args.route, 'messages': messages, **params}).encode()
        req = urllib.request.Request(os.environ['NPCBENCH_BASE_URL'].rstrip('/') + '/chat/completions',
            data=body, headers={'Content-Type': 'application/json',
                               'Authorization': 'Bearer ' + os.environ['NPCBENCH_API_KEY']})
        with urllib.request.urlopen(req, timeout=900) as f:
            result = json.load(f)
        choice = result['choices'][0]
        response = choice['message'].get('content')
        if isinstance(response, list):
            response = '\n'.join(x.get('text', '') for x in response if isinstance(x, dict))
        return {'response': response,
                'finish_reason': choice.get('finish_reason'), 'provider_response': result}
    with (args.output/'attempts.jsonl').open('w') as log, (args.output/'responses.jsonl').open('w') as out:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for result in pool.map(lambda sid: recover(requests[sid], answers[sid], send), requests):
                log.write(json.dumps(result, ensure_ascii=False)+'\n'); log.flush()
                out.write(json.dumps({'id':result['id'], **result['selected']}, ensure_ascii=False)+'\n'); out.flush()


if __name__ == '__main__':
    main()
