"""Reproduce all Full100 model scores offline from frozen responses and gold labels."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from npcbench_eval.l6_scoring import NPCFullCarePath25 as Parser
from npcbench_eval import l6_scoring

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load(p):
    return json.loads(p.read_text())

def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    a = ap.parse_args()
    root = a.root.resolve()
    manifest = load(root / 'artifact_manifest.json')
    for r in manifest:
        if not sha(root / r['path']) == r['sha256']:
            raise ValueError(r['path'])
    if not sha(Path(l6_scoring.__file__)) == sha(root / 'code/l6_scoring.py'):
        raise ValueError('Installed parser differs from frozen archive parser')
    episodes = {r['patient_id']: r for r in rows(root / 'data/patient_episodes.jsonl')}
    pathology = {r['sample_id']: r for r in rows(root / 'data/pathology_samples.jsonl')}
    mri = {r['sample_id']: r for r in rows(root / 'data/mri_samples.jsonl')}
    care = {r['sample_id']: r for r in rows(root / 'data/care_streaming_templates.jsonl')}
    hard = set(load(root / 'data/hardcore93_ids.json'))
    if not (len(episodes) == 100 and len(care) == 223 and (len(hard) == 93)):
        raise ValueError('Frozen archive consistency check failed')
    with (root / 'metrics.csv').open(encoding='utf-8-sig') as f:
        expected = list(csv.DictReader(f))
    if not len(expected) == 21:
        raise ValueError('Frozen archive consistency check failed')
    results = []
    for reference in expected:
        model = reference['model']
        data = load(root / 'models' / model / 'replayed/results.json')
        if not (len(data) == 100 and {r['patient_id'] for r in data} == set(episodes)):
            raise ValueError('Frozen archive consistency check failed')
        counts = Counter()
        f1sum = 0.0
        seen = set()
        for r in data:
            e = episodes[r['patient_id']]
            p = r['pathology']
            m = r['mri']
            pg = pathology[e['pathology_sample_id']]['answer']
            mg = mri[e['mri_sample_id']]['answer']
            if not (p['gold'] == pg and m['gold_T_stage'] == mg['T_stage'] and (m['gold_structure_invasion'] == mg['structure_invasion'])):
                raise ValueError('Frozen archive consistency check failed')
            pp = Parser._parse_binary(p['response'])
            pt = Parser._parse_t_stage(m['response'])
            ps = Parser._parse_structure_dict(m['response'])
            pok = pp == str(pg)
            tok = pt == mg['T_stage']
            f1 = Parser._structure_scores(ps, mg['structure_invasion'])['f1']
            f1sum += f1
            counts['pathology_parse_failures'] += not bool(pp)
            counts['mri_t_parse_failures'] += not bool(pt)
            selected = {c['sample_id'] for c in e['care_turns']}
            if not ({c['sample_id'] for c in r['care_results']} == selected and len(r['care_results']) == len(selected)):
                raise ValueError('Frozen archive consistency check failed')
            correct = []
            for c in r['care_results']:
                sid = c['sample_id']
                if not sid not in seen:
                    raise ValueError('Frozen archive consistency check failed')
                seen.add(sid)
                if not care[sid]['patient_id'] == r['patient_id']:
                    raise ValueError('Frozen archive consistency check failed')
                gold = care[sid]['gold_letter']
                pred = Parser._parse_letter(c['response'])
                right = pred == gold
                correct.append(right)
                if not (c['gold_letter'] == gold and c['pred_answer'] == pred and (c['correct'] == right)):
                    raise ValueError('Frozen archive consistency check failed')
                counts['care_right'] += right
                counts['hard_right'] += bool(right and sid in hard)
                counts['care_parse_failures'] += not bool(pred)
            allcare = all(correct)
            strict = pok and tok and allcare
            if not (allcare == r['all_care_correct'] and strict == r['strict_episode_correct']):
                raise ValueError('Frozen archive consistency check failed')
            counts.update(pathology_right=int(pok), mri_t_right=int(tok), patient_all_right=int(allcare), strict_right=int(strict))
        if not seen == set(care):
            raise ValueError('Frozen archive consistency check failed')
        for k, v in counts.items():
            if not v == int(reference[k]):
                raise ValueError((model, k, v, reference[k]))
        if not abs(f1sum / 100 - float(reference['mri_structure_macro_f1'])) < 1e-12:
            raise ValueError('Frozen archive consistency check failed')
        results.append(dict(model=model, **dict(counts), mri_structure_macro_f1=f1sum / 100))
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    verification = dict(status='passed', models=21, patients_per_model=100, pathology_responses=2100, mri_responses=2100, care_responses=4683, total_raw_responses_reparsed=8883, score_differences=0, model_calls=0, GPU_used=False, network_used=False, manifest_files=len(manifest))
    (out / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps(verification, indent=2))
if __name__ == '__main__':
    main()
