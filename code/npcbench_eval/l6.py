"""Score an explicit L6 membership version; parser policies remain diagnostic."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from . import l6_scoring
from .l6_scoring import NPCFullCarePath25 as Parser
from .response_channels import gate_response
from .final_answer import POLICY, parse_choice, json_payload, parse_t_stage

POLICIES = ('frozen_v1', 'reject_unfinished_think_v1', POLICY)
FILES = {'episodes': 'data/patient_episodes.jsonl', 'pathology': 'data/pathology_samples.jsonl',
         'mri': 'data/mri_samples.jsonl', 'care': 'data/care_streaming_templates.jsonl',
         'unscored': 'unscored/care_pending_review.jsonl'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def unique(records, key):
    result = {}
    for row in records:
        value = row[key]
        if not isinstance(value, str) or not value or value in result:
            raise ValueError('Duplicate or empty ' + key)
        result[value] = row
    return result


def validate_membership(data):
    contract = data['contract']
    indexed = {key: unique(data[key], 'patient_id' if key == 'episodes' else 'sample_id') for key in FILES}
    patients = set(indexed['episodes'])
    if len(patients) != contract['patients']:
        raise ValueError('Patient denominator differs from membership contract')
    care, excluded = set(indexed['care']), set(indexed['unscored'])
    hard = set(data['hardcore'])
    if (care != set(contract['scored_ids']) or excluded != set(contract['excluded_ids']) or care & excluded
            or len(care) != contract['scored_care_nodes'] or len(excluded) != contract['unscored_pending_review_nodes']
            or len(hard) != len(data['hardcore']) or len(hard) != contract['hardcore_nodes'] or not hard <= care):
        raise ValueError('Scored, excluded or hard-core mask mismatch')
    denominators = {'care_micro': len(care), 'patient_all_care': len(patients),
                    'strict_episode': len(patients), 'pathology': len(patients),
                    'MRI_T': len(patients), 'hardcore_care': len(hard)}
    if contract['metric_denominators'] != denominators:
        raise ValueError('Metric denominators contradict the scoring membership')
    if any(any(k.startswith('gold') or k == 'answer' for k in row) for row in data['unscored']):
        raise ValueError('Unadjudicated excluded gold cannot be an answer key')
    joined_care, joined_unscored = [], []
    modalities = {'pathology': [], 'mri': []}
    for patient, episode in indexed['episodes'].items():
        turns = [t['sample_id'] for t in episode['care_turns']]
        if not turns:
            raise ValueError('Empty scored care episode is not a valid success denominator')
        unscored = episode.get('unscored_care_turn_ids', [])
        for sid in turns:
            if sid not in care or indexed['care'][sid]['patient_id'] != patient:
                raise ValueError('Care item belongs to a different patient or mask')
        for sid in unscored:
            if sid not in excluded or indexed['unscored'][sid]['patient_id'] != patient:
                raise ValueError('Excluded item patient join mismatch')
        joined_care.extend(turns)
        joined_unscored.extend(unscored)
        for kind in modalities:
            sid = episode[kind + '_sample_id']
            if sid not in indexed[kind] or indexed[kind][sid]['patient_id'] != patient:
                raise ValueError('Frontend patient join mismatch')
            modalities[kind].append(sid)
    if len(joined_care) != len(care) or set(joined_care) != care or len(joined_unscored) != len(excluded) or set(joined_unscored) != excluded:
        raise ValueError('Episode membership must cover each scored/excluded item exactly once')
    for kind, ids in modalities.items():
        if len(set(ids)) != len(patients) or set(ids) != set(indexed[kind]):
            raise ValueError('Frontend samples must exactly cover retained patients')
    all_ids = [sid for kind in ('pathology', 'mri', 'care') for sid in indexed[kind]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError('Sample identity collides across endpoints')
    return indexed


def load_dataset(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'dataset_manifest.json').read_text())
    contract_path = root / 'scoring_contract.json'
    if manifest['schema_version'] != 'npcbench-full100-public-input-v2' or sha(contract_path) != manifest['scoring_contract_sha256']:
        raise ValueError('Unsupported input version or contract hash mismatch')
    contract = json.loads(contract_path.read_text())
    if contract['membership_status'] != 'confirmed' or (contract['patients'], contract['scored_care_nodes'], contract['unscored_pending_review_nodes']) != (100, 217, 6):
        raise ValueError('Expected the confirmed Full100/217 contract')
    files = {r['path']: r['sha256'] for r in json.loads((root / 'file_manifest.json').read_text())}
    for relative in [*FILES.values(), 'data/hardcore93_ids.json', 'scoring_contract.json', 'dataset_manifest.json']:
        if relative not in files or sha(root / relative) != files[relative]:
            raise ValueError('Scoring input is absent from manifest or has changed')
    data = {key: rows(root / relative) for key, relative in FILES.items()}
    data.update(contract=contract, hardcore=json.loads((root / 'data/hardcore93_ids.json').read_text()))
    validate_membership(data)
    return data


def score(data, predictions, policy):
    if policy not in POLICIES:
        raise ValueError('Choose an explicit diagnostic parser policy')
    indexed = validate_membership(data)
    responses = unique(predictions, 'sample_id')
    expected = {sid for kind in ('pathology', 'mri', 'care') for sid in indexed[kind]}
    if set(responses) != expected:
        raise ValueError('Predictions must exactly cover scored samples; excluded answers are not accepted')
    transform = gate_response if policy == 'reject_unfinished_think_v1' else lambda x: x
    parsed, per_patient, counts, frontend_changes = [], [], Counter(), []
    hard = set(data['hardcore'])
    f1sum = 0.0
    for patient, episode in indexed['episodes'].items():
        def text(sid):
            value = responses[sid].get('response') or ''
            if not isinstance(value, str):
                raise ValueError('Response must be text or null')
            return transform(value)
        psid, msid = episode['pathology_sample_id'], episode['mri_sample_id']
        pg, mg = indexed['pathology'][psid]['answer'], indexed['mri'][msid]['answer']
        ptext, mtext = text(psid), text(msid)
        if policy == POLICY:
            panswer = parse_choice(ptext, {'0': '', '1': ''})
            tanswer = parse_t_stage(mtext)
            payload, mstatus, mchannel = json_payload(mtext)
            pp = panswer.prediction
            mtext = json.dumps(payload, ensure_ascii=False) if payload is not None else ''
            mt = tanswer.prediction
            original_p, original_m = responses[psid].get('response'), responses[msid].get('response')
            changed_fields = []
            if pp != Parser._parse_binary(original_p):
                changed_fields.append('pathology')
            if mt != Parser._parse_t_stage(original_m):
                changed_fields.append('MRI_T')
            if Parser._parse_structure_dict(mtext) != Parser._parse_structure_dict(original_m):
                changed_fields.append('MRI_structures')
            if changed_fields:
                frontend_changes.append({'patient_id': patient, 'changed_fields': changed_fields,
                                         'care_nodes': len(episode['care_turns'])})
        else:
            pp, mt = Parser._parse_binary(ptext), Parser._parse_t_stage(mtext)
        pok, tok = pp == str(pg), mt == mg['T_stage']
        f1sum += Parser._structure_scores(Parser._parse_structure_dict(mtext), mg['structure_invasion'])['f1']
        counts.update(pathology_right=int(pok), mri_t_right=int(tok),
            pathology_parse_failures=int(not pp), mri_t_parse_failures=int(not mt))
        parsed.extend([{'sample_id': psid, 'prediction': pp, 'correct': pok},
                       {'sample_id': msid, 'prediction': mt, 'correct': tok}])
        if policy == POLICY:
            parsed[-2].update(answer_status=panswer.status, answer_channel=panswer.channel)
            parsed[-1].update(answer_status=tanswer.status, answer_channel=tanswer.channel,
                              structure_json_status=mstatus)
        care_correct = []
        for turn in episode['care_turns']:
            sid = turn['sample_id']
            if policy == POLICY:
                answer = parse_choice(text(sid), indexed['care'][sid]['options'])
                pred = answer.prediction
            else:
                pred = Parser._parse_letter(text(sid))
            correct = pred == indexed['care'][sid]['gold_letter']
            care_correct.append(correct)
            counts.update(care_right=int(correct), hardcore_right=int(correct and sid in hard), care_parse_failures=int(not pred))
            parsed.append({'sample_id': sid, 'prediction': pred, 'correct': correct})
            if policy == POLICY:
                parsed[-1].update(answer_status=answer.status, answer_channel=answer.channel)
        allcare = all(care_correct)
        strict = pok and tok and allcare
        counts.update(patient_all_care_right=int(allcare), strict_episode_right=int(strict))
        per_patient.append({'patient_id': patient, 'scored_care_nodes': len(care_correct),
            'care_right': sum(care_correct), 'all_care_correct': allcare, 'strict_episode_correct': strict})
    n, care_n = len(indexed['episodes']), len(indexed['care'])
    metrics = {'patients': n, 'scored_care_nodes': care_n, 'unscored_pending_review_nodes': len(indexed['unscored']),
        'hardcore_nodes': len(hard), 'response_count': len(responses), **dict(counts),
        'care_micro_percent': 100 * counts['care_right'] / care_n,
        'patient_all_care_percent': 100 * counts['patient_all_care_right'] / n,
        'strict_episode_percent': 100 * counts['strict_episode_right'] / n,
        'pathology_percent': 100 * counts['pathology_right'] / n, 'mri_t_percent': 100 * counts['mri_t_right'] / n,
        'hardcore_micro_percent': 100 * counts['hardcore_right'] / len(hard) if hard else None,
        'mri_structure_macro_f1': f1sum / n}
    result = {'status': 'diagnostic_pending_final_answer_parser', 'membership_version': data['contract']['version'],
            'policy': policy, 'frozen_parser_sha256': sha(l6_scoring.__file__),
            'metrics': metrics, 'patient_results': per_patient, 'parsed_predictions': parsed}
    if policy == POLICY:
        result['frontend_interpretation_changes'] = frontend_changes
        result['frontend_inputs_require_review'] = bool(frontend_changes)
        result['answer_parser_sha256'] = sha(Path(__file__).with_name('final_answer.py'))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-root', type=Path, required=True)
    ap.add_argument('--predictions', type=Path, required=True, help='417 unique sample_id/response records; excluded items absent')
    ap.add_argument('--policy', choices=POLICIES, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    result = score(load_dataset(args.dataset_root), rows(args.predictions), args.policy)
    result['predictions_sha256'] = sha(args.predictions)
    result['contract_sha256'] = sha(args.dataset_root / 'scoring_contract.json')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('patient_results', 'parsed_predictions')}, indent=2))


if __name__ == '__main__':
    main()
