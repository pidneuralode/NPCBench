"""Export versioned Full100 inputs with explicit scoring/exclusion membership."""
import argparse
import hashlib
import hmac
import json
from pathlib import Path
import re
import secrets
import shutil


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def literal_replacer(replacements):
    """Replace source literals once, without rewriting generated identifiers."""
    pattern = re.compile('|'.join(re.escape(key) for key in sorted(replacements, key=len, reverse=True)))
    return lambda text: pattern.sub(lambda match: replacements[match.group(0)], text)


def validate_image_references(records, assets):
    for row in records:
        if not row['images'] or any(name not in assets for name in row['images']):
            raise ValueError('Public image reference does not resolve to an exported asset')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle-root', type=Path, required=True)
    ap.add_argument('--frozen-results-root', type=Path, required=True)
    ap.add_argument('--asset-manifest', type=Path, required=True)
    ap.add_argument('--asset-root', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--private-map', type=Path, required=True)
    args = ap.parse_args()
    bundle, frozen = args.bundle_root.resolve(), args.frozen_results_root.resolve()
    out, mapping_path = args.output_dir.resolve(), args.private_map.resolve()
    if out.exists():
        raise ValueError('Output directory must be new; never overwrite an existing export')
    if mapping_path == out or out in mapping_path.parents:
        raise ValueError('Private mapping must be outside the public output directory')
    revision = json.loads((bundle / 'revision.json').read_text())
    contract_path = bundle / 'scoring_contract.json'
    contract = json.loads(contract_path.read_text()) if contract_path.exists() else None
    expected_care = 217 if contract else 223
    exclusions = read_rows(bundle / 'exclusions.internal.jsonl') if contract else []
    unscored = read_rows(bundle / 'unscored/care_pending_review.internal.jsonl') if contract else []
    excluded_ids = {r['sample_id'] for r in exclusions}
    if contract:
        if contract['schema'] != 'npcbench-l6-membership-v1' or contract['membership_status'] != 'confirmed':
            raise ValueError('Unsupported or unconfirmed scoring membership')
        if revision['scoring_contract_sha256'] != sha(contract_path):
            raise ValueError('Membership contract hash mismatch')
        for field, filename in [('care_sha256', 'care_streaming_templates.jsonl'),
                                ('episodes_sha256', 'patient_episodes.jsonl'), ('hardcore_sha256', 'hardcore93_ids.json')]:
            if sha(bundle / 'data' / filename) != contract[field]:
                raise ValueError('Membership source file hash mismatch')
        if len(excluded_ids) != len(exclusions) or len(exclusions) != 6 or len(unscored) != 6:
            raise ValueError('Expected six unique quality exclusions')
        if excluded_ids != set(contract['excluded_ids']) or excluded_ids != {r['sample_id'] for r in unscored}:
            raise ValueError('Exclusion identities do not join')
        if any(r['phase'] != 'P4' for r in unscored) or any(not r['reason_codes'] for r in exclusions):
            raise ValueError('Excluded treatment items require quality reasons')
    filenames = ['patient_episodes.jsonl', 'pathology_samples.jsonl', 'mri_samples.jsonl', 'care_streaming_templates.jsonl']
    data = {name: read_rows(bundle / 'data' / name) for name in filenames}
    episodes = data['patient_episodes.jsonl']
    care = data['care_streaming_templates.jsonl']
    if (len(episodes), len(data['pathology_samples.jsonl']), len(data['mri_samples.jsonl']), len(care)) != (100, 100, 100, expected_care):
        raise ValueError('This exporter expects the declared Full100 inventory')
    old_care = {row['sample_id']: row for row in read_rows(frozen / 'data/care_streaming_templates.jsonl')}
    changed = {row['sample_id'] for row in care if row['sample_id'] not in old_care or any(row.get(k) != old_care[row['sample_id']].get(k) for k in ('prompt', 'question', 'options', 'gold_letter'))}
    change_rows = read_rows(bundle / 'changes.internal.jsonl')
    declared = {row['sample_id'] for row in change_rows}
    if changed != declared:
        raise ValueError('Declared changes do not match actual input changes')
    retained_ids = {row['sample_id'] for row in care}
    if set(old_care) != retained_ids | excluded_ids or retained_ids & excluded_ids:
        raise ValueError('Changed membership requires its matching exclusion contract')
    if contract and retained_ids != set(contract['scored_ids']):
        raise ValueError('Scored input IDs differ from the membership contract')
    previous = json.loads(mapping_path.read_text()) if mapping_path.exists() else {'salt': secrets.token_hex(32), 'ids': {}}
    salt, ids = bytes.fromhex(previous['salt']), previous['ids']
    aliases = {}

    def public_id(kind, value):
        key = kind + ':' + str(value)
        identifier = kind + '-' + hmac.new(salt, key.encode(), hashlib.sha256).hexdigest()[:24]
        if key in ids and ids[key] != identifier:
            raise ValueError('Existing private mapping is inconsistent')
        ids[key] = identifier
        return identifier

    for records in [*data.values(), unscored]:
        for row in records:
            for key, kind in [('patient_id', 'patient'), ('sample_id', 'sample'), ('episode_id', 'episode'), ('case_id', 'specimen')]:
                if row.get(key):
                    aliases[str(row[key])] = public_id(kind, row[key])
            if row.get('patient_display_id'):
                aliases[str(row['patient_display_id'])] = public_id('patient', row['patient_id'])
    asset_entries = read_rows(args.asset_manifest)
    asset_by_source = {source: entry for entry in asset_entries for source in entry['source_paths']}
    image_map, assets = {}, {}
    for name in ['pathology_samples.jsonl', 'mri_samples.jsonl']:
        for row in data[name]:
            for image in row['images']:
                if image not in asset_by_source:
                    raise ValueError('Image reference missing from the supplied asset manifest')
                entry = asset_by_source[image]
                source = (args.asset_root.resolve() / entry['stage_path']).resolve()
                if args.asset_root.resolve() not in source.parents:
                    raise ValueError('Asset path escapes asset root')
                if not source.is_file() or sha(source) != entry['sha256']:
                    raise ValueError('Asset missing or hash mismatch')
                relative = 'assets/' + entry['sha256'] + source.suffix.lower()
                image_map[image] = relative
                assets[relative] = (source, entry['sha256'])
    replace = literal_replacer({**aliases, **image_map})
    internal_path = re.compile(r'/(?:Users|zju_\d+|home|mnt|data_private)/|[A-Za-z]:\\')

    def clean(value):
        if isinstance(value, str):
            value = replace(value)
            if internal_path.search(value):
                raise ValueError('Whitelisted output still contains a private absolute path')
            return value
        if isinstance(value, list):
            return [clean(x) for x in value]
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        return value

    def take(row, fields):
        return clean({key: row[key] for key in fields if key in row})

    public = {}
    for name in ['pathology_samples.jsonl', 'mri_samples.jsonl']:
        public[name] = []
        for row in data[name]:
            entry = take(row, ['sample_id', 'patient_id', 'task', 'prompt', 'images', 'answer'])
            entry['modality_source'] = 'pathology_specimen_proxy' if name.startswith('pathology') and row['source_track'] == 'physician_reviewed86' else 'patient_matched'
            public[name].append(entry)
        validate_image_references(public[name], assets)
    public['care_streaming_templates.jsonl'] = []
    for row in care:
        entry = take(row, ['sample_id', 'patient_id', 'episode_id', 'turn_id', 'turn_number', 'phase', 'task', 'question', 'prompt', 'gold_letter', 'gold_action', 'context_placeholders'])
        options = row['options']
        entry['options'] = clean({key: value for key, value in options.items()}) if isinstance(options, dict) else [take(option, ['letter', 'text']) for option in options]
        entry['input_changed'] = row['sample_id'] in changed
        entry['evaluation_status'] = 'pending_evaluation'
        public['care_streaming_templates.jsonl'].append(entry)
    public['patient_episodes.jsonl'] = []
    for row in episodes:
        entry = take(row, ['episode_id', 'patient_id', 'pathology_sample_id', 'mri_sample_id', 'oracle_tnm_context', 'streaming_tnm_context_template', 'ajcc_v9_gold', 'structure_invasion_19_gold', 'episode_scoring_rule', 'unscored_care_turn_ids', 'selected_care_turn_count', 'scoring_membership_version'])
        entry['care_turns'] = [take(turn, ['sample_id', 'turn_id', 'turn_type']) for turn in row['care_turns']]
        public['patient_episodes.jsonl'].append(entry)
    hard = clean(json.loads((bundle / 'data/hardcore93_ids.json').read_text()))
    # Validate joins after remapping, before any public output is created.
    patient_ids = {x['patient_id'] for x in public['patient_episodes.jsonl']}
    care_ids = {x['sample_id'] for x in public['care_streaming_templates.jsonl']}
    if len(patient_ids) != 100 or len(care_ids) != expected_care or len(set(hard)) != 93 or not set(hard) <= care_ids:
        raise ValueError('Public identifiers or scoring-mask joins are inconsistent')
    if any(row['patient_id'] not in patient_ids for row in public['care_streaming_templates.jsonl']):
        raise ValueError('Public care patient join is inconsistent')
    turns = [turn['sample_id'] for ep in public['patient_episodes.jsonl'] for turn in ep['care_turns']]
    if set(turns) != care_ids or len(turns) != expected_care or any(not ep['care_turns'] for ep in public['patient_episodes.jsonl']):
        raise ValueError('Public episode scoring-mask join is inconsistent')
    public_unscored = []
    if contract:
        reasons = {r['sample_id']: r for r in exclusions}
        for row in unscored:
            entry = take(row, ['sample_id', 'patient_id', 'episode_id', 'turn_id', 'phase', 'question', 'prompt', 'context_placeholders'])
            options = row['options']
            entry['options'] = clean(dict(options)) if isinstance(options, dict) else [take(o, ['letter', 'text']) for o in options]
            entry.update(take(reasons[row['sample_id']], ['status', 'reason_codes', 'quality_exclusion_reason']))
            public_unscored.append(entry)
        unscored_ids = {r['sample_id'] for r in public_unscored}
        declared_unscored = [sid for ep in public['patient_episodes.jsonl'] for sid in ep.get('unscored_care_turn_ids', [])]
        if unscored_ids & care_ids or set(declared_unscored) != unscored_ids or len(declared_unscored) != 6:
            raise ValueError('Excluded items entered the scoring mask or lost episode linkage')
        if any(r['patient_id'] not in patient_ids for r in public_unscored):
            raise ValueError('Excluded-item patient join is inconsistent')
    model_registry = json.loads((frozen / 'source_registry.json').read_text())
    models = [row['model'] for row in model_registry]
    if len(set(models)) != 21:
        raise ValueError('Expected the frozen21model registry')
    out.mkdir(parents=True)
    for name, records in public.items():
        write_rows(out / 'data' / name, records)
    (out / 'data/hardcore93_ids.json').write_text(json.dumps(hard, indent=2) + '\n')
    if contract:
        write_rows(out / 'unscored/care_pending_review.jsonl', public_unscored)
        public_contract = dict(contract)
        # Only identity fields are remapped; provenance digests are literal bytes.
        public_contract['scored_ids'] = clean(contract['scored_ids'])
        public_contract['excluded_ids'] = clean(contract['excluded_ids'])
        public_contract.update(care_sha256=sha(out / 'data/care_streaming_templates.jsonl'),
            episodes_sha256=sha(out / 'data/patient_episodes.jsonl'), hardcore_sha256=sha(out / 'data/hardcore93_ids.json'))
        (out / 'scoring_contract.json').write_text(json.dumps(public_contract, ensure_ascii=False, indent=2) + '\n')
    for relative, (source, digest) in assets.items():
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha(target) != digest:
            raise ValueError('Copied asset hash mismatch')
    write_rows(out / 'evaluation_status.jsonl', [{'model': model, 'dataset_version': revision['version'], 'status': 'pending_evaluation', 'changed_care_nodes': len(changed), 'historical_predictions_exported': False} for model in models])
    summary = {'schema_version': 'npcbench-full100-public-input-v2' if contract else 'npcbench-full100-public-input-v1', 'dataset_version': revision['version'], 'status': 'membership_confirmed_results_pending' if contract else 'candidate_pending_evaluation', 'patients': 100, 'pathology_samples': 100, 'mri_samples': 100, 'care_nodes': expected_care, 'unscored_pending_review_nodes': len(public_unscored), 'changed_care_nodes': len(changed), 'unchanged_care_nodes': expected_care - len(changed), 'assets': len(assets), 'pathology_proxy_samples': sum(x['modality_source'] == 'pathology_specimen_proxy' for x in public['pathology_samples.jsonl']), 'models_pending': len(models), 'model_results_included': 0, 'historical_predictions_exported': False, 'not_exported': ['full_context811 historical clinical material', 'internal source and reviewer identities', 'historical model predictions', 'unadjudicated gold labels for unscored items'], 'source_revision_sha256': sha(bundle / 'revision.json'), 'source_data_sha256': {name: sha(bundle / 'data' / name) for name in filenames}, 'image_path_format': 'relative assets/<sha256>.<extension>'}
    if contract:
        summary['scoring_contract_sha256'] = sha(out / 'scoring_contract.json')
        summary['answer_parser_status'] = contract['answer_parser_status']
    summary['source_input_status'] = revision['status']
    summary['remaining_material_findings'] = revision.get('remaining_material_findings', [])
    (out / 'dataset_manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    files = [{'path': str(p.relative_to(out)), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()]
    (out / 'file_manifest.json').write_text(json.dumps(files, indent=2) + '\n')
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(json.dumps({'salt': previous['salt'], 'ids': ids}, indent=2) + '\n')
    mapping_path.chmod(0o600)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
