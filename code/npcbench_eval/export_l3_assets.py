"""Export canonical L3 primary inference inputs and unchanged image bytes.

This input view is distinct from archived model-specific prompts/wrappers. Gold
is stored separately; model_request() exposes only prompt and image references.
No model inference, network requests, image conversion, or pixel edits occur.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

from .l3 import ENDPOINTS, canonical_model, content_sha, index, project_rows, read_jsonl


PROTOCOL = 'npcbench-l3-canonical-inputs-v1'
# Reviewed static prompts: identical for every item within the reference run.
# Pinning these templates prevents exporting arbitrary clinical prompt content.
PROMPT_SHA = {
    'Path': 'ccc9e3249b905730abb5d6b4b6df4a45fc92329879ac0ddd06c211bc55a3d07f',
    'Seq': '5490181391b7ce5c120ab72b587471b15a07459d36a64106dd5c55b3bbfc3f44',
    'Struct': '79d5ea8a0b756e7e6f8a22edd33bff573b97be04e368de9a8ee3a54c80ea5069',
    'T-Stg': 'af074aff4d0c48a5e4e5eead1b2f0b7d5c923068601035b3b94f80f6e5dc368e',
}
SOURCE_FILES = {
    'cards': 'mri_v2/metadata/manifests/cards_test.jsonl',
    'card_answers': 'mri_v2/metadata/answers/cards_test_answers.jsonl',
    'patients': 'mri_v2/metadata/manifests/patients_test.jsonl',
    'patient_answers': 'mri_v2/metadata/answers/patients_test_answers.jsonl',
    'sequence': 'sequence/metadata/manifests/test.jsonl',
    'sequence_answers': 'sequence/metadata/answers/test_answers.jsonl',
    'pathology': 'pathology/manifests/questions_overview.csv',
    'pathology_cases': 'pathology/manifests/cases.csv',
}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def public_item_id(subset, raw_id):
    return subset + '-' + hashlib.sha256((subset + '\0' + str(raw_id)).encode()).hexdigest()[:24]


def unique(rows, field):
    result = {str(row[field]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate source identity')
    return result


def rows(path):
    if path.suffix == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))
    return read_jsonl(path)


def inside(root, relative):
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError('File path escapes its bundle root')
    return path


def geometry(subset, source, item_id):
    """Only numeric display geometry; no labels, identities, paths or answers."""
    if subset == 'Struct':
        return {'id': item_id, 'image_size': source['image_size'], 'tile_size': source['tile_size'],
                'panels': [{key: panel[key] for key in ('panel_id', 'row_index', 'canvas_bbox_norm')}
                           for panel in source['panels']]}
    if subset == 'Seq':
        return {'id': item_id, 'panel_shape': source['panel_shape'], 'panel_size': source['panel_size'],
                'tile_size': source['tile_size'],
                'tiles': [{key: tile[key] for key in ('grid_index', 'grid_row', 'grid_col')}
                          for tile in source['tiles']]}
    raise ValueError('No geometry sidecar for this endpoint')


def model_request(record):
    """Adapter boundary: deliberately do not serialize the whole dataset row."""
    if not isinstance(record.get('prompt'), str) or not record.get('images'):
        raise ValueError('A model request requires prompt text and images')
    return {'prompt': record['prompt'], 'images': list(record['images'])}


def prepare(stage, scorer_bundle, source_inventory):
    stage, scorer_bundle = Path(stage).resolve(), Path(scorer_bundle).resolve()
    inventory = json.loads(Path(source_inventory).read_text())
    source_locks = {entry['dataset'] + '/' + entry['relative_path']: entry for entry in inventory}
    data, source_hashes = {}, {}
    for name, relative in SOURCE_FILES.items():
        path = inside(stage, 'l3_sources/' + relative)
        lock = source_locks[relative]
        if file_sha(path) != lock['sha256']:
            raise ValueError('L3 source metadata hash mismatch')
        data[name] = rows(path)
        if len(data[name]) != lock['rows']:
            raise ValueError('L3 metadata row count mismatch')
        source_hashes[relative] = lock['sha256']
    frozen = json.loads((scorer_bundle / 'manifest.json').read_text())
    if frozen['schema'] != 'npcbench-l3-frozen-v1' or frozen.get('class_macro_reduction') != 'sorted_labels_v1':
        raise ValueError('Use the current L3 v3 scorer bundle')
    keys = {}
    for subset, (count, _, _, _) in ENDPOINTS.items():
        relative = 'data/' + subset + '.jsonl'
        if file_sha(scorer_bundle / relative) != frozen['files'][relative]:
            raise ValueError('Frozen scorer dataset hash mismatch')
        keys[subset] = index(read_jsonl(scorer_bundle / relative))
        if len(keys[subset]) != count:
            raise ValueError('Frozen scorer cohort mismatch')
    assets_path = stage / 'manifests/multimodal_assets.internal.jsonl'
    references_path = stage / 'manifests/multimodal_references.internal.jsonl'
    assets = {r['sha256']: r for r in read_jsonl(assets_path)}
    references = {}
    for reference in read_jsonl(references_path):
        if reference['level'] != 'L3' or reference['task'] not in {'mri_card', 'mri_sequence', 'pathology'}:
            continue
        identity = (reference['task'], str(reference['item_id']), reference['role'])
        if identity in references:
            raise ValueError('Duplicate primary asset reference')
        references[identity] = reference
    card_answers = unique(data['card_answers'], 'card_id')
    patient_answers = unique(data['patient_answers'], 'patient_case_id')
    sequence_answers = unique(data['sequence_answers'], 'question_id')
    path_cases = unique(data['pathology_cases'], 'case_id')
    records, layouts, selected_images = {s: [] for s in ENDPOINTS}, {}, {}

    def attach(task, raw_id, role, metadata_path, dataset_name):
        reference = references[(task, str(raw_id), role)]
        asset = assets[reference['sha256']]
        if reference['stage_path'] != asset['stage_path']:
            raise ValueError('Asset content path mismatch')
        root = next(x['dataset_root'] for x in inventory if x['dataset'] == dataset_name)
        source_path = str(Path(root) / metadata_path)
        if source_path not in asset['source_paths']:
            raise ValueError('Metadata image/layout does not match archived source provenance')
        path = inside(stage, asset['stage_path'])
        if not path.is_file() or path.stat().st_size != asset['bytes']:
            raise ValueError('Missing or truncated staged asset')
        return asset, reference

    def add(subset, raw_id, patient, gold, image_assets, **extra):
        item_id = public_item_id(subset, raw_id)
        if item_id not in keys[subset] or gold != keys[subset][item_id]['gold']:
            raise ValueError('Metadata answer or sample ID differs from frozen scorer')
        for asset in image_assets:
            if asset['kind'] != 'image':
                raise ValueError('Primary tasks require static image assets')
            selected_images[asset['sha256']] = asset
        records[subset].append({'id': item_id, '_internal_id': str(raw_id), '_patient_key': patient,
                                'images': [asset['stage_path'] for asset in image_assets], **extra})

    for row in data['pathology']:
        case = path_cases[str(row['case_id'])]
        if str(row['answer']) != str(case['label']):
            raise ValueError('Pathology case label and question answer differ')
        image_paths = json.loads(row['image_paths'])
        if len(image_paths) != 1 or int(row['num_images']) != 1:
            raise ValueError('Expected one overview image per case')
        asset, _ = attach('pathology', row['question_id'], 'image', image_paths[0], 'pathology')
        add('Path', row['question_id'], 'pathology\0' + str(row['case_id']), str(row['answer']), [asset],
            options=[{'option_id': '0', 'label': '正常'}, {'option_id': '1', 'label': '鼻咽癌'}],
            _source_relative_images=image_paths)
    for row in data['sequence']:
        raw_id = str(row['question_id'])
        answer = sequence_answers[raw_id]
        if str(answer['patient_id']) != str(row['patient_id']):
            raise ValueError('Sequence answer patient mismatch')
        asset, reference = attach('mri_sequence', raw_id, 'asset_path', row['asset_path'], 'sequence')
        if str(reference['patient_key_internal']) != str(row['patient_id']):
            raise ValueError('Sequence asset patient mismatch')
        item_id = public_item_id('Seq', raw_id)
        options = [{key: option[key] for key in ('option_id', 'sequence_label', 'sequence_cn')} for option in row['options']]
        if options != keys['Seq'][item_id]['options']:
            raise ValueError('Sequence options differ from frozen scorer')
        layout, _ = attach('mri_sequence', raw_id, 'layout_path', row['layout_path'], 'sequence')
        if file_sha(stage / layout['stage_path']) != layout['sha256']:
            raise ValueError('Sequence layout hash mismatch')
        layouts[item_id] = geometry('Seq', json.loads((stage / layout['stage_path']).read_text()), item_id)
        add('Seq', raw_id, 'mri\0' + str(row['patient_id']), answer['correct_option_id'], [asset], options=options,
            layout_paths=['layouts/' + item_id + '.json'], _source_relative_images=[row['asset_path']])
    cards_by_id = {}
    for row in data['cards']:
        raw_id = str(row['card_id'])
        answer = card_answers[raw_id]
        if (str(answer['patient_id']), str(answer['patient_case_id'])) != (str(row['patient_id']), str(row['patient_case_id'])):
            raise ValueError('Structure answer patient mismatch')
        item_id = public_item_id('Struct', raw_id)
        choices = row['answer_space']['structure_identification']
        if sorted(choices) != keys['Struct'][item_id]['structure_choices']:
            raise ValueError('Structure choices differ from frozen scorer')
        asset, reference = attach('mri_card', raw_id, 'asset_path', row['asset_path'], 'mri_v2')
        if str(reference['patient_key_internal']) != str(row['patient_id']):
            raise ValueError('Structure asset patient mismatch')
        layout, _ = attach('mri_card', raw_id, 'layout_path', row['layout_path'], 'mri_v2')
        if file_sha(stage / layout['stage_path']) != layout['sha256']:
            raise ValueError('Structure layout hash mismatch')
        layouts[item_id] = geometry('Struct', json.loads((stage / layout['stage_path']).read_text()), item_id)
        add('Struct', raw_id, 'mri\0' + str(row['patient_id']), answer['base_structure_id_gt'], [asset],
            structure_choices=choices, layout_paths=['layouts/' + item_id + '.json'],
            _source_relative_images=[row['asset_path']])
        cards_by_id[raw_id] = (row, asset)
    for row in data['patients']:
        raw_id = str(row['patient_case_id'])
        answer = patient_answers[raw_id]
        if str(answer['patient_id']) != str(row['patient_id']):
            raise ValueError('T-stage answer patient mismatch')
        item_id = public_item_id('T-Stg', raw_id)
        choices = row['answer_space']['base_structure_invasion']
        if sorted(choices) != keys['T-Stg'][item_id]['base_structure_choices']:
            raise ValueError('T-stage choices differ from frozen scorer')
        card_ids = [str(card['card_id']) for card in row['cards']]
        if len(card_ids) != 27 or len(set(card_ids)) != 27 or set(card_ids) != set(answer['card_ids']):
            raise ValueError('T-stage card inventory mismatch')
        image_assets = []
        for card in row['cards']:
            source, asset = cards_by_id[str(card['card_id'])]
            if (str(source['patient_id']), str(source['patient_case_id']), source['asset_path'], source['layout_path']) != (
                    str(row['patient_id']), raw_id, card['asset_path'], card['layout_path']):
                raise ValueError('T-stage card links disagree with structure metadata')
            image_assets.append(asset)
        public_cards = [public_item_id('Struct', key) for key in card_ids]
        add('T-Stg', raw_id, 'mri\0' + str(row['patient_id']), answer['t_stage_gt'], image_assets,
            structure_choices=choices, card_ids=public_cards,
            layout_paths=['layouts/' + key + '.json' for key in public_cards],
            _source_relative_images=[card['asset_path'] for card in row['cards']])
    for subset in ENDPOINTS:
        if {r['id'] for r in records[subset]} != set(keys[subset]):
            raise ValueError('Canonical input IDs do not exactly cover frozen scorer')
    lock_path = stage / 'primary_l3_source_lock.json'
    if file_sha(lock_path) != frozen['source_lock_sha256']:
        raise ValueError('Frozen response source lock mismatch')
    prompts, history = {}, []
    archived_paths = {path: asset['stage_path'] for asset in assets.values() for path in asset['source_paths']}
    frozen_entries = {(e['model'], e['subset']): e for e in frozen['entries']}
    canonical_by_id = {subset: index(records[subset]) for subset in records}
    for lock in json.loads(lock_path.read_text()):
        model, subset = canonical_model(lock['model']), lock['subset']
        path = inside(stage, lock['stage_path'])
        if file_sha(path) != lock['sha256']:
            raise ValueError('Archived response source hash mismatch')
        archive = json.loads(path.read_text())
        archive = list(archive.values()) if isinstance(archive, dict) else archive
        projected, _ = project_rows(subset, archive)
        if index(projected) != keys[subset]:
            raise ValueError('Archived run labels/options/IDs differ from candidate scorer')
        if model == 'Claude-Sonnet-4.6':
            templates = {row['prompt'] for row in archive}
            if len(templates) != 1:
                raise ValueError('Canonical prompt must be static within an endpoint')
            template = templates.pop()
            if hashlib.sha256(template.encode()).hexdigest() != PROMPT_SHA[subset]:
                raise ValueError('Canonical prompt is not the reviewed static template')
            prompts[subset] = template
        comparisons = Counter()
        for item, row in zip(projected, archive):
            comparison = canonical_by_id[subset][item['id']]
            comparisons['prompt_matches_reviewed_template'] += hashlib.sha256(row['prompt'].encode()).hexdigest() == PROMPT_SHA[subset]
            raw_images = row.get('asset_abspaths', row.get('images'))
            if raw_images is None:
                raw_images = [row.get('asset_abspath')]
            resolved = [archived_paths.get(str(image)) for image in raw_images]
            comparisons['image_references_resolved'] += all(resolved)
            comparisons['image_order_matches_canonical'] += resolved == comparison['images']
            relative_images = row.get('asset_paths')
            if relative_images is None and row.get('asset_path') is not None:
                relative_images = [row['asset_path']]
            if relative_images is None and subset == 'Path':
                dataset_root = next(x['dataset_root'] for x in inventory if x['dataset'] == 'pathology')
                relative_images = [str(Path(image).relative_to(dataset_root))
                                   if Path(image).is_relative_to(dataset_root) else None for image in raw_images]
            comparisons['relative_image_order_matches_canonical'] += relative_images == comparison['_source_relative_images']
        evidence = frozen_entries[(model, subset)]
        history.append({'model': model, 'subset': subset, 'items': len(archive), **dict(comparisons),
                        **{key: evidence[key] for key in ('source_sha256', 'archived_input_view_sha256', 'archived_prompt_list_sha256')}})
    if set(prompts) != set(ENDPOINTS) or len(history) != 84:
        raise ValueError('Incomplete historical source coverage')
    return {'stage': stage, 'scorer_bundle': scorer_bundle, 'records': records, 'layouts': layouts,
            'images': selected_images, 'prompts': prompts, 'history': history,
            'source_hashes': source_hashes, 'scorer_manifest_sha256': file_sha(scorer_bundle / 'manifest.json'),
            'source_inventory_sha256': file_sha(source_inventory),
            'asset_manifest_sha256': file_sha(assets_path), 'reference_manifest_sha256': file_sha(references_path)}


def plan(prepared):
    return {'scope': 'L3 primary static inputs only; cine excluded',
            'items': {s: len(rs) for s, rs in prepared['records'].items()},
            'unique_images': len(prepared['images']),
            'image_bytes': sum(a['bytes'] for a in prepared['images'].values()),
            'geometry_json_count': len(prepared['layouts']),
            'scorer_manifest_sha256': prepared['scorer_manifest_sha256']}


def private_mapping(prepared, path):
    path = Path(path)
    patient_keys = {r['_patient_key'] for rs in prepared['records'].values() for r in rs}
    if path.exists():
        if path.stat().st_mode & 0o777 != 0o600:
            raise ValueError('Private mapping must have mode 0600')
        mapping = json.loads(path.read_text())
        patients = mapping['patients']
        if set(patients) != patient_keys:
            raise ValueError('Existing private mapping belongs to a different cohort')
        return patients
    patients = {key: 'L3P-' + uuid.uuid4().hex for key in sorted(patient_keys)}
    mapping = {'schema': PROTOCOL + '-private-map', 'patients': patients,
               'samples': [{'subset': subset, 'internal_id': r['_internal_id'], 'public_id': r['id'],
                            'internal_patient_key': r['_patient_key']}
                           for subset, rs in prepared['records'].items() for r in rs]}
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(mapping, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return patients


def export(prepared, output, private_map):
    output, private_map = Path(output).resolve(), Path(private_map).resolve()
    if output.exists() or output == private_map or output in private_map.parents:
        raise ValueError('Output must be new and private mapping must stay outside it')
    output.parent.mkdir(parents=True, exist_ok=True)
    estimate = plan(prepared)
    free = shutil.disk_usage(output.parent).free
    if free < estimate['image_bytes'] + 512 * 2**20:
        raise ValueError('Insufficient disk space for primary assets')
    patients = private_mapping(prepared, private_map)
    output.mkdir()
    files, image_files, layout_bytes = {}, {}, 0

    def write(relative, content):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        files[relative] = file_sha(path)
        return path.stat().st_size

    for asset in prepared['images'].values():
        relative = asset['stage_path']
        if not re.fullmatch(r'assets/sha256/[0-9a-f]{2}/[0-9a-f]{64}\.(jpg|jpeg|png)', relative):
            raise ValueError('Image filename is not content-addressed')
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with (prepared['stage'] / relative).open('rb') as source, destination.open('wb') as target:
            for block in iter(lambda: source.read(2**20), b''):
                target.write(block)
                digest.update(block)
        if digest.hexdigest() != asset['sha256'] or destination.stat().st_size != asset['bytes']:
            raise ValueError('Copied image bytes differ from archive')
        files[relative] = asset['sha256']
        image_files[relative] = {'sha256': asset['sha256'], 'bytes': asset['bytes']}
    for key, layout in prepared['layouts'].items():
        layout_bytes += write('layouts/' + key + '.json', json.dumps(layout, ensure_ascii=False, sort_keys=True) + '\n')
    participants = defaultdict(set)
    for subset, records in prepared['records'].items():
        exported = []
        for row in records:
            record = {key: value for key, value in row.items() if not key.startswith('_')}
            record.update(participant_id=patients[row['_patient_key']], protocol=PROTOCOL,
                          prompt=prepared['prompts'][subset])
            participants[record['participant_id']].add(subset)
            exported.append(record)
        write('inputs/' + subset + '.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in exported))
        write('prompts/' + subset + '.txt', prepared['prompts'][subset])
        source = prepared['scorer_bundle'] / 'data' / (subset + '.jsonl')
        destination = output / 'answer_keys' / (subset + '.jsonl')
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, destination)
        files['answer_keys/' + subset + '.jsonl'] = file_sha(destination)
    write('participants.jsonl', ''.join(json.dumps({'participant_id': key, 'tasks': sorted(tasks)}) + '\n'
                                       for key, tasks in sorted(participants.items())))
    write('historical_input_comparison.json', json.dumps(prepared['history'], indent=2) + '\n')
    manifest = {'schema': PROTOCOL, **estimate, 'geometry_json_bytes': layout_bytes,
                'participant_count': len(participants), 'files': files, 'image_files': image_files,
                'canonical_prompt_sha256': PROMPT_SHA, 'source_metadata_sha256': prepared['source_hashes'],
                'asset_manifest_sha256': prepared['asset_manifest_sha256'],
                'reference_manifest_sha256': prepared['reference_manifest_sha256'],
                'source_inventory_sha256': prepared['source_inventory_sha256'],
                'source_pixels_modified': False, 'model_calls': 0, 'public_release': False,
                'protocol_notes': {
                    'canonical_prompts': 'Reviewed static Claude-Sonnet-4.6 archived prompt templates; all item-specific content excluded.',
                    'historical_identity_claim': False,
                    'history': 'Prompt, exact archived image-path, and relative image-order matches are reported separately. Matching a relative path does not verify the pixels at an alternate historical asset root. Unarchived wrapper transformations and wire requests are not reconstructed.',
                    't_stage_images': 'Original metadata card order retained; only opaque IDs and content-addressed image paths are exposed.',
                    'geometry': 'Numeric display geometry only; no structure/sequence gold, source paths, volume names, coordinate labels or identifiers.',
                    'participants': 'Random stable pseudonyms; matching MRI source patient IDs are linked across static tasks. Pathology cases use a separate namespace. The total counts public grouping entities, not independently verified unique people across modalities.',
                    'answers': 'Scorer keys are separate from inference inputs. Pass only model_request(record) to inference adapters.',
                    'limits': 'Static primary tasks only; cine, WSI originals, raw MRI volumes, model-specific wrappers and secondary invasion gold are not included.'}}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return verify(output, prepared['scorer_bundle'])


def verify(bundle, scorer_bundle=None):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest['schema'] != PROTOCOL:
        raise ValueError('Unsupported canonical input bundle')
    for relative, digest in manifest['files'].items():
        if file_sha(inside(bundle, relative)) != digest:
            raise ValueError('Canonical bundle hash mismatch')
    if scorer_bundle is not None and file_sha(Path(scorer_bundle) / 'manifest.json') != manifest['scorer_manifest_sha256']:
        raise ValueError('Candidate was built against another scorer bundle')
    inputs, counts = {}, {}
    for subset, (count, _, _, _) in ENDPOINTS.items():
        candidates = index(read_jsonl(bundle / 'inputs' / (subset + '.jsonl')))
        answers = index(read_jsonl(bundle / 'answer_keys' / (subset + '.jsonl')))
        if len(candidates) != count or set(candidates) != set(answers):
            raise ValueError('Input/answer sample join mismatch')
        if scorer_bundle is not None and answers != index(read_jsonl(Path(scorer_bundle) / 'data' / (subset + '.jsonl'))):
            raise ValueError('Answer keys differ from frozen scorer')
        for key, item in candidates.items():
            if hashlib.sha256(item['prompt'].encode()).hexdigest() != PROMPT_SHA[subset]:
                raise ValueError('Canonical prompt changed')
            if any(field in item for field in ('gold', 'answer', 'structure_id_gt', 't_stage_gt', 'invasion_gt', 'response')):
                raise ValueError('Scoring fields leaked into inference row')
            if any(path not in manifest['image_files'] for path in item['images']):
                raise ValueError('Inference references an unmanifested image')
            if any(path not in manifest['files'] for path in item.get('layout_paths', [])):
                raise ValueError('Inference references an unmanifested geometry file')
            if len(item['images']) != (27 if subset == 'T-Stg' else 1):
                raise ValueError('Wrong primary image count')
            if subset == 'Seq' and item['options'] != answers[key]['options']:
                raise ValueError('Sequence input/scorer choices disagree')
            if subset in {'Struct', 'T-Stg'} and sorted(item['structure_choices']) != answers[key][
                    'structure_choices' if subset == 'Struct' else 'base_structure_choices']:
                raise ValueError('Structure input/scorer choices disagree')
            if set(model_request(item)) != {'prompt', 'images'}:
                raise ValueError('Model adapter exposed unrelated fields')
        inputs[subset], counts[subset] = candidates, len(candidates)
    for patient in inputs['T-Stg'].values():
        cards = [inputs['Struct'][key] for key in patient['card_ids']]
        if [card['images'][0] for card in cards] != patient['images'] or any(card['participant_id'] != patient['participant_id'] for card in cards):
            raise ValueError('T-stage/structure patient-image linkage failed')
    return {'verified': True, 'items': counts, 'unique_images': len(manifest['image_files']),
            'image_bytes': sum(v['bytes'] for v in manifest['image_files'].values()),
            'geometry_json_count': manifest['geometry_json_count'], 'geometry_json_bytes': manifest['geometry_json_bytes'],
            'participant_count': manifest['participant_count'], 'source_pixels_modified': False,
            'scorer_manifest_sha256': manifest['scorer_manifest_sha256'], 'model_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path)
    parser.add_argument('--scorer-bundle', type=Path)
    parser.add_argument('--source-inventory', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--private-map', type=Path)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify(args.verify, args.scorer_bundle)
    else:
        if not (args.stage and args.scorer_bundle and args.source_inventory):
            parser.error('Export/plan requires --stage, --scorer-bundle and --source-inventory')
        prepared = prepare(args.stage, args.scorer_bundle, args.source_inventory)
        if args.plan:
            result = plan(prepared)
        else:
            if not (args.output and args.private_map):
                parser.error('Export requires --output and an external --private-map')
            result = export(prepared, args.output, args.private_map)
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
