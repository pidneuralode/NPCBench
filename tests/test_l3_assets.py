import json
from pathlib import Path
import tempfile
import unittest

from npcbench_eval.export_l3_assets import geometry, model_request, private_mapping, public_item_id
from npcbench_eval.l3 import project_rows


class L3AssetTests(unittest.TestCase):
    def test_public_id_joins_existing_scorer(self):
        source = {'question_id': 'synthetic-item', 'answer': '1', 'response': ''}
        frozen, _ = project_rows('Path', [source])
        self.assertEqual(public_item_id('Path', source['question_id']), frozen[0]['id'])
        self.assertNotEqual(public_item_id('Path', source['question_id']), public_item_id('Seq', source['question_id']))

    def test_structure_geometry_omits_target_and_private_fields(self):
        source = {'image_size': [1536, 2160], 'tile_size': [512, 512],
                  'patient_id': 'PRIVATE_PERSON', 'structure_en': 'TARGET_GOLD',
                  'source_path': '/private/record',
                  'panels': [{'panel_id': 1, 'row_index': 0, 'canvas_bbox_norm': [0, 0, 1, 1],
                              'sequence': 'private-series', 'candidate_count': 9}],
                  'visible_candidates': [{'coord_field': 'TARGET_GOLD', 'bbox_norm': [0, 0, 1, 1]}]}
        result = geometry('Struct', source, 'public-item')
        self.assertEqual(set(result), {'id', 'image_size', 'tile_size', 'panels'})
        self.assertEqual(set(result['panels'][0]), {'panel_id', 'row_index', 'canvas_bbox_norm'})
        serialized = json.dumps(result)
        for private in ['PRIVATE_PERSON', 'TARGET_GOLD', 'private-series', '/private/record']:
            self.assertNotIn(private, serialized)

    def test_sequence_geometry_drops_series_identity_and_gold(self):
        source = {'panel_shape': [3, 3], 'panel_size': [1536, 1536], 'tile_size': [512, 512],
                  'sequence_label_gt': 'SECRET_GOLD', 'resolved_volume_dir': '/private/volume',
                  'tiles': [{'grid_index': 1, 'grid_row': 0, 'grid_col': 0, 'slice_index': 99}]}
        result = geometry('Seq', source, 'public-item')
        self.assertEqual(set(result['tiles'][0]), {'grid_index', 'grid_row', 'grid_col'})
        self.assertNotIn('SECRET_GOLD', json.dumps(result))
        self.assertNotIn('/private/volume', json.dumps(result))

    def test_model_adapter_does_not_serialize_answers_or_participant(self):
        row = {'prompt': 'A task.', 'images': ['assets/sha256/example.jpg'],
               'gold': 'DO_NOT_SEND', 'participant_id': 'DO_NOT_SEND', 'structure_id_gt': 'DO_NOT_SEND'}
        self.assertEqual(model_request(row), {'prompt': 'A task.', 'images': ['assets/sha256/example.jpg']})
        self.assertNotIn('DO_NOT_SEND', json.dumps(model_request(row)))

    def test_private_mapping_is_stable_and_mode_0600(self):
        prepared = {'records': {'Struct': [{'id': 'public-card', '_internal_id': 'internal-card', '_patient_key': 'mri\0internal-patient'}],
                                'T-Stg': [{'id': 'public-patient', '_internal_id': 'internal-case', '_patient_key': 'mri\0internal-patient'}]}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'private.json'
            first = private_mapping(prepared, path)
            self.assertEqual(len(first), 1)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(first, private_mapping(prepared, path))
            self.assertTrue(next(iter(first.values())).startswith('L3P-'))
            self.assertNotIn('internal', next(iter(first.values())))

    def test_overbroad_private_mapping_permissions_are_rejected(self):
        prepared = {'records': {'Path': [{'id': 'public', '_internal_id': 'source', '_patient_key': 'pathology\0case'}]}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'private.json'
            private_mapping(prepared, path)
            path.chmod(0o644)
            with self.assertRaisesRegex(ValueError, '0600'):
                private_mapping(prepared, path)


if __name__ == '__main__':
    unittest.main()
