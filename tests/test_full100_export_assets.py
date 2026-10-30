import unittest

from npcbench_eval.export_full100 import literal_replacer, validate_image_references


class ExportAssetTests(unittest.TestCase):
    def test_private_identifier_cannot_rewrite_generated_asset_digest(self):
        replace = literal_replacer({'/private/image.jpg': 'assets/ab123456cd.jpg', '123456': 'specimen-public'})
        self.assertEqual(replace('/private/image.jpg'), 'assets/ab123456cd.jpg')
        self.assertEqual(replace('specimen 123456'), 'specimen specimen-public')

    def test_overlapping_source_names_use_the_longest_match(self):
        replace = literal_replacer({'patient01': 'patient-public', 'patient01_turn2': 'sample-public'})
        self.assertEqual(replace('patient01_turn2 patient01'), 'sample-public patient-public')

    def test_missing_public_asset_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_image_references([{'images': ['assets/missing.jpg']}], {'assets/present.jpg'})
        validate_image_references([{'images': ['assets/present.jpg']}], {'assets/present.jpg'})


if __name__ == '__main__':
    unittest.main()
