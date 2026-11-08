"""Parse the anatomical structure named in a model answer."""
import json
import re


class StructureCardParser:
    @staticmethod
    def _canonicalize_text(text):
        text = str(text).strip().lower()
        text = re.sub('[\\s\\-]+', '_', text)
        text = re.sub('_+', '_', text)
        return text.strip('_')

    @classmethod
    def _alias_candidates(cls, text):
        if text is None:
            return set()
        raw = str(text).strip()
        if not raw:
            return set()
        canonical = cls._canonicalize_text(raw)
        compact = re.sub('[\\s_\\-]+', '', raw.strip().lower())
        return {raw.lower(), canonical, compact}

    def _load_structure_aliases(self):
        alias_map = {}
        if not self.structure_rows:
            return alias_map
        for row in self.structure_rows:
            structure_id = row.get('structure_id')
            base_structure = row.get('base_structure_en') or structure_id
            if not structure_id:
                continue
            for value in (row.get('structure_id'), row.get('structure_en'), row.get('structure_cn')):
                for alias in self._alias_candidates(value):
                    alias_map[alias] = structure_id
            for value in (row.get('base_structure_en'), row.get('base_structure_cn')):
                for alias in self._alias_candidates(value):
                    alias_map[alias] = base_structure
        return alias_map

    def _structure_to_base(self, structure_id):
        if structure_id is None:
            return None
        normalized = self._canonicalize_text(structure_id)
        if normalized in self.structure_id_to_base:
            return self.structure_id_to_base[normalized]
        return re.sub('_(left|right)$', '', normalized)

    @staticmethod
    def _strip_think_blocks(text):
        return re.sub('<think>.*?</think>', '', text or '', flags=re.IGNORECASE | re.DOTALL)

    @staticmethod
    def _find_json_objects(text):
        objects = []
        start_idx = None
        depth = 0
        in_string = False
        escape = False
        for (idx, char) in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif char == '\\':
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == '{':
                if depth == 0:
                    start_idx = idx
                depth += 1
            elif char == '}':
                if depth == 0:
                    continue
                depth -= 1
                if depth == 0 and start_idx is not None:
                    objects.append(text[start_idx:idx + 1])
                    start_idx = None
        return objects

    def _extract_json_payload(self, response):
        cleaned = self._strip_think_blocks(response).strip()
        candidates = []
        answer_match = re.search('<answer>\\s*(.*?)\\s*</answer>', cleaned, flags=re.IGNORECASE | re.DOTALL)
        if answer_match:
            candidates.append(answer_match.group(1).strip())
        fenced_matches = re.findall('```(?:json)?\\s*(.*?)\\s*```', cleaned, flags=re.IGNORECASE | re.DOTALL)
        candidates.extend((match.strip() for match in fenced_matches if match.strip()))
        candidates.append(cleaned)
        seen = set()
        expanded_candidates = []
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                expanded_candidates.append(candidate)
            for obj in self._find_json_objects(candidate):
                if obj not in seen:
                    seen.add(obj)
                    expanded_candidates.append(obj)
        for candidate in expanded_candidates:
            try:
                payload = json.loads(candidate)
                if not isinstance(payload, dict):
                    return (None, 'json_not_object')
                return (payload, None)
            except json.JSONDecodeError:
                continue
        return (None, 'json_not_found')

    def _normalize_structure_id(self, value, structure_choices):
        if isinstance(value, dict):
            for key in ('structure_id', 'structure', 'name', 'label', 'answer', 'value'):
                if key in value:
                    value = value[key]
                    break
        if value is None:
            return None
        structure_choice_set = set(structure_choices)
        for alias in self._alias_candidates(value):
            if alias in self.structure_alias_to_id:
                candidate = self.structure_alias_to_id[alias]
                if candidate in structure_choice_set:
                    return candidate
                base_candidate = self._structure_to_base(candidate)
                if base_candidate in structure_choice_set:
                    return base_candidate
        normalized = self._canonicalize_text(value)
        if normalized in structure_choice_set:
            return normalized
        base_normalized = self._structure_to_base(normalized)
        if base_normalized in structure_choice_set:
            return base_normalized
        return None

    @staticmethod
    def _normalize_invasion(value):
        if isinstance(value, dict):
            for key in ('invaded', 'value', 'label', 'answer'):
                if key in value:
                    value = value[key]
                    break
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            value = int(value)
            if value in (0, 1):
                return value
            return None
        if value is None:
            return None
        normalized = str(value).strip().lower()
        positive = {'1', 'yes', 'true', 'positive', '受侵', '侵犯', '有', '是'}
        negative = {'0', 'no', 'false', 'negative', '未受侵', '无受侵', '未侵犯', '没有', '否'}
        if normalized in positive:
            return 1
        if normalized in negative:
            return 0
        return None

    def _parse_card_prediction(self, response, structure_choices):
        (payload, payload_error) = self._extract_json_payload(response)
        parsed_prediction = {'structure_id': None, 'invasion': None}
        parse_errors = []
        if payload_error:
            parse_errors.append(payload_error)
            return (parsed_prediction, parse_errors)
        structure_source = payload.get('structure_id')
        if structure_source is None:
            for key in ('structure', 'structure_name', 'anatomy', 'target_structure'):
                if key in payload:
                    structure_source = payload.get(key)
                    break
        parsed_prediction['structure_id'] = self._normalize_structure_id(structure_source, structure_choices)
        if parsed_prediction['structure_id'] is None:
            parse_errors.append('invalid_structure_id')
        invasion_source = payload.get('invasion')
        if invasion_source is None and 'invaded' in payload:
            invasion_source = payload.get('invaded')
        parsed_prediction['invasion'] = self._normalize_invasion(invasion_source)
        if parsed_prediction['invasion'] is None:
            parse_errors.append('invalid_invasion')
        return (parsed_prediction, parse_errors)


STRUCTURE_ROWS = [{'structure_id': 'nasal_passage',
  'structure_en': 'nasal_passage',
  'structure_cn': '鼻腔',
  'base_structure_en': 'nasal_passage',
  'base_structure_cn': '鼻腔'},
 {'structure_id': 'oropharynx',
  'structure_en': 'oropharynx',
  'structure_cn': '口咽',
  'base_structure_en': 'oropharynx',
  'base_structure_cn': '口咽'},
 {'structure_id': 'levator_veli_palatini_left',
  'structure_en': 'levator_veli_palatini_left',
  'structure_cn': '左侧腭帆提肌',
  'base_structure_en': 'levator_veli_palatini',
  'base_structure_cn': '腭帆提肌'},
 {'structure_id': 'levator_veli_palatini_right',
  'structure_en': 'levator_veli_palatini_right',
  'structure_cn': '右侧腭帆提肌',
  'base_structure_en': 'levator_veli_palatini',
  'base_structure_cn': '腭帆提肌'},
 {'structure_id': 'parapharyngeal_space_left',
  'structure_en': 'parapharyngeal_space_left',
  'structure_cn': '左侧咽旁间隙',
  'base_structure_en': 'parapharyngeal_space',
  'base_structure_cn': '咽旁间隙'},
 {'structure_id': 'parapharyngeal_space_right',
  'structure_en': 'parapharyngeal_space_right',
  'structure_cn': '右侧咽旁间隙',
  'base_structure_en': 'parapharyngeal_space',
  'base_structure_cn': '咽旁间隙'},
 {'structure_id': 'carotid_sheath_left',
  'structure_en': 'carotid_sheath_left',
  'structure_cn': '左侧颈动脉鞘',
  'base_structure_en': 'carotid_sheath',
  'base_structure_cn': '颈动脉鞘'},
 {'structure_id': 'carotid_sheath_right',
  'structure_en': 'carotid_sheath_right',
  'structure_cn': '右侧颈动脉鞘',
  'base_structure_en': 'carotid_sheath',
  'base_structure_cn': '颈动脉鞘'},
 {'structure_id': 'longus_capitis_left',
  'structure_en': 'longus_capitis_left',
  'structure_cn': '左侧头长肌',
  'base_structure_en': 'longus_capitis',
  'base_structure_cn': '头长肌'},
 {'structure_id': 'longus_capitis_right',
  'structure_en': 'longus_capitis_right',
  'structure_cn': '右侧头长肌',
  'base_structure_en': 'longus_capitis',
  'base_structure_cn': '头长肌'},
 {'structure_id': 'medial_pterygoid_left',
  'structure_en': 'medial_pterygoid_left',
  'structure_cn': '左侧翼内肌',
  'base_structure_en': 'medial_pterygoid',
  'base_structure_cn': '翼内肌'},
 {'structure_id': 'medial_pterygoid_right',
  'structure_en': 'medial_pterygoid_right',
  'structure_cn': '右侧翼内肌',
  'base_structure_en': 'medial_pterygoid',
  'base_structure_cn': '翼内肌'},
 {'structure_id': 'lateral_pterygoid_left',
  'structure_en': 'lateral_pterygoid_left',
  'structure_cn': '左侧翼外肌',
  'base_structure_en': 'lateral_pterygoid',
  'base_structure_cn': '翼外肌'},
 {'structure_id': 'lateral_pterygoid_right',
  'structure_en': 'lateral_pterygoid_right',
  'structure_cn': '右侧翼外肌',
  'base_structure_en': 'lateral_pterygoid',
  'base_structure_cn': '翼外肌'},
 {'structure_id': 'pterygoid_base_left',
  'structure_en': 'pterygoid_base_left',
  'structure_cn': '左侧翼突基底部',
  'base_structure_en': 'pterygoid_base',
  'base_structure_cn': '翼突基底部'},
 {'structure_id': 'pterygoid_base_right',
  'structure_en': 'pterygoid_base_right',
  'structure_cn': '右侧翼突基底部',
  'base_structure_en': 'pterygoid_base',
  'base_structure_cn': '翼突基底部'},
 {'structure_id': 'sphenoid_base',
  'structure_en': 'sphenoid_base',
  'structure_cn': '蝶骨基底部',
  'base_structure_en': 'sphenoid_base',
  'base_structure_cn': '蝶骨基底部'},
 {'structure_id': 'clivus',
  'structure_en': 'clivus',
  'structure_cn': '斜坡',
  'base_structure_en': 'clivus',
  'base_structure_cn': '斜坡'},
 {'structure_id': 'petrous_apex_left',
  'structure_en': 'petrous_apex_left',
  'structure_cn': '左侧岩尖',
  'base_structure_en': 'petrous_apex',
  'base_structure_cn': '岩尖'},
 {'structure_id': 'petrous_apex_right',
  'structure_en': 'petrous_apex_right',
  'structure_cn': '右侧岩尖',
  'base_structure_en': 'petrous_apex',
  'base_structure_cn': '岩尖'},
 {'structure_id': 'atlas',
  'structure_en': 'atlas',
  'structure_cn': '寰椎',
  'base_structure_en': 'atlas',
  'base_structure_cn': '寰椎'},
 {'structure_id': 'axis',
  'structure_en': 'axis',
  'structure_cn': '枢椎',
  'base_structure_en': 'axis',
  'base_structure_cn': '枢椎'},
 {'structure_id': 'sphenoid_sinus',
  'structure_en': 'sphenoid_sinus',
  'structure_cn': '蝶窦',
  'base_structure_en': 'sphenoid_sinus',
  'base_structure_cn': '蝶窦'},
 {'structure_id': 'ethmoid_sinus',
  'structure_en': 'ethmoid_sinus',
  'structure_cn': '筛窦',
  'base_structure_en': 'ethmoid_sinus',
  'base_structure_cn': '筛窦'},
 {'structure_id': 'maxillary_sinus_left',
  'structure_en': 'maxillary_sinus_left',
  'structure_cn': '左侧上颌窦',
  'base_structure_en': 'maxillary_sinus',
  'base_structure_cn': '上颌窦'},
 {'structure_id': 'maxillary_sinus_right',
  'structure_en': 'maxillary_sinus_right',
  'structure_cn': '右侧上颌窦',
  'base_structure_en': 'maxillary_sinus',
  'base_structure_cn': '上颌窦'},
 {'structure_id': 'cavernous_sinus',
  'structure_en': 'cavernous_sinus',
  'structure_cn': '海绵窦',
  'base_structure_en': 'cavernous_sinus',
  'base_structure_cn': '海绵窦'}]
