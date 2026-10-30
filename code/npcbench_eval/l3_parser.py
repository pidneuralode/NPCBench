"""Frozen L3 parsers, extracted without semantic edits from the CPU replay.

Only pure parser methods and their dependencies are retained. Provenance records
source and AST hashes; no clinical examples, inference, or networking is included.
"""
from __future__ import annotations
import json
import re

class NPCMMBenchBox:
    DEFAULT_EXCLUDED_BASE_STRUCTURES = ('nasopharynx',)

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

    @staticmethod
    def _normalize_t_stage(value):
        if isinstance(value, dict):
            for key in ('t_stage', 'stage', 'value', 'answer'):
                if key in value:
                    value = value[key]
                    break
        if value is None:
            return None
        match = re.search('t\\s*([1-4])', str(value).strip(), flags=re.IGNORECASE)
        if not match:
            return None
        return f'T{match.group(1)}'

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

    def _normalize_invasion_map(self, value, base_structure_choices):
        if value is None:
            return {}
        if isinstance(value, dict):
            for wrapper_key in ('structure_invasion', 'structure_invasions', 'base_structure_invasion', 'invasion', 'invasions'):
                if wrapper_key in value and isinstance(value[wrapper_key], (dict, list)):
                    return self._normalize_invasion_map(value[wrapper_key], base_structure_choices)
            normalized = {}
            for (key, raw_value) in value.items():
                if isinstance(raw_value, dict):
                    structure_source = raw_value.get('structure_id') or raw_value.get('structure')
                    if structure_source is not None:
                        key = structure_source
                    for invasion_key in ('invasion', 'invaded', 'value', 'answer'):
                        if invasion_key in raw_value:
                            raw_value = raw_value[invasion_key]
                            break
                structure_id = self._normalize_structure_id(key, base_structure_choices)
                invasion = self._normalize_invasion(raw_value)
                if structure_id is not None and invasion is not None:
                    normalized[structure_id] = max(normalized.get(structure_id, 0), invasion)
            return normalized
        if isinstance(value, list):
            normalized = {}
            for item in value:
                if not isinstance(item, dict):
                    continue
                structure_source = item.get('structure_id') or item.get('structure') or item.get('name') or item.get('label')
                invasion_source = None
                for invasion_key in ('invasion', 'invaded', 'value', 'answer'):
                    if invasion_key in item:
                        invasion_source = item[invasion_key]
                        break
                structure_id = self._normalize_structure_id(structure_source, base_structure_choices)
                invasion = self._normalize_invasion(invasion_source)
                if structure_id is not None and invasion is not None:
                    normalized[structure_id] = max(normalized.get(structure_id, 0), invasion)
            return normalized
        return {}

    def _parse_patient_prediction(self, response, base_structure_choices):
        (payload, payload_error) = self._extract_json_payload(response)
        parsed_prediction = {'structure_invasion': {}, 't_stage': None}
        parse_errors = []
        if payload_error:
            parse_errors.append(payload_error)
            return (parsed_prediction, parse_errors)
        invasion_source = None
        for key in ('structure_invasion', 'structure_invasions', 'base_structure_invasion', 'invasion', 'invasions'):
            if key in payload:
                invasion_source = payload.get(key)
                break
        parsed_prediction['structure_invasion'] = self._normalize_invasion_map(invasion_source, base_structure_choices)
        missing_invasion = [choice for choice in base_structure_choices if choice not in parsed_prediction['structure_invasion']]
        if missing_invasion:
            parse_errors.append(f'missing_structure_invasion:{len(missing_invasion)}')
        t_stage_source = payload.get('t_stage')
        if t_stage_source is None and 'stage' in payload:
            t_stage_source = payload.get('stage')
        parsed_prediction['t_stage'] = self._normalize_t_stage(t_stage_source)
        if parsed_prediction['t_stage'] is None:
            parse_errors.append('invalid_t_stage')
        return (parsed_prediction, parse_errors)

class NPCMMBenchSequence:

    @staticmethod
    def _extract_boxed_content(text):
        if not text:
            return ''
        matches = re.findall('\\\\boxed\\{([^{}]+)\\}', str(text))
        if not matches:
            return ''
        return matches[-1].strip()

    @staticmethod
    def _strip_latex_text_wrappers(text):
        text = re.sub('\\\\text(?:bf|it)?\\{([^{}]+)\\}', '\\1', text)
        text = re.sub('\\\\mathrm\\{([^{}]+)\\}', '\\1', text)
        return text

    def _normalize_option_id(self, value, sample):
        if value is None:
            return None
        option_ids = {option['option_id'] for option in sample['options']}
        text = self._strip_latex_text_wrappers(str(value))
        text = text.replace('{', ' ').replace('}', ' ')
        text = ' '.join(text.split()).strip()
        if not text:
            return None
        upper = text.upper()
        if upper in option_ids:
            return upper
        leading_match = re.match('^\\(?\\s*([A-Z])(?:[\\s\\.\\):]|$)', upper)
        if leading_match:
            candidate = leading_match.group(1)
            if candidate in option_ids:
                return candidate
        lowered = text.lower()
        for option in sample['options']:
            aliases = {option['sequence_label'].lower(), option['sequence_cn'].lower(), f"{option['option_id'].lower()}. {option['sequence_label'].lower()}", f"{option['option_id'].lower()}. {option['sequence_cn'].lower()}", f"{option['option_id'].lower()} {option['sequence_label'].lower()}", f"{option['option_id'].lower()} {option['sequence_cn'].lower()}"}
            if lowered in aliases:
                return option['option_id']
        if len(lowered) <= 32:
            for option in sample['options']:
                if option['sequence_label'].lower() in lowered:
                    return option['option_id']
                if option['sequence_cn'].lower() in lowered:
                    return option['option_id']
        return None

    def _parse_prediction(self, response, sample):
        boxed_content = self._extract_boxed_content(response)
        strict_box_option = None
        if boxed_content:
            strict_box_match = re.fullmatch('\\s*([A-Z])\\s*', boxed_content.upper())
            if strict_box_match:
                candidate = strict_box_match.group(1)
                if candidate in {option['option_id'] for option in sample['options']}:
                    strict_box_option = candidate
        normalized_response = parse_response('' if response is None else str(response))
        predicted_option_id = self._normalize_option_id(normalized_response, sample)
        parse_errors = []
        if not boxed_content:
            parse_errors.append('boxed_not_found')
        elif strict_box_option is None:
            parse_errors.append('invalid_box_content')
        if predicted_option_id is None:
            parse_errors.append('invalid_option')
        option_to_label = {option['option_id']: option['sequence_label'] for option in sample['options']}
        return ({'boxed_content': boxed_content or None, 'strict_box_option_id': strict_box_option, 'normalized_response': normalized_response, 'option_id': predicted_option_id, 'sequence_label': option_to_label.get(predicted_option_id)}, parse_errors)

def _extract_between_tags(text, tag_name):
    pattern = f'<{tag_name}>(.*?)</{tag_name}>'
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if not matches:
        return ''
    return ' '.join((match.strip() for match in matches if str(match).strip())).strip()

def _extract_boxed_content(text):
    matches = re.findall('\\\\boxed\\{([^{}]+)\\}', text)
    if not matches:
        return ''
    return matches[-1].strip()

class MMBench_Pathology:

    @staticmethod
    def _normalize_text(text):
        return ' '.join(str(text).strip().split())

    def _parse_prediction(self, response):
        if response is None:
            return ''
        response = str(response).strip()
        if not response:
            return ''
        boxed = _extract_boxed_content(response)
        if boxed:
            boxed = self._normalize_text(boxed)
            if boxed in {'0', '1'}:
                return boxed
        tagged = _extract_between_tags(response, 'answer')
        tagged = self._normalize_text(tagged)
        if tagged in {'0', '1'}:
            return tagged
        cleaned = self._normalize_text(response)
        if cleaned in {'0', '1'}:
            return cleaned
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        if lines:
            last_line = self._normalize_text(lines[-1]).rstrip('。.,:：;；')
            if last_line in {'0', '1'}:
                return last_line
        digit_candidates = re.findall('(?<!\\d)([01])(?!\\d)', response)
        if digit_candidates:
            return digit_candidates[-1]
        lowered = response.lower()
        if any((token in lowered for token in ['nasopharyngeal carcinoma', 'npc', '鼻咽癌'])):
            return '1'
        if any((token in lowered for token in ['normal', '正常'])):
            return '0'
        return ''

def parse_response(response):
    response = response.lower()
    if 'boxed' in response:
        response = extract_boxed_content(response)
    elif '<answer>' in response:
        response = extract(response, 'answer')
    tail = response[-1200:]
    explicit_letter_patterns = ['assistant\\s*final\\s*[:：]?\\s*([a-j])\\b', 'assistantfinal\\s*[:：]?\\s*([a-j])\\b', '(?:final answer is|final answer|answer is|answer:|答案是|答案[:：]|最终选择)\\s*(?:option|选项)?\\s*([a-j])\\b', '(?:therefore|thus|so|因此|所以)[^a-j]{0,30}(?:answer|答案|选项)[^a-j]{0,15}([a-j])\\b']
    for pattern in explicit_letter_patterns:
        matches = re.findall(pattern, tail)
        if matches:
            return matches[-1]
    trailing_letter = re.search('(?:^|[\\s\\"\'`：:])([a-j])\\s*$', response.strip())
    if trailing_letter:
        return trailing_letter.group(1)
    answer_patterns = ['final answer is', 'final answer', '**answer**:', '**answer**', '*answer*:', '**answer:**', 'answer is', 'answer:', '答案是', '答案:', '答案：']
    for answer_pattern in answer_patterns:
        if answer_pattern in response:
            response = response.split(answer_pattern)[-1]
    return response

def get_content_between_a_b(start_tag, end_tag, text):
    extracted_text = ''
    start_index = text.find(start_tag)
    while start_index != -1:
        end_index = text.find(end_tag, start_index + len(start_tag))
        if end_index != -1:
            extracted_text += text[start_index + len(start_tag):end_index] + ' '
            start_index = text.find(start_tag, end_index + len(end_tag))
        else:
            break
    return extracted_text.strip()

def extract(text, type, hard=True):
    if text:
        target_str = get_content_between_a_b(f'<{type}>', f'</{type}>', text)
        if target_str:
            return target_str
        elif hard:
            return text
        else:
            return ''
    else:
        return ''

def extract_boxed_content(text: str) -> str:
    """
    Extracts answers in \\boxed{}.
    """
    depth = 0
    start_pos = text.rfind('\\boxed{')
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len('\\boxed{'):]
        for (i, char) in enumerate(content):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
            if depth == -1:
                end_pos = i
                break
    if end_pos != -1:
        return content[:end_pos].strip()
    return 'None'

STRUCTURE_ROWS = [{'structure_id': 'nasal_passage', 'structure_en': 'nasal_passage', 'structure_cn': '鼻腔', 'base_structure_en': 'nasal_passage', 'base_structure_cn': '鼻腔'}, {'structure_id': 'oropharynx', 'structure_en': 'oropharynx', 'structure_cn': '口咽', 'base_structure_en': 'oropharynx', 'base_structure_cn': '口咽'}, {'structure_id': 'levator_veli_palatini_left', 'structure_en': 'levator_veli_palatini_left', 'structure_cn': '左侧腭帆提肌', 'base_structure_en': 'levator_veli_palatini', 'base_structure_cn': '腭帆提肌'}, {'structure_id': 'levator_veli_palatini_right', 'structure_en': 'levator_veli_palatini_right', 'structure_cn': '右侧腭帆提肌', 'base_structure_en': 'levator_veli_palatini', 'base_structure_cn': '腭帆提肌'}, {'structure_id': 'parapharyngeal_space_left', 'structure_en': 'parapharyngeal_space_left', 'structure_cn': '左侧咽旁间隙', 'base_structure_en': 'parapharyngeal_space', 'base_structure_cn': '咽旁间隙'}, {'structure_id': 'parapharyngeal_space_right', 'structure_en': 'parapharyngeal_space_right', 'structure_cn': '右侧咽旁间隙', 'base_structure_en': 'parapharyngeal_space', 'base_structure_cn': '咽旁间隙'}, {'structure_id': 'carotid_sheath_left', 'structure_en': 'carotid_sheath_left', 'structure_cn': '左侧颈动脉鞘', 'base_structure_en': 'carotid_sheath', 'base_structure_cn': '颈动脉鞘'}, {'structure_id': 'carotid_sheath_right', 'structure_en': 'carotid_sheath_right', 'structure_cn': '右侧颈动脉鞘', 'base_structure_en': 'carotid_sheath', 'base_structure_cn': '颈动脉鞘'}, {'structure_id': 'longus_capitis_left', 'structure_en': 'longus_capitis_left', 'structure_cn': '左侧头长肌', 'base_structure_en': 'longus_capitis', 'base_structure_cn': '头长肌'}, {'structure_id': 'longus_capitis_right', 'structure_en': 'longus_capitis_right', 'structure_cn': '右侧头长肌', 'base_structure_en': 'longus_capitis', 'base_structure_cn': '头长肌'}, {'structure_id': 'medial_pterygoid_left', 'structure_en': 'medial_pterygoid_left', 'structure_cn': '左侧翼内肌', 'base_structure_en': 'medial_pterygoid', 'base_structure_cn': '翼内肌'}, {'structure_id': 'medial_pterygoid_right', 'structure_en': 'medial_pterygoid_right', 'structure_cn': '右侧翼内肌', 'base_structure_en': 'medial_pterygoid', 'base_structure_cn': '翼内肌'}, {'structure_id': 'lateral_pterygoid_left', 'structure_en': 'lateral_pterygoid_left', 'structure_cn': '左侧翼外肌', 'base_structure_en': 'lateral_pterygoid', 'base_structure_cn': '翼外肌'}, {'structure_id': 'lateral_pterygoid_right', 'structure_en': 'lateral_pterygoid_right', 'structure_cn': '右侧翼外肌', 'base_structure_en': 'lateral_pterygoid', 'base_structure_cn': '翼外肌'}, {'structure_id': 'pterygoid_base_left', 'structure_en': 'pterygoid_base_left', 'structure_cn': '左侧翼突基底部', 'base_structure_en': 'pterygoid_base', 'base_structure_cn': '翼突基底部'}, {'structure_id': 'pterygoid_base_right', 'structure_en': 'pterygoid_base_right', 'structure_cn': '右侧翼突基底部', 'base_structure_en': 'pterygoid_base', 'base_structure_cn': '翼突基底部'}, {'structure_id': 'sphenoid_base', 'structure_en': 'sphenoid_base', 'structure_cn': '蝶骨基底部', 'base_structure_en': 'sphenoid_base', 'base_structure_cn': '蝶骨基底部'}, {'structure_id': 'clivus', 'structure_en': 'clivus', 'structure_cn': '斜坡', 'base_structure_en': 'clivus', 'base_structure_cn': '斜坡'}, {'structure_id': 'petrous_apex_left', 'structure_en': 'petrous_apex_left', 'structure_cn': '左侧岩尖', 'base_structure_en': 'petrous_apex', 'base_structure_cn': '岩尖'}, {'structure_id': 'petrous_apex_right', 'structure_en': 'petrous_apex_right', 'structure_cn': '右侧岩尖', 'base_structure_en': 'petrous_apex', 'base_structure_cn': '岩尖'}, {'structure_id': 'atlas', 'structure_en': 'atlas', 'structure_cn': '寰椎', 'base_structure_en': 'atlas', 'base_structure_cn': '寰椎'}, {'structure_id': 'axis', 'structure_en': 'axis', 'structure_cn': '枢椎', 'base_structure_en': 'axis', 'base_structure_cn': '枢椎'}, {'structure_id': 'sphenoid_sinus', 'structure_en': 'sphenoid_sinus', 'structure_cn': '蝶窦', 'base_structure_en': 'sphenoid_sinus', 'base_structure_cn': '蝶窦'}, {'structure_id': 'ethmoid_sinus', 'structure_en': 'ethmoid_sinus', 'structure_cn': '筛窦', 'base_structure_en': 'ethmoid_sinus', 'base_structure_cn': '筛窦'}, {'structure_id': 'maxillary_sinus_left', 'structure_en': 'maxillary_sinus_left', 'structure_cn': '左侧上颌窦', 'base_structure_en': 'maxillary_sinus', 'base_structure_cn': '上颌窦'}, {'structure_id': 'maxillary_sinus_right', 'structure_en': 'maxillary_sinus_right', 'structure_cn': '右侧上颌窦', 'base_structure_en': 'maxillary_sinus', 'base_structure_cn': '上颌窦'}, {'structure_id': 'cavernous_sinus', 'structure_en': 'cavernous_sinus', 'structure_cn': '海绵窦', 'base_structure_en': 'cavernous_sinus', 'base_structure_cn': '海绵窦'}]
