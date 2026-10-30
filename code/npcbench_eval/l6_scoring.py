import re
import json

class NPCFullCarePath25:
    OPTION_LABELS = 'ABCDEFGHIJKL'
    STRUCTURE_KEYS = ['鼻咽', '腭帆提肌', '咽旁间隙', '头长肌', '翼内肌', '蝶窦', '鼻腔', '颈动脉鞘', '翼外肌', '筛窦', '上颌窦', '斜坡', '海绵窦', '口咽', '翼突基底部', '蝶骨基底部', '岩尖', '寰椎', '枢椎']
    EXCLUDED_STRUCTURE_KEYS = {'鼻咽'}

    @staticmethod
    def _strip_think_blocks(text):
        return re.sub('<think>.*?</think>', '', str(text or ''), flags=re.DOTALL).strip()

    @classmethod
    def _parse_binary(cls, response):
        text = cls._strip_think_blocks(response)
        boxed = re.findall('\\\\boxed\\{([^{}]+)\\}', text)
        if boxed and boxed[-1].strip() in {'0', '1'}:
            return boxed[-1].strip()
        tagged = re.findall('<answer>(.*?)</answer>', text, flags=re.DOTALL)
        if tagged:
            value = ' '.join(tagged[-1].strip().split())
            if value in {'0', '1'}:
                return value
        cleaned = ' '.join(text.split()).strip()
        if cleaned in {'0', '1'}:
            return cleaned
        digit_candidates = re.findall('(?<!\\d)([01])(?!\\d)', text)
        if digit_candidates:
            return digit_candidates[-1]
        lowered = text.lower()
        if any((token in lowered for token in ['nasopharyngeal carcinoma', 'npc', '鼻咽癌'])):
            return '1'
        if any((token in lowered for token in ['normal', '正常'])):
            return '0'
        return ''

    @classmethod
    def _parse_letter(cls, response):
        text = cls._strip_think_blocks(response).strip()
        boxed = re.findall('\\\\boxed\\{([A-L])\\}', text, flags=re.IGNORECASE)
        if boxed:
            return boxed[-1].upper()
        tagged = re.findall('<answer>(.*?)</answer>', text, flags=re.DOTALL | re.IGNORECASE)
        if tagged:
            match = re.search('\\b([A-L])\\b', tagged[-1], flags=re.IGNORECASE)
            if match:
                return match.group(1).upper()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines[-5:]):
            cleaned = line.strip().rstrip('。.,:：;；')
            if re.fullmatch('[A-L]', cleaned, flags=re.IGNORECASE):
                return cleaned.upper()
            match = re.search('(?:答案|选项|answer)\\s*[:：]?\\s*([A-L])\\b', cleaned, flags=re.IGNORECASE)
            if match:
                return match.group(1).upper()
        candidates = re.findall('\\b([A-L])\\b', text, flags=re.IGNORECASE)
        return candidates[-1].upper() if candidates else ''

    @classmethod
    def _json_candidates(cls, response):
        text = cls._strip_think_blocks(response)
        candidates = []
        candidates.extend(re.findall('```json\\s*(\\{.*?\\})\\s*```', text, flags=re.DOTALL))
        candidates.extend(re.findall('```\\s*(\\{.*?\\})\\s*```', text, flags=re.DOTALL))
        starts = [idx for idx, char in enumerate(text) if char == '{']
        for start in starts:
            depth = 0
            for end in range(start, len(text)):
                if text[end] == '{':
                    depth += 1
                elif text[end] == '}':
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start:end + 1])
                        break
        return candidates

    @classmethod
    def _parse_json(cls, response):
        for candidate in cls._json_candidates(response):
            try:
                return json.loads(candidate)
            except Exception:
                continue
        return {}

    @classmethod
    def _parse_t_stage(cls, response):
        payload = cls._parse_json(response)
        for key in ['T_stage', 't_stage', 'T', 't']:
            value = payload.get(key) if isinstance(payload, dict) else None
            if value:
                match = re.search('T\\s*([1-4])', str(value), flags=re.IGNORECASE)
                if match:
                    return f'T{match.group(1)}'
        match = re.search('\\bT\\s*([1-4])\\b', cls._strip_think_blocks(response), flags=re.IGNORECASE)
        return f'T{match.group(1)}' if match else ''

    @classmethod
    def _parse_structure_dict(cls, response):
        payload = cls._parse_json(response)
        result = {key: 0 for key in cls.STRUCTURE_KEYS}
        if isinstance(payload, dict):
            raw_structure = payload.get('structure_invasion') or payload.get('structure_invasion_19')
            if isinstance(raw_structure, dict):
                for key in cls.STRUCTURE_KEYS:
                    value = raw_structure.get(key)
                    if isinstance(value, str):
                        value = value.strip()
                    result[key] = 1 if value in {1, True, '1', 'yes', 'Yes', '阳性', '受侵'} else 0
            involved = payload.get('involved_structures')
            if isinstance(involved, list):
                involved_text = ' '.join((str(item) for item in involved))
                for key in cls.STRUCTURE_KEYS:
                    if key in involved_text:
                        result[key] = 1
        return result

    @classmethod
    def _scored_structure_keys(cls, gold):
        return [key for key in cls.STRUCTURE_KEYS if key in gold and key not in cls.EXCLUDED_STRUCTURE_KEYS]

    @classmethod
    def _structure_scores(cls, pred, gold):
        keys = cls._scored_structure_keys(gold)
        tp = sum((1 for key in keys if pred.get(key, 0) == 1 and gold.get(key, 0) == 1))
        fp = sum((1 for key in keys if pred.get(key, 0) == 1 and gold.get(key, 0) == 0))
        fn = sum((1 for key in keys if pred.get(key, 0) == 0 and gold.get(key, 0) == 1))
        right = sum((1 for key in keys if pred.get(key, 0) == gold.get(key, 0)))
        total = len(keys)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        accuracy = right / total if total else 0.0
        complete_correct = bool(keys) and right == total
        return {'keys': keys, 'tp': tp, 'fp': fp, 'fn': fn, 'right': right, 'total': total, 'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy, 'complete_correct': complete_correct}
