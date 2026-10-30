"""Frozen LongBenchQA parser; extracted verbatim from primary_parsers_v20261025.py.
Source SHA256: 73698a38532329e1defd3f12a062bb600de2091f2efb43f9634b995dffa4caba
"""
from __future__ import annotations
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

class LongBenchQA:
    OPTION_LABELS = 'ABCDEFGHIJKL'

    def _labels(self, sample):
        return set((str(label).upper() for label in sample.get('options', {}).keys()))

    def parse_answer(self, response: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        text = response or ''
        labels = self._labels(sample)
        if not labels:
            return self._parse_result('', False, 'missing_options', [], False)
        label_class = ''.join(sorted(labels))
        patterns = [('boxed', f'\\\\boxed\\s*\\{{\\s*([{label_class}{label_class.lower()}])\\s*\\}}'), ('boxed_text', f'\\\\boxed\\s*\\{{\\s*\\\\(?:text|mathrm|mathbf|mathit|operatorname|textbf)\\s*\\{{\\s*([{label_class}{label_class.lower()}])\\s*\\}}\\s*\\}}'), ('xml_answer', f'<answer>\\s*([{label_class}{label_class.lower()}])\\s*</answer>'), ('html_span_answer', f'<span\\b[^>]*>\\s*([{label_class}{label_class.lower()}])\\s*</span>'), ('explicit_marker', f'(?:final\\s+answer|answer|答案|选择|我选|正确选项)\\s*(?:is|为|是|:|：)?\\s*(?:option|选项)?\\s*([{label_class}{label_class.lower()}])\\b')]
        for method, pattern in patterns:
            candidates = [(method, match.group(1).upper(), match.group(0)) for match in re.finditer(pattern, text, flags=re.IGNORECASE) if match.group(1).upper() in labels]
            if candidates:
                pred, valid, conflict = self._resolve_candidates(candidates)
                return self._parse_result(pred, valid, method, candidates, conflict)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            last = re.sub('^[\\-\\*\\d\\.\\)\\s]+', '', lines[-1].strip().strip('`*_# '))
            match = re.fullmatch(f'(?:option|选项)?\\s*([{label_class}{label_class.lower()}])\\s*(?:[\\.。!！]?)', last, flags=re.IGNORECASE)
            if match and match.group(1).upper() in labels:
                candidate = [('last_line_letter', match.group(1).upper(), lines[-1])]
                return self._parse_result(candidate[0][1], True, 'last_line_letter', candidate, False)
        concise = text.strip().strip('`*_# ')
        match = re.fullmatch(f'(?:\\\\boxed\\{{)?\\s*([{label_class}{label_class.lower()}])\\s*(?:\\}})?\\s*[\\.。]?', concise, flags=re.IGNORECASE)
        if match and match.group(1).upper() in labels:
            candidate = [('direct_letter', match.group(1).upper(), concise)]
            return self._parse_result(candidate[0][1], True, 'direct_letter', candidate, False)
        containment = []
        normalized_tail = self._normalize_text(text[-2000:])
        for label, option_text in sample.get('options', {}).items():
            normalized_option = self._normalize_text(option_text)
            if len(normalized_option) >= 8 and normalized_option in normalized_tail:
                containment.append(('option_text_containment', str(label).upper(), str(option_text)[:120]))
        if containment:
            pred, valid, conflict = self._resolve_candidates(containment)
            return self._parse_result(pred, valid, 'option_text_containment', containment, conflict)
        best_label = ''
        best_score = 0.0
        comparison_window = self._normalize_text(text[-2000:])
        for label, option_text in sample.get('options', {}).items():
            normalized_option = self._normalize_text(option_text)
            if not normalized_option:
                continue
            window = comparison_window[-max(120, len(normalized_option) * 2):]
            score = SequenceMatcher(None, normalized_option, window).ratio()
            if score > best_score:
                best_label = str(label).upper()
                best_score = score
        if best_label and best_score >= 0.88:
            candidate = [('option_text_similarity', best_label, round(best_score, 4))]
            return self._parse_result(best_label, True, 'option_text_similarity', candidate, False)
        return self._parse_result('', False, 'unparsed', [], False)

    def _resolve_candidates(self, candidates: List[Tuple[str, str, Any]]):
        values = [candidate[1] for candidate in candidates]
        unique = sorted(set(values))
        if len(unique) == 1:
            return (values[-1], True, False)
        return ('', False, True)

    def _parse_result(self, pred_answer, valid, method, candidates, conflict):
        return {'pred_answer': pred_answer, 'parse_valid': valid, 'parse_method': method, 'parse_conflict': conflict, 'parse_candidates': [{'method': method_name, 'answer': answer, 'evidence': evidence} for method_name, answer, evidence in candidates]}

    def _normalize_text(self, text):
        return re.sub('\\s+', ' ', str(text).lower()).strip()

    def _add_stat(self, stats, key, correct):
        key = str(key or 'UNKNOWN')
        if key not in stats:
            stats[key] = {'total': 0, 'right': 0}
        stats[key]['total'] += 1
        if correct:
            stats[key]['right'] += 1

    def _finalize_stats(self, stats):
        return {key: {'total': value['total'], 'right': value['right'], 'acc': value['right'] / value['total'] if value['total'] else 0.0} for key, value in sorted(stats.items())}

    def cal_metrics(self, out_samples):
        total = 0
        right = 0
        parse_valid = 0
        empty_response = 0
        subgroup_stats = {'turn_id': {}, 'release_status': {}, 'is_revised': {}, 'gate': {}, 'micro_axis': {}, 'answer_label': {}}
        parser_methods = {}
        patient_stats = {}
        for idx, sample in enumerate(out_samples):
            response = sample.get('response') or ''
            parse = self.parse_answer(response, sample)
            ground_truth = str(sample.get('answer') or '').strip().upper()
            correct = bool(parse['parse_valid'] and parse['pred_answer'] == ground_truth)
            metadata = sample.get('metadata', {})
            patient_id = sample.get('patient_id') or metadata.get('patient_id')
            out_samples[idx].update(parse)
            out_samples[idx]['correct'] = correct
            out_samples[idx]['response_empty'] = not bool(str(response).strip())
            if correct:
                right += 1
            if parse['parse_valid']:
                parse_valid += 1
            if out_samples[idx]['response_empty']:
                empty_response += 1
            total += 1
            parser_methods[parse['parse_method']] = parser_methods.get(parse['parse_method'], 0) + 1
            self._add_stat(patient_stats, patient_id, correct)
            self._add_stat(subgroup_stats['turn_id'], sample.get('turn_id'), correct)
            self._add_stat(subgroup_stats['release_status'], metadata.get('release_status'), correct)
            self._add_stat(subgroup_stats['is_revised'], metadata.get('is_revised'), correct)
            self._add_stat(subgroup_stats['gate'], metadata.get('gate'), correct)
            self._add_stat(subgroup_stats['micro_axis'], metadata.get('micro_axis'), correct)
            self._add_stat(subgroup_stats['answer_label'], ground_truth, correct)
        patient_accs = [value['right'] / value['total'] for value in patient_stats.values() if value['total'] > 0]
        patient_full_success = [1.0 if value['right'] == value['total'] and value['total'] > 0 else 0.0 for value in patient_stats.values()]
        metrics = {'total metrics': {'total': total, 'right': right, 'acc': right / total if total else 0.0}, 'parser metrics': {'parse_valid': parse_valid, 'parse_valid_rate': parse_valid / total if total else 0.0, 'empty_response': empty_response, 'parse_methods': dict(sorted(parser_methods.items()))}, 'subgroup metrics': {key: self._finalize_stats(value) for key, value in subgroup_stats.items()}, 'patient metrics': {'patient_count': len(patient_stats), 'patient_macro_acc': sum(patient_accs) / len(patient_accs) if patient_accs else 0.0, 'patient_full_success': sum(patient_full_success) / len(patient_full_success) if patient_full_success else 0.0, 'patient_detail': self._finalize_stats(patient_stats)}}
        return (metrics, out_samples)
