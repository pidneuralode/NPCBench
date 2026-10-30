"""Render L6 downstream inputs from recorded frontend outputs.

The input policy is explicit: missing pathology is unknown, not a negative
diagnosis. Historical rendering is supported only to verify archived prompts.
Neither renderer reads a care answer or infers a clinical decision.
"""
import json

from .final_answer import POLICY, json_payload, parse_choice, parse_t_stage
from .l6_scoring import NPCFullCarePath25 as Parser

INPUT_POLICY = 'final_answer_input_v1'


def frontend_values(pathology, mri, policy=POLICY):
    if policy == 'frozen_v1':
        return {'pathology': Parser._parse_binary(pathology), 'mri_t': Parser._parse_t_stage(mri),
                'structures': Parser._parse_structure_dict(mri)}
    if policy != POLICY:
        raise ValueError('Unknown frontend answer policy')
    payload, _, _ = json_payload(mri)
    return {'pathology': parse_choice(pathology, {'0': '', '1': ''}).prediction,
            'mri_t': parse_t_stage(mri).prediction,
            'structures': Parser._parse_structure_dict(json.dumps(payload, ensure_ascii=False)) if payload is not None else {}}


def render_care(template, values, ajcc_v9, policy=INPUT_POLICY):
    if policy not in (INPUT_POLICY, 'frozen_v1'):
        raise ValueError('Unknown downstream input policy')
    pathology = values['pathology']
    pathology_text = ('支持鼻咽癌诊断' if pathology == '1' else '不支持鼻咽癌诊断'
                      if pathology == '0' or policy == 'frozen_v1' else '病理结果未明')
    involved = [key for key, value in values['structures'].items()
                if value == 1 and key not in Parser.EXCLUDED_STRUCTURE_KEYS]
    t = values['mri_t'] or 'T未明'
    n, m = ajcc_v9.get('N') or 'N未明', ajcc_v9.get('M') or 'M未明'
    replacements = {'{{PATHOLOGY_RESULT}}': pathology_text, '{{MRI_T_STAGE}}': t,
                    '{{MRI_INVOLVED_STRUCTURES}}': '、'.join(involved) if involved else '未列出明确受侵结构',
                    '{{N_STAGE}}': n, '{{M_STAGE}}': m, '{{TNM_STATUS}}': t + n + m,
                    '{{OVERALL_STAGE}}': ajcc_v9.get('overall_stage') or '未明'}
    for placeholder, text in replacements.items():
        template = template.replace(placeholder, text)
    return template
