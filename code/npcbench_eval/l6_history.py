"""Versioned, gold-independent answer-to-history rendering for original25 L6."""
from .final_answer import parse_choice
from .l6_scoring import NPCFullCarePath25 as FrozenParser

POLICY = 'final_answer_history_v1'
UNKNOWN_ACTION = '未形成可解析的处理意见'


def action_from_response(response, options, policy):
    choices = options if isinstance(options, dict) else {x['letter']: x['text'] for x in options}
    if policy == POLICY:
        answer = parse_choice(response, choices)
        letter = answer.prediction if answer.valid else ''
        return {'letter': letter, 'action_text': str(choices[letter]).strip() if letter else UNKNOWN_ACTION}
    if policy == 'frozen_history_v1':
        letter = FrozenParser._parse_letter(response)
        action = str(choices.get(letter, '')).strip() if letter else ''
        if not action:
            action = FrozenParser._strip_think_blocks(response).strip()
        return {'letter': letter, 'action_text': action or UNKNOWN_ACTION}
    raise ValueError('Unknown history input policy')


def render_history(prompt, actions):
    """Replace only the frozen auxiliary-history block, preserving other bytes."""
    heading = '【历史辅助决策节点】'
    start = prompt.rfind(heading)
    if start < 0:
        raise ValueError('L6 prompt missing historical auxiliary heading')
    body_start = start + len(heading)
    candidates = [prompt.find(marker, body_start) for marker in
                  ('\n\n请在以下', '\n\n结合该患者', '\n\n结合当前', '\n\n问题')]
    candidates = [x for x in candidates if x >= 0]
    if not candidates:
        raise ValueError('L6 prompt missing question boundary after history')
    lines = [f"- 历史辅助决策节点{i}：{entry['action_text'].rstrip('。')}。" for i, entry in enumerate(actions, 1)]
    if not lines:
        lines = ['- 暂无可展示的历史辅助决策节点。']
    return prompt[:body_start] + '\n' + '\n'.join(lines) + prompt[min(candidates):]
