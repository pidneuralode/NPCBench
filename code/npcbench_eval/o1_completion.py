"""Parse a fresh judge completion after separating its explicit final answer."""
from .final_answer import select_final
from .o1_release import scorer


def parse_completion(response):
    final=select_final(response)
    if final.status!='ok':
        return dict(criteria_met=None,explanation='',judge_status=final.status)
    try:
        parsed=scorer.parse_judge_json(final.text)
    except ValueError:
        return dict(criteria_met=None,explanation='',judge_status='invalid_final_json')
    valid=isinstance(parsed,dict) and scorer.validate_judge_output(parsed)[0]
    return dict(criteria_met=parsed['criteria_met'] if valid else None,
                explanation=parsed['explanation'] if valid else '',
                judge_status='ok' if valid else 'invalid_final_json')
