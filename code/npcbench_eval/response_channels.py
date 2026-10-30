"""A versioned, gold-independent gate for unfinished explicit reasoning blocks.

This is a candidate policy, not a modification of any frozen archive parser.
It does not infer truncation from text length or a missing finish_reason.
"""
import re


POLICY = 'reject_unfinished_think_v1'
_THINK = re.compile(r'<(/?)(think|thinking)>', re.IGNORECASE)


def has_unfinished_think(response):
    """Recognize literal, case-insensitive think/thinking tags.

    Closing tags without a visible opener are accepted: some model templates
    omit the opening token from decoded completions. Matching completed blocks
    leave the archived response untouched. Unclosed or mismatched open blocks
    fail this candidate gate even if they contain a boxed answer.
    """
    stack = []
    for match in _THINK.finditer(response or ''):
        closing, name = match.group(1), match.group(2).lower()
        if not closing:
            stack.append(name)
        elif stack and stack[-1] == name:
            stack.pop()
    return bool(stack)


def gate_response(response):
    """Pass all other responses through byte-for-byte; never inspect a gold."""
    return '' if has_unfinished_think(response) else response
