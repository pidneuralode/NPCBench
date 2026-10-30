"""Gold-independent final-answer extraction, separately versioned from archives.

The caller supplies valid option labels/text, never a reference answer. There
is no fuzzy matching, clinical inference or retry. Incompatible explicit
answers are invalid rather than resolved by choosing the last convenient one.
"""
from dataclasses import dataclass
import json
import re
import unicodedata

POLICY = 'final_answer_v1'
_THINK = re.compile(r'<(/?)(think|thinking)>', re.I)
_FINAL = re.compile(r'<\|channel\|>final<\|message\|>')
_COMPACT = re.compile(r'assistant\s*final')
_FINAL_HEADING = re.compile(r'^\s*#{1,6}\s*(?:Final Response|Final Answer|最终回答|最终答案)\s*[:：]?\s*$', re.I | re.M)
_END = re.compile(r'<\|(?:im_end|eot_id|endoftext|return|fim_suffix|end)\|>')
_MARKER = re.compile(
    r'(?<![A-Za-z])(?:(?:final\s+answer|the\s+(?:correct|best)\s+(?:answer|option)|answer)\b|'
    r'最终答案|最终选择|正确答案|正确选项|答案|我选|应选|推荐选择|(?:是|为|即)\s*选项)'
    r'\s*(?:(?:is|为|是)\s*)?[:：]?\s*', re.I)
_SELECTION = re.compile(r'(?<![A-Za-z])(?:is|would be|choose|select|recommend)\s+(?:option|choice)\s*|'
                        r'(?:选择|首选)\s*(?:选项\s*)?|'
                        r'(?:选项|答案|方案|医嘱|决策|策略|处理|组合)(?:是|为)\s*[:：]?\s*', re.I)


@dataclass(frozen=True)
class Channel:
    text: str
    status: str
    method: str


@dataclass(frozen=True)
class Answer:
    prediction: str
    valid: bool
    status: str
    method: str
    channel: str
    candidates: tuple = ()


def select_final(response):
    """Separate explicit reasoning from the answer without guessing a stop reason."""
    if response is None:
        response = ''
    if not isinstance(response, str):
        raise ValueError('Response must be text or null')
    text, method = response, 'unmarked'
    matches = list(_FINAL.finditer(text))
    headings = list(_FINAL_HEADING.finditer(text))
    compact_analysis = bool(re.match(r'^(?:assistant)?analysis(?=[A-Z\n<])', text.lstrip()))
    if not matches and (compact_analysis or re.match(r'^\s*assistant\s*final', text)):
        matches = list(_COMPACT.finditer(text))
    if not matches and headings:
        matches = headings
    if matches:
        if len(matches) != 1:
            return Channel('', 'multiple_final_channels', 'explicit_final')
        text, method = text[matches[0].end():], 'explicit_final'
    elif compact_analysis or '<|channel|>analysis<|message|>' in text:
        return Channel('', 'unfinished_analysis_channel', 'analysis')
    elif re.match(r'^\s*#{1,6}\s*Thinking\s*$', text.splitlines()[0] if text.splitlines() else '', re.I):
        return Channel('', 'unfinished_markdown_reasoning', 'markdown_thinking')
    stack, visible, cursor = [], [], 0
    for match in _THINK.finditer(text):
        closing, name = match.group(1), match.group(2).lower()
        if not closing:
            if not stack:
                visible.append(text[cursor:match.start()])
                if method == 'unmarked':
                    method = 'closed_reasoning'
            stack.append(name)
        elif stack:
            if stack[-1] != name:
                return Channel('', 'mismatched_reasoning_tags', method)
            stack.pop()
            if not stack:
                cursor = match.end()
        else:
            # Some decoders omit the opening reasoning token. Its closing tag
            # still delimits the preceding reasoning from the final answer.
            visible, cursor = [], match.end()
            method = 'implicit_reasoning_close' if method == 'unmarked' else method
    if stack:
        return Channel('', 'unfinished_reasoning', method)
    visible.append(text[cursor:])
    text = _END.sub('', ''.join(visible)).strip()
    return Channel(text, 'ok' if text else 'empty_final', method)


def _clean(text):
    text = unicodedata.normalize('NFKC', text).strip()
    text = re.sub(r'^```(?:json|text|latex)?\s*|\s*```$', '', text, flags=re.I)
    text = text.strip(' \t\r\n`*_#$')
    previous = None
    while text != previous:
        previous = text
        text = re.sub(r'^<(b|strong|em|i)>\s*(.*?)\s*</\1>$', r'\2', text, flags=re.I | re.S).strip()
        text = re.sub(r'^\\(?:boxed|text|textbf|textit|mathrm|mathbf|mathit|operatorname)\s*\{(.*)\}$',
                      r'\1', text, flags=re.S).strip()
        if len(text) >= 2 and (text[0], text[-1]) in [('(', ')'), ('[', ']'), ('{', '}')]:
            text = text[1:-1].strip()
    return text


def _normal(text):
    return re.sub(r'\s+', ' ', _clean(text)).strip(' .。!！;；').casefold()


def _boxed(text):
    values, spans = [], []
    for match in re.finditer(r'\\boxed\s*\{', text):
        depth = 1
        for end in range(match.end(), len(text)):
            depth += (text[end] == '{') - (text[end] == '}')
            if depth == 0:
                values.append(text[match.end():end])
                spans.append((match.start(), end + 1))
                break
        else:
            return [], [], 'unfinished_answer_box'
    return values, spans, None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result and result[key] != value:
            raise ValueError('duplicate_json_key')
        result[key] = value
    return result


def json_payload(response):
    """Require one unambiguous complete JSON object in the selected channel."""
    channel = select_final(response)
    if channel.status != 'ok':
        return None, channel.status, channel.method
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    text, pos, objects = channel.text, 0, []
    while True:
        start = text.find('{', pos)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text, start)
        except (ValueError, json.JSONDecodeError):
            return None, 'invalid_json', channel.method
        if isinstance(value, dict):
            objects.append(value)
        pos = end
    if not objects:
        return None, 'missing_json', channel.method
    if any(value != objects[0] for value in objects[1:]):
        return None, 'conflicting_json', channel.method
    return objects[0], 'ok', channel.method


def parse_t_stage(response):
    """Read the stage field, not T1-weighted images or thoracic vertebra names.

    A complete explicit field can be scored even if a later ancillary JSON
    field is incomplete. This preserves field-level rather than all-JSON
    correctness. Conflicting stage fields never receive a guessed value.
    """
    channel = select_final(response)
    if channel.status != 'ok':
        return Answer('', False, channel.status, 'none', channel.method)
    # Walk string tokens so quoted explanations (including examples of JSON)
    # cannot be mistaken for a stage field. Keep fields preceding a truncated
    # ancillary string, but never extract an answer from that string.
    decoder, text, pos, values, depth = json.JSONDecoder(), channel.text, 0, [], 0
    invalid_field = False
    while pos < len(text):
        start, key = pos, None
        if text[pos] == '"':
            try:
                token, pos = decoder.raw_decode(text, pos)
            except ValueError:
                break
            if token in {'T_stage', 't_stage', 'T stage', 't stage', 'T', 't'}:
                key = token
        else:
            match = re.match(r'(?:T_stage|t_stage|T stage|t stage|T|t)\b', text[pos:])
            if match:
                key, pos = match.group(), pos + match.end()
            else:
                depth += (text[pos] == '{') - (text[pos] == '}')
                depth = max(depth, 0)
                pos += 1
        if key is None or depth > 1 or not re.search(r'(?:^|[\n,{])\s*$', text[:start]):
            continue
        colon = re.match(r'\s*:\s*', text[pos:])
        if colon is None:
            continue
        pos += colon.end()
        if text[pos:pos + 1] == '"':
            try:
                value, pos = decoder.raw_decode(text, pos)
            except ValueError:
                invalid_field = True
                break
        else:
            value_match = re.match(r'[^\s,}]+', text[pos:])
            value = value_match.group() if value_match else ''
            pos += len(value)
        value = re.sub(r'\s+', '', str(value)).upper()
        if value in {'T1', 'T2', 'T3', 'T4'}:
            values.append(value)
        else:
            invalid_field = True
    if invalid_field:
        return Answer('', False, 'invalid_stage_field', 'stage_field', channel.method, tuple(sorted(set(values))))
    if not values:
        direct = _clean(channel.text).upper().strip(' .。')
        if direct in {'T1', 'T2', 'T3', 'T4'}:
            values = [direct]
    unique = sorted(set(values))
    if len(unique) != 1:
        return Answer('', False, 'conflicting_stage_fields' if unique else 'no_final_stage',
                      'stage_field', channel.method, tuple(unique))
    return Answer(unique[0], True, 'ok', 'stage_field', channel.method, tuple(unique))


def parse_choice(response, options):
    """Extract a single option from final markers or an exact option value.

    Valid signals: boxed/XML/JSON answers, explicit answer markers, a leading
    labeled answer, or a bare final line. Plain full option text is accepted
    only on exact normalized equality. Declared final outputs must agree;
    incidental option references in accompanying prose are lower priority.
    """
    if isinstance(options, list):
        # Original25 care uses letter/text rows; expanded care and text tasks
        # use a mapping. The labels remain explicit in both schemas.
        mapped = {str(row['letter']).upper(): row['text'] for row in options}
        if len(mapped) != len(options):
            raise ValueError('Duplicate option labels')
        options = mapped
    channel = select_final(response)
    if channel.status != 'ok':
        return Answer('', False, channel.status, 'none', channel.method)
    labels = {str(key).upper() for key in options}
    if not labels:
        raise ValueError('Options must contain valid labels')
    choices = {str(k).upper(): str(v) for k, v in options.items()}
    signals, malformed = [], []

    def add(value, method, prefix=False):
        value = _clean(str(value))
        direct = re.sub(r'^(?:option|选项)\s*', '', value, flags=re.I)
        normalized = direct.strip(' .。!！;；').upper()
        if normalized in labels:
            signals.append((normalized, method))
            return
        exact = [key for key, choice in choices.items() if choice and _normal(choice) == _normal(value)]
        if exact:
            signals.extend((key, 'exact_option_text') for key in exact)
            return
        match = re.match(r'^([A-Z]|[01])(?=$|[^A-Za-z0-9])', direct, re.I)
        if prefix and match:
            signals.append((match.group(1).upper(), method))
            rest = direct[match.end():]
            multiple = re.match(r'\s*(?:/|,|、|\band\b|\bor\b|和|或)\s*([A-Z01])(?=$|[^A-Za-z0-9])', rest, re.I)
            if multiple:
                signals.append((multiple.group(1).upper(), method))
        elif method in {'boxed', 'xml_answer', 'json_answer'}:
            malformed.append(method)

    text = channel.text
    # Decode complete JSON containers before scanning prose/LaTeX. A quoted
    # reasoning string may contain answer examples; only top-level answer
    # fields (or answer fields in a top-level array) declare the choice.
    json_spans = []
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    for match in re.finditer(r'[\[{]', text):
        if any(start <= match.start() < end for start, end in json_spans):
            continue
        try:
            payload, end = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        if not isinstance(payload, (dict, list)):
            continue
        for obj in payload if isinstance(payload, list) else [payload]:
            if isinstance(obj, dict):
                for key in ('answer', 'option', 'box'):
                    if key in obj:
                        add(obj[key], 'json_answer')
        json_spans.append((match.start(), end))
    for start, end in reversed(json_spans):
        text = text[:start] + ' ' * (end - start) + text[end:]
    if re.search(r'"(?:answer|option|box)"\s*:', text):
        malformed.append('json_answer')
    boxes, spans, error = _boxed(text)
    if error:
        return Answer('', False, error, 'boxed', channel.method)
    for value in boxes:
        add(value, 'boxed')
    for match in re.finditer(r'<answer>(.*?)</answer>', text, flags=re.I | re.S):
        add(match.group(1), 'xml_answer')
        spans.append(match.span())
    for match in re.finditer(r'<span\b[^>]*>(.*?)</span>', text, flags=re.I | re.S):
        add(match.group(1), 'html_span_answer')
        spans.append(match.span())
    if re.search(r'<answer>', text, re.I) and not re.search(r'</answer>', text, re.I):
        malformed.append('xml_answer')
    declarations, prose = [], []
    for match in _MARKER.finditer(text):
        if not any(start <= match.start() < end for start, end in spans):
            line_prefix = text[text.rfind('\n', 0, match.start()) + 1:match.start()].strip(' \t*_#`')
            strong = not line_prefix or bool(re.match(r'(?:final\s+answer|最终)', match.group(), re.I))
            (declarations if strong else prose).append(match)
    for match in declarations:
        add(text[match.end():].splitlines()[0] if text[match.end():].splitlines() else '',
            'declared_answer', prefix=True)
    if not signals and not malformed:
        for match in [*prose, *_SELECTION.finditer(text)]:
            if not any(start <= match.start() < end for start, end in spans):
                add(text[match.end():].splitlines()[0] if text[match.end():].splitlines() else '',
                    'selection_prose', prefix=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        add(lines[-1], 'final_line')
        if not signals:
            add(lines[0], 'first_line')
    cleaned = _clean(text)
    label_pattern = r'^(?:' + '|'.join(re.escape(x) for x in sorted(labels)) + r')\s*[.\):：、]\s*\S'
    if not signals:
        labeled_lines = [line for line in lines if re.match(label_pattern, line, re.I)]
        for line in labeled_lines:
            add(line, 'leading_label', prefix=True)
    add(cleaned, 'exact_answer')
    unique = sorted({value for value, _ in signals})
    if malformed:
        return Answer('', False, 'malformed_explicit_answer', malformed[0], channel.method, tuple(unique))
    if any(value not in labels for value in unique):
        return Answer('', False, 'invalid_option', 'explicit', channel.method, tuple(unique))
    if len(unique) > 1:
        return Answer('', False, 'conflicting_answers', 'conflict', channel.method, tuple(unique))
    if not unique:
        return Answer('', False, 'no_final_answer', 'none', channel.method)
    return Answer(unique[0], True, 'ok', signals[0][1], channel.method, tuple(unique))
