from npcbench_eval.o1_completion import parse_completion


def test_false_is_a_valid_judgment():
    value=parse_completion('{"criteria_met": false, "explanation": "The criterion is absent."}')
    assert value['judge_status']=='ok' and value['criteria_met'] is False


def test_unfinished_reasoning_json_is_not_a_judgment():
    raw='<think>I could return {"criteria_met": true, "explanation": "Draft only."}'
    value=parse_completion(raw)
    assert value['judge_status']=='unfinished_reasoning' and value['criteria_met'] is None


def test_malformed_final_remains_unavailable():
    value=parse_completion('{"criteria_met":')
    assert value['judge_status']=='invalid_final_json' and value['criteria_met'] is None


def test_final_channel_overrides_a_reasoning_draft():
    raw='<|channel|>analysis<|message|>{"criteria_met": true, "explanation": "Draft."}'
    raw+='<|channel|>final<|message|>{"criteria_met": false, "explanation": "Final."}'
    value=parse_completion(raw)
    assert value['judge_status']=='ok' and value['criteria_met'] is False


def test_harmony_json_final_metadata():
    raw='<|channel|>analysis<|message|>A draft might say true.'
    raw+='<|end|><|start|>assistant<|channel|>final <|constrain|>JSON<|message|>'
    raw+='{"criteria_met": false, "explanation": "The criterion is absent."}'
    value=parse_completion(raw)
    assert value['judge_status']=='ok' and value['criteria_met'] is False


def test_json_metadata_does_not_turn_analysis_into_final():
    raw='<|channel|>analysis<|message|><|constrain|>JSON'
    raw+='{"criteria_met": true, "explanation": "Draft only."}'
    assert parse_completion(raw)['judge_status']=='unfinished_analysis_channel'


def test_lowercase_harmony_json_metadata():
    raw='<|channel|>analysis<|message|>Reasoning.'
    raw+='<|end|><|start|>assistant<|channel|>final <|constrain|>json<|message|>'
    raw+='{"criteria_met": true, "explanation": "Explicit final output."}'
    assert parse_completion(raw)['criteria_met'] is True


def test_two_harmony_final_channels_remain_ambiguous():
    raw='<|channel|>final<|message|>{"criteria_met": true, "explanation": "First."}'
    raw+='<|channel|>final<|constrain|>JSON<|message|>{"criteria_met": false, "explanation": "Second."}'
    assert parse_completion(raw)['judge_status']=='multiple_final_channels'
