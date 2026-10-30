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
