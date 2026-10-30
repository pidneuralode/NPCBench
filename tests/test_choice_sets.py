from npcbench_eval.choice_sets import parse_choice_set

def test_partial_answer_is_not_the_complete_correct_set():
    answer=parse_choice_set(r'\boxed{B}', 'ABCD')
    assert answer.valid and answer.prediction!='B,D'

def test_complete_set_accepts_order_and_unicode_separator():
    assert parse_choice_set(r'\boxed{D，B}', 'ABCD').prediction=='B,D'

def test_conflicting_final_boxes_are_invalid():
    assert not parse_choice_set(r'\boxed{B,D} \boxed{B}', 'ABCD').valid

def test_unfinished_reasoning_does_not_supply_a_final_set():
    assert not parse_choice_set(r'<think>We might select \boxed{B,D}', 'ABCD').valid

def test_explicit_final_channel_takes_precedence_over_reasoning():
    response='<|channel|>analysis<|message|>Maybe \\boxed{A}.<|channel|>final<|message|>\\boxed{B,D}'
    assert parse_choice_set(response,'ABCD').prediction=='B,D'

def test_prose_is_not_mined_for_option_letters():
    assert not parse_choice_set('Both B and D may be considered.', 'ABCD').valid

def test_repeated_and_out_of_range_options_are_invalid():
    assert not parse_choice_set(r'\boxed{B,B}', 'ABCD').valid
    assert not parse_choice_set(r'\boxed{B,Z}', 'ABCD').valid

def test_latex_wrapper_retains_the_full_set():
    assert parse_choice_set(r'\boxed{\mathrm{B,D}}', 'ABCD').prediction=='B,D'
