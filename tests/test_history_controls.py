from copy import deepcopy
from npcbench_eval.history_controls import audit_triplet,without_temporal_markers


def triplet():
    prefix='Current D20\n【既往检查检验与治疗记录】\n'
    lines=['- [第1次入院 全程D4] Prior treatment: cisplatin 71 mg, cycle 2.',
           '- [第1次入院 全程D8] CT: no progression; continue observation.']
    tail='\n\nWhat next?\n\nA. Assess.\nB. Repeat.\n\nReturn one letter.'
    row=dict(id='example',patient_id='public-patient',options={'A':'Assess.','B':'Repeat.'},answer='A')
    full=dict(row,prompt=prefix+'\n'.join(lines)+tail)
    snapshot=dict(row,prompt=prefix+'\n'.join(without_temporal_markers(l) for l in lines[::-1])+tail)
    shuffled=dict(row,prompt=prefix+'\n'.join(lines[::-1])+tail)
    return full,snapshot,shuffled


def test_time_stripped_payload_parity_is_distinct_from_exact_record_parity():
    result=audit_triplet(*triplet())
    assert result['snapshot_payload_multiset_equal_after_time_removal']
    assert not result['snapshot_exact_history_multiset_equal']
    assert result['full_history_records_with_time']==2 and result['snapshot_history_records_with_time']==0
    assert result['shuffled_exact_prompt_line_multiset_equal']


def test_changed_dose_or_negation_does_not_pass_payload_check():
    full,snapshot,shuffled=triplet()
    for before,after in [('71 mg','100 mg'),('no progression','progression')]:
        changed=deepcopy(snapshot);changed['prompt']=changed['prompt'].replace(before,after)
        assert not audit_triplet(full,changed,shuffled)['snapshot_payload_multiset_equal_after_time_removal']


def test_changed_question_is_visible_independently_of_history_payload():
    full,snapshot,shuffled=triplet();snapshot['prompt']=snapshot['prompt'].replace('What next?','What first?')
    result=audit_triplet(full,snapshot,shuffled)
    assert result['snapshot_payload_multiset_equal_after_time_removal']
    assert not result['question_options_instructions_identical']
