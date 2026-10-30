import json
from npcbench_eval.l3_diagnostics import BASE_STRUCTURES, analyze_model, derive_t_stage


def fixtures():
    vector={key:0 for key in BASE_STRUCTURES};vector['clivus']=1
    cards=[dict(id='Struct-example',patient_case_id='T-Stg-example',base_structure_id_gt='clivus',invasion_gt=1)]
    patients=[dict(id='T-Stg-example',patient_case_id='T-Stg-example',t_stage_gt='T3',base_structure_invasion_gt=vector)]
    return cards,patients,vector


def test_correct_stage_without_invasion_evidence_is_not_grounded():
    cards,patients,_=fixtures()
    component,_,patient=analyze_model('example',cards,patients,
        [dict(id='Struct-example',response=json.dumps(dict(structure_id='clivus',invasion=1)))],
        [dict(id='T-Stg-example',response=json.dumps(dict(t_stage='T3')))],{'T-Stg-example'})
    assert patient['direct_t']==100 and patient['complete_inv']==0 and patient['eg_t']==0
    assert component['patient_invasion_label_micro']==0


def test_complete_correct_evidence_supports_joint_success():
    cards,patients,vector=fixtures()
    _,ablation,patient=analyze_model('example',cards,patients,
        [dict(id='Struct-example',response=json.dumps(dict(structure_id='clivus',invasion=1)))],
        [dict(id='T-Stg-example',response=json.dumps(dict(t_stage='T3',structure_invasion=vector)))],{'T-Stg-example'})
    assert patient['complete_inv']==patient['eg_t']==100
    assert ablation['oracle_slot_rule_t']==ablation['model_slot_rule_t']==100


def test_zero_default_is_a_rule_assumption_not_complete_evidence():
    assert derive_t_stage({})=='T1'
    assert derive_t_stage({'clivus':1,'cavernous_sinus':1})=='T4'
    assert derive_t_stage({'medial_pterygoid':1})=='T2'
