"""Inspect what the frozen L5 history interventions preserve and remove."""
from collections import Counter
import hashlib
import re

HISTORY_START='【既往检查检验与治疗记录】'
TIME_PATTERNS=(
    r'\[(?:第\d+次入院|住院间期/门诊)?\s*(?:全程D-?\d+|入院前D-?\d+)\]',
    r'\[(?:第\d+次入院|住院间期/门诊)\]',
    r'(?:全程D|入院前D)-?\d+',r'第\d+次入院',r'\bD-?\d+\b',
)


def without_temporal_markers(line):
    """The archived snapshot transformation, without punctuation deletion."""
    for pattern in TIME_PATTERNS:line=re.sub(pattern,'',line)
    line=re.sub(r'\s+',' ',line).strip()
    return re.sub(r'^-\s*','- ',line)


def audit_triplet(full,snapshot,shuffled):
    """Check the specified inputs; time-stripped parity is not clinical equivalence."""
    for key in ['id','patient_id','answer','options']:
        if not full[key]==snapshot[key]==shuffled[key]:raise ValueError('Paired identity, options or answer differ: '+key)
    prompts=[r['prompt'] for r in [full,snapshot,shuffled]]
    parts=[p.split(HISTORY_START,1) for p in prompts]
    if any(len(p)!=2 for p in parts):raise ValueError('Missing frozen history boundary')
    histories=[[l.strip() for l in body.splitlines() if l.strip().startswith('- ')] for _,body in parts]
    suffixes=[]
    for p in prompts:
        option_start=p.index('\n\nA. ')
        question_start=p.rfind('\n\n',0,option_start)+2
        suffixes.append(p[question_start:])
    sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
    return dict(id=full['id'],patient_id=full['patient_id'],
        current_and_baseline_identical=len({a for a,_ in parts})==1,
        question_options_instructions_identical=len(set(suffixes))==1,
        snapshot_payload_multiset_equal_after_time_removal=
            Counter(map(without_temporal_markers,histories[0]))==Counter(map(without_temporal_markers,histories[1])),
        snapshot_exact_history_multiset_equal=Counter(histories[0])==Counter(histories[1]),
        shuffled_exact_prompt_line_multiset_equal=Counter(prompts[0].splitlines())==Counter(prompts[2].splitlines()),
        history_records=len(histories[0]),
        full_history_records_with_time=sum(any(re.search(p,x) for p in TIME_PATTERNS) for x in histories[0]),
        snapshot_history_records_with_time=sum(any(re.search(p,x) for p in TIME_PATTERNS) for x in histories[1]),
        full_prompt_sha256=sha(prompts[0]),snapshot_prompt_sha256=sha(prompts[1]),shuffled_prompt_sha256=sha(prompts[2]))
