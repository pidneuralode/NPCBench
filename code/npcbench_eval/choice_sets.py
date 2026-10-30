"""Exact option-set extraction for explicitly multiple-select revision tasks."""
import re
from .final_answer import Answer, _boxed, _clean, select_final

POLICY='exact_choice_set_v1'

def parse_choice_set(response, options):
    """Read a complete final set; incidental prose and reasoning are not labels."""
    labels={str(x).upper() for x in options}
    if not labels:raise ValueError('Options must be nonempty')
    channel=select_final(response)
    if channel.status!='ok':return Answer('',False,channel.status,'none',channel.method)
    boxes,_,error=_boxed(channel.text)
    if error:return Answer('',False,error,'boxed_set',channel.method)
    candidates=[]
    for text in boxes or [channel.text]:
        text=_clean(text).upper()
        if not re.fullmatch(r'[A-Z](?:\s*[,，+、;；]?\s*[A-Z])*',text):
            return Answer('',False,'invalid_option_set','boxed_set' if boxes else 'standalone_set',channel.method)
        values=re.findall('[A-Z]',text)
        if len(values)!=len(set(values)) or not set(values)<=labels:
            return Answer('',False,'invalid_option_set','boxed_set' if boxes else 'standalone_set',channel.method)
        candidates.append(','.join(sorted(values)))
    if len(set(candidates))!=1:return Answer('',False,'conflicting_answers','boxed_set',channel.method,tuple(candidates))
    return Answer(candidates[0],True,'ok','boxed_set' if boxes else 'standalone_set',channel.method,tuple(candidates))
