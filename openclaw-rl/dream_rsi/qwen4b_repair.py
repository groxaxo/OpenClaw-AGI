"""Verified-repair SFT protocol for a fixed set of known coding task families.

The previous confirmation set is explicitly retired into development. New final
inputs are preregistered separately. Prompt/algorithm families are trained, so
this validates known-task program correctness, not unseen-task generalization.
"""
from .qwen4b_curriculum import SFTExample
from .coding_tasks import CodingTask
from .qwen4b_harder import TRAIN as PRIOR_TRAIN, EXTRA, GUARDS, build_hard_suite, DESCRIPTIONS

KINDS=('chunks','range','union','minpath','prefix','smaller','runs','sum','rpn','lru','bfs','edit')
ALIASES=tuple(SFTExample('repair_alias_'+kind,DESCRIPTIONS[kind],example.solution,example.cases)
              for kind,example in zip(KINDS,EXTRA))
TRAIN=PRIOR_TRAIN+ALIASES
# These are the exposed prior-round confirmation cases. They are development
# data from this point onward and must never again be labeled sealed/heldout.
DEV=tuple(CodingTask('repair_dev_'+r['task_id'],r['description'],tuple((a,b) for a,b in r['cases']))
          for r in build_hard_suite(52917083))


def choose_target(example,record):
    """Reuse verified own answers; repair failures with verified reference code."""
    result=record['result']
    correct=result['passed']==result['total'] and result['total']>0
    target=record['response'] if correct else example.solution
    if not isinstance(target,str) or not target.strip():raise ValueError('empty repair target')
    return {'task_id':example.task_id,'description':example.description,'solution':target,
            'weight':1.0 if correct else 4.0,'target_source':'verified_self_replay' if correct else 'verified_reference_repair',
            'prior_passed':result['passed'],'prior_total':result['total']}
