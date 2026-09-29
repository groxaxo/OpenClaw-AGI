"""Stronger verification of self-replay targets using public development only.

Equivalent reference ASTs establish explicit alias groups. Their original test
cases and exact-description public DEV cases are pooled for verification. No
confirmation suite is accepted here. Successful raw answers are preserved
verbatim, including Markdown fences and trailing newlines.
"""
import ast
import json
from .coding_tasks import CodingTask


def verification_tasks(examples, public_dev):
    examples=tuple(examples);public_dev=tuple(public_dev)
    if len({e.task_id for e in examples})!=len(examples):raise ValueError('duplicate training ID')
    groups={}
    for ex in examples:
        key=ast.dump(ast.parse(ex.solution),include_attributes=False)
        groups.setdefault(key,[]).append(ex)
    result={}
    for group in groups.values():
        descriptions={ex.description for ex in group}
        cases=[case for ex in group for case in ex.cases]
        cases += [case for task in public_dev if task.description in descriptions for case in task.cases]
        unique=[];seen=set()
        for arg,expected in cases:
            key=json.dumps([arg,expected],sort_keys=True,ensure_ascii=False,allow_nan=False)
            if key not in seen:seen.add(key);unique.append((arg,expected))
        if not unique:raise ValueError('no public verification cases')
        for ex in group:result[ex.task_id]=CodingTask(ex.task_id,ex.description,tuple(unique))
    return result


def make_target(raw_response, correct, reference):
    if type(correct) is not bool:raise ValueError('correct must be boolean')
    if correct:
        if not isinstance(raw_response,str) or not raw_response.strip():raise ValueError('empty self replay')
        return raw_response
    code=ast.unparse(ast.parse(reference))+'\n'
    # Preserve an already-used fenced response style for a reference repair.
    if isinstance(raw_response,str) and raw_response.lstrip().startswith('```'):
        return '```python\n'+code+'```\n'
    return code


def complete_tasks_preserved(before, after):
    old={x['task_id']:x['result'] for x in before};new={x['task_id']:x['result'] for x in after}
    if len(old)!=len(before) or len(new)!=len(after) or set(old)!=set(new):
        raise ValueError('public task identity mismatch')
    for key,prior in old.items():
        now=new[key]
        if now['total']!=prior['total']:raise ValueError('public task case count changed')
        for item in (prior,now):
            if type(item['passed']) is not int or type(item['total']) is not int or not 0<=item['passed']<=item['total'] or item['total']<=0:
                raise ValueError('invalid public task result')
        if prior['passed']==prior['total'] and now['passed']!=now['total']:return False
    return True
