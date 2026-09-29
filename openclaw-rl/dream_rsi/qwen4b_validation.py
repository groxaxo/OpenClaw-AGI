"""Deterministic, independent small-input oracles and an in-domain smoke gate.

Training covers several of these algorithm families. New prompts and inputs
are not a claim of unseen-algorithm generalization or statistical significance.
"""
from __future__ import annotations
import itertools
import json
import math
import random
from collections import deque
from pathlib import Path
from .coding_tasks import CodingTask


def grid_paths(data):
    rows,cols,blocked=data
    bad={tuple(x) for x in blocked}
    def walk(r,c):
        if (r,c) in bad or r>=rows or c>=cols:return 0
        if (r,c)==(rows-1,cols-1):return 1
        return walk(r+1,c)+walk(r,c+1)
    return walk(0,0)


def merge_closed(data):
    # Repeated pairwise union, independent of the sorted sweep in SFT targets.
    out=[list(x) for x in data]
    changed=True
    while changed:
        changed=False
        for i in range(len(out)):
            for j in range(i+1,len(out)):
                a,b=out[i];c,d=out[j]
                if max(a,c)<=min(b,d):
                    out[i]=[min(a,c),max(b,d)];out.pop(j);changed=True;break
            if changed:break
    return sorted(out)


def min_coins(data):
    coins,amount=data
    queue=deque([(0,0)]);seen={0}
    while queue:
        value,depth=queue.popleft()
        if value==amount:return depth
        for coin in coins:
            nxt=value+coin
            if nxt<=amount and nxt not in seen:seen.add(nxt);queue.append((nxt,depth+1))
    return -1


def spiral(data):
    if not data or not data[0]:return []
    rows,cols=len(data),len(data[0]);seen=set();out=[];r=c=d=0
    directions=((0,1),(1,0),(0,-1),(-1,0))
    for _ in range(rows*cols):
        out.append(data[r][c]);seen.add((r,c))
        dr,dc=directions[d];nr,nc=r+dr,c+dc
        if not(0<=nr<rows and 0<=nc<cols) or (nr,nc) in seen:
            d=(d+1)%4;dr,dc=directions[d]
        r,c=r+dr,c+dc
    return out


def brackets(text):
    # Repeated adjacent-pair reduction instead of a stack.
    text=''.join(c for c in text if c in '()[]{}')
    while True:
        reduced=text.replace('()','').replace('[]','').replace('{}','')
        if reduced==text:return not text
        text=reduced


def unique_substring(text):
    return max([0]+[j-i for i in range(len(text)) for j in range(i+1,len(text)+1)
                    if len(set(text[i:j]))==j-i])


def oracle(kind,data):
    if kind=='rotate':
        items,k=data
        if not items:return []
        return [items[(i-k)%len(items)] for i in range(len(items))]
    if kind=='window':
        nums,k=data;return [sorted(nums[i:i+k])[-1] for i in range(len(nums)-k+1)]
    if kind=='merge':return merge_closed(data)
    if kind=='paths':return grid_paths(data)
    if kind=='subarrays':
        nums,target=data;return sum(sum(nums[i:j])==target for i in range(len(nums)) for j in range(i+1,len(nums)+1))
    if kind=='next':return [next((x for x in data[i+1:] if x>v),-1) for i,v in enumerate(data)]
    if kind=='product':return [math.prod(data[:i]+data[i+1:]) for i in range(len(data))]
    if kind=='rle':return [[char,len(list(items))] for char,items in itertools.groupby(data)]
    if kind=='coins':return min_coins(data)
    if kind=='unique':return unique_substring(data)
    if kind=='spiral':return spiral(data)
    if kind=='brackets':return brackets(data)
    raise ValueError('unknown task kind')


DESCRIPTIONS={
'rotate':'data is [items,k]. Return items rotated right by k places; negative k means left. Empty items yields [].',
'window':'data is [numbers,k] with 1<=k<=len(numbers). Return the largest number in every consecutive k-element window.',
'merge':'data contains closed intervals [start,end]. Return their sorted union as intervals, merging overlap and shared endpoints.',
'paths':'data is [rows,cols,blocked]. Count paths from [0,0] to [rows-1,cols-1] that move right or down and never enter a blocked [r,c] cell.',
'subarrays':'data is [numbers,target]. Count nonempty consecutive subarrays summing exactly to target. Values can be negative or zero.',
'next':'For each element of data, return the first strictly larger value found to its right, or -1 when absent.',
'product':'For each index in data, return the product of all OTHER integers. Do not divide. [] gives []; one item gives [1].',
'rle':'Encode consecutive equal-character runs in data (a string) into a list of [character,run_length] pairs. Preserve Unicode characters.',
'coins':'data is [coins,amount]. Each positive integer coin denomination can be reused. Return the fewest coins summing to amount, -1 if impossible; zero amount needs zero coins.',
'unique':'data is a string. Return the greatest length of a consecutive substring whose characters are all distinct.',
'spiral':'data is a rectangular matrix. Return its elements in clockwise spiral order starting at top-left. [] or zero-width rows gives [].',
'brackets':'data is a string. Ignore characters except (), [] and {} and return whether the brackets are correctly matched and nested.'}


def build_suite(seed:int):
    rng=random.Random(seed);tasks=[]
    for kind,desc in DESCRIPTIONS.items():
        inputs=[]
        for j in range(20):
            values=[rng.randrange(-3,7) for _ in range(j%8)]
            if kind=='rotate':data=[values,rng.randrange(-20,21)]
            elif kind=='window':
                nums=values or [rng.randrange(-5,6)];data=[nums,rng.randrange(1,len(nums)+1)]
            elif kind=='merge':data=[sorted([rng.randrange(-6,10),rng.randrange(-6,10)]) for _ in range(j%5)]
            elif kind=='paths':
                rows,cols=rng.randrange(1,5),rng.randrange(1,5)
                blocked=[[r,c] for r in range(rows) for c in range(cols) if rng.random()<.15]
                data=[rows,cols,blocked]
            elif kind=='subarrays':data=[values,rng.randrange(-4,9)]
            elif kind in ('next','product'):data=values
            elif kind=='rle':data=''.join(rng.choice('ab🙂')*rng.randrange(1,4) for _ in range(j%6))
            elif kind=='coins':data=[sorted(rng.sample(range(1,8),1+j%4)),j%18]
            elif kind=='unique':data=''.join(rng.choice('abc🙂d') for _ in range(j%11))
            elif kind=='spiral':
                rows,cols=j%5,(j//5)%4
                data=[[r*cols+c+1 for c in range(cols)] for r in range(rows)]
            else:
                known=['','text','([]{})','{[(])}',']','a{b[c](d)}','([{}])','()[]{}','(()','())']
                data=known[j%len(known)] if j<10 else ''.join(rng.choice('([]){}ab') for _ in range(j))
            inputs.append(data)
        tasks.append({'task_id':'confirm_'+kind,'description':desc,
                      'cases':[[data,oracle(kind,data)] for data in inputs]})
    return tasks


def load_suite(path,expected_sha):
    import hashlib
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha:raise ValueError('confirmation suite hash mismatch')
    rows=json.loads(raw);ids=set();tasks=[]
    for row in rows:
        if row['task_id'] in ids:raise ValueError('duplicate confirmation task')
        ids.add(row['task_id'])
        if not row['cases']:raise ValueError('empty confirmation task')
        tasks.append(CodingTask(row['task_id'],row['description'],tuple((x[0],x[1]) for x in row['cases'])))
    if len(tasks)!=12:raise ValueError('expected 12 preregistered tasks')
    return tuple(tasks)


def promotion_gate(before,after):
    if {r['task_id'] for r in before}!={r['task_id'] for r in after}:raise ValueError('task sets differ')
    def summarize(rows):
        passed=sum(x['result']['passed'] for x in rows);total=sum(x['result']['total'] for x in rows)
        return {'passed_cases':passed,'total_cases':total,'case_accuracy':passed/total,
                'complete_tasks':sum(x['result']['passed']==x['result']['total'] for x in rows),'task_count':len(rows)}
    old={r['task_id']:r['result'] for r in before};new={r['task_id']:r['result'] for r in after}
    wins=losses=0
    for tid,x in old.items():
        y=new[tid]
        if x['total']!=y['total']:raise ValueError('case count changed')
        for z in (x,y):
            if type(z['passed']) is not int or type(z['total']) is not int or not 0<=z['passed']<=z['total'] or z['total']<=0:
                raise ValueError('invalid case counts')
        bc=x['passed']==x['total'];ac=y['passed']==y['total']
        wins+=ac and not bc;losses+=bc and not ac
    b,a=summarize(before),summarize(after)
    delta=a['case_accuracy']-b['case_accuracy']
    return {'case_gain':delta,'final_case_accuracy':a['case_accuracy'],'complete_task_wins':wins,
            'complete_task_losses':losses,'passed':delta>=.08 and a['case_accuracy']>=.75 and wins>=2 and losses==0},b,a
