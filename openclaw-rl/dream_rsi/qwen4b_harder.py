"""Second, preregistered in-domain distribution with stronger regression checks.

The prior near-ceiling suite stays rejected. These tasks are fixed before this
round's model evaluation; the absolute +8-point promotion gate is unchanged.
"""
from collections import OrderedDict,deque
from functools import lru_cache
import random
from .coding_tasks import CodingTask
from .qwen4b_curriculum import SFTExample
from .qwen4b_curriculum_c import TRAIN as PRIOR_TRAIN,DEV as PRIOR_DEV
from .qwen4b_validation import merge_closed,oracle as basic_oracle

EXTRA=(
SFTExample('d_chunks','Input [items,width]. Reverse each consecutive block of width items; reverse the final shorter block too.',
'''def solve(data):
    items,width=data
    out=[]
    for i in range(0,len(items),width):out.extend(items[i:i+width][::-1])
    return out
''',(([[1,2,3,4,5],2],[2,1,4,3,5]),([[],4],[]))),
SFTExample('d_window_range','Input [numbers,k]. Return the difference between the maximum and minimum in each consecutive width-k window.',
'''def solve(data):
    numbers,k=data
    return [max(numbers[i:i+k])-min(numbers[i:i+k]) for i in range(len(numbers)-k+1)]
''',(([[1,4,2,7],2],[3,2,5]),([[-2,-5,-1],2],[3,4]))),
SFTExample('d_union_measure','Given integer half-open intervals [a,b), return the length of their union.',
'''def solve(data):
    if not data:return 0
    intervals=sorted(data);left,right=intervals[0];total=0
    for a,b in intervals[1:]:
        if a<=right:right=max(right,b)
        else:total+=right-left;left,right=a,b
    return total+right-left
''',(([[1,4],[3,6],[8,10]],7),([],0))),
SFTExample('d_path_sum','Return the minimum path sum in the nonempty rectangular input matrix, from top-left to bottom-right using only right and down moves.',
'''def solve(data):
    dp=[float('inf')]*len(data[0]);dp[0]=0
    for row in data:
        for c,x in enumerate(row):dp[c]=x+min(dp[c],dp[c-1] if c else float('inf'))
    return dp[-1]
''',(([[1,3,1],[1,5,1],[4,2,1]],7),([[5]],5))),
SFTExample('d_prefix_count','Input [numbers,target]. Count nonempty prefixes whose total equals target.',
'''def solve(data):
    numbers,target=data;total=0;count=0
    for x in numbers:
        total+=x
        if total==target:count+=1
    return count
''',(([[1,-1,2,-2],0],2),([[],0],0))),
SFTExample('d_previous_smaller','For each integer, return the nearest strictly smaller value to its left, or -1.',
'''def solve(data):
    stack=[];out=[]
    for x in data:
        while stack and stack[-1]>=x:stack.pop()
        out.append(stack[-1] if stack else -1);stack.append(x)
    return out
''',(([3,1,2,0],[-1,-1,1,-1]),([2,2],[-1,-1]))),
SFTExample('d_run_sizes','Given a string, return the lengths of consecutive runs of equal characters.',
'''def solve(data):
    if not data:return []
    out=[];previous=data[0];count=1
    for ch in data[1:]:
        if ch==previous:count+=1
        else:out.append(count);previous=ch;count=1
    out.append(count);return out
''',(('aaabbcca',[3,2,2,1]),('',[]))),
SFTExample('d_sum_others','Return the sum of all other integers at each input index. Empty input returns [].',
'''def solve(data):
    total=sum(data)
    return [total-x for x in data]
''',(([1,2,3],[5,4,3]),([7],[0]))),
SFTExample('d_rpn','Evaluate valid reverse Polish notation tokens: signed integers and +,-,*,/. Division truncates toward zero. Input is a token list; return one integer.',
'''def solve(data):
    stack=[]
    for token in data:
        if token not in ('+','-','*','/'):stack.append(int(token));continue
        b=stack.pop();a=stack.pop()
        if token=='+':value=a+b
        elif token=='-':value=a-b
        elif token=='*':value=a*b
        else:value=int(a/b)
        stack.append(value)
    return stack[0]
''',((['4','13','5','/','+'],6),(['-7','2','/'],-3))),
SFTExample('d_lru','Input [capacity,operations] for an LRU cache initially empty. Operations are ["put",key,value] or ["get",key]. Return get results, -1 for missing. Capacity can be zero.',
'''def solve(data):
    capacity,ops=data;cache={};out=[]
    for op in ops:
        key=op[1]
        if op[0]=='get':
            if key in cache:
                value=cache.pop(key);cache[key]=value;out.append(value)
            else:out.append(-1)
        elif capacity:
            cache.pop(key,None);cache[key]=op[2]
            if len(cache)>capacity:cache.pop(next(iter(cache)))
    return out
''',(([2,[['put',1,7],['put',2,8],['get',1],['put',3,9],['get',2]]],[7,-1]),([0,[['put',1,2],['get',1]]],[-1]))),
SFTExample('d_grid_distance','Input [grid,start,goal]. Grid is nonempty, 0=open and 1=blocked. Return minimum four-direction moves between [row,col] coordinates, -1 if blocked/unreachable.',
'''def solve(data):
    grid,start,goal=data;rows=len(grid);cols=len(grid[0]);s=tuple(start);g=tuple(goal)
    if grid[s[0]][s[1]] or grid[g[0]][g[1]]:return -1
    frontier={s};seen={s};distance=0
    while frontier:
        if g in frontier:return distance
        nxt=set()
        for r,c in frontier:
            for a,b in ((r+1,c),(r-1,c),(r,c+1),(r,c-1)):
                if 0<=a<rows and 0<=b<cols and not grid[a][b] and (a,b) not in seen:seen.add((a,b));nxt.add((a,b))
        frontier=nxt;distance+=1
    return -1
''',(([[[0,0,0],[1,1,0],[0,0,0]],[0,0],[2,0]],6),([[[0,1],[1,0]],[0,0],[1,1]],-1))),
SFTExample('d_edit_distance','Input [a,b] strings. Return Levenshtein distance with insertion, deletion and substitution costing one each.',
'''def solve(data):
    a,b=data;dp=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row=[i]
        for j,y in enumerate(b,1):row.append(min(row[-1]+1,dp[j]+1,dp[j-1]+(x!=y)))
        dp=row
    return dp[-1]
''',((['kitten','sitting'],3),(['','abc'],3),(['🙂a','🙂b'],1))),
)
# Preserve the known-good closed-endpoint behavior explicitly, without using
# any future confirmation outcomes to set this curriculum.
closed=next(x for x in PRIOR_TRAIN if x.task_id=='train_touch_merge')
ANCHOR_TRAIN=(
SFTExample('d_closed_boundaries','Return the union of closed intervals as sorted intervals. Shared endpoints connect intervals and MUST be merged.',closed.solution,closed.cases),
SFTExample('d_closed_zero','Combine overlapping or endpoint-touching closed intervals, including zero-length intervals.',closed.solution,(([[0,0],[0,2],[2,2]],[[0,2]]),([],[]))),
)
TRAIN=PRIOR_TRAIN+EXTRA+ANCHOR_TRAIN
DEV=PRIOR_DEV+(
CodingTask('ddev_rpn','Input is a valid postfix expression token list with integers and +,-,*,/. Return its value with division truncated toward zero.',((['2','1','+','3','*'],9),(['7','-3','/'],-2),(['8'],8))),
CodingTask('ddev_lru','Input [capacity,operations]. Simulate least-recently-used cache. ["put",key,value] sets a key; ["get",key] returns value or -1 and updates recency. Return all get values.',(([1,[['put',1,5],['put',2,6],['get',1],['get',2]]],[-1,6]),([0,[['get',1]]],[-1]),([2,[['put',1,3],['get',1],['get',1]]],[3,3]))),
CodingTask('ddev_bfs','Input [grid,start,goal], with 0=open,1=wall. Return shortest four-neighbor path length or -1; blocked start/goal means -1.',(([[[0,0],[0,0]],[0,0],[1,1]],2),([[[0]],[0,0],[0,0]],0),([[[1]],[0,0],[0,0]],-1))),
CodingTask('ddev_edit','Input [a,b]. Return minimum single-character insertions, deletions and substitutions to turn string a into b.',((['abc','adc'],1),(['abc',''],3),(['same','same'],0))),
)
GUARDS=(
CodingTask('guard_closed_merge','Given closed intervals [a,b], merge overlaps and shared endpoints and return sorted intervals.',(([[1,2],[2,4]],[[1,4]]),([[0,0],[0,2]],[[0,2]]),([[1,5],[2,3]],[[1,5]]),([],[]))),
CodingTask('guard_brackets','Ignore non-bracket characters and return whether (), [], {} are balanced and correctly nested.',(('([]{})',True),('([)]',False),('abc',True),(')',False))),
CodingTask('guard_coins','Input [coins,amount]. Return minimum count of reusable positive coins summing to amount, -1 if impossible.',(([[1,3,4],6],2),([[2],3],-1),([[7],0],0))),
)


def hard_oracle(kind,data):
    if kind=='chunks':
        items,n=data;return [items[min(i//n*n+n,len(items))-1-i%n] for i in range(len(items))]
    if kind=='range':
        nums,n=data;return [sorted(nums[i:i+n])[-1]-sorted(nums[i:i+n])[0] for i in range(len(nums)-n+1)]
    if kind=='union':return len({x for a,b in data for x in range(a,b)})
    if kind=='minpath':
        def walk(r,c):
            x=data[r][c]
            if r==len(data)-1 and c==len(data[0])-1:return x
            return x+min(([walk(r+1,c)] if r+1<len(data) else [])+([walk(r,c+1)] if c+1<len(data[0]) else []))
        return walk(0,0)
    if kind=='prefix':
        xs,t=data;return sum(sum(xs[:j])==t for j in range(1,len(xs)+1))
    if kind=='smaller':return [next((v for v in data[:i][::-1] if v<x),-1) for i,x in enumerate(data)]
    if kind=='runs':return [n for _,n in basic_oracle('rle',data)]
    if kind=='sum':return [sum(data[:i]+data[i+1:]) for i in range(len(data))]
    if kind=='rpn':
        stack=[]
        for t in data:
            if t not in ['+','-','*','/']:stack.append(int(t));continue
            b,a=stack.pop(),stack.pop()
            if t=='+':v=a+b
            elif t=='-':v=a-b
            elif t=='*':v=a*b
            else:v=(abs(a)//abs(b))*(-1 if (a<0)!=(b<0) else 1)
            stack.append(v)
        return stack[0]
    if kind=='lru':
        cap,ops=data;cache=OrderedDict();out=[]
        for op in ops:
            k=op[1]
            if op[0]=='get':
                out.append(cache.get(k,-1))
                if k in cache:cache.move_to_end(k)
            elif cap:
                cache[k]=op[2];cache.move_to_end(k)
                if len(cache)>cap:cache.popitem(last=False)
        return out
    if kind=='bfs':
        grid,s,g=data;s=tuple(s);g=tuple(g)
        if grid[s[0]][s[1]] or grid[g[0]][g[1]]:return -1
        queue=deque([(s,0)]);seen={s}
        while queue:
            (r,c),d=queue.popleft()
            if (r,c)==g:return d
            for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                p=(r+dr,c+dc)
                if 0<=p[0]<len(grid) and 0<=p[1]<len(grid[0]) and grid[p[0]][p[1]]==0 and p not in seen:seen.add(p);queue.append((p,d+1))
        return -1
    if kind=='edit':
        a,b=data
        @lru_cache(None)
        def f(i,j):
            if i==len(a):return len(b)-j
            if j==len(b):return len(a)-i
            return min(1+f(i+1,j),1+f(i,j+1),(a[i]!=b[j])+f(i+1,j+1))
        return f(0,0)
    raise ValueError('unknown task')

DESCRIPTIONS={
'chunks':'Input [items,size]. Reverse each consecutive group of size elements independently; reverse a shorter last group too.',
'range':'Input [numbers,k]. Return maximum minus minimum for every consecutive width-k window, in order.',
'union':'Given half-open integer intervals [start,end), return the total length covered by at least one interval.',
'minpath':'For a nonempty rectangular matrix, return the minimum sum along a path from top-left to bottom-right using right and down moves.',
'prefix':'Input [numbers,target]. Count nonempty initial prefixes whose sum equals target.',
'smaller':'For every integer, report the closest strictly smaller value to its left, or -1 if there is none.',
'runs':'Given a string, output just the lengths of successive runs of identical characters.',
'sum':'For each index, return the sum of every other integer in the input list. An empty list returns [].',
'rpn':'Input is a valid reverse Polish notation expression as string tokens. Operators + - * / are binary. Return its integer result; division truncates toward zero.',
'lru':'Input [capacity,operations]. Simulate an initially empty LRU cache: ["put",key,value], ["get",key]. Return get results, -1 for missing. Every access refreshes recency. Capacity zero is allowed.',
'bfs':'Input [grid,start,goal]. Grid cells 0 are open and 1 are walls. Return the minimum number of four-direction moves, or -1 when unreachable or an endpoint is blocked.',
'edit':'Input [a,b] strings. Compute Levenshtein distance: each character insertion, deletion, or substitution costs one.'}


def build_hard_suite(seed):
    rng=random.Random(seed);out=[]
    for kind,description in DESCRIPTIONS.items():
        cases=[]
        for j in range(20):
            xs=[rng.randrange(-4,8) for _ in range(j%9)]
            if kind=='chunks':data=[xs,1+j%5]
            elif kind=='range':
                xs=xs or [3];data=[xs,rng.randrange(1,len(xs)+1)]
            elif kind=='union':data=[sorted([rng.randrange(-5,10),rng.randrange(-5,10)]) for _ in range(j%6)]
            elif kind=='minpath':data=[[rng.randrange(-2,9) for _ in range(1+j%4)] for _ in range(1+(j//4)%4)]
            elif kind=='prefix':data=[xs,rng.randrange(-5,6)]
            elif kind in ['smaller','sum']:data=xs
            elif kind=='runs':data=''.join(rng.choice('ab🙂')*rng.randrange(1,4) for _ in range(j%7))
            elif kind=='rpn':
                a,b,c=[rng.choice([-7,-3,-1,1,2,4,8]) for _ in range(3)]
                data=[str(a),str(b),rng.choice(['+','-','*','/']),str(c),rng.choice(['+','-','*','/'])]
            elif kind=='lru':
                ops=[]
                for k in range(12):
                    key=rng.randrange(4)
                    ops.append(['get',key] if rng.random()<.5 else ['put',key,rng.randrange(-3,20)])
                data=[j%4,ops]
            elif kind=='bfs':
                rows,cols=1+j%4,1+(j//4)%4
                data=[[[int(rng.random()<.25) for _ in range(cols)] for _ in range(rows)],[0,0],[rows-1,cols-1]]
            else:data=[''.join(rng.choice('abc🙂') for _ in range(j%7)),''.join(rng.choice('abc🙂') for _ in range((j+2)%7))]
            cases.append([data,hard_oracle(kind,data)])
        out.append({'task_id':'hard_confirm_'+kind,'description':description,'cases':cases})
    return out
