"""Reference-tested curriculum for Qwen3.5-4B coding-format improvement.

Training and development tasks are public and adaptive. Promotion cases are
loaded from a separately hashed machine-local file and are never used to
choose examples, epochs, learning rate, or stopping.
"""
from dataclasses import dataclass
from .coding_tasks import CodingTask

@dataclass(frozen=True)
class SFTExample:
    task_id: str
    description: str
    solution: str
    cases: tuple

TRAIN = (
SFTExample("train_left_rotate","Input [items,k]. Rotate items left by k positions; negative k rotates right. Empty items returns [].",
"""def solve(data):
    items,k=data
    if not items:return []
    k%=len(items)
    return items[k:]+items[:k]
""", (([[1,2,3,4],1],[2,3,4,1]),([[1,2,3],-1],[3,1,2]),([[],3],[]))),
SFTExample("train_window_min","Input [numbers,k], 1<=k<=len(numbers). Return the minimum in each consecutive window of width k.",
"""def solve(data):
    nums,k=data
    return [min(nums[i:i+k]) for i in range(len(nums)-k+1)]
""", (([[4,2,7,1],2],[2,2,1]),([[3],1],[3]))),
SFTExample("train_touch_merge","Merge overlapping or touching closed integer intervals [a,b]. Return sorted merged intervals.",
"""def solve(data):
    if not data:return []
    xs=sorted(data)
    out=[xs[0][:]]
    for a,b in xs[1:]:
        if a<=out[-1][1]:
            out[-1][1]=max(out[-1][1],b)
        else:out.append([a,b])
    return out
""", (([[3,5],[1,3],[8,9]],[[1,5],[8,9]]),([],[]))),
SFTExample("train_grid_paths","Input [rows,cols,blocked]. Count right/down paths from [0,0] to bottom-right avoiding blocked cells.",
"""def solve(data):
    rows,cols,blocked=data
    bad={tuple(x) for x in blocked}
    if rows<1 or cols<1 or (0,0) in bad:return 0
    dp=[0]*cols;dp[0]=1
    for r in range(rows):
        for c in range(cols):
            if (r,c) in bad:dp[c]=0
            elif c:dp[c]+=dp[c-1]
    return dp[-1]
""", (([3,3,[[1,1]]],2),([2,3,[]],3),([2,2,[[0,0]]],0))),
SFTExample("train_zero_subarrays","Input [numbers,target]. Count nonempty contiguous subarrays summing to target.",
"""def solve(data):
    nums,target=data
    count=0;prefix=0;seen={0:1}
    for x in nums:
        prefix+=x
        count+=seen.get(prefix-target,0)
        seen[prefix]=seen.get(prefix,0)+1
    return count
""", (([[1,-1,0],0],3),([[1,1,1],2],2),([[],0],0))),
SFTExample("train_prev_greater","For each number, return the closest strictly greater number to its left, or -1.",
"""def solve(data):
    out=[];stack=[]
    for x in data:
        while stack and stack[-1]<=x:stack.pop()
        out.append(stack[-1] if stack else -1)
        stack.append(x)
    return out
""", (([3,1,2,5],[-1,3,3,-1]),([2,2],[-1,-1]))),
SFTExample("train_suffix_products","Return suffix products: output[i] is product of input[i:] for each i. Empty input returns [].",
"""def solve(data):
    out=[1]*len(data);p=1
    for i in range(len(data)-1,-1,-1):
        p*=data[i];out[i]=p
    return out
""", (([2,3,4],[24,12,4]),([0,2],[0,2]),([],[]))),
SFTExample("train_rle_decode","Input is a list of [character,count]. Return the decoded string; counts are nonnegative integers.",
"""def solve(data):
    return ''.join(ch*count for ch,count in data)
""", (([["a",3],["b",1]],"aaab"),([], ""),([["🙂",2]],"🙂🙂"))),
SFTExample("train_min_coins","Input [coins,amount]. Minimum number of unlimited positive coins making amount; -1 if impossible, 0 for zero.",
"""def solve(data):
    coins,amount=data
    best=[amount+1]*(amount+1);best[0]=0
    for value in range(1,amount+1):
        for coin in coins:
            if coin<=value:best[value]=min(best[value],best[value-coin]+1)
    return -1 if best[amount]>amount else best[amount]
""", (([[1,3,4],6],2),([[2],3],-1),([[7],0],0))),
SFTExample("train_longest_run","Return the length of the longest consecutive run of equal characters in the input string.",
"""def solve(data):
    if not data:return 0
    best=cur=1
    for i in range(1,len(data)):
        cur=cur+1 if data[i]==data[i-1] else 1
        best=max(best,cur)
    return best
""", (("aaabbc",3),("",0),("🙂🙂x",2))),
SFTExample("train_outer_ring","Return the outer ring of a rectangular matrix clockwise, without repeating corners. Empty matrix returns [].",
"""def solve(data):
    if not data or not data[0]:return []
    r=len(data);c=len(data[0])
    if r==1:return data[0][:]
    if c==1:return [row[0] for row in data]
    out=data[0][:]
    out += [data[i][c-1] for i in range(1,r)]
    out += data[r-1][c-2::-1]
    out += [data[i][0] for i in range(r-2,0,-1)]
    return out
""", (([[1,2,3],[4,5,6],[7,8,9]],[1,2,3,6,9,8,7,4]),([[1,2]],[1,2]))),
SFTExample("train_parens_only","Ignore non-parenthesis characters and return whether parentheses () are balanced.",
"""def solve(data):
    depth=0
    for ch in data:
        if ch=='(':depth+=1
        elif ch==')':
            depth-=1
            if depth<0:return False
    return depth==0
""", (("(a(b)c)",True),(")(",False),("",True))),
SFTExample("train_dedupe_sorted","Input is a sorted list. Return a new list with duplicates removed.",
"""def solve(data):
    out=[]
    for x in data:
        if not out or out[-1]!=x:out.append(x)
    return out
""", (([1,1,2,3,3],[1,2,3]),([],[]))),
SFTExample("train_transpose","Return the transpose of a rectangular matrix. Empty input returns [].",
"""def solve(data):
    if not data:return []
    return [list(row) for row in zip(*data)]
""", (([[1,2,3],[4,5,6]],[[1,4],[2,5],[3,6]]),([],[]))),
SFTExample("train_prefix_max","Return a list where output[i] is max(input[:i+1]). Empty input returns [].",
"""def solve(data):
    out=[];best=None
    for x in data:
        best=x if best is None else max(best,x)
        out.append(best)
    return out
""", (([2,1,5,3],[2,2,5,5]),([],[]))),
SFTExample("train_anagram_flag","Input [a,b]. Return whether strings a and b are anagrams, with exact character counts.",
"""def solve(data):
    a,b=data
    if len(a)!=len(b):return False
    counts={}
    for ch in a:counts[ch]=counts.get(ch,0)+1
    for ch in b:
        if counts.get(ch,0)==0:return False
        counts[ch]-=1
    return True
""", ((["listen","silent"],True),(["aab","abb"],False),(["",""],True))),
SFTExample("train_flatten_once","Flatten a list whose elements are either scalars or lists by exactly one level.",
"""def solve(data):
    out=[]
    for x in data:
        out.extend(x if isinstance(x,list) else [x])
    return out
""", (([1,[2,3],4],[1,2,3,4]),([],[]),([[1],[2]], [1,2]))),
SFTExample("train_first_index","Input [items,target]. Return first index of target or -1.",
"""def solve(data):
    items,target=data
    for i,x in enumerate(items):
        if x==target:return i
    return -1
""", (([[3,1,3],3],0),([[1,2],4],-1),([[],1],-1))),
)

DEV = (
CodingTask("dev_right_rotate","Input [items,k]. Rotate items right by k positions; negative k rotates left. Empty items returns [].", (([[1,2,3,4],1],[4,1,2,3]),([[1,2,3],-1],[2,3,1]),([[],8],[]))),
CodingTask("dev_window_max","Input [numbers,k], 1<=k<=len(numbers). Return maximum for each consecutive window.", (([[1,3,-1,-3,5],3],[3,3,5]),([[2,2],1],[2,2]))),
CodingTask("dev_strict_merge","Merge closed intervals only when they overlap (not merely touch). Return sorted intervals.", (([[1,2],[2,3],[5,7],[6,8]],[[1,2],[2,3],[5,8]]),([],[]))),
CodingTask("dev_paths_blocked","Input [rows,cols,blocked]. Count right/down paths through a grid avoiding blocked [r,c].", (([4,3,[[1,1],[2,1]]],2),([1,4,[]],1),([2,2,[[1,1]]],0))),
CodingTask("dev_next_greater","For each number return the next strictly greater number to its right, or -1.", (([2,1,2,4,3],[4,2,4,-1,-1]),([2,2],[-1,-1]),([],[]))),
CodingTask("dev_rle","Run-length encode consecutive characters into [character,count] pairs.", (("aaabbcca",[["a",3],["b",2],["c",2],["a",1]]),("",[]),("🙂🙂x",[["🙂",2],["x",1]]))),
CodingTask("dev_product_except","Return products of all input integers except self, without division. []=>[], singleton=>[1].", (([1,2,3,4],[24,12,8,6]),([0,2,3],[6,0,0]),([7],[1]))),
CodingTask("dev_spiral","Flatten a rectangular matrix in clockwise spiral order from top-left. Empty matrix returns [].", (([[1,2,3],[4,5,6],[7,8,9]],[1,2,3,6,9,8,7,4,5]),([[1],[2],[3]],[1,2,3]),([],[]))),
)

def as_coding(example):
    return CodingTask(example.task_id, example.description, example.cases)
