from .qwen4b_curriculum import TRAIN as BASE_TRAIN, SFTExample
from .coding_tasks import CodingTask

EXTRA = (
SFTExample("c_strict_overlap","Merge closed intervals [a,b] only when they overlap with positive interior; intervals that only touch remain separate. Return sorted intervals.",
'''def solve(data):
    if not data:return []
    xs=sorted(data);out=[xs[0][:]]
    for a,b in xs[1:]:
        if a<out[-1][1]:out[-1][1]=max(out[-1][1],b)
        else:out.append([a,b])
    return out
''', (([[1,2],[2,3],[5,7],[6,8]],[[1,2],[2,3],[5,8]]),([[1,4],[3,5]],[[1,5]]),([],[]))),
SFTExample("c_blocked_paths","Input [rows,cols,blocked]. Count paths from top-left to bottom-right moving only right/down, avoiding blocked coordinates.",
'''def solve(data):
    rows,cols,blocked=data;bad={tuple(x) for x in blocked}
    if rows<1 or cols<1 or (0,0) in bad or (rows-1,cols-1) in bad:return 0
    dp=[0]*cols;dp[0]=1
    for r in range(rows):
        for c in range(cols):
            if (r,c) in bad:dp[c]=0
            elif c:dp[c]+=dp[c-1]
    return dp[-1]
''', (([4,3,[[1,1],[2,1]]],2),([1,4,[]],1),([2,2,[[1,1]]],0))),
SFTExample("c_spiral","Flatten a rectangular matrix clockwise in spiral order. Empty input or zero-width rows returns [].",
'''def solve(data):
    if not data or not data[0]:return []
    out=[];top=0;bottom=len(data)-1;left=0;right=len(data[0])-1
    while top<=bottom and left<=right:
        out+=data[top][left:right+1];top+=1
        for r in range(top,bottom+1):out.append(data[r][right])
        right-=1
        if top<=bottom:out+=data[bottom][left:right+1][::-1];bottom-=1
        if left<=right:
            for r in range(bottom,top-1,-1):out.append(data[r][left])
            left+=1
    return out
''', (([[1,2,3],[4,5,6],[7,8,9]],[1,2,3,6,9,8,7,4,5]),([[1],[2],[3]],[1,2,3]),([],[]))),
SFTExample("c_right_rotate","Input [items,k]. Rotate items right by k; negative k rotates left. Empty items returns [].",
'''def solve(data):
    items,k=data
    if not items:return []
    k%=len(items)
    return items[-k:]+items[:-k] if k else items[:]
''', (([[1,2,3,4],1],[4,1,2,3]),([[1,2,3],-1],[2,3,1]),([[],9],[]))),
SFTExample("c_window_max","Input [numbers,k]. Return maximum of every consecutive width-k window.",
'''def solve(data):
    nums,k=data
    return [max(nums[i:i+k]) for i in range(len(nums)-k+1)]
''', (([[1,3,-1,-3,5],3],[3,3,5]),([[2,2],1],[2,2]))),
SFTExample("c_next_greater","For each number return the next strictly greater value to its right, else -1.",
'''def solve(data):
    out=[-1]*len(data);stack=[]
    for i,x in enumerate(data):
        while stack and data[stack[-1]]<x:out[stack.pop()]=x
        stack.append(i)
    return out
''', (([2,1,2,4,3],[4,2,4,-1,-1]),([2,2],[-1,-1]),([],[]))),
SFTExample("c_rle","Run-length encode consecutive characters into [character,count] pairs.",
'''def solve(data):
    if not data:return []
    out=[];ch=data[0];n=1
    for x in data[1:]:
        if x==ch:n+=1
        else:out.append([ch,n]);ch=x;n=1
    out.append([ch,n]);return out
''', (("aaabbcca",[["a",3],["b",2],["c",2],["a",1]]),("",[]),("🙂🙂x",[["🙂",2],["x",1]]))),
SFTExample("c_product_except","Return products of all input integers except self without division. []=>[], singleton=>[1].",
'''def solve(data):
    n=len(data);out=[1]*n;p=1
    for i in range(n):out[i]=p;p*=data[i]
    p=1
    for i in range(n-1,-1,-1):out[i]*=p;p*=data[i]
    return out
''', (([1,2,3,4],[24,12,8,6]),([0,2,3],[6,0,0]),([7],[1]),([],[]))),
SFTExample("c_brackets","Ignore non-bracket characters and decide whether (), [] and {} are properly balanced and nested.",
'''def solve(data):
    pairs={')':'(',']':'[','}':'{'};stack=[]
    for ch in data:
        if ch in '([{':stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop()!=pairs[ch]:return False
    return not stack
''', (("a{b[c](d)}",True),("{[(])}",False),("text",True),("]",False))),
SFTExample("c_two_distinct","Return the length of the longest substring containing at most two distinct characters.",
'''def solve(data):
    left=0;counts={};best=0
    for right,ch in enumerate(data):
        counts[ch]=counts.get(ch,0)+1
        while len(counts)>2:
            x=data[left];counts[x]-=1
            if counts[x]==0:del counts[x]
            left+=1
        best=max(best,right-left+1)
    return best
''', (("eceba",3),("ccaabbb",5),("",0),("🙂a🙂bb",3))),
SFTExample("c_coin_ways","Input [coins,amount]. Count order-independent combinations using unlimited positive coin denominations.",
'''def solve(data):
    coins,amount=data;dp=[0]*(amount+1);dp[0]=1
    for coin in coins:
        for value in range(coin,amount+1):dp[value]+=dp[value-coin]
    return dp[amount]
''', (([[1,2,5],5],4),([[2],3],0),([[3],0],1),([[2,3],7],1))),
SFTExample("c_stable_unique","Remove duplicates from a list while preserving first-occurrence order.",
'''def solve(data):
    out=[];seen=set()
    for x in data:
        if x not in seen:seen.add(x);out.append(x)
    return out
''', (([3,1,3,2,1],[3,1,2]),([],[]),(["a","a","b"],["a","b"]))),
SFTExample("c_intersections","Input [a,b] where each is sorted disjoint closed intervals. Return their closed intersections.",
'''def solve(data):
    a,b=data;i=j=0;out=[]
    while i<len(a) and j<len(b):
        lo=max(a[i][0],b[j][0]);hi=min(a[i][1],b[j][1])
        if lo<=hi:out.append([lo,hi])
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return out
''', (([[[1,4],[7,9]],[[2,5],[8,10]]],[[2,4],[8,9]]),([[[0,1]],[[2,3]]],[]))),
SFTExample("c_moving_sum","Input [numbers,k]. Return sums of every consecutive width-k window.",
'''def solve(data):
    nums,k=data
    if not nums:return []
    total=sum(nums[:k]);out=[total]
    for i in range(k,len(nums)):total+=nums[i]-nums[i-k];out.append(total)
    return out
''', (([[1,2,3,4],2],[3,5,7]),([[7],1],[7]),([[-2,5,-1],2],[3,4]))),
)

TRAIN = BASE_TRAIN + EXTRA

DEV = (
CodingTask("cdev_chunk_reverse","Input [items,size]. Reverse each consecutive chunk of at most size items independently.", (([[1,2,3,4,5],2],[2,1,4,3,5]),([[1,2,3,4],3],[3,2,1,4]),([[],3],[]))),
CodingTask("cdev_window_range","Input [numbers,k]. For each width-k window return max(window)-min(window).", (([[1,4,2,7],2],[3,2,5]),([[-2,-5,-1],2],[3,4]),([[8],1],[0]))),
CodingTask("cdev_union_length","Given half-open integer intervals [start,end), return total length of their union.", (([[1,4],[3,6],[8,10]],7),([[0,1],[1,2]],2),([],0))),
CodingTask("cdev_grid_min","Given a nonempty rectangular integer matrix, return minimum top-left to bottom-right path sum moving only right/down.", (([[1,3,1],[1,5,1],[4,2,1]],7),([[5]],5),([[1,2],[1,1]],3))),
CodingTask("cdev_prefix_hits","Input [numbers,target]. Count nonempty prefixes whose sum equals target.", (([[1,-1,2,-2],0],2),([[2,2,2],4],1),([[],0],0),([[0,0],0],2))),
CodingTask("cdev_prev_smaller","For each number return the closest strictly smaller value to its left, else -1.", (([3,1,2,0],[-1,-1,1,-1]),([2,2],[-1,-1]),([],[]))),
CodingTask("cdev_run_lengths","Return lengths of consecutive equal-character runs in a string.", (("aaabbcca",[3,2,2,1]),("abcd",[1,1,1,1]),("",[]),("🙂🙂x",[2,1]))),
CodingTask("cdev_sum_except","Return a list where each item is the sum of all input numbers except itself. []=>[].", (([1,2,3],[5,4,3]),([0,-2,5],[3,5,-2]),([7],[0]),([],[]))),
)
