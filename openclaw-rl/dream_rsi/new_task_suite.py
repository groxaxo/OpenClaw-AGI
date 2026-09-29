"""Preregisterable synthetic software tasks, separate from prior curricula.

References are trusted author code. Their outputs are checked against independent
oracles before model evaluation. New inputs, not unknown pretraining exposure.
"""
from __future__ import annotations
import copy, csv, io, ipaddress, itertools, json, posixpath, random, re
from decimal import Decimal, ROUND_HALF_EVEN
from fractions import Fraction
from functools import lru_cache
from .external_judge import canonical, digest
from .coding_tasks import CodingTask

SPECS = {
'versions': ('Input [a,b] contains dotted nonnegative-integer version strings. Compare numerically component by component, treating missing trailing components as zero. Return -1, 0 or 1.', '''def solve(data):
    a, b = [list(map(int, s.split('.'))) for s in data]
    n = max(len(a), len(b))
    a += [0] * (n-len(a)); b += [0] * (n-len(b))
    return (a > b) - (a < b)
'''),
'wildcard': ('Input [text,pattern]. Return whether the entire text matches the pattern: ? matches one character, * matches any number including zero. All other characters are literal. Case sensitive.', '''def solve(data):
    text, pattern = data
    dp = [True] + [False]*len(text)
    for c in pattern:
        new = [False]*(len(text)+1)
        if c == '*': new[0] = dp[0]
        for i in range(1,len(text)+1):
            new[i] = (dp[i] or new[i-1]) if c == '*' else (dp[i-1] and (c == '?' or c == text[i-1]))
        dp = new
    return dp[-1]
'''),
'ordering': ('Input [n,edges] is a directed graph on integers 0..n-1; each [u,v] requires u before v. Return the lexicographically smallest topological ordering. Duplicate edges have no extra meaning. Return [] on a cycle.', '''def solve(data):
    import heapq
    n, edges = data
    adj = [set() for i in range(n)]; indeg = [0]*n
    for u,v in edges:
        if v not in adj[u]: adj[u].add(v); indeg[v] += 1
    ready = [i for i in range(n) if indeg[i] == 0]; heapq.heapify(ready); out = []
    while ready:
        u = heapq.heappop(ready); out.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0: heapq.heappush(ready,v)
    return out if len(out) == n else []
'''),
'csv_record': ('Input is one valid CSV record without line breaks. Return the list of fields. Commas separate fields; a quoted field starts with double quote, doubled double quotes represent one literal quote. Spaces are preserved. Empty input means one empty field, not zero fields.', '''def solve(data):
    fields = []; field = ''; quoted = False; i = 0
    while i < len(data):
        c = data[i]
        if quoted:
            if c == '"':
                if i+1 < len(data) and data[i+1] == '"': field += '"'; i += 1
                else: quoted = False
            else: field += c
        elif c == '"' and field == '': quoted = True
        elif c == ',': fields.append(field); field = ''
        else: field += c
        i += 1
    return fields + [field]
'''),
'paths': ('Normalize a POSIX-style path string. Remove repeated / and . components. Resolve .. against a previous ordinary component; preserve unresolved .. in relative paths, never climb above / in absolute paths. Empty relative result is .; absolute result starts with exactly one /. Ignore trailing slashes.', '''def solve(data):
    absolute = data.startswith('/'); stack = []
    for part in data.split('/'):
        if part in ('', '.'): continue
        if part == '..':
            if stack and stack[-1] != '..': stack.pop()
            elif not absolute: stack.append('..')
        else: stack.append(part)
    text = '/'.join(stack)
    return '/' + text if absolute else (text or '.')
'''),
'ipv4': ('Input [address,routes]. Address is dotted IPv4; each route is [cidr,label], cidr such as 10.0.0.0/8; host bits may be set. Return the label of the matching route with longest prefix. Equal prefixes choose the first matching route. Return null/None when no route matches.', '''def solve(data):
    address, routes = data
    def number(s):
        value = 0
        for part in s.split('.'): value = value*256 + int(part)
        return value
    ip = number(address); best = -1; answer = None
    for cidr, label in routes:
        base, length = cidr.split('/'); length = int(length)
        if length > best and (ip >> (32-length)) == (number(base) >> (32-length)):
            best = length; answer = label
    return answer
'''),
'pointer': ('Input [document,pointer] uses JSON Pointer. Empty pointer selects the whole document; otherwise split after the first /, decode ~1 to / then ~0 to ~. Dictionary keys are literal; list indices must be canonical nonnegative integers (0 or no leading zero), not -. Return {"found":true,"value":selected} or {"found":false} on any missing/invalid step.', '''def solve(data):
    value, pointer = data
    if pointer == '': return {'found':True,'value':value}
    if not pointer.startswith('/'): return {'found':False}
    for token in pointer[1:].split('/'):
        token = token.replace('~1','/').replace('~0','~')
        if isinstance(value,dict):
            if token not in value: return {'found':False}
            value = value[token]
        elif isinstance(value,list):
            if not token.isdigit() or (len(token)>1 and token[0]=='0'): return {'found':False}
            i = int(token)
            if i >= len(value): return {'found':False}
            value = value[i]
        else: return {'found':False}
    return {'found':True,'value':value}
'''),
'bucket': ('Input [capacity,refill_rate,events]. A token bucket starts full at time 0. Events [time,cost] are sorted, with nonnegative integer times/costs; capacity and refill_rate are nonnegative integers. Before each event add elapsed*rate tokens capped at capacity. Accept and deduct cost iff enough tokens; rejected events deduct nothing. Return acceptance booleans.', '''def solve(data):
    capacity, rate, events = data; tokens = capacity; last = 0; out = []
    for time, cost in events:
        tokens = min(capacity,tokens+(time-last)*rate); last = time
        ok = tokens >= cost; out.append(ok)
        if ok: tokens -= cost
    return out
'''),
'merge_patch': ('Input [target,patch] is a JSON merge patch. A non-dictionary patch replaces the target entirely. A dictionary patch treats a non-dictionary target as {}, removes keys whose patch value is null, and recursively merges every other key. Arrays are replacement values. Return the resulting JSON value.', '''def solve(data):
    target, patch = data
    def apply(target, patch):
        if not isinstance(patch,dict): return patch
        out = dict(target) if isinstance(target,dict) else {}
        for key,value in patch.items():
            if value is None: out.pop(key,None)
            else: out[key] = apply(out.get(key),value)
        return out
    return apply(target,patch)
'''),
'round_even': ('Input [decimal_string,places], 0<=places<=4. Round the finite signed base-10 decimal to exactly places fractional digits, using round-half-to-even. Avoid binary floating-point. Return a string with exactly places digits after the dot when places>0; normalize negative zero to positive zero.', '''def solve(data):
    text, places = data; negative = text.startswith('-'); text = text.lstrip('+-')
    pieces = text.split('.'); whole = pieces[0] or '0'; frac = pieces[1] if len(pieces)>1 else ''
    numerator = int(whole+frac); scale = 10**len(frac)
    q, r = divmod(numerator*10**places,scale)
    if 2*r > scale or (2*r == scale and q%2): q += 1
    s = str(q).zfill(places+1)
    if places: s = s[:-places]+'.'+s[-places:]
    return ('-' if negative and q else '') + s
'''),
}

TRANSFER = {
'roman': 'Input is an integer 1..3999. Return its canonical uppercase Roman numeral, using subtractive pairs IV, IX, XL, XC, CD and CM.',
'rectangle': 'Input is a rectangular matrix of integer 0/1. Return the area of the largest axis-aligned all-1 rectangle. Empty input or empty rows has area zero.',
'bipartite': 'Input [n,edges] is an undirected graph on 0..n-1; duplicate edges allowed. Return whether it is bipartite, including disconnected vertices; any self-loop makes it false.',
'range_add': 'Input [n,updates] describes an initially zero array of length n. Each [left,right,delta] adds delta at every inclusive index left..right. Return the final array. Indices are valid; n may be zero only with no updates.',
'shell_words': 'Split the input command text into words. Space separates words outside quotes. Single and double quotes remove themselves and preserve contents; adjacent quoted/unquoted parts concatenate. Backslash escapes the next character everywhere except inside single quotes. Empty quoted strings produce empty words. Input has balanced quotes and no dangling backslash. No expansion or other shell syntax.',
'rational': 'Input [a,b,operator,c,d], b,d nonzero, operator one of +,-,*,/. Evaluate (a/b) operator (c/d); division has c!=0. Return [numerator,denominator] in lowest terms with positive denominator; zero is [0,1].',
}


def oracle(name, data):
    """Trusted small-input or standard-library gold, never executed model code."""
    if name == 'versions':
        a,b=(tuple(map(int,x.split('.'))) for x in data)
        for x,y in itertools.zip_longest(a,b,fillvalue=0):
            if x!=y: return 1 if x>y else -1
        return 0
    if name == 'wildcard':
        text,pattern=data
        @lru_cache(None)
        def f(i,j):
            if j==len(pattern): return i==len(text)
            if pattern[j]=='*': return f(i,j+1) or (i<len(text) and f(i+1,j))
            return i<len(text) and (pattern[j]=='?' or pattern[j]==text[i]) and f(i+1,j+1)
        return f(0,0)
    if name == 'ordering':
        n,edges=data
        for perm in itertools.permutations(range(n)):
            pos={v:i for i,v in enumerate(perm)}
            if all(pos[u]<pos[v] for u,v in edges): return list(perm)
        return []
    if name == 'csv_record': return next(csv.reader([data])) if data else ['']
    if name == 'paths':
        value=posixpath.normpath(data)
        return '/'+value.lstrip('/') if data.startswith('/') else value
    if name == 'ipv4':
        address,routes=data; ip=ipaddress.IPv4Address(address); matches=[]
        for i,(cidr,label) in enumerate(routes):
            network=ipaddress.IPv4Network(cidr,strict=False)
            if ip in network: matches.append((-network.prefixlen,i,label))
        return min(matches)[2] if matches else None
    if name == 'pointer':
        obj,pointer=data
        if not pointer: return {'found':True,'value':obj}
        if not pointer.startswith('/'): return {'found':False}
        try:
            for token in pointer[1:].split('/'):
                token=re.sub(r'~[01]',lambda m:'~' if m[0]=='~0' else '/',token)
                if isinstance(obj,list):
                    if not re.fullmatch('0|[1-9][0-9]*',token): return {'found':False}
                    obj=obj[int(token)]
                elif isinstance(obj,dict): obj=obj[token]
                else: return {'found':False}
            return {'found':True,'value':obj}
        except (KeyError,IndexError): return {'found':False}
    if name == 'bucket':
        cap,rate,events=data; deficit=0; previous=0; result=[]
        for time,cost in events:
            for _ in range(time-previous): deficit=max(0,deficit-rate)
            allowed=deficit+cost<=cap
            if allowed: deficit+=cost
            result.append(allowed); previous=time
        return result
    if name == 'merge_patch':
        target,patch=copy.deepcopy(data)
        if not isinstance(patch,dict): return patch
        root=target if isinstance(target,dict) else {}; pending=[(root,patch)]
        while pending:
            dst,src=pending.pop()
            for k,v in src.items():
                if v is None: dst.pop(k,None)
                elif isinstance(v,dict):
                    if not isinstance(dst.get(k),dict): dst[k]={}
                    pending.append((dst[k],v))
                else: dst[k]=v
        return root
    if name == 'round_even':
        text,places=data; value=Decimal(text).quantize(Decimal(1).scaleb(-places),rounding=ROUND_HALF_EVEN)
        if value==0: value=abs(value)
        return f'{value:.{places}f}'
    if name == 'roman':
        thousands=['','M','MM','MMM']; hundreds=['','C','CC','CCC','CD','D','DC','DCC','DCCC','CM']
        tens=['','X','XX','XXX','XL','L','LX','LXX','LXXX','XC']; units=['','I','II','III','IV','V','VI','VII','VIII','IX']
        return thousands[data//1000]+hundreds[data//100%10]+tens[data//10%10]+units[data%10]
    if name == 'rectangle':
        if not data or not data[0]: return 0
        h,w=len(data),len(data[0]); best=0
        for r0 in range(h):
            for r1 in range(r0+1,h+1):
                for c0 in range(w):
                    for c1 in range(c0+1,w+1):
                        if all(data[r][c] for r in range(r0,r1) for c in range(c0,c1)):
                            best=max(best,(r1-r0)*(c1-c0))
        return best
    if name == 'bipartite':
        n,edges=data
        return any(all(((bits>>u)&1)!=((bits>>v)&1) for u,v in edges) for bits in range(1<<n))
    if name == 'range_add':
        n,updates=data
        return [sum(delta for left,right,delta in updates if left<=i<=right) for i in range(n)]
    if name == 'shell_words':
        # Generated words carry their exact decoded gold alongside the text;
        # the independent oracle uses a regex scanner of the stipulated grammar.
        pattern=r'''(?:[^\s'"\\]+|'[^']*'|"(?:\\.|[^"\\])*"|\\.)+'''
        words=[]
        for m in re.finditer(pattern,data):
            token=m.group(); pieces=re.findall(r'''[^'"\\]+|'[^']*'|"(?:\\.|[^"\\])*"|\\.''',token)
            parts=[]
            for x in pieces:
                if x.startswith("'"): parts.append(x[1:-1])
                elif x.startswith('"'): parts.append(re.sub(r'\\(.)',r'\1',x[1:-1]))
                elif x.startswith('\\'): parts.append(x[1:])
                else: parts.append(x)
            words.append(''.join(parts))
        return words
    if name == 'rational':
        a,b,op,c,d=data; x=Fraction(a,b); y=Fraction(c,d)
        value={'+':lambda:x+y,'-':lambda:x-y,'*':lambda:x*y,'/':lambda:x/y}[op]()
        return [value.numerator,value.denominator]
    raise ValueError('unknown oracle '+name)


def draw(name,rng):
    ints=lambda n:[rng.randint(-9,12) for _ in range(n)]
    if name=='versions':
        a=[rng.randrange(15) for _ in range(rng.randint(1,5))]
        b=a+[0]*rng.randrange(1,4) if rng.random()<.35 else [rng.randrange(15) for _ in range(rng.randint(1,5))]
        return ['.'.join(map(str,a)),'.'.join(('0'+str(x) if rng.random()<.3 else str(x)) for x in b)]
    if name=='wildcard':
        text=''.join(rng.choices('abc',k=rng.randrange(9)))
        pattern=''.join(rng.choices('abc?*',k=rng.randrange(9)))
        if rng.random()<.4: pattern='*'+text[:len(text)//2]+'?*'
        return [text,pattern]
    if name=='ordering':
        n=rng.randrange(1,7); edges=[]
        for _ in range(rng.randrange(n*n+1)):
            u,v=rng.randrange(n),rng.randrange(n)
            if rng.random()<.75: u,v=sorted((u,v))
            edges.append([u,v])
        return [n,edges]
    if name=='csv_record':
        fields=[''.join(rng.choices('ab ,"',k=rng.randrange(9))) for _ in range(rng.randrange(1,6))]
        s=io.StringIO(); csv.writer(s,lineterminator='\n').writerow(fields)
        return s.getvalue()[:-1]
    if name=='paths':
        return ('/'*rng.randrange(1,4) if rng.random()<.5 else '') + '/'.join(rng.choices(['.','..','a','b','c',''],k=rng.randrange(1,9)))
    if name=='ipv4':
        ip=rng.randrange(2**32); routes=[]
        for i in range(rng.randrange(0,7)):
            length=rng.choice([0,8,16,24,28,32]); base=ip if rng.random()<.55 else rng.randrange(2**32)
            routes.append([str(ipaddress.IPv4Address(base))+'/'+str(length),'route'+str(i)])
        return [str(ipaddress.IPv4Address(ip)),routes]
    if name=='pointer':
        doc={'a/b':ints(rng.randrange(1,5)), '~key':{'':rng.randrange(100)},'01':rng.choice([None,True,False]),'nested':[{'x':rng.randrange(99)}]}
        pointer=rng.choice(['','/a~1b/0','/a~1b/9','/~0key/','/01','/nested/0/x','/nested/01','/nested/-','/missing','/a~1b/00'])
        return [doc,pointer]
    if name=='bucket':
        capacity=rng.randrange(0,13); rate=rng.randrange(0,5); time=0;events=[]
        for _ in range(rng.randrange(0,12)):
            time+=rng.randrange(0,4);events.append([time,rng.randrange(0,capacity+5)])
        return [capacity,rate,events]
    if name=='merge_patch':
        target={'a':rng.randrange(99),'b':{'x':rng.randrange(99),'y':[1,2]},'c':[rng.randrange(99)]}
        patch=rng.choice([None,[rng.randrange(99)],{'a':None,'b':{'x':rng.randrange(99),'y':None},'d':{'z':None}}, {'a':{'q':rng.randrange(99)},'b':[rng.randrange(99)]}])
        if rng.random()<.3: target=[rng.randrange(99)]
        return [target,patch]
    if name=='round_even':
        places=rng.randrange(5); whole=str(rng.randrange(50)); frac=''.join(rng.choices('0123456789',k=places))+'5'
        if rng.random()<.35: frac+=''.join(rng.choices('0123456789',k=rng.randrange(1,4)))
        return [rng.choice(['','-','+'])+whole+'.'+frac,places]
    if name=='roman': return rng.randrange(1,4000)
    if name=='rectangle':
        h,w=rng.randrange(1,6),rng.randrange(1,6)
        return [[rng.randrange(2) for _ in range(w)] for _ in range(h)]
    if name=='bipartite':
        n=rng.randrange(1,8)
        return [n,[[rng.randrange(n),rng.randrange(n)] for _ in range(rng.randrange(12))]]
    if name=='range_add':
        n=rng.randrange(1,12);updates=[]
        for _ in range(rng.randrange(9)):
            a,b=sorted([rng.randrange(n),rng.randrange(n)]);updates.append([a,b,rng.randrange(-7,8)])
        return [n,updates]
    if name=='shell_words':
        words=[''.join(rng.choices('ab $*',k=rng.randrange(0,7))) for _ in range(rng.randrange(1,6))]
        encoded=[]
        for s in words:
            form=rng.randrange(3)
            encoded.append("'"+s+"'" if form==0 else ('"'+s+'"' if form==1 else (''.join('\\'+c if c==' ' else c for c in s) or "''")))
        text='  '.join(encoded)
        assert oracle(name,text)==words,(text,words,oracle(name,text))
        return text
    if name=='rational':
        op=rng.choice('+-*/');c=rng.choice([x for x in range(-12,13) if x!=0])
        return [rng.randrange(-15,16),rng.choice([-9,-5,-2,1,3,7]),op,c,rng.choice([-8,-3,1,2,5,11])]
    raise ValueError(name)


def build_suite(seed=20260929,count=24):
    if type(seed) is not int or type(count) is not int or not 4<=count<=40: raise ValueError('bad suite budget')
    rng=random.Random(seed); bundle={'schema':1,'seed':seed,'count_per_task':count,'train':[],'dev':[],'confirmation':[],'transfer':[]}
    for name,(description,solution) in SPECS.items():
        seen=set()
        for split in ('train','dev','confirmation'):
            cases=[]
            for _ in range(10000):
                arg=draw(name,rng);key=canonical(arg)
                if key in seen: continue
                seen.add(key);cases.append([arg,oracle(name,copy.deepcopy(arg))])
                if len(cases)==count: break
            if len(cases)!=count: raise ValueError('insufficient distinct cases '+name)
            row={'task_id':split+'_'+name,'family':name,'description':description,'cases':cases}
            if split=='train': row['solution']=solution
            bundle[split].append(row)
    for name,description in TRANSFER.items():
        cases=[];seen=set()
        while len(cases)<count:
            arg=draw(name,rng);key=canonical(arg)
            if key in seen: continue
            seen.add(key);cases.append([arg,oracle(name,copy.deepcopy(arg))])
        bundle['transfer'].append({'task_id':'transfer_'+name,'family':name,'description':description,'cases':cases})
    return bundle


def as_task(row): return CodingTask(row['task_id'],row['description'],tuple((a,b) for a,b in row['cases']))
