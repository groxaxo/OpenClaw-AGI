"""Small immutable coding smoke suite; NOT a capability benchmark.

Generated code only executes in a digest-pinned, unprivileged Docker container
with no network, no host mounts, read-only root and hard resource limits.
"""
from __future__ import annotations
import ast
import json
import re
import subprocess
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CodingTask:
    task_id: str
    description: str
    cases: tuple[tuple[Any, Any], ...]


TRAIN_TASKS = (
    CodingTask("train_merge", "Merge overlapping or touching closed intervals given as a list of [start,end]. Return sorted merged intervals; empty input gives [].", (([], []), ([[3,4],[1,3],[8,10]], [[1,4],[8,10]]), ([[1,8],[2,3],[8,9]], [[1,9]]))),
    CodingTask("train_window", "Input is [numbers,k], with 1<=k<=len(numbers). Return the maximum in every consecutive window of width k.", (([[1,3,-1,-3,5,3,6,7],3],[3,3,5,5,6,7]), ([[2,2,1],2],[2,2]), ([[-4],1],[-4]))),
    CodingTask("train_brackets", "Input is a string. Ignore non-bracket characters and decide whether (), [] and {} are properly balanced and nested. Return a boolean.", (("([]){x}",True), ("([)]",False), ("a)b",False), ("",True))),
    CodingTask("train_unique", "Input is a string. Return the length of its longest substring with no repeated characters. Unicode characters count as single characters.", (("abcabcbb",3),("abba",2),("",0),("🙂a🙂b",3))),
    CodingTask("train_paths", "Input [rows,cols,blocked] describes a grid starting at [0,0], ending at [rows-1,cols-1]. Count paths moving only right or down and avoiding blocked [r,c] cells. Return the integer count.", (([3,3,[[1,1]]],2),([1,1,[]],1),([2,2,[[0,0]]],0),([2,3,[]],3))),
    CodingTask("train_subarray", "Input [numbers,target]. Count all nonempty contiguous subarrays whose sum equals target. Numbers may be negative or zero.", (([[1,1,1],2],2), ([[0,0,0],0],6), ([[1,-1,0],0],3), ([[],0],0))),
)

HOLDOUT_TASKS = (
    CodingTask("holdout_rotate", "Input [items,k]. Rotate the list right by k positions; negative k rotates left. Empty input list stays empty.", (([[1,2,3,4],1],[4,1,2,3]), ([[1,2,3],-1],[2,3,1]), ([[],7],[]), ([[1,2],5],[2,1]))),
    CodingTask("holdout_runs", "Run-length encode the input string as a list of [character,count] for consecutive runs. Empty input returns [].", (("aaabbcca",[["a",3],["b",2],["c",2],["a",1]]),("",[]),("🙂🙂x",[["🙂",2],["x",1]]))),
    CodingTask("holdout_coins", "Input [coins,amount]. Return the minimum number of coins needed to make amount using unlimited coins of each positive denomination. Return -1 if impossible and 0 for amount zero.", (([[1,3,4],6],2), ([[2],3],-1), ([[7],0],0), ([[2,5],11],4))),
    CodingTask("holdout_next", "For each number in a list, return the next strictly greater number to its right, or -1 if none. Return a list of equal length.", (([2,1,2,4,3],[4,2,4,-1,-1]), ([2,2],[-1,-1]), ([],[]))),
    CodingTask("holdout_product", "Return a list where position i is the product of all input integers except input[i]. Do not use division. An empty input returns []; singleton returns [1].", (([1,2,3,4],[24,12,8,6]), ([0,2,0],[0,0,0]), ([0,2,3],[6,0,0]), ([7],[1]))),
    CodingTask("holdout_spiral", "Flatten a rectangular matrix into clockwise spiral order starting at its top-left corner. An empty matrix or rows of length zero returns [].", (([[1,2,3],[4,5,6],[7,8,9]],[1,2,3,6,9,8,7,4,5]), ([[1,2,3]],[1,2,3]), ([[],[]],[]), ([],[]))),
)


def extract_code(response: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", response, re.S)
    code = (blocks[-1] if blocks else response).strip()
    if not code or len(code.encode()) > 20000:
        raise ValueError("empty or oversized code")
    tree = ast.parse(code)
    if not any(isinstance(x, ast.FunctionDef) and x.name == "solve" for x in tree.body):
        raise ValueError("missing solve function")
    banned = {"exec", "eval", "open", "compile", "globals", "locals", "vars", "getattr", "setattr", "delattr", "breakpoint", "input", "help", "exit", "quit"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in banned):
            raise ValueError("forbidden name")
        if isinstance(node, ast.alias) and any(part.startswith("_") for part in node.name.split(".")):
            raise ValueError("private import names forbidden")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("private attributes forbidden")
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef)):
            raise ValueError("unsupported code construct")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module]
            if any(m not in {"math", "collections", "heapq", "bisect", "itertools", "functools", "csv", "decimal", "fractions", "ipaddress", "json", "posixpath", "re", "shlex", "io"} for m in modules):
                raise ValueError("import not allowlisted")
    return code


# The candidate sees inputs only; gold comparisons run in the trusted parent.
HARNESS = '''import json,sys,io,contextlib
obj=json.load(sys.stdin)
ns={}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(obj["code"], "candidate.py", "exec"), ns)
    results=[]
    for arg in obj["inputs"]:
        try:
            actual=ns["solve"](arg)
            encoded=json.dumps(actual,allow_nan=False)
            if len(encoded)>12000:
                raise ValueError("result too large")
            results.append({"ok":True,"value":actual})
        except BaseException:
            results.append({"ok":False})
print(json.dumps({"results":results},allow_nan=False))
'''



def grade(task: CodingTask, response: str, image: str, timeout_s: int = 12) -> dict:
    if "@sha256:" not in image:
        raise ValueError("evaluator image must be pinned by digest")
    try:
        code = extract_code(response)
    except (ValueError, SyntaxError) as exc:
        return {"task_id": task.task_id, "passed": 0, "total": len(task.cases), "score": -1.0, "error": str(exc)}
    name = "dream-eval-" + uuid.uuid4().hex
    cmd = ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
           "--memory", "256m", "--memory-swap", "256m", "--cpus", "1", "--pids-limit", "32",
           "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "65534:65534",
           "-i", image, "python", "-I", "-S", "-c", HARNESS]
    try:
        result = subprocess.run(cmd, input=json.dumps({"code": code, "inputs": [arg for arg,_ in task.cases]}),
                                text=True, capture_output=True, timeout=timeout_s)
        if result.returncode != 0 or len(result.stdout) > 10000:
            raise ValueError("candidate execution failed")
        values = json.loads(result.stdout)["results"]
        if not isinstance(values,list) or len(values)!=len(task.cases):
            raise ValueError("malformed evaluator result")
        passed=0
        for value,(_,expected) in zip(values,task.cases):
            if not isinstance(value,dict) or type(value.get("ok")) is not bool:
                raise ValueError("malformed evaluator result")
            # Gold answers remain in the trusted parent, never the candidate container.
            actual=value.get("value")
            passed+=value["ok"] and type(actual) is type(expected) and actual==expected
        return {"task_id": task.task_id, "passed": passed, "total": len(values),
                "score": 2.0 * passed / len(values) - 1.0}
    except (subprocess.TimeoutExpired, ValueError, KeyError) as exc:
        return {"task_id": task.task_id, "passed": 0, "total": len(task.cases), "score": -1.0,
                "error": type(exc).__name__}
    finally:
        subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=8)
