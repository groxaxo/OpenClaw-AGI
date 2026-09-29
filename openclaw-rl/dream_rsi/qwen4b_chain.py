"""Checkpoint lineage and preregistration helpers for incremental promotions.

A successful generation must improve over its immediate accepted parent. The
original base is not a substitute comparator for subsequent generations.
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from .coding_tasks import CodingTask
from .external_judge import canonical
from .qwen4b_harder import build_hard_suite


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_parent(path):
    record = json.loads(Path(path).read_text())
    if record.get('status') != 'PASS' or record.get('cold_reproduction_passed') is not True:
        raise ValueError('parent must have passed promotion and cold reproduction')
    if record.get('model') != 'Qwen/Qwen3.5-4B':
        raise ValueError('wrong parent model')
    for name in ('adapter_file_sha256', 'adapter_tensor_state_sha256'):
        if not re.fullmatch(r'[0-9a-f]{64}', str(record.get(name, ''))):
            raise ValueError('invalid parent digest')
    folder = Path(record['adapter_path']).resolve()
    if file_sha256(folder/'adapter_model.safetensors') != record['adapter_file_sha256']:
        raise ValueError('parent adapter bytes do not match acceptance record')
    config = json.loads((folder/'adapter_config.json').read_text())
    expected = {'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}
    if (config.get('r') != 8 or config.get('lora_alpha') != 16
            or set(config.get('target_modules', ())) != expected
            or config.get('peft_type') != 'LORA'):
        raise ValueError('parent adapter configuration is incompatible')
    generation = record.get('successful_generation', 1)
    if type(generation) is not int or generation < 1:
        raise ValueError('invalid parent generation')
    return {**record, 'successful_generation': generation,
            'approval_file_sha256': file_sha256(path)}


def expand_public_development(development, retired_rows):
    """Prior published confirmation is now DEVELOPMENT, not an unseen test.

    Merge by exact prompt and deduplicate identical inputs. Conflicting oracle
    answers fail validation instead of allowing input-count manipulation.
    """
    by_description = {row['description']: row for row in retired_rows}
    if len(by_description) != len(retired_rows):
        raise ValueError('duplicate retired prompt')
    result = []
    for task in development:
        row = by_description.pop(task.description, None)
        if row is None:
            raise ValueError('retired parent prompt missing from development')
        cases = {}
        for arg, expected in tuple(task.cases) + tuple((a,b) for a,b in row['cases']):
            key = canonical(arg)
            if key in cases and canonical(cases[key][1]) != canonical(expected):
                raise ValueError('conflicting public oracle results')
            cases[key] = (arg, expected)
        result.append(CodingTask(task.task_id, task.description, tuple(cases.values())))
    if by_description:
        raise ValueError('retired prompts were silently omitted')
    return tuple(result)


def build_fresh_confirmation(seed, training, development, count=20):
    """Draw new case inputs without consulting any model outcomes.

    Exclude all exact public inputs globally, including known aliases, and
    deduplicate each family. Same known-prompt algorithm distribution; not a
    claim of unseen-task generalization.
    """
    if type(seed) is not int or type(count) is not int or not 1 <= count <= 100:
        raise ValueError('invalid fresh suite configuration')
    public_inputs = {canonical(arg) for task in tuple(training)+tuple(development)
                     for arg, _ in task.cases}
    selected = None
    seen = {}
    for batch in range(100):
        rows = build_hard_suite(seed+104729*batch)
        if selected is None:
            selected = [{'task_id':r['task_id'],'description':r['description'],'cases':[]} for r in rows]
            seen = {r['task_id']:set() for r in rows}
        for target, row in zip(selected, rows):
            if target['task_id'] != row['task_id']:
                raise ValueError('generator family changed')
            for arg, answer in row['cases']:
                key = canonical(arg)
                if len(target['cases']) < count and key not in public_inputs and key not in seen[row['task_id']]:
                    target['cases'].append([arg, answer])
                    seen[row['task_id']].add(key)
        if all(len(r['cases']) == count for r in selected):
            return selected
    raise ValueError('insufficient unseen inputs; no silent reduction in test size')


def lineage_summary(root_hash, attempts):
    """Count verified promotions, not optimizer steps or repeated base wins."""
    tip = root_hash
    successful = 1
    rejected = 0
    for attempt in attempts:
        if attempt['parent_sha256'] != tip:
            raise ValueError('attempt is not based on current accepted tip')
        if attempt['status'] == 'PASS':
            if (attempt.get('promotion_approved') is not True
                    or attempt.get('final_external_review_ok') is not True
                    or attempt.get('cold_reproduction_passed') is not True
                    or attempt.get('gate', {}).get('passed') is not True):
                raise ValueError('incomplete promotion evidence cannot advance lineage')
            candidate = attempt['candidate_sha256']
            if candidate == tip:
                raise ValueError('unchanged checkpoint is not an improvement')
            tip = candidate
            successful += 1
        elif attempt['status'] in ('NOT_OK','INTERRUPTED'):
            rejected += 1
        else:
            raise ValueError('pending attempt is not a completed result')
    return {'successful_promotions':successful, 'continuation_attempts':len(attempts),
            'rejected_or_interrupted_continuations':rejected, 'accepted_tip_sha256':tip}


def development_ready(parent, candidate):
    """Early stopping must meet the same gain/family requirements as promotion.

    These are public development counts, not final-suite results. Reaching
    this condition only permits moving to the independently locked final test.
    """
    for row in (parent,candidate):
        for key in ('passed_cases','total_cases','complete_tasks','task_count'):
            if type(row.get(key)) is not int:
                raise ValueError('invalid development count')
        if not 0<=row['passed_cases']<=row['total_cases'] or row['total_cases']<=0:
            raise ValueError('invalid development case counts')
        if not 0<=row['complete_tasks']<=row['task_count'] or row['task_count']<=0:
            raise ValueError('invalid development task counts')
    if parent['total_cases']!=candidate['total_cases'] or parent['task_count']!=candidate['task_count']:
        raise ValueError('development distribution changed')
    total=parent['total_cases']
    return ((candidate['passed_cases']-parent['passed_cases'])/total>=.08
            and candidate['passed_cases']/total>=.75
            and candidate['complete_tasks']-parent['complete_tasks']>=2)
