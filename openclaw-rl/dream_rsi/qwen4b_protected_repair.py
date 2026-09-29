"""Public minimal repairs and first-order protection of correct replay losses.

Projection is a local approximation, not a guarantee that generated programs
stay correct. Immutable development/guard tests remain the actual acceptance
criterion. This module never reads final confirmation cases.
"""
from __future__ import annotations
import math


def minimal_repairs(tasks, records):
    """Repair the two known parent defects without rewriting unrelated answers.

    The literal edit is guarded against unexpected responses. The caller must
    independently execute every repaired program on all public cases before
    using it as a target. This is explicit known-task supervised adaptation.
    """
    by_id = {row['task_id']: row for row in records}
    if len(by_id) != len(records):
        raise ValueError('duplicate development output')
    out = []
    for task in tasks:
        record = by_id[task.task_id]
        result = record['result']
        if result['passed'] == result['total']:
            continue
        old = record['response']
        if task.task_id.endswith('_range'):
            before, after = '    k = data[-1]\n    nums = data[:-1]\n', '    nums, k = data\n'
        elif task.task_id.endswith('_sum'):
            before, after = 'sum(data[i::2]) for i in range(2)', 'sum(data) - x for x in data'
        else:
            raise ValueError('unexpected failed task; require a new explicit repair')
        if old.count(before) != 1:
            raise ValueError('expected defective span not present exactly once')
        out.append({'task_id': task.task_id, 'description': task.description,
                    'solution': old.replace(before, after), 'parent_response': old,
                    'target_source': 'verified_minimal_reference_repair', 'weight': 1.0})
    if len(out) != 2:
        raise ValueError('expected exactly two failed public families')
    return out


def projected_delta(delta, anchors, *, rcond=1e-7):
    """Remove the proposed displacement along measured anchor-loss gradients.

    A is the row-normalized anchor Jacobian; delta_perp = delta - A^+ A delta.
    The tiny Gram eigensolve runs in float64; vectors remain float32. Correlated
    rows are handled by the documented relative eigenvalue cutoff.
    """
    import torch
    if delta.ndim != 1 or anchors.ndim != 2 or anchors.shape[1] != delta.numel():
        raise ValueError('incompatible projected-update dimensions')
    if not math.isfinite(rcond) or not 0 < rcond < 1:
        raise ValueError('invalid projection cutoff')
    if not torch.isfinite(delta).all() or not torch.isfinite(anchors).all():
        raise ValueError('nonfinite projection input')
    if anchors.shape[0] == 0:
        return delta.clone(), {'rank': 0, 'retained_norm_ratio': 1.0}
    norms = anchors.norm(dim=1)
    valid = norms > 1e-12
    if not valid.any():
        return delta.clone(), {'rank': 0, 'retained_norm_ratio': 1.0}
    a = anchors[valid] / norms[valid, None]
    gram = (a @ a.T).double()
    values, vectors = torch.linalg.eigh(gram)
    keep = values > values.max() * rcond
    inv = torch.where(keep, values.clamp_min(1e-30).reciprocal(), torch.zeros_like(values))
    rhs = (a @ delta).double()
    coeff = vectors @ (inv * (vectors.T @ rhs))
    protected = delta - a.T @ coeff.to(a.dtype)
    if not torch.isfinite(protected).all():
        raise ValueError('nonfinite projected delta')
    original_norm = float(delta.norm())
    metrics = {'rank': int(keep.sum()), 'anchor_count': int(a.shape[0]),
               'raw_update_norm': original_norm, 'projected_update_norm': float(protected.norm()),
               'retained_norm_ratio': float(protected.norm()) / max(original_norm, 1e-30),
               'max_anchor_dot_before': float((a @ delta).abs().max()),
               'max_anchor_dot_after': float((a @ protected).abs().max()), 'rcond': rcond}
    return protected, metrics
