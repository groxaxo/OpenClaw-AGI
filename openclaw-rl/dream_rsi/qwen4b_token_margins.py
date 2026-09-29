"""Public-only token-level anchors for a measured continuation experiment.

A contrast is the chosen token logit minus a competitor logit under the same
prefix. Protecting its gradient is still a local linear approximation, not a
promise of unchanged generation. Executable guard/DEV tests remain decisive.
"""
from __future__ import annotations
import math


def first_divergence(correct: list[int], rejected: list[int]):
    """Return first (position, correct token, rejected token), or no contrast."""
    for seq in (correct,rejected):
        if not isinstance(seq,list) or any(type(x) is not int or x<0 for x in seq):
            raise ValueError('token IDs must be nonnegative integers')
    for position,(chosen,other) in enumerate(zip(correct,rejected)):
        if chosen!=other:
            return position,chosen,other
    return None


def choose_margins(logits, targets, *, count=3, counterexamples=()):
    """Choose weak top-competitor contrasts plus first-divergence contrasts.

    Counterexamples must have exactly the same prefix before their first
    difference. No final-suite text or results are accepted here.
    """
    import torch
    if type(count) is not int or not 1<=count<=8:
        raise ValueError('margin count must be 1..8')
    if logits.ndim!=2 or targets.ndim!=1 or logits.shape[0]!=targets.numel() or logits.shape[1]<2:
        raise ValueError('unaligned token logits')
    if targets.dtype not in (torch.int32,torch.int64) or targets.numel()==0:
        raise ValueError('empty or noninteger targets')
    if not torch.isfinite(logits).all() or targets.min()<0 or targets.max()>=logits.shape[1]:
        raise ValueError('nonfinite logits or invalid vocabulary index')
    top=logits.topk(2,dim=1)
    competitors=torch.where(top.indices[:,0]==targets,top.indices[:,1],top.indices[:,0])
    chosen_logits=logits.gather(1,targets[:,None]).squeeze(1).float()
    margins=chosen_logits-logits.gather(1,competitors[:,None]).squeeze(1).float()
    positions=torch.argsort(margins,stable=True)[:min(count,targets.numel())].tolist()
    selected={}
    def add(pos,other,source):
        key=(pos,other)
        if key in selected:
            selected[key]['sources'].append(source);return
        chosen=int(targets[pos]);gap=float(logits[pos,chosen].float()-logits[pos,other].float())
        selected[key]={'position':pos,'chosen_token':chosen,'competitor_token':other,
                       'teacher_forced_margin':gap,'sources':[source]}
    for pos in positions:add(pos,int(competitors[pos]),'lowest_current_margin')
    correct=targets.tolist()
    for rejected in counterexamples:
        contrast=first_divergence(correct,rejected)
        if contrast is None:continue
        pos,chosen,other=contrast
        if other>=logits.shape[1]:raise ValueError('counterexample vocabulary mismatch')
        add(pos,other,'prior_public_regression')
    return list(selected.values())
