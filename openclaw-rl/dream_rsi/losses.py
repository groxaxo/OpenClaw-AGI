"""Response-masked clipped policy loss with KL against a frozen reference.

The caller supplies advantages. Raw signed rewards implement a zero-baseline
policy gradient; they must not be described as group-normalized GRPO.
"""
import torch


def clipped_policy_loss(current, old, reference, advantages, mask, *, clip=0.2, kl_coef=0.02):
    if current.shape != old.shape or current.shape != reference.shape or current.shape != mask.shape:
        raise ValueError("unaligned response log probabilities")
    if advantages.shape != current.shape[:1] or not 0 < clip < 1 or kl_coef < 0:
        raise ValueError("invalid advantage shape or coefficients")
    if not all(torch.isfinite(x).all() for x in (current, old, reference, advantages, mask)):
        raise ValueError("non-finite loss input")
    if torch.any(mask < 0) or mask.sum() <= 0:
        raise ValueError("empty or negative loss mask")
    delta = current - old.detach()
    ref_delta = reference.detach() - current
    if delta.abs().max() > 30 or ref_delta.abs().max() > 30:
        raise ValueError("log-ratio exceeds conservative stability bound")
    ratio = delta.exp()
    adv = advantages[:, None]
    pg = -torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
    kl = ref_delta.exp() - 1 - ref_delta
    denominator = mask.sum()
    loss = ((pg + kl_coef * kl) * mask).sum() / denominator
    return loss, {"loss": float(loss.detach()), "policy_loss": float((pg * mask).sum().detach() / denominator),
                  "reference_kl": float((kl * mask).sum().detach() / denominator)}
