from __future__ import annotations

import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class PolicyConfig:
    """Bounded, replayable controller policy.

    This independent curriculum experiment evolves numeric parameters, not
    controller source. It is not a reference Dream-RSI implementation.
    """

    version: int = 1
    beta: float = 0.60
    exploit_fraction: float = 0.55
    explore_fraction: float = 0.25
    recover_fraction: float = 0.20
    score_weight: float = 1.00
    novelty_weight: float = 0.35
    recovery_weight: float = 0.45
    delta_weight: float = 0.30
    depth_penalty: float = 0.05

    def validated(self) -> "PolicyConfig":
        bounded = {
            "beta": (0.0, 1.0),
            "exploit_fraction": (0.0, 1.0),
            "explore_fraction": (0.0, 1.0),
            "recover_fraction": (0.0, 1.0),
            "score_weight": (0.0, 4.0),
            "novelty_weight": (0.0, 4.0),
            "recovery_weight": (0.0, 4.0),
            "delta_weight": (0.0, 4.0),
            "depth_penalty": (0.0, 1.0),
        }
        if type(self.version) is not int or self.version < 1:
            raise ValueError("version must be a positive integer")
        values = asdict(self)
        for key, (lo, hi) in bounded.items():
            if type(values[key]) not in (int, float) or not math.isfinite(values[key]):
                raise ValueError(f"{key} must be a finite number")
            value = float(values[key])
            if not lo <= value <= hi:
                raise ValueError(f"{key}={value} outside [{lo}, {hi}]")
        total = self.exploit_fraction + self.explore_fraction + self.recover_fraction
        if not 0.999 <= total <= 1.001:
            raise ValueError("exploit/explore/recover fractions must sum to 1")
        return self

    @classmethod
    def from_dict(cls, data: dict) -> "PolicyConfig":
        if not isinstance(data, dict):
            raise ValueError("policy must be an object")
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown policy fields: {sorted(unknown)}")
        return cls(**data).validated()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Candidate:
    node_id: str
    session_id: str
    turn: int
    score: float
    valid: bool = True
    fail_class: str | None = None
    parent_id: str | None = None
    delta_vs_parent: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("score", "delta_vs_parent"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("node_id must be a nonempty string")
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must be a nonempty string")
        if type(self.turn) is not int or self.turn < 0:
            raise ValueError("turn must be a nonnegative integer")
        if type(self.valid) is not bool:
            raise ValueError("valid must be boolean")


@dataclass(frozen=True)
class ReplayMetrics:
    objective: float
    best_score: float
    mean_score: float
    mean_abs_score: float
    selected: int
    rounds: int
    distinct_sessions: int
    positive_rate: float


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    mean_delta: float
    wins: int
    losses: int
    ties: int
    one_sided_sign_p: float
    reason: str


def _quota(n: int, policy: PolicyConfig) -> tuple[int, int, int]:
    exploit = int(round(n * policy.exploit_fraction))
    explore = int(round(n * policy.explore_fraction))
    recover = n - exploit - explore
    if recover < 0:
        recover = 0
        exploit = max(0, n - explore)
    return exploit, explore, recover


def _candidate_features(candidates: Sequence[Candidate]) -> dict[str, dict[str, float]]:
    by_session: dict[str, list[Candidate]] = {}
    for c in sorted(candidates, key=lambda x: (x.session_id, x.turn, x.node_id)):
        by_session.setdefault(c.session_id, []).append(c)

    out: dict[str, dict[str, float]] = {}
    for items in by_session.values():
        seen = 0
        last = 0.0
        for c in items:
            delta = c.delta_vs_parent if c.parent_id is not None else (c.score - last if seen else 0.0)
            novelty = 1.0 / math.sqrt(1.0 + seen)
            recovery = 1.0 if seen and last < 0.0 <= c.score else 0.0
            out[c.node_id] = {
                "novelty": novelty,
                "delta": delta,
                "recovery": recovery,
                "depth": max(0.0, float(c.turn - 1)),
            }
            seen += 1
            last = c.score
    return out


def select_candidates(candidates: Sequence[Candidate], target: int, policy: PolicyConfig) -> list[Candidate]:
    """Deterministic exploit/explore/recover selection on frozen observations."""

    policy = policy.validated()
    if type(target) is not int or target < 0:
        raise ValueError("target must be a nonnegative integer")
    if len({c.node_id for c in candidates}) != len(candidates):
        raise ValueError("candidate node IDs must be unique within a pool")
    if target == 0:
        return []
    usable = [c for c in candidates if c.valid]
    if len(usable) <= target:
        return list(usable)

    features = _candidate_features(usable)

    def exploit_key(c: Candidate):
        f = features[c.node_id]
        utility = (
            policy.score_weight * (0.80 * abs(c.score) + 0.20 * c.score)
            + policy.delta_weight * f["delta"]
            - policy.depth_penalty * f["depth"] * (1.0 - 0.5 * policy.beta)
        )
        return (utility, c.score, -c.turn, c.node_id)

    def explore_key(c: Candidate):
        f = features[c.node_id]
        utility = (
            policy.novelty_weight * f["novelty"] * (0.5 + policy.beta)
            + 0.20 * abs(c.score)
            - 0.5 * policy.depth_penalty * f["depth"]
        )
        return (utility, -c.turn, c.node_id)

    def recover_key(c: Candidate):
        f = features[c.node_id]
        utility = (
            policy.recovery_weight * f["recovery"]
            + policy.delta_weight * max(0.0, f["delta"])
            + 0.15 * abs(c.score)
            + 0.10 * policy.beta * f["novelty"]
        )
        return (utility, f["recovery"], f["delta"], c.node_id)

    n_exploit, n_explore, n_recover = _quota(target, policy)
    buckets = [
        (sorted(usable, key=exploit_key, reverse=True), n_exploit),
        (sorted(usable, key=explore_key, reverse=True), n_explore),
        (sorted(usable, key=recover_key, reverse=True), n_recover),
    ]

    chosen: list[Candidate] = []
    chosen_ids: set[str] = set()
    for bucket, want in buckets:
        for c in bucket:
            if len(chosen) >= target or want <= 0:
                break
            if c.node_id in chosen_ids:
                continue
            chosen.append(c)
            chosen_ids.add(c.node_id)
            want -= 1

    if len(chosen) < target:
        for c in sorted(usable, key=exploit_key, reverse=True):
            if c.node_id not in chosen_ids:
                chosen.append(c)
                chosen_ids.add(c.node_id)
                if len(chosen) >= target:
                    break
    return chosen[:target]


def objective(
    selected: Sequence[Candidate],
    rounds: int = 1,
    call_penalty: float = 0.01,
    parallel_bonus: float = 0.01,
) -> ReplayMetrics:
    if not selected:
        # Empty pools carry no evidence. Keep serialized metrics finite;
        # evolution excludes pools with fewer than the target valid samples.
        return ReplayMetrics(0.0, 0.0, 0.0, 0.0, 0, max(1, rounds), 0, 0.0)
    scores = [float(c.score) for c in selected]
    best = max(scores)
    mean = statistics.fmean(scores)
    mean_abs = statistics.fmean(abs(s) for s in scores)
    n = len(selected)
    k = max(1, int(rounds))
    distinct = len({c.session_id for c in selected})
    positive = sum(1 for s in scores if s > 0) / n
    diversity = distinct / n
    # With OpenClaw's {-1, 0, +1} PRM, max(score) saturates quickly.  Keep
    # Dream-RSI's best/compute/parallel structure, but add training-signal
    # informativeness and a small diversity term so replay remains meaningful.
    value = (
        best
        + 0.25 * mean_abs
        + 0.05 * mean
        - call_penalty * n
        + parallel_bonus * (n / k)
        + 0.05 * diversity
    )
    return ReplayMetrics(value, best, mean, mean_abs, n, k, distinct, positive)


def replay_policy(
    pools: Sequence[Sequence[Candidate]],
    policy: PolicyConfig,
    target: int,
    call_penalty: float = 0.01,
    parallel_bonus: float = 0.01,
) -> ReplayMetrics:
    chosen: list[Candidate] = []
    for pool in pools:
        chosen.extend(select_candidates(pool, target, policy))
    return objective(chosen, rounds=max(1, len(pools)), call_penalty=call_penalty, parallel_bonus=parallel_bonus)


def mutate_policy(policy: PolicyConfig, seed: int, count: int = 24) -> list[PolicyConfig]:
    """Deterministically generate bounded challengers around an incumbent."""

    policy.validated()
    if type(count) is not int or count < 1:
        raise ValueError("mutation count must be a positive integer")
    rng = random.Random(seed)
    out: list[PolicyConfig] = []
    seen: set[str] = set()
    for _ in range(max(1, count * 4)):
        if len(out) >= count:
            break
        beta = min(1.0, max(0.0, policy.beta + rng.uniform(-0.20, 0.20)))
        raw = [
            max(0.03, policy.exploit_fraction + rng.uniform(-0.16, 0.16)),
            max(0.03, policy.explore_fraction + rng.uniform(-0.12, 0.12)),
            max(0.03, policy.recover_fraction + rng.uniform(-0.12, 0.12)),
        ]
        total = sum(raw)
        fractions = [x / total for x in raw]
        candidate = replace(
            policy,
            version=policy.version + 1,
            beta=round(beta, 6),
            exploit_fraction=round(fractions[0], 6),
            explore_fraction=round(fractions[1], 6),
            recover_fraction=round(1.0 - fractions[0] - fractions[1], 6),
            score_weight=round(min(4.0, max(0.0, policy.score_weight * rng.uniform(0.80, 1.20))), 6),
            novelty_weight=round(min(4.0, max(0.0, policy.novelty_weight * rng.uniform(0.70, 1.35))), 6),
            recovery_weight=round(min(4.0, max(0.0, policy.recovery_weight * rng.uniform(0.70, 1.35))), 6),
            delta_weight=round(min(4.0, max(0.0, policy.delta_weight * rng.uniform(0.70, 1.35))), 6),
            depth_penalty=round(min(1.0, max(0.0, policy.depth_penalty * rng.uniform(0.60, 1.40))), 6),
        )
        key = json.dumps(candidate.to_dict(), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate.validated())
    return out


def _sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0 or wins <= losses:
        return 1.0
    return min(1.0, sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n))


def holdout_gate(
    incumbent_values: Sequence[float],
    challenger_values: Sequence[float],
    *,
    min_pairs: int = 8,
    min_mean_delta: float = 0.0,
    alpha: float = 0.10,
) -> PromotionDecision:
    if type(min_pairs) is not int or min_pairs < 1:
        raise ValueError("min_pairs must be a positive integer")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    if not math.isfinite(min_mean_delta) or min_mean_delta < 0.0:
        raise ValueError("min_mean_delta must be finite and nonnegative")
    if len(incumbent_values) != len(challenger_values):
        raise ValueError("incumbent/challenger holdout vectors must have equal length")
    if any(not math.isfinite(float(v)) for v in (*incumbent_values, *challenger_values)):
        raise ValueError("holdout values must be finite")
    if len(incumbent_values) < min_pairs:
        return PromotionDecision(False, 0.0, 0, 0, 0, 1.0, f"need at least {min_pairs} paired holdout episodes")
    deltas = [float(c) - float(i) for i, c in zip(incumbent_values, challenger_values)]
    if any(not math.isfinite(d) for d in deltas):
        raise ValueError("holdout differences must be finite")
    wins = sum(d > 1e-12 for d in deltas)
    losses = sum(d < -1e-12 for d in deltas)
    ties = len(deltas) - wins - losses
    mean_delta = statistics.fmean(deltas)
    p = _sign_test_p(wins, losses)
    promote = mean_delta > min_mean_delta and wins > losses and p <= alpha
    if promote:
        reason = "challenger passed paired holdout gate"
    elif mean_delta <= min_mean_delta:
        reason = "mean holdout improvement below threshold"
    elif wins <= losses:
        reason = "challenger did not win a majority of non-tied holdout episodes"
    else:
        reason = f"one-sided sign-test p={p:.4f} exceeds alpha={alpha:.4f}"
    return PromotionDecision(promote, mean_delta, wins, losses, ties, p, reason)


class TraceStore:
    """Append-only compact trace for replay. No prompts/responses are stored."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append_pool(self, rollout_id: int, candidates: Sequence[Candidate]) -> None:
        record = {
            "type": "candidate_pool",
            "timestamp": time.time(),
            "rollout_id": int(rollout_id),
            "candidates": [
                {
                    "node_id": c.node_id,
                    "session_id": c.session_id,
                    "turn": c.turn,
                    "score": c.score,
                    "valid": c.valid,
                    "fail_class": c.fail_class,
                    "parent_id": c.parent_id,
                    "delta_vs_parent": c.delta_vs_parent,
                    "metadata": {"has_next_state": bool(c.metadata.get("has_next_state", False))},
                }
                for c in candidates
            ],
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")

    def append_event(self, event: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")

    def load_pools(self) -> list[list[Candidate]]:
        if not self.path.exists():
            return []
        pools: list[list[Candidate]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("type") != "candidate_pool":
                    continue
                pools.append([Candidate(**item) for item in rec.get("candidates", [])])
        return pools


class DreamRSIController:
    """Frozen-replay controller for OpenClaw-RL curriculum selection."""

    def __init__(
        self,
        state_dir: str | os.PathLike[str],
        *,
        target_batch_size: int,
        pool_factor: float = 2.0,
        evolve_interval: int = 8,
        mutation_count: int = 24,
        holdout_pools: int = 4,
        min_holdout_pairs: int = 4,
        seed: int = 42,
        call_penalty: float = 0.01,
        parallel_bonus: float = 0.01,
        promotion_judge=None,
    ):
        for name, value in {
            "target_batch_size": target_batch_size,
            "evolve_interval": evolve_interval,
            "mutation_count": mutation_count,
            "holdout_pools": holdout_pools,
            "min_holdout_pairs": min_holdout_pairs,
        }.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(pool_factor) or pool_factor < 1.0:
            raise ValueError("pool_factor must be finite and >= 1.0")
        if min_holdout_pairs > holdout_pools:
            raise ValueError("min_holdout_pairs cannot exceed holdout_pools")
        if any(not math.isfinite(v) or v < 0 for v in (call_penalty, parallel_bonus)):
            raise ValueError("objective coefficients must be finite and nonnegative")
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.policy_path = self.state_dir / "policy.json"
        self.history_path = self.state_dir / "policy_history.jsonl"
        self.evaluation_state_path = self.state_dir / "evaluation_state.json"
        self.trace = TraceStore(self.state_dir / "replay.jsonl")
        self.target_batch_size = int(target_batch_size)
        self.pool_factor = float(pool_factor)
        self.evolve_interval = max(1, int(evolve_interval))
        self.mutation_count = max(1, int(mutation_count))
        self.holdout_pools = max(1, int(holdout_pools))
        self.min_holdout_pairs = max(1, int(min_holdout_pairs))
        self.seed = int(seed)
        self.promotion_judge = promotion_judge
        self.call_penalty = float(call_penalty)
        self.parallel_bonus = float(parallel_bonus)
        self.policy = self._load_policy()
        state = (json.loads(self.evaluation_state_path.read_text(encoding="utf-8"))
                 if self.evaluation_state_path.exists() else {})
        self._last_evolution_pool_count = state.get("consumed_pool_count", 0)
        if type(self._last_evolution_pool_count) is not int or self._last_evolution_pool_count < 0:
            raise ValueError("invalid persisted replay cursor")

    @property
    def pool_target(self) -> int:
        return max(self.target_batch_size, int(math.ceil(self.target_batch_size * self.pool_factor)))

    def _load_policy(self) -> PolicyConfig:
        if self.policy_path.exists():
            return PolicyConfig.from_dict(json.loads(self.policy_path.read_text(encoding="utf-8")))
        policy = PolicyConfig().validated()
        self._write_policy(policy)
        return policy

    def _write_policy(self, policy: PolicyConfig) -> None:
        tmp = self.policy_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(policy.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self.policy_path)

    @staticmethod
    def _sample_to_candidate(sample) -> Candidate:
        md = dict(getattr(sample, "metadata", {}) or {})
        session_id = str(md.get("session_id", "unknown"))
        turn = int(md.get("turn", 0) or 0)
        reward = getattr(sample, "reward", 0.0)
        score = float(reward.get("score", 0.0) if isinstance(reward, dict) else reward or 0.0)
        node_id = str(md.get("dream_node_id") or f"{session_id}:{turn}:{getattr(sample, 'index', 'na')}")
        parent_id = md.get("dream_parent_id")
        status = getattr(sample, "status", "completed")
        status = getattr(status, "value", status)
        mask = getattr(sample, "loss_mask", None)
        trainable = mask is None or any(value > 0 for value in mask)
        valid = (not bool(getattr(sample, "remove_sample", False))
                 and status in ("completed", "truncated") and trainable)
        return Candidate(
            node_id=node_id,
            session_id=session_id,
            turn=turn,
            score=score,
            valid=valid,
            fail_class=md.get("fail_class"),
            parent_id=str(parent_id) if parent_id else None,
            delta_vs_parent=float(md.get("delta_vs_parent", 0.0) or 0.0),
            metadata={"has_next_state": bool(md.get("has_next_state", False))},
        )

    def select_samples(self, samples: Sequence, rollout_id: int):
        candidates = [self._sample_to_candidate(s) for s in samples]
        selected = select_candidates(candidates, self.target_batch_size, self.policy)
        self.trace.append_pool(rollout_id, candidates)
        by_id = {candidate.node_id: sample for sample, candidate in zip(samples, candidates)}
        # Never pad a short valid batch with excluded or zero-mask samples.
        return [by_id[c.node_id] for c in selected]

    def maybe_evolve(self, rollout_id: int) -> dict | None:
        if (rollout_id + 1) % self.evolve_interval != 0:
            return None
        pools = self.trace.load_pools()
        if len(pools) < self.holdout_pools + 2:
            return None

        if len(pools) < self._last_evolution_pool_count:
            raise ValueError("replay trace was truncated behind its evaluation cursor")
        if len(pools) - self._last_evolution_pool_count < self.holdout_pools:
            return None  # Do not re-use holdout pools, including after restart.
        holdout = pools[-self.holdout_pools:]
        if any(sum(c.valid for c in pool) < self.target_batch_size for pool in holdout):
            return None
        heldout_sessions = {c.session_id for pool in holdout for c in pool}
        train_pools = [
            [c for c in pool if c.session_id not in heldout_sessions]
            for pool in pools[:-self.holdout_pools]
        ]
        train_pools = [pool for pool in train_pools
                       if sum(c.valid for c in pool) >= self.target_batch_size]
        if len(train_pools) < 2:
            return None
        # Each statistical pair must represent disjoint recorded sessions.
        seen_sessions: set[str] = set()
        for pool in holdout:
            sessions = {c.session_id for c in pool}
            if seen_sessions & sessions:
                return None
            seen_sessions.update(sessions)
        # Claim this evaluation window before scoring. A failure consumes the
        # window conservatively instead of allowing repeated attempts on it.
        state_tmp = self.evaluation_state_path.with_suffix(".json.tmp")
        state_tmp.write_text(json.dumps({"consumed_pool_count": len(pools)}) + "\n", encoding="utf-8")
        os.replace(state_tmp, self.evaluation_state_path)
        self._last_evolution_pool_count = len(pools)
        incumbent = self.policy
        incumbent_train = replay_policy(
            train_pools, incumbent, self.target_batch_size, self.call_penalty, self.parallel_bonus
        )
        challengers = mutate_policy(incumbent, self.seed + rollout_id, self.mutation_count)
        scored = [
            (
                replay_policy(train_pools, p, self.target_batch_size, self.call_penalty, self.parallel_bonus),
                p,
            )
            for p in challengers
        ]
        scored.sort(key=lambda item: (item[0].objective, item[0].mean_score), reverse=True)
        challenger_metrics, challenger = scored[0]

        if challenger_metrics.objective <= incumbent_train.objective:
            event = {
                "type": "policy_evolution",
                "rollout_id": rollout_id,
                "promoted": False,
                "reason": "no replay challenger beat incumbent",
                "incumbent": incumbent.to_dict(),
                "incumbent_replay": asdict(incumbent_train),
                "best_challenger": challenger.to_dict(),
                "best_challenger_replay": asdict(challenger_metrics),
            }
            self.trace.append_event(event)
            self._append_history(event)
            return event

        incumbent_holdout = [
            replay_policy([pool], incumbent, self.target_batch_size, self.call_penalty, self.parallel_bonus).objective
            for pool in holdout
        ]
        challenger_holdout = [
            replay_policy([pool], challenger, self.target_batch_size, self.call_penalty, self.parallel_bonus).objective
            for pool in holdout
        ]
        gate = holdout_gate(
            incumbent_holdout,
            challenger_holdout,
            min_pairs=self.min_holdout_pairs,
            min_mean_delta=0.0,
            alpha=0.10,
        )
        external_ok = True
        if gate.promote and self.promotion_judge is not None:
            external_ok = self.promotion_judge({"rollout_id": rollout_id,
                "incumbent": incumbent.to_dict(), "challenger": challenger.to_dict(),
                "gate": asdict(gate), "note": "Retrospective curriculum surrogate, not live model capability evidence"}) is True
        promoted = gate.promote and external_ok
        if promoted:
            self._write_policy(challenger)
            self.policy = challenger

        event = {
            "type": "policy_evolution",
            "rollout_id": rollout_id,
            "promoted": promoted,
            "reason": gate.reason if external_ok else "external judge vetoed policy promotion",
            "gate": asdict(gate),
            "incumbent": incumbent.to_dict(),
            "incumbent_replay": asdict(incumbent_train),
            "challenger": challenger.to_dict(),
            "challenger_replay": asdict(challenger_metrics),
        }
        self.trace.append_event(event)
        self._append_history(event)
        return event

    def _append_history(self, event: dict) -> None:
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
