from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .core import PolicyConfig, TraceStore, mutate_policy, replay_policy


def main() -> int:
    p = argparse.ArgumentParser(description="Offline frozen-replay inspection for OpenClaw Dream-RSI")
    p.add_argument("trace", help="Path to dream_rsi/replay.jsonl")
    p.add_argument("--target", type=int, default=16, help="Samples selected from each candidate pool")
    p.add_argument("--policy", help="Optional policy.json; defaults to built-in policy")
    p.add_argument("--mutations", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    store = TraceStore(args.trace)
    pools = store.load_pools()
    if not pools:
        raise SystemExit("No candidate_pool events found")
    if args.policy:
        with open(args.policy, "r", encoding="utf-8") as f:
            policy = PolicyConfig.from_dict(json.load(f))
    else:
        policy = PolicyConfig().validated()

    incumbent = replay_policy(pools, policy, args.target)
    challengers = mutate_policy(policy, args.seed, args.mutations)
    ranked = sorted(
        ((replay_policy(pools, c, args.target), c) for c in challengers),
        key=lambda item: (item[0].objective, item[0].mean_score),
        reverse=True,
    )
    best_metrics, best_policy = ranked[0]
    print(json.dumps({
        "pools": len(pools),
        "incumbent": {"policy": policy.to_dict(), "metrics": asdict(incumbent)},
        "best_challenger": {"policy": best_policy.to_dict(), "metrics": asdict(best_metrics)},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
