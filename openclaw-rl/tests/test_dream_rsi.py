import tempfile
import unittest
from pathlib import Path

from dream_rsi import (
    Candidate,
    DreamRSIController,
    PolicyConfig,
    TraceStore,
    holdout_gate,
    mutate_policy,
    replay_policy,
    select_candidates,
)


class DummySample:
    def __init__(self, session, turn, score, index):
        self.metadata = {"session_id": session, "turn": turn, "has_next_state": True}
        self.reward = {"score": score}
        self.index = index
        self.remove_sample = False


class DreamRSITests(unittest.TestCase):
    def pool(self):
        return [
            Candidate("a1", "a", 1, -1.0),
            Candidate("a2", "a", 2, 1.0, parent_id="a1", delta_vs_parent=2.0),
            Candidate("b1", "b", 1, 1.0),
            Candidate("c1", "c", 1, 0.0),
            Candidate("d1", "d", 1, 1.0),
            Candidate("e1", "e", 1, -1.0),
        ]

    def test_policy_validation(self):
        PolicyConfig().validated()
        with self.assertRaises(ValueError):
            PolicyConfig(exploit_fraction=0.9, explore_fraction=0.9, recover_fraction=0.1).validated()

    def test_selection_is_deterministic_and_target_bounded(self):
        p = PolicyConfig().validated()
        a = select_candidates(self.pool(), 4, p)
        b = select_candidates(self.pool(), 4, p)
        self.assertEqual([x.node_id for x in a], [x.node_id for x in b])
        self.assertEqual(len(a), 4)
        self.assertEqual(len({x.node_id for x in a}), 4)

    def test_mutations_are_valid_and_reproducible(self):
        p = PolicyConfig().validated()
        a = mutate_policy(p, seed=7, count=6)
        b = mutate_policy(p, seed=7, count=6)
        self.assertEqual([x.to_dict() for x in a], [x.to_dict() for x in b])
        self.assertEqual(len(a), 6)
        for x in a:
            x.validated()

    def test_holdout_gate(self):
        decision = holdout_gate(
            [0] * 8, [1] * 8, min_pairs=8, alpha=0.10
        )
        self.assertTrue(decision.promote)
        rejected = holdout_gate([1] * 8, [0] * 8, min_pairs=8, alpha=0.10)
        self.assertFalse(rejected.promote)

    def test_trace_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            store = TraceStore(Path(td) / "trace.jsonl")
            store.append_pool(3, self.pool())
            loaded = store.load_pools()
            self.assertEqual(len(loaded), 1)
            self.assertEqual([c.node_id for c in loaded[0]], [c.node_id for c in self.pool()])

    def test_controller_selects_and_persists(self):
        with tempfile.TemporaryDirectory() as td:
            controller = DreamRSIController(
                td,
                target_batch_size=2,
                pool_factor=2.0,
                evolve_interval=2,
                holdout_pools=1,
                min_holdout_pairs=1,
            )
            samples = [
                DummySample("a", 1, -1, 1),
                DummySample("a", 2, 1, 2),
                DummySample("b", 1, 1, 3),
                DummySample("c", 1, 0, 4),
            ]
            chosen = controller.select_samples(samples, rollout_id=0)
            self.assertEqual(len(chosen), 2)
            self.assertTrue((Path(td) / "policy.json").exists())
            self.assertTrue((Path(td) / "replay.jsonl").exists())

    def test_replay_scores_policy(self):
        p = PolicyConfig().validated()
        metrics = replay_policy([self.pool(), self.pool()], p, target=3)
        self.assertEqual(metrics.selected, 6)
        self.assertGreaterEqual(metrics.distinct_sessions, 2)


if __name__ == "__main__":
    unittest.main()
