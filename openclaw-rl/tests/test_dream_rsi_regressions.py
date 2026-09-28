"""CPU-only regression checks for the opt-in curriculum controller."""
import argparse
import ast
import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from dream_rsi import (Candidate, DreamRSIController, PolicyConfig,
                       TraceStore, holdout_gate, objective, select_candidates)
from dream_rsi.cli import main as cli_main

ROOT = Path(__file__).resolve().parents[1]


def sample(index, *, removed=False, mask=None, status="completed"):
    return SimpleNamespace(index=index, metadata={"session_id": "s", "turn": index},
                           reward={"score": 1.0}, remove_sample=removed,
                           loss_mask=[1] if mask is None else mask, status=status)


def extracted_function(path, name, namespace):
    """Exercise the actual function source without importing CUDA dependencies."""
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class RegressionTests(unittest.TestCase):
    def test_removed_samples_never_reenter_batch(self):
        with tempfile.TemporaryDirectory() as d:
            c = DreamRSIController(d, target_batch_size=2)
            result = c.select_samples([sample(1), sample(2, removed=True)], 0)
            self.assertEqual([s.index for s in result], [1])

    def test_zero_loss_masks_are_not_training_candidates(self):
        with tempfile.TemporaryDirectory() as d:
            c = DreamRSIController(d, target_batch_size=2)
            self.assertEqual(c.select_samples([sample(1, mask=[0]), sample(2, mask=[0])], 0), [])

    def test_failed_or_aborted_samples_are_not_selected(self):
        with tempfile.TemporaryDirectory() as d:
            c = DreamRSIController(d, target_batch_size=2)
            self.assertEqual(c.select_samples([sample(1, status="aborted"), sample(2, status="failed")], 0), [])

    def test_duplicate_node_ids_fail_closed(self):
        with self.assertRaises(ValueError):
            select_candidates([Candidate("dup", "s", 1, 1), Candidate("dup", "s", 1, 1)], 2, PolicyConfig())

    def test_nonfinite_candidates_rejected(self):
        for score in [float("nan"), float("inf"), -float("inf")]:
            with self.subTest(score=score), self.assertRaises(ValueError):
                Candidate("a", "s", 1, score)

    def test_nonfinite_holdout_never_promotes(self):
        for score in [float("nan"), float("inf"), -float("inf")]:
            with self.subTest(score=score), self.assertRaises(ValueError):
                holdout_gate([0.0] * 8, [score] * 8)

    def test_overflowing_holdout_differences_never_promote(self):
        with self.assertRaises(ValueError):
            holdout_gate([-1e308] * 8, [1e308] * 8)

    def test_holdout_parameters_validated(self):
        for kw in [{"min_pairs": 0}, {"alpha": 0.0}, {"alpha": 1.0}, {"min_mean_delta": -1}]:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                holdout_gate([0] * 8, [1] * 8, **kw)

    def test_insufficient_holdout_cannot_promote(self):
        self.assertFalse(holdout_gate([0] * 4, [1] * 4, min_pairs=8).promote)

    def test_tied_holdout_cannot_promote(self):
        self.assertFalse(holdout_gate([1] * 8, [1] * 8).promote)

    def test_policy_rejects_string_bool_and_unknown_fields(self):
        for data in [{"beta": "0.6"}, {"beta": True}, {"bogus": 1}, {"version": 0}]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                PolicyConfig.from_dict(data)

    def test_controller_rejects_bad_configuration(self):
        for kw in [{"pool_factor": float("nan")}, {"evolve_interval": 0},
                   {"mutation_count": 0}, {"holdout_pools": 1, "min_holdout_pairs": 4}]:
            with tempfile.TemporaryDirectory() as d, self.subTest(kw=kw), self.assertRaises(ValueError):
                DreamRSIController(d, target_batch_size=2, **kw)

    def test_trace_metadata_is_allowlisted(self):
        with tempfile.TemporaryDirectory() as d:
            store = TraceStore(Path(d) / "trace.jsonl")
            store.append_pool(0, [Candidate("a", "s", 1, 1, metadata={"prompt": "SECRET_PAYLOAD", "has_next_state": True})])
            self.assertNotIn("SECRET_PAYLOAD", store.path.read_text())
            self.assertTrue(store.load_pools()[0][0].metadata["has_next_state"])

    def test_empty_objective_is_finite_json(self):
        self.assertTrue(math.isfinite(objective([]).objective))

    def test_cli_rejects_zero_target_or_mutations(self):
        with tempfile.TemporaryDirectory() as d:
            store = TraceStore(Path(d) / "trace.jsonl")
            store.append_pool(0, [Candidate("a", "s", 1, 1)])
            for flag in ["--target", "--mutations"]:
                with self.subTest(flag=flag), patch.object(sys, "argv", ["dream-rsi", str(store.path), flag, "0"]), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as ex:
                        cli_main()
                    self.assertEqual(ex.exception.code, 2)

    def test_trainer_flag_off_remains_default(self):
        parse = extracted_function(ROOT / "unsloth_qlora_trainer.py", "parse_args", {"argparse": argparse, "math": math})
        with patch.object(sys, "argv", ["trainer", "--save", "/tmp/unused"]):
            self.assertFalse(parse().dream_rsi_enable)

    def test_trainer_rejects_dream_without_reward_judge(self):
        parse = extracted_function(ROOT / "unsloth_qlora_trainer.py", "parse_args", {"argparse": argparse, "math": math})
        with patch.object(sys, "argv", ["trainer", "--save", "/tmp/unused", "--dream-rsi-enable"]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ex:
                parse()
            self.assertEqual(ex.exception.code, 2)

    def test_prm_buffer_works_without_conversation_logging(self):
        tree = ast.parse((ROOT / "openclaw_api_server.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OpenClawAPIServer")
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_buffer_record")
        import time
        ns = {"time": time}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "proxy-buffer", "exec"), ns)
        owner = SimpleNamespace(_record_file="", _prm_enabled=True, _pending_records={})
        ns["_buffer_record"](owner, "s", 1, [], "prompt", "response", [])
        self.assertIn("s", owner._pending_records)


class EvolutionIntegrationTests(unittest.TestCase):
    # Synthetic fixtures test plumbing, not empirical model improvement.
    def pool(self, namespace):
        scores = [0.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 0.0]
        return [Candidate(f"{namespace}:n{i}", f"{namespace}:s{i//2}", i % 2 + 1, score)
                for i, score in enumerate(scores)]

    def controller(self, directory):
        c = DreamRSIController(directory, target_batch_size=4, evolve_interval=1)
        for i in range(6):
            c.trace.append_pool(i, self.pool(str(i)))
        return c

    def challenger(self, base):
        return replace(base, version=base.version + 1, exploit_fraction=1.0,
                       explore_fraction=0.0, recover_fraction=0.0,
                       delta_weight=0.0, depth_penalty=0.0)

    def test_promotion_and_restart_persist_policy(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.controller(d)
            proposal = self.challenger(c.policy)
            with patch("dream_rsi.core.mutate_policy", return_value=[proposal]):
                event = c.maybe_evolve(5)
            self.assertTrue(event["promoted"])
            self.assertEqual(event["gate"]["wins"], 4)
            restarted = DreamRSIController(d, target_batch_size=4, evolve_interval=1)
            self.assertEqual(restarted.policy, proposal)
            self.assertEqual(len(restarted.history_path.read_text().splitlines()), 1)

    def test_no_improvement_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.controller(d)
            with patch("dream_rsi.core.mutate_policy", return_value=[c.policy]):
                event = c.maybe_evolve(5)
            self.assertFalse(event["promoted"])
            self.assertEqual(c.policy.version, 1)

    def test_holdout_cannot_be_reused_after_restart(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.controller(d)
            with patch("dream_rsi.core.mutate_policy", return_value=[c.policy]):
                self.assertIsNotNone(c.maybe_evolve(5))
            restarted = DreamRSIController(d, target_batch_size=4, evolve_interval=1)
            self.assertIsNone(restarted.maybe_evolve(6))
            restarted.trace.append_pool(6, self.pool("6"))
            self.assertIsNone(restarted.maybe_evolve(6))

    def test_new_disjoint_window_can_be_evaluated(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.controller(d)
            with patch("dream_rsi.core.mutate_policy", return_value=[c.policy]):
                self.assertIsNotNone(c.maybe_evolve(5))
                for i in range(6, 10):
                    c.trace.append_pool(i, self.pool(str(i)))
                self.assertIsNotNone(c.maybe_evolve(9))
            self.assertEqual(len(c.history_path.read_text().splitlines()), 2)

    def test_session_overlap_prevents_false_independence(self):
        with tempfile.TemporaryDirectory() as d:
            c = DreamRSIController(d, target_batch_size=4, evolve_interval=1)
            for i in range(6):
                c.trace.append_pool(i, self.pool("same-session"))
            self.assertIsNone(c.maybe_evolve(5))

    def test_underfilled_holdout_is_not_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.controller(d)
            c.trace.append_pool(6, [Candidate("bad", "bad", 1, 0, valid=False)])
            self.assertIsNone(c.maybe_evolve(6))

    def test_failed_policy_write_does_not_change_memory(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.controller(d)
            incumbent = c.policy
            with patch("dream_rsi.core.mutate_policy", return_value=[self.challenger(incumbent)]), patch.object(c, "_write_policy", side_effect=OSError("test failure")):
                with self.assertRaises(OSError):
                    c.maybe_evolve(5)
            self.assertEqual(c.policy, incumbent)
            self.assertEqual(json.loads(c.policy_path.read_text())["version"], 1)

    def test_offline_cli_smoke(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.controller(d)
            output = io.StringIO()
            with patch.object(sys, "argv", ["dream-rsi", str(c.trace.path), "--target", "4", "--mutations", "8"]), contextlib.redirect_stdout(output):
                self.assertEqual(cli_main(), 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["pools"], 6)
            self.assertTrue(math.isfinite(report["best_challenger"]["metrics"]["objective"]))

    def test_trainer_accepts_valid_opt_in_configuration(self):
        parse = extracted_function(ROOT / "unsloth_qlora_trainer.py", "parse_args", {"argparse": argparse, "math": math})
        with patch.object(sys, "argv", ["trainer", "--save", "/tmp/unused", "--dream-rsi-enable", "--prm-enable"]):
            args = parse()
            self.assertTrue(args.dream_rsi_enable)
            self.assertTrue(args.prm_enable)

    def test_prm_buffer_stays_off_when_prm_and_logging_are_off(self):
        tree = ast.parse((ROOT / "openclaw_api_server.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OpenClawAPIServer")
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_buffer_record")
        import time
        ns = {"time": time}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "proxy-buffer", "exec"), ns)
        owner = SimpleNamespace(_record_file="", _prm_enabled=False, _pending_records={})
        ns["_buffer_record"](owner, "s", 1, [], "prompt", "response", [])
        self.assertEqual(owner._pending_records, {})


if __name__ == "__main__":
    unittest.main()
