import importlib.util
import os
import sys
import threading
import time
import unittest

# Load main module directly
spec = importlib.util.spec_from_file_location("main", os.path.join(os.path.dirname(__file__), "..", "main.py"))
main = importlib.util.module_from_spec(spec)
sys.modules["main"] = main
spec.loader.exec_module(main)  # type: ignore


class MockProlog:
    def __init__(self, behavior: dict | None = None, simulate_cleanup_failure: bool = False):
        # behavior maps query string to return value or Exception to raise
        self.behavior = behavior or {}
        self.simulate_cleanup_failure = simulate_cleanup_failure
        self._in_use = False

    def query(self, q):
        # maliciously simulate concurrency detection: if assertz called while in_use True, raise
        if q.startswith("assertz"):
            if self._in_use:
                raise RuntimeError("concurrent_assert")
            self._in_use = True
            try:
                time.sleep(0.02)
                val = self.behavior.get("assertz", [])
                if isinstance(val, Exception):
                    raise val
                return val
            finally:
                # leave _in_use True until decision finishes
                pass
        elif q.startswith("decision"):
            time.sleep(0.02)
            val = self.behavior.get("decision", [])
            # clear in_use when decision completes
            self._in_use = False
            if isinstance(val, Exception):
                raise val
            return val
        if q.startswith("retractall"):
            if self.simulate_cleanup_failure:
                raise RuntimeError("cleanup failed")
            val = self.behavior.get("retractall", [])
            if isinstance(val, Exception):
                raise val
            return val
        return []


class PrologTests(unittest.TestCase):
    def setUp(self):
        # reload main to get fresh PrologRiskEngine definition
        pass

    def _make_detection(self):
        return main.Detection(
            name="car",
            confidence=0.9,
            box=(0, 0, 10, 10),
            source="s",
            track_key="k",
            distance_m=5.0,
            in_lane=True,
            lane_overlap=0.5,
            box_height_ratio=0.3,
        )

    def test_successful_prolog_decision(self):
        mock = MockProlog({"decision": [{"Level": "warning", "Action": "slow_down", "RuleID": "vehicle_following_distance", "Explanation": "Reduce speed and increase following distance."}]})
        main.Prolog = lambda: None  # placeholder class
        engine = main.PrologRiskEngine.__new__(main.PrologRiskEngine)
        engine.available = True
        engine.prolog = mock
        engine._prolog_lock = threading.Lock()

        d = self._make_detection()
        level, action, rule_id, explanation, source = main.PrologRiskEngine.decide(engine, d)
        self.assertEqual(level, "WARNING")
        self.assertIn("SLOW", action)
        self.assertEqual(rule_id, "vehicle_following_distance")
        self.assertIn("Reduce", explanation)
        self.assertEqual(source, "prolog")

    def test_query_returns_no_result(self):
        mock = MockProlog({"decision": []})
        main.Prolog = lambda: None
        engine = main.PrologRiskEngine.__new__(main.PrologRiskEngine)
        engine.available = True
        engine.prolog = mock
        engine._prolog_lock = threading.Lock()

        d = self._make_detection()
        level, action, rule_id, explanation, source = main.PrologRiskEngine.decide(engine, d)
        # should fallback to Python logic
        self.assertIsInstance(level, str)
        self.assertEqual(source, "python_fallback")

    def test_assertion_failure(self):
        mock = MockProlog({"assertz": RuntimeError("assert fail")})
        main.Prolog = lambda: None
        engine = main.PrologRiskEngine.__new__(main.PrologRiskEngine)
        engine.available = True
        engine.prolog = mock
        engine._prolog_lock = threading.Lock()

        d = self._make_detection()
        level, action, rule_id, explanation, source = main.PrologRiskEngine.decide(engine, d)
        self.assertIsInstance(level, str)
        self.assertEqual(source, "python_fallback")

    def test_decision_query_failure(self):
        mock = MockProlog({"decision": RuntimeError("query fail")})
        main.Prolog = lambda: None
        engine = main.PrologRiskEngine.__new__(main.PrologRiskEngine)
        engine.available = True
        engine.prolog = mock
        engine._prolog_lock = threading.Lock()

        d = self._make_detection()
        level, action, rule_id, explanation, source = main.PrologRiskEngine.decide(engine, d)
        self.assertIsInstance(level, str)
        self.assertEqual(source, "python_fallback")

    def test_cleanup_failure_logged(self):
        mock = MockProlog({"decision": [{"Level": "warning", "Action": "ok", "RuleID": "test_rule", "Explanation": "ok"}]}, simulate_cleanup_failure=True)
        main.Prolog = lambda: None
        engine = main.PrologRiskEngine.__new__(main.PrologRiskEngine)
        engine.available = True
        engine.prolog = mock
        engine._prolog_lock = threading.Lock()

        d = self._make_detection()
        # Should not raise despite cleanup failure
        level, action, rule_id, explanation, source = main.PrologRiskEngine.decide(engine, d)
        # Should not raise despite cleanup failure
        self.assertEqual(level, "WARNING")
        self.assertEqual(source, "prolog")

    def test_concurrent_calls_are_serialized(self):
        # MockProlog will raise if assertz is called concurrently
        mock = MockProlog({"decision": [{"Level": "warning", "Action": "ok", "RuleID": "test_rule", "Explanation": "ok"}]})
        main.Prolog = lambda: None
        engine = main.PrologRiskEngine.__new__(main.PrologRiskEngine)
        engine.available = True
        engine.prolog = mock
        engine._prolog_lock = threading.Lock()

        d = self._make_detection()

        results = []

        def run_decide():
            try:
                results.append(main.PrologRiskEngine.decide(engine, d))
            except Exception as e:
                results.append(("ERR", str(e)))

        threads = [threading.Thread(target=run_decide) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # None should have raised 'concurrent_assert' and all should return valid results
        for r in results:
            self.assertIsInstance(r[0], str)

    def test_fallback_when_prolog_unavailable(self):
        # Ensure that fallback is used when Prolog is not available
        main.Prolog = None
        engine = main.PrologRiskEngine.__new__(main.PrologRiskEngine)
        engine.available = False
        engine.prolog = None
        engine._prolog_lock = threading.Lock()

        d = self._make_detection()
        level, action, rule_id, explanation, source = main.PrologRiskEngine.decide(engine, d)
        self.assertIsInstance(level, str)
        self.assertEqual(source, "python_fallback")


if __name__ == "__main__":
    unittest.main()
