import importlib.util
import os
import sys
import unittest

spec = importlib.util.spec_from_file_location("main", os.path.join(os.path.dirname(__file__), "..", "main.py"))
main = importlib.util.module_from_spec(spec)
sys.modules["main"] = main
spec.loader.exec_module(main)  # type: ignore


def make_detection(name: str, distance: float, ttc: float, box_ratio: float,
                    in_lane: bool = True, confidence: float = 0.9) -> "main.Detection":
    return main.Detection(
        name=name,
        confidence=confidence,
        box=(0, 0, 10, 10),
        source="s",
        track_key="k",
        distance_m=distance,
        in_lane=in_lane,
        lane_overlap=0.5,
        box_height_ratio=box_ratio,
        ttc_s=ttc,
    )


class ObstacleRuleTests(unittest.TestCase):
    """New OBSTACLE_CLASSES / obstacle(1) coverage, checked against the real
    Prolog engine (expert_system.pl) and its Python fallback_decision mirror."""

    @classmethod
    def setUpClass(cls):
        cls.engine = main.PrologRiskEngine(main.PROLOG_FILE)
        # If pyswip/SWI-Prolog isn't installed in this environment, skip the
        # Prolog-backed assertions but still run the fallback-only ones.
        cls.prolog_available = cls.engine.available

    def test_prolog_obstacle_critical(self):
        if not self.prolog_available:
            self.skipTest("Prolog engine unavailable in this environment")
        d = make_detection("tree", distance=2.0, ttc=1.0, box_ratio=0.3)
        risk, action, rule_id, explanation, source = self.engine.decide(d)
        self.assertEqual(risk, "CRITICAL")
        self.assertEqual(rule_id, "brake_and_avoid_obstacle")
        self.assertEqual(source, "prolog")

    def test_prolog_obstacle_warning(self):
        if not self.prolog_available:
            self.skipTest("Prolog engine unavailable in this environment")
        d = make_detection("cone", distance=5.0, ttc=2.5, box_ratio=0.2)
        risk, action, rule_id, explanation, source = self.engine.decide(d)
        self.assertEqual(risk, "WARNING")
        self.assertEqual(rule_id, "obstacle_warning")
        self.assertEqual(source, "prolog")

    def test_prolog_non_obstacle_object_unaffected(self):
        if not self.prolog_available:
            self.skipTest("Prolog engine unavailable in this environment")
        # a plain "car" at the same critical distance must still use the
        # generic collision rule, not the new obstacle rule
        d = make_detection("car", distance=2.0, ttc=1.0, box_ratio=0.3)
        risk, action, rule_id, explanation, source = self.engine.decide(d)
        self.assertEqual(risk, "CRITICAL")
        self.assertEqual(rule_id, "brake_now_object_too_close")

    def test_fallback_obstacle_critical_matches_prolog(self):
        d = make_detection("tree", distance=2.0, ttc=1.0, box_ratio=0.3)
        risk, action, rule_id, explanation, source = main.fallback_decision(d)
        self.assertEqual(risk, "CRITICAL")
        self.assertEqual(rule_id, "brake_and_avoid_obstacle_py")
        self.assertEqual(source, "python_fallback")

    def test_fallback_obstacle_warning_matches_prolog(self):
        d = make_detection("cone", distance=5.0, ttc=2.5, box_ratio=0.2)
        risk, action, rule_id, explanation, source = main.fallback_decision(d)
        self.assertEqual(risk, "WARNING")
        self.assertEqual(rule_id, "obstacle_warning_py")
        self.assertEqual(source, "python_fallback")

    def test_obstacle_classes_membership(self):
        for name in ("tree", "fallen_tree", "obstacle", "cone", "barrier", "debris", "log", "branch"):
            self.assertIn(name, main.OBSTACLE_CLASSES)


if __name__ == "__main__":
    unittest.main()
