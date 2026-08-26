"""Rule-priority and conflict-resolution tests.

These prove that when several Prolog rules are true at once, the winner is
chosen by explicit priority (risk level first, then rule priority) rather than
by clause order -- and that CRITICAL rules override less important ones.
"""

import importlib.util
import os
import sys
import unittest

spec = importlib.util.spec_from_file_location("main", os.path.join(os.path.dirname(__file__), "..", "main.py"))
assert spec is not None
main = importlib.util.module_from_spec(spec)
sys.modules["main"] = main
spec.loader.exec_module(main)  # type: ignore


def make_detection(name: str, distance: float, ttc: float, box_ratio: float,
                    in_lane: bool = True, confidence: float = 0.9):
    return main.Detection(
        name=name, confidence=confidence, box=(0, 0, 10, 10), source="s",
        track_key="k", distance_m=distance, in_lane=in_lane, lane_overlap=0.5,
        box_height_ratio=box_ratio, ttc_s=ttc,
    )


class RulePriorityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = main.PrologRiskEngine(main.PROLOG_FILE)

    def setUp(self):
        if not self.engine.available:
            self.skipTest("SWI-Prolog/pyswip unavailable in this environment")

    # -- the core guarantee -------------------------------------------------

    def test_critical_overrides_lower_priority_rules(self):
        """A person very close in-lane triggers CRITICAL, WARNING, CAUTION and
        SAFE rules simultaneously; CRITICAL must win."""
        d = make_detection("person", distance=2.0, ttc=1.0, box_ratio=0.3)
        risk, _, rule_id, _, source = self.engine.decide(d)

        self.assertEqual(risk, "CRITICAL")
        self.assertEqual(rule_id, "brake_immediately_person_ahead")
        self.assertEqual(source, "prolog")

        # the lower-priority rules really did fire and really were overridden
        levels = {t["risk"] for t in d.decision_trace}
        self.assertIn("WARNING", levels)
        self.assertIn("CAUTION", levels)
        self.assertIn("SAFE", levels)

    def test_winner_has_highest_priority_in_trace(self):
        d = make_detection("person", distance=2.0, ttc=1.0, box_ratio=0.3)
        self.engine.decide(d)

        winners = [t for t in d.decision_trace if t["winner"]]
        self.assertEqual(len(winners), 1)
        top_priority = max(t["priority"] for t in d.decision_trace)
        self.assertEqual(winners[0]["priority"], top_priority)
        self.assertEqual(winners[0]["priority"], 100)

    def test_trace_is_sorted_by_descending_priority(self):
        d = make_detection("pothole", distance=2.0, ttc=1.0, box_ratio=0.3)
        self.engine.decide(d)
        priorities = [t["priority"] for t in d.decision_trace]
        self.assertEqual(priorities, sorted(priorities, reverse=True))

    # -- specific hazards outrank the generic collision rule ----------------

    def test_road_damage_beats_generic_collision_rule(self):
        d = make_detection("pothole", distance=2.0, ttc=1.0, box_ratio=0.3)
        risk, _, rule_id, _, _ = self.engine.decide(d)
        self.assertEqual(risk, "CRITICAL")
        self.assertEqual(rule_id, "brake_and_avoid_road_damage")
        # the generic rule fired too, but lost
        overridden = {t["rule_id"] for t in d.decision_trace if not t["winner"]}
        self.assertIn("brake_now_object_too_close", overridden)

    def test_obstacle_beats_generic_collision_rule(self):
        d = make_detection("tree", distance=2.0, ttc=1.0, box_ratio=0.3)
        risk, _, rule_id, _, _ = self.engine.decide(d)
        self.assertEqual(risk, "CRITICAL")
        self.assertEqual(rule_id, "brake_and_avoid_obstacle")
        overridden = {t["rule_id"] for t in d.decision_trace if not t["winner"]}
        self.assertIn("brake_now_object_too_close", overridden)

    def test_generic_rule_wins_for_unclassified_object(self):
        """A car has no specific critical rule, so the generic one should win."""
        d = make_detection("car", distance=2.0, ttc=1.0, box_ratio=0.3)
        risk, _, rule_id, _, _ = self.engine.decide(d)
        self.assertEqual(risk, "CRITICAL")
        self.assertEqual(rule_id, "brake_now_object_too_close")

    # -- warning tier -------------------------------------------------------

    def test_specific_warning_beats_generic_warning(self):
        d = make_detection("car", distance=5.0, ttc=2.5, box_ratio=0.2)
        risk, _, rule_id, _, _ = self.engine.decide(d)
        self.assertEqual(risk, "WARNING")
        self.assertEqual(rule_id, "vehicle_following_distance")
        overridden = {t["rule_id"] for t in d.decision_trace if not t["winner"]}
        self.assertIn("generic_warning", overridden)

    def test_out_of_lane_object_stays_safe(self):
        d = make_detection("car", distance=50.0, ttc=99.0, box_ratio=0.05, in_lane=False)
        risk, _, rule_id, _, _ = self.engine.decide(d)
        self.assertEqual(risk, "SAFE")
        self.assertEqual(rule_id, "object_outside_vehicle_lane")

    # -- no duplicate rules in the conflict set -----------------------------

    def test_each_rule_appears_once_in_trace(self):
        """The threshold disjunctions must not report a rule several times."""
        d = make_detection("person", distance=2.0, ttc=1.0, box_ratio=0.6)
        self.engine.decide(d)
        rule_ids = [t["rule_id"] for t in d.decision_trace]
        self.assertEqual(len(rule_ids), len(set(rule_ids)))

    # -- priority table integrity ------------------------------------------

    def test_priority_ordering_respects_risk_levels(self):
        """Every CRITICAL rule must outrank every WARNING rule, and so on."""
        critical = ["brake_immediately_person_ahead", "brake_and_avoid_road_damage",
                    "brake_and_avoid_obstacle", "brake_now_object_too_close"]
        warning = ["animal_warning", "road_damage_warning", "obstacle_warning",
                   "vehicle_following_distance", "generic_warning"]
        caution = ["observe_traffic_control", "caution_object_in_vehicle_lane"]
        safe = ["object_outside_vehicle_lane", "object_at_safe_distance"]

        p = main.RULE_PRIORITIES
        self.assertGreater(min(p[r] for r in critical), max(p[r] for r in warning))
        self.assertGreater(min(p[r] for r in warning), max(p[r] for r in caution))
        self.assertGreater(min(p[r] for r in caution), max(p[r] for r in safe))

    def test_python_priorities_match_prolog(self):
        """RULE_PRIORITIES in main.py must mirror rule_priority/2 in Prolog."""
        rows = list(self.engine.prolog.query("rule_priority(RuleID, Priority)"))
        prolog_priorities = {str(r["RuleID"]): int(r["Priority"]) for r in rows}
        self.assertEqual(prolog_priorities, main.RULE_PRIORITIES)


class FallbackPriorityTests(unittest.TestCase):
    """The Python fallback path must report priorities too, so the dashboard
    still works when Prolog is unavailable."""

    def test_fallback_sets_priority_and_trace(self):
        d = make_detection("person", distance=2.0, ttc=1.0, box_ratio=0.3)
        risk, _, rule_id, _, source = main.fallback_decision(d)
        self.assertEqual(risk, "CRITICAL")
        self.assertEqual(source, "python_fallback")
        self.assertEqual(d.rule_priority, 100)
        self.assertEqual(len(d.decision_trace), 1)
        self.assertTrue(d.decision_trace[0]["winner"])


class TraceFormattingTests(unittest.TestCase):
    def test_format_decision_trace_marks_winner(self):
        trace = [
            {"rule_id": "brake_and_avoid_road_damage", "risk": "CRITICAL", "priority": 95, "winner": True},
            {"rule_id": "brake_now_object_too_close", "risk": "CRITICAL", "priority": 90, "winner": False},
        ]
        out = main.format_decision_trace(trace)
        self.assertEqual(
            out,
            "critical:brake_and_avoid_road_damage:95* | critical:brake_now_object_too_close:90",
        )

    def test_format_decision_trace_empty(self):
        self.assertEqual(main.format_decision_trace([]), "")


if __name__ == "__main__":
    unittest.main()
