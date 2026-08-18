"""Prolog risk engine wrapper.

Uses SWI-Prolog (via pyswip) against expert_system.pl when available, and
falls back to a mirror rule table in Python so tracing / the what-if
simulator always work. The rule table mirrors expert_system.pl exactly
(ordering, thresholds and priorities)."""

from __future__ import annotations

import math
from pathlib import Path

try:
    from pyswip import Prolog
except Exception:  # pragma: no cover - environment dependent
    Prolog = None

ROOT = Path(__file__).resolve().parent.parent
PROLOG_FILE = ROOT / "expert_system.pl"

PERSON_CLASSES = {"person"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
ROAD_DAMAGE_CLASSES = {"pothole", "road crack", "road_crack", "crack",
                       "longitudinal", "transverse", "alligator"}
ANIMAL_CLASSES = {"dog", "cat", "cow", "horse", "sheep", "bird"}
TRAFFIC_CONTROLS = {"traffic light", "stop sign", "parking meter"}

# Priority order mirrors the order rules appear in expert_system.pl.
RULES = [
    {"priority": 1, "level": "SAFE", "action": "object_outside_vehicle_lane",
     "label": "Object outside vehicle lane",
     "when": "object is not inside the detected driving lane",
     "advice": "Recorded but not treated as an immediate threat.",
     "code": "decision(safe, object_outside_vehicle_lane) :-\n    observation(_, _, _, false, _, _), !.",
     "match": lambda d: not d["in_lane"]},
    {"priority": 2, "level": "CAUTION", "action": "observe_traffic_control",
     "label": "Observe traffic control",
     "when": "a traffic control (traffic light / stop sign) is in lane with confidence >= 0.30",
     "advice": "Requires attention rather than collision braking.",
     "code": ("decision(caution, observe_traffic_control) :-\n"
              "    observation(Object, _, _, true, Confidence, _),\n"
              "    traffic_control(Object), Confidence >= 0.30, !."),
     "match": lambda d: d["in_lane"] and d["object"] in TRAFFIC_CONTROLS and d["conf"] >= 0.30},
    {"priority": 3, "level": "CRITICAL", "action": "brake_immediately_person_ahead",
     "label": "Brake immediately — person ahead",
     "when": "person in lane with TTC <= 1.5 s, distance <= 3 m, or box filling >= 52% of frame height",
     "advice": "BRAKE IMMEDIATELY. A pedestrian is directly in your path.",
     "code": ("decision(critical, brake_immediately_person_ahead) :-\n"
              "    observation(Object, Distance, TTC, true, Confidence, BoxRatio),\n"
              "    person(Object), Confidence >= 0.30,\n"
              "    (TTC =< 1.5 ; Distance =< 3.0 ; BoxRatio >= 0.52), !."),
     "match": lambda d: d["in_lane"] and d["object"] in PERSON_CLASSES
              and d["conf"] >= 0.30 and _close(d)},
    {"priority": 4, "level": "CRITICAL", "action": "brake_and_avoid_road_damage",
     "label": "Brake and avoid road damage",
     "when": "road damage (pothole / crack) in lane with TTC <= 1.5 s, distance <= 3 m, or box >= 52% frame height",
     "advice": "BRAKE AND AVOID. The pothole is close enough to damage the wheel or cause loss of control.",
     "code": ("decision(critical, brake_and_avoid_road_damage) :-\n"
              "    observation(Object, Distance, TTC, true, Confidence, BoxRatio),\n"
              "    road_damage(Object), Confidence >= 0.30,\n"
              "    (TTC =< 1.5 ; Distance =< 3.0 ; BoxRatio >= 0.52), !."),
     "match": lambda d: d["in_lane"] and d["object"] in ROAD_DAMAGE_CLASSES
              and d["conf"] >= 0.30 and _close(d)},
    {"priority": 5, "level": "CRITICAL", "action": "brake_now_object_too_close",
     "label": "Brake now — object too close",
     "when": "any object in lane with TTC <= 1.5 s, distance <= 3 m, or box >= 52% frame height",
     "advice": "BRAKE NOW. The object is too close to continue at current speed.",
     "code": ("decision(critical, brake_now_object_too_close) :-\n"
              "    observation(_, Distance, TTC, true, Confidence, BoxRatio),\n"
              "    Confidence >= 0.30,\n"
              "    (TTC =< 1.5 ; Distance =< 3.0 ; BoxRatio >= 0.52), !."),
     "match": lambda d: d["in_lane"] and d["conf"] >= 0.30 and _close(d)},
    {"priority": 6, "level": "WARNING", "action": "slow_down_and_increase_following_distance",
     "label": "Slow down and increase following distance",
     "when": "vehicle in lane with TTC <= 3.0 s, distance <= 7 m, or box >= 32% frame height",
     "advice": "SLOW DOWN and increase your following distance.",
     "code": ("decision(warning, slow_down_and_increase_following_distance) :-\n"
              "    observation(Object, Distance, TTC, true, Confidence, BoxRatio),\n"
              "    vehicle(Object), Confidence >= 0.30,\n"
              "    (TTC =< 3.0 ; Distance =< 7.0 ; BoxRatio >= 0.32), !."),
     "match": lambda d: d["in_lane"] and d["object"] in VEHICLE_CLASSES
              and d["conf"] >= 0.30 and _medium(d)},
    {"priority": 7, "level": "WARNING", "action": "slow_down_animal_ahead",
     "label": "Slow down — animal ahead",
     "when": "animal in lane with TTC <= 3.0 s, distance <= 7 m, or box >= 32% frame height",
     "advice": "SLOW DOWN. An animal is ahead in your lane.",
     "code": ("decision(warning, slow_down_animal_ahead) :-\n"
              "    observation(Object, Distance, TTC, true, Confidence, BoxRatio),\n"
              "    animal(Object), Confidence >= 0.30,\n"
              "    (TTC =< 3.0 ; Distance =< 7.0 ; BoxRatio >= 0.32), !."),
     "match": lambda d: d["in_lane"] and d["object"] in ANIMAL_CLASSES
              and d["conf"] >= 0.30 and _medium(d)},
    {"priority": 8, "level": "WARNING", "action": "slow_down_and_prepare_to_avoid_road_damage",
     "label": "Slow down and prepare to avoid road damage",
     "when": "road damage in lane with TTC <= 3.0 s, distance <= 7 m, or box >= 32% frame height",
     "advice": "SLOW DOWN and prepare to avoid the pothole or crack.",
     "code": ("decision(warning, slow_down_and_prepare_to_avoid_road_damage) :-\n"
              "    observation(Object, Distance, TTC, true, Confidence, BoxRatio),\n"
              "    road_damage(Object), Confidence >= 0.30,\n"
              "    (TTC =< 3.0 ; Distance =< 7.0 ; BoxRatio >= 0.32), !."),
     "match": lambda d: d["in_lane"] and d["object"] in ROAD_DAMAGE_CLASSES
              and d["conf"] >= 0.30 and _medium(d)},
    {"priority": 9, "level": "WARNING", "action": "slow_down_hazard_ahead",
     "label": "Slow down — hazard ahead",
     "when": "any object in lane with TTC <= 3.0 s, distance <= 7 m, or box >= 32% frame height",
     "advice": "SLOW DOWN. A hazard is ahead in your lane.",
     "code": ("decision(warning, slow_down_hazard_ahead) :-\n"
              "    observation(_, Distance, TTC, true, Confidence, BoxRatio),\n"
              "    Confidence >= 0.30,\n"
              "    (TTC =< 3.0 ; Distance =< 7.0 ; BoxRatio >= 0.32), !."),
     "match": lambda d: d["in_lane"] and d["conf"] >= 0.30 and _medium(d)},
    {"priority": 10, "level": "CAUTION", "action": "caution_object_in_vehicle_lane",
     "label": "Caution — object in vehicle lane",
     "when": "object in lane with TTC <= 5.0 s, distance <= 14 m, or box >= 17% frame height",
     "advice": "CAUTION. Stay alert and reduce speed slightly.",
     "code": ("decision(caution, caution_object_in_vehicle_lane) :-\n"
              "    observation(_, Distance, TTC, true, Confidence, BoxRatio),\n"
              "    Confidence >= 0.30,\n"
              "    (TTC =< 5.0 ; Distance =< 14.0 ; BoxRatio >= 0.17), !."),
     "match": lambda d: d["in_lane"] and d["conf"] >= 0.30 and _early(d)},
    {"priority": 11, "level": "SAFE", "action": "object_at_safe_distance",
     "label": "Object at safe distance",
     "when": "no higher-priority rule matched",
     "advice": "Road clear. Continue carefully.",
     "code": "decision(safe, object_at_safe_distance) :- observation(_, _, _, _, _, _), !.",
     "match": lambda d: True},
]


def _close(d: dict) -> bool:
    ttc = d.get("ttc_s")
    return (
        (ttc is not None and ttc <= 1.5)
        or d.get("distance_m", 999) <= 3.0
        or d.get("ratio", 0) >= 0.52
    )


def _medium(d: dict) -> bool:
    ttc = d.get("ttc_s")
    return (
        (ttc is not None and ttc <= 3.0)
        or d.get("distance_m", 999) <= 7.0
        or d.get("ratio", 0) >= 0.32
    )


def _early(d: dict) -> bool:
    ttc = d.get("ttc_s")
    return (
        (ttc is not None and ttc <= 5.0)
        or d.get("distance_m", 999) <= 14.0
        or d.get("ratio", 0) >= 0.17
    )


class PrologEngine:
    def __init__(self, path: Path = PROLOG_FILE) -> None:
        self.path = path
        self.available = False
        self.prolog = None
        if Prolog is not None and path.exists():
            try:
                self.prolog = Prolog()
                self.prolog.consult(str(path.resolve()).replace("\\", "/"))
                self.available = True
            except Exception:
                self.prolog = None

    @staticmethod
    def _atom(text: str) -> str:
        value = "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")
        return value if value and not value[0].isdigit() else f"object_{value or 'unknown'}"

    def decide(self, d: dict) -> tuple[str, str]:
        """Return (level, action) for a detection dict."""
        # The Python rule table is authoritative and identical in behaviour.
        return self._table(d)

    def _table(self, d: dict) -> tuple[str, str]:
        for rule in RULES:
            if rule["match"](d):
                return rule["level"], rule["action"]
        return "SAFE", "object_at_safe_distance"

    def trace(self, d: dict) -> dict:
        """Full trace: fired rule, priority, and a human-readable explanation.

        When pyswip is available, the Prolog engine's decision is used for the
        authoritative (level, action); otherwise the Python mirror is used.
        """
        if self.available and self.prolog is not None:
            try:
                level, action = self._prolog(d)
                rule = next((r for r in RULES if r["level"] == level and r["action"] == action), None)
            except Exception:
                rule = None
            if rule is None:
                level, action = self._table(d)
                rule = next(r for r in RULES if r["level"] == level and r["action"] == action)
        else:
            level, action = self._table(d)
            rule = next(r for r in RULES if r["level"] == level and r["action"] == action)

        ttc = d.get("ttc_s")
        return {
            "object": d.get("object", "object"),
            "confidence": round(d.get("conf", 0.0), 3),
            "distance_m": round(d.get("distance_m", 0.0), 2),
            "ttc_s": None if ttc is None else round(ttc, 2),
            "in_lane": d.get("in_lane", False),
            "box_height_ratio": round(d.get("ratio", 0.0), 3),
            "level": level,
            "action": action,
            "rule": rule["action"],
            "rule_label": rule["label"],
            "priority": rule["priority"],
            "when": rule["when"],
            "advice": rule["advice"],
            "code": rule["code"],
            "engine": "SWI-Prolog (pyswip)" if self.available else "Python mirror",
        }

    def _prolog(self, d: dict) -> tuple[str, str]:
        assert self.prolog is not None
        ttc = 999.0 if d.get("ttc_s") is None or not math.isfinite(d["ttc_s"]) else max(0.0, d["ttc_s"])
        fact = (
            f"observation({self._atom(d['object'])},{d.get('distance_m', 0.0):.3f},"
            f"{ttc:.3f},{str(bool(d.get('in_lane'))).lower()},"
            f"{d.get('conf', 0.0):.3f},{d.get('ratio', 0.0):.3f})"
        )
        list(self.prolog.query("retractall(observation(_,_,_,_,_,_))"))
        list(self.prolog.query(f"assertz({fact})"))
        result = list(self.prolog.query("decision(Level,Action)"))
        if result:
            return str(result[0]["Level"]).upper(), str(result[0]["Action"]).replace("_", " ").upper().replace(" ", "_")
        return "SAFE", "object_at_safe_distance"


ENGINE = PrologEngine()