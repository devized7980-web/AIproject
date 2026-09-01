"""Authoritative risk-engine adapter used by the API and simulator."""

from __future__ import annotations

import math
from pathlib import Path

try:
    from pyswip import Prolog
except Exception:  # pragma: no cover - optional dependency
    Prolog = None  # type: ignore[misc, assignment]

ROOT = Path(__file__).resolve().parent.parent
PROLOG_FILE = ROOT / "expert_system.pl"

VEHICLES = {"car", "truck", "bus", "motorcycle", "bicycle"}
ROAD_DAMAGE = {"pothole", "road crack", "road_crack", "crack", "longitudinal", "transverse", "alligator"}
ANIMALS = {"dog", "cat", "cow", "horse", "sheep", "bird"}
TRAFFIC = {"traffic light", "stop sign", "parking meter"}
OBSTACLES = {"tree", "fallen_tree", "fallen tree", "obstacle", "cone", "barrier", "debris", "log", "branch"}


def _close(d: dict) -> bool:
    return ((d.get("ttc_s") is not None and d["ttc_s"] <= 1.5)
            or d.get("distance_m", 999) <= 3.0 or d.get("ratio", 0) >= 0.52)


def _medium(d: dict) -> bool:
    return ((d.get("ttc_s") is not None and d["ttc_s"] <= 3.0)
            or d.get("distance_m", 999) <= 7.0 or d.get("ratio", 0) >= 0.32)


def _early(d: dict) -> bool:
    return ((d.get("ttc_s") is not None and d["ttc_s"] <= 5.0)
            or d.get("distance_m", 999) <= 14.0 or d.get("ratio", 0) >= 0.17)


def _rule(priority, level, rule_id, display, label, when, advice, match):
    return {"priority": priority, "level": level, "rule": rule_id, "display": display,
            "label": label, "when": when, "advice": advice, "match": match}


RULES = [
    _rule(25, "SAFE", "object_outside_vehicle_lane", "OUTSIDE VEHICLE LANE", "Object outside vehicle lane", "object is not inside the detected driving lane", "Recorded but not treated as an immediate threat.", lambda d: not d.get("in_lane", False)),
    _rule(45, "CAUTION", "observe_traffic_control", "OBSERVE TRAFFIC CONTROL", "Observe traffic control", "a traffic control is in lane with confidence >= 0.30", "Requires attention rather than collision braking.", lambda d: d.get("in_lane") and d.get("object") in TRAFFIC and d.get("conf", 0) >= 0.30),
    _rule(100, "CRITICAL", "brake_immediately_person_ahead", "BRAKE IMMEDIATELY - PERSON AHEAD", "Brake immediately - person ahead", "person in lane with a close TTC, distance or box", "BRAKE IMMEDIATELY. A pedestrian is directly in your path.", lambda d: d.get("in_lane") and d.get("object") == "person" and d.get("conf", 0) >= 0.30 and _close(d)),
    _rule(95, "CRITICAL", "brake_and_avoid_road_damage", "BRAKE AND AVOID ROAD DAMAGE", "Brake and avoid road damage", "road damage in lane with a close TTC, distance or box", "BRAKE AND AVOID. Road damage is close enough to cause loss of control.", lambda d: d.get("in_lane") and d.get("object") in ROAD_DAMAGE and d.get("conf", 0) >= 0.30 and _close(d)),
    _rule(94, "CRITICAL", "brake_and_avoid_obstacle", "BRAKE AND AVOID OBSTACLE", "Brake and avoid obstacle", "obstacle in lane with a close TTC, distance or box", "BRAKE AND AVOID. An obstacle is blocking the lane.", lambda d: d.get("in_lane") and d.get("object") in OBSTACLES and d.get("conf", 0) >= 0.30 and _close(d)),
    _rule(90, "CRITICAL", "brake_now_object_too_close", "BRAKE NOW - OBJECT TOO CLOSE", "Brake now - object too close", "any object in lane with a close TTC, distance or box", "BRAKE NOW. The object is too close to continue at current speed.", lambda d: d.get("in_lane") and d.get("conf", 0) >= 0.30 and _close(d)),
    _rule(65, "WARNING", "animal_warning", "SLOW DOWN - ANIMAL AHEAD", "Slow down - animal ahead", "animal in lane with a medium TTC, distance or box", "SLOW DOWN. An animal is ahead in your lane.", lambda d: d.get("in_lane") and d.get("object") in ANIMALS and d.get("conf", 0) >= 0.30 and _medium(d)),
    _rule(62, "WARNING", "road_damage_warning", "SLOW DOWN AND PREPARE TO AVOID ROAD DAMAGE", "Slow down and prepare to avoid road damage", "road damage in lane with a medium TTC, distance or box", "SLOW DOWN and prepare to avoid road damage.", lambda d: d.get("in_lane") and d.get("object") in ROAD_DAMAGE and d.get("conf", 0) >= 0.30 and _medium(d)),
    _rule(61, "WARNING", "obstacle_warning", "SLOW DOWN AND PREPARE TO AVOID OBSTACLE", "Slow down and prepare to avoid obstacle", "obstacle in lane with a medium TTC, distance or box", "SLOW DOWN and prepare to avoid the obstacle.", lambda d: d.get("in_lane") and d.get("object") in OBSTACLES and d.get("conf", 0) >= 0.30 and _medium(d)),
    _rule(60, "WARNING", "vehicle_following_distance", "SLOW DOWN AND INCREASE FOLLOWING DISTANCE", "Slow down and increase following distance", "vehicle in lane with a medium TTC, distance or box", "SLOW DOWN and increase your following distance.", lambda d: d.get("in_lane") and d.get("object") in VEHICLES and d.get("conf", 0) >= 0.30 and _medium(d)),
    _rule(55, "WARNING", "generic_warning", "SLOW DOWN - HAZARD AHEAD", "Slow down - hazard ahead", "any object in lane with a medium TTC, distance or box", "SLOW DOWN. A hazard is ahead in your lane.", lambda d: d.get("in_lane") and d.get("conf", 0) >= 0.30 and _medium(d)),
    _rule(40, "CAUTION", "caution_object_in_vehicle_lane", "CAUTION - OBJECT IN VEHICLE LANE", "Caution - object in vehicle lane", "object in lane with an early TTC, distance or box", "CAUTION. Stay alert and reduce speed slightly.", lambda d: d.get("in_lane") and d.get("conf", 0) >= 0.30 and _early(d)),
    _rule(20, "SAFE", "object_at_safe_distance", "OBJECT AT SAFE DISTANCE", "Object at safe distance", "no higher-priority rule matched", "Road clear. Continue carefully.", lambda d: True),
]


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

    def _table(self, d: dict) -> dict:
        for rule in RULES:
            if rule["match"](d):
                return rule
        return RULES[-1]

    def _prolog_result(self, d: dict) -> tuple[str, str, str, int] | None:
        if not self.available or self.prolog is None:
            return None
        ttc = d.get("ttc_s")
        ttc = 999.0 if ttc is None or not math.isfinite(ttc) else max(0.0, ttc)
        fact = f"observation({self._atom(d.get('object', 'object'))},{float(d.get('distance_m', 999)):.3f},{ttc:.3f},{str(bool(d.get('in_lane'))).lower()},{float(d.get('conf', 0)):.3f},{float(d.get('ratio', 0)):.3f})"
        list(self.prolog.query("retractall(observation(_,_,_,_,_,_))"))
        list(self.prolog.query(f"assertz({fact})"))
        result = list(self.prolog.query("decision_with_priority(Level,Action,RuleID,Explanation,Priority)"))
        if not result:
            return None
        row = result[0]
        return (str(row["Level"]).upper(), str(row["Action"]).replace("_", " ").upper(), str(row["RuleID"]), int(row["Priority"]))

    def decide(self, d: dict) -> tuple[str, str, str, int, str]:
        try:
            result = self._prolog_result(d)
        except Exception:
            result = None
        if result is not None:
            level, action, rule_id, priority = result
            return level, action, rule_id, priority, "prolog"
        rule = self._table(d)
        return rule["level"], rule["display"], rule["rule"], rule["priority"], "python_fallback"

    def trace(self, d: dict) -> dict:
        level, action, rule_id, priority, source = self.decide(d)
        rule = next((r for r in RULES if r["rule"] == rule_id), self._table(d))
        return {
            "object": d.get("object", "object"), "confidence": round(d.get("conf", 0.0), 3),
            "distance_m": round(d.get("distance_m", 0.0), 2), "ttc_s": d.get("ttc_s"),
            "in_lane": d.get("in_lane", False), "box_height_ratio": round(d.get("ratio", 0.0), 3),
            "level": level, "action": action, "rule": rule_id, "rule_label": rule["label"],
            "priority": priority, "when": rule["when"], "advice": rule["advice"],
            "code": f"{rule['level'].lower()}:{rule_id}",
            "engine": "SWI-Prolog" if source == "prolog" else "Python fallback",
            "decision_source": source,
        }


ENGINE = PrologEngine()
