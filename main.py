from __future__ import annotations

import argparse
import csv
import json
import math
import re
import threading
import queue
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from pyswip import Prolog
except Exception:
    Prolog: Any = None

try:
    import pyttsx3
except Exception:
    pyttsx3: Any = None


ROOT = Path(__file__).resolve().parent
COMMON_MODEL_PATH = ROOT / "models" / "yolo11n.pt"
CUSTOM_MODEL_PATH = ROOT / "models" / "best.pt"
PROLOG_FILE = ROOT / "expert_system.pl"
VIDEO_FOLDER = ROOT / "videos"
OUTPUT_FOLDER = ROOT / "output"
INCIDENT_FOLDER = OUTPUT_FOLDER / "incidents"

DETECTION_INTERVAL = 1

# Per-model confidence thresholds.
# best.pt (road damage) scores far lower than the COCO model on real footage:
# on road_video_3 its pothole confidences run 0.25-0.68 with a median of 0.33,
# so a 0.60 gate kept only 4 of 301 detections and the pipeline was effectively
# blind to potholes. 0.30 matches the project's original single CONFIDENCE
# value and keeps ~190 of them.
COMMON_MODEL_CONFIDENCE = 0.50
CUSTOM_MODEL_CONFIDENCE = 0.30
IOU = 0.50
IMAGE_SIZE = 640
# Frames wider than this are downscaled before processing (see process_video).
MAX_PROCESSING_WIDTH = 1920
# best.pt was evidently trained near 416: on road_video_3 it finds 439 potholes
# at imgsz=416 versus 190 at 640 and only 38 at 960, and 416 is also the
# fastest (5.3 ms vs 5.7 ms). Larger is emphatically not better for this model.
ROAD_DAMAGE_IMAGE_SIZE = 416

# Visualization reference resolution.  All overlay sizes (fonts, thickness,
# padding, panel height) are designed for 1280x720 and scaled proportionally
# when the actual output frame differs.
VIS_REF_W = 1280
VIS_REF_H = 720
VIS_SCALE_MIN = 0.45
VIS_SCALE_MAX = 1.8
TRACKER = "bytetrack.yaml"

# Inference device. Resolved once at startup by select_device(); YOLO inference
# is ~90% of per-frame cost, and on Apple Silicon MPS runs it ~37% faster than
# CPU with identical detections.
INFERENCE_DEVICE: str | None = None

# Set True to print a per-stage timing breakdown when a video finishes.
DEBUG_PERFORMANCE = False

# Smoothness / performance settings
# 1 = detect on every frame (best accuracy, slower)
# 2 = detect every second frame (recommended for smoother playback)
DISTANCE_SMOOTH_ALPHA = 0.35
TTC_SMOOTH_ALPHA = 0.45
# Fresh matched detections should lead the rendered box; the Kalman filter is
# retained for association and short misses, not used as a lagging display box.
BOX_SMOOTH_ALPHA = 0.70
# Road damage closes on the camera quickly and is detected sporadically, so its
# distance follows the measurement more closely than a tracked vehicle's.
ROAD_DAMAGE_DISTANCE_ALPHA = 0.75
LANE_SMOOTH_ALPHA = 0.20
# Lane temporal-tracking gates (see LaneTracker).
LIVE_READER_QUEUE_SIZE = 2          # shallow queue for the live preview path
MAX_LANE_MISSED_FRAMES = 3          # coast this long through detection gaps
LANE_MAX_CENTRE_SHIFT_RATIO = 0.08  # max lane-centre move per frame, as a fraction of width
LANE_MAX_WIDTH_RATIO = 1.5          # max lane-width change factor per frame
LANE_MIN_DRAW_CONFIDENCE = 0.70     # needs 2+ consecutive good detections before drawing,
                                    # so one isolated hit cannot flash the overlay on

# Ego-vehicle filter.  The camera car's hood/bumper appears as a large
# vehicle detection at the bottom-centre of the frame.  We suppress it
# when it overlaps a small centred ROI at the bottom, using normalised
# coordinates so the filter works at any resolution.
# Normalized camera-hood ROI. It is evaluated on the processing frame, so it
# remains correct when inference downsizes a source video.
EGO_ROI_X1 = 0.25
EGO_ROI_X2 = 0.75
EGO_ROI_Y1 = 0.78
EGO_OVERLAP_MIN_RATIO = 0.25
EGO_MIN_WIDTH_RATIO = 0.45
EGO_MIN_TOP_RATIO = 0.68
EGO_MIN_BOTTOM_RATIO = 0.96

# Forget a track after it has not been seen for this many detection cycles.
TRACK_FORGET_AFTER = 20
TRACK_MAX_AGE = TRACK_FORGET_AFTER
TRACK_MIN_HITS = 2

# Road damage is stationary in the world but sweeps through image space as the
# camera vehicle moves, so a constant-velocity prediction diverges from it fast.
# Vehicles move predictably relative to the camera and tolerate longer gaps.
ROAD_DAMAGE_FORGET_AFTER = 2

# Maximum fraction of the frame diagonal a track may move between detection
# cycles and still be considered the same object. Stops a detection on one
# pothole being associated with a different pothole across the road.
MAX_TRACK_JUMP_RATIO = 0.12

# A track whose box top is at least this many pixels below the frame top gets
# its label above the box; otherwise below. Evaluated once per track.
LABEL_MIN_TOP_ROOM = 26

# Render a box only while it is backed by recent detection evidence. A track
# surviving purely on prediction is not drawn -- a box briefly disappearing is
# preferable to one floating over empty road.
MAX_PREDICTION_ONLY_FRAMES = 0

# Frames with mean grayscale brightness below this (0-255) are treated as
# low-light/night footage and get a CLAHE contrast boost before detection.
LOW_LIGHT_THRESHOLD = 70.0

# Extra, optionally-present detector models (e.g. a trained obstacle/tree
# model) auto-loaded from this folder if any *.pt files are dropped in it.
EXTRA_MODELS_DIR = ROOT / "models" / "extra"

FOCAL_LENGTH_PX = 700.0
DEFAULT_WIDTH_M = 0.8
KNOWN_WIDTHS_M = {
    "person": 0.50, "bicycle": 0.60, "car": 1.80, "motorcycle": 0.80,
    "bus": 2.50, "truck": 2.50, "dog": 0.45, "cat": 0.30,
    "traffic light": 0.30, "stop sign": 0.75, "pothole": 0.80,
    "road crack": 0.60, "road_crack": 0.60, "cone": 0.35,
    "barrier": 1.20, "debris": 0.50, "tree": 3.00, "fallen_tree": 3.50,
    "fallen tree": 3.50, "obstacle": 0.80, "log": 0.40, "branch": 0.30,
}

LEVEL_PRIORITY = {"SAFE": 0, "CAUTION": 1, "WARNING": 2, "CRITICAL": 3, "ERROR": 4}
LEVEL_COLORS = {
    "SAFE": (0, 210, 106), "CAUTION": (0, 176, 255),
    "WARNING": (0, 176, 255), "CRITICAL": (45, 31, 255),
    "ERROR": (255, 255, 255),
}

TRAFFIC_CONTROLS = {"traffic light", "stop sign", "parking meter"}
PERSON_CLASSES = {"person"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
ROAD_DAMAGE_CLASSES = {"pothole", "road crack", "road_crack", "crack",
                       "longitudinal", "transverse", "alligator"}
ANIMAL_CLASSES = {"dog", "cat", "cow", "horse", "sheep", "bird"}
OBSTACLE_CLASSES = {"tree", "fallen_tree", "fallen tree", "obstacle", "cone",
                     "barrier", "debris", "log", "branch"}

# Mirror of the rule_priority/2 facts in expert_system.pl, used only by the
# Python fallback path so priorities still appear when Prolog is unavailable.
# Prolog remains the authority whenever it is loaded.
RULE_PRIORITIES = {
    "brake_immediately_person_ahead": 100,
    "brake_and_avoid_road_damage": 95,
    "brake_and_avoid_obstacle": 94,
    "brake_now_object_too_close": 90,
    "animal_warning": 65,
    "road_damage_warning": 62,
    "obstacle_warning": 61,
    "vehicle_following_distance": 60,
    "generic_warning": 55,
    "observe_traffic_control": 45,
    "caution_object_in_vehicle_lane": 40,
    "object_outside_vehicle_lane": 25,
    "object_at_safe_distance": 20,
}

Decision: TypeAlias = tuple[str, str, str, str, str]


@dataclass
class Detection:
    name: str
    confidence: float
    box: tuple[int, int, int, int]
    source: str
    track_key: str
    distance_m: float
    in_lane: bool
    lane_overlap: float
    box_height_ratio: float
    closing_speed_mps: float = 0.0
    ttc_s: float = math.inf
    risk: str = "SAFE"
    action: str = "CONTINUE CAREFULLY"
    distance_method: str = "unknown"
    # measured bounding box from detector; used for drawing
    measured_box: tuple[int, int, int, int] = field(default_factory=lambda: (0, 0, 0, 0))
    # Structured decision fields
    rule_id: str = ""
    explanation: str = ""
    decision_source: str = "python_fallback"
    # Rule-priority / conflict-resolution fields. `decision_trace` holds every
    # rule that fired for this observation (winner included), highest priority
    # first, so the dashboard can show the full reasoning behind one decision.
    rule_priority: int = 0
    decision_trace: list[dict] = field(default_factory=list)
    # Which stage produced the rendered box: "yolo" (measured this frame) or
    # "kalman" (filtered). Nothing is drawn from prediction alone.
    tracking_source: str = "yolo"
    # "top" or "bottom", chosen once when the track is created and then held for
    # the track's lifetime so the label cannot flip sides between frames.
    label_side: str = "top"
    track_hits: int = 0

    @property
    def x1(self) -> int: return self.box[0]
    @property
    def y1(self) -> int: return self.box[1]
    @property
    def x2(self) -> int: return self.box[2]
    @property
    def y2(self) -> int: return self.box[3]


@dataclass
class RunStats:
    frames: int = 0
    total_detections: int = 0
    level_counts: Counter = field(default_factory=Counter)
    object_counts: Counter = field(default_factory=Counter)
    incidents: int = 0
    min_ttc: float = math.inf
    low_light_frames: int = 0
    processing_fps_samples: list[float] = field(default_factory=list)
    # Prolog/fallback decision stats
    triggered_rules: Counter = field(default_factory=Counter)
    decision_source_counts: Counter = field(default_factory=Counter)
    # Rule-conflict stats: how often more than one rule fired at once, and
    # which rule beat which (losing rule id -> times it was overridden).
    conflicts: int = 0
    overridden_rules: Counter = field(default_factory=Counter)
    conflict_examples: list[dict] = field(default_factory=list)


class VoiceAlert:
    """Asynchronous TTS queue with a single long-running worker thread.

    - Uses queue.Queue as the backing buffer (bounded).
    - Engine is initialized exactly once inside the worker thread.
    - `speak()` never starts threads or calls `pyttsx3.init()`.
    - Duplicate queued keys are ignored.
    - When full, the oldest non-CRITICAL alert is discarded to make room.
    """

    DEFAULT_CAPACITY = 64

    def __init__(self, enabled: bool = True, capacity: int | None = None) -> None:
        self.enabled = enabled and pyttsx3 is not None
        self.last_spoken: dict[str, float] = {}
        self._queue: "queue.Queue[tuple[str, str]]" = queue.Queue(maxsize=(capacity or self.DEFAULT_CAPACITY))
        self._queued_keys: set[str] = set()
        self._lock = threading.Lock()
        self._sentinel = ("__SENTINEL__", "__SENTINEL__")
        self._worker_thread: threading.Thread | None = None
        if self.enabled:
            # start long-running daemon worker
            self._worker_thread = threading.Thread(target=self._run, daemon=True)
            self._worker_thread.start()

    def speak(self, key: str, message: str, cooldown: float = 3.0) -> None:
        """Enqueue a message for speech if cooldown permits. Does not start threads.

        Duplicate queued keys are ignored. If the queue is full we try to remove
        the oldest noncritical message to make room; otherwise the new message
        is skipped.
        """
        if not self.enabled:
            return
        now = time.time()
        if now - self.last_spoken.get(key, 0.0) < cooldown:
            return
        # update last spoken to enforce cooldown immediately
        self.last_spoken[key] = now

        with self._lock:
            if key in self._queued_keys:
                return
            item = (key, message)
            try:
                self._queue.put_nowait(item)
                self._queued_keys.add(key)
                return
            except Exception:
                # queue full: try to drop oldest noncritical
                drained: list[tuple[str, str]] = []
                try:
                    while True:
                        drained.append(self._queue.get_nowait())
                except Exception:
                    pass

                # find first noncritical to remove (level assumed after last ':' in key)
                removed_index = None
                for i, (k, m) in enumerate(drained):
                    level = k.split(":")[-1] if ":" in k else ""
                    if level.upper() != "CRITICAL":
                        removed_index = i
                        break

                if removed_index is not None:
                    removed = drained.pop(removed_index)
                    self._queued_keys.discard(removed[0])
                    drained.append(item)
                    self._queued_keys.add(key)
                else:
                    # no removable noncritical item: skip enqueueing
                    # put drained items back
                    drained.append(item)  # append new so we will trim it below

                # put back items up to capacity; if we appended the new item and
                # no removal occurred, this will effectively drop the new item
                for pair in drained[: self._queue.maxsize]:
                    try:
                        self._queue.put_nowait(pair)
                    except Exception:
                        break

                # rebuild queued_keys
                self._queued_keys = {k for k, _ in list(self._queue.queue)}

    def _run(self) -> None:
        engine = None
        try:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 175)
            except Exception as exc:
                print(f"Voice engine initialization failed: {exc!r}")
                engine = None

            while True:
                try:
                    item = self._queue.get()
                except Exception:
                    time.sleep(0.01)
                    continue

                if item == self._sentinel:
                    break

                key, message = item
                # remove from queued set immediately so duplicates can be queued again
                with self._lock:
                    self._queued_keys.discard(key)

                if engine is None:
                    # engine failed to initialize earlier; skip speaking
                    continue

                try:
                    engine.say(message)
                    engine.runAndWait()
                except Exception as exc:
                    print(f"TTS speaking failed for '{message}': {exc!r}")
                    try:
                        engine.stop()
                    except Exception:
                        pass
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass

    def close(self, wait: float = 0.5) -> None:
        """Send sentinel and wait briefly for the worker to exit.

        This method is safe to call even if voice was disabled.
        """
        if not self.enabled:
            return
        try:
            # clear queued keys and send sentinel to stop the worker
            with self._lock:
                self._queued_keys.clear()
            # put sentinel into queue to stop worker
            try:
                self._queue.put_nowait(self._sentinel)
            except Exception:
                # queue full: discard one item to make room for sentinel
                try:
                    _ = self._queue.get_nowait()
                except Exception:
                    pass
                try:
                    self._queue.put_nowait(self._sentinel)
                except Exception:
                    pass

            if self._worker_thread is not None:
                self._worker_thread.join(timeout=wait)
        except Exception:
            pass


class PrologRiskEngine:
    def __init__(self, path: Path) -> None:
        self.available = False
        self.prolog: Any = None
        self._prolog_lock = threading.Lock()
        if Prolog is None or not path.exists():
            print("Prolog unavailable; Python fallback rules will be used.")
            return
        try:
            self.prolog = Prolog()
            self.prolog.consult(str(path.resolve()).replace("\\", "/"))
            self.available = True
            print("Prolog expert system loaded.")
        except Exception as exc:
            print(f"Prolog loading failed: {exc!r}")

    @staticmethod
    def atom(text: str) -> str:
        value = "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")
        return value if value and not value[0].isdigit() else f"object_{value or 'unknown'}"

    def _collect_trace(self, winning_rule_id: str) -> list[dict]:
        """Every rule that fired for the current observation, highest priority
        first. Read-only: conflict resolution already happened in Prolog, this
        just exposes the losing rules so the reasoning can be inspected."""
        try:
            rows = list(self.prolog.query(
                "triggered_rule(Level,_Action,RuleID,Explanation,Priority)"
            ))
        except Exception as exc:
            print(f"Prolog trace query failed: {exc!r}")
            return []

        trace: list[dict] = []
        seen: set[str] = set()
        for row in rows:
            try:
                rule_id = str(row.get("RuleID") or "")
                if not rule_id or rule_id in seen:
                    continue
                seen.add(rule_id)
                trace.append({
                    "rule_id": rule_id,
                    "risk": str(row.get("Level") or "").upper(),
                    "priority": int(row.get("Priority") or 0),
                    "explanation": str(row.get("Explanation") or ""),
                    "winner": rule_id == winning_rule_id,
                })
            except Exception:
                continue
        trace.sort(key=lambda r: r["priority"], reverse=True)
        return trace
    def decide(self, d: Detection) -> Decision:
        # New interface: return (risk, action_display, rule_id, explanation, decision_source)
        if not self.available:
            return fallback_decision(d)

        try:
            assert self.prolog is not None
            # Normalize numeric values and reject non-finite inputs
            def _norm(value: float, default: float) -> float:
                try:
                    return float(value) if math.isfinite(value) else default
                except Exception:
                    return default

            distance = _norm(d.distance_m, 999.0)
            confidence = _norm(d.confidence, 0.0)
            ttc = _norm(d.ttc_s, 999.0)
            # ttc must be non-negative for the fact
            ttc = max(0.0, ttc)
            box_height_ratio = _norm(d.box_height_ratio, 0.0)
            in_lane_str = str(bool(d.in_lane)).lower()
            atom_name = self.atom(d.name)

            with self._prolog_lock:
                # Ensure any previous temporary observations are cleared first.
                try:
                    list(self.prolog.query("retractall(observation(_,_,_,_,_,_))"))
                except Exception as exc:
                    print(f"Prolog cleanup(before) failed: {exc!r}")

                fact = (
                    f"observation({atom_name},{distance:.3f},{ttc:.3f},"
                    f"{in_lane_str},{confidence:.3f},{box_height_ratio:.3f})"
                )

                try:
                    # assert the fact and query decision; any failure yields fallback
                    try:
                        list(self.prolog.query(f"assertz({fact})"))
                    except Exception as exc_assert:
                        print(f"Prolog assertion failed: {exc_assert!r}")
                        return fallback_decision(d)

                    try:
                        # Prolog resolves the conflict itself (highest risk
                        # level, then highest rule priority) and reports the
                        # winning rule's priority alongside the decision.
                        result = list(self.prolog.query(
                            "decision_with_priority(Level,Action,RuleID,Explanation,Priority)"
                        ))
                    except Exception as exc_query:
                        print(f"Prolog decision query failed: {exc_query!r}")
                        return fallback_decision(d)

                    if result:
                        try:
                            level_atom = str(result[0]["Level"]).upper()
                            action_atom = str(result[0]["Action"]).replace("_", " ").upper()
                            rule_id_atom = str(result[0].get("RuleID") or result[0].get("RuleId") or "")
                            # Explanation may be an atom or a string; convert to readable form
                            explanation_val = result[0].get("Explanation") or result[0].get("explanation") or ""
                            explanation = str(explanation_val)
                            try:
                                d.rule_priority = int(result[0].get("Priority") or 0)
                            except Exception:
                                d.rule_priority = 0
                            # Collect the losing rules purely for display; the
                            # winner above was already chosen by Prolog.
                            d.decision_trace = self._collect_trace(rule_id_atom)
                            return (level_atom, action_atom, rule_id_atom, explanation, "prolog")
                        except Exception:
                            return fallback_decision(d)
                    return fallback_decision(d)
                finally:
                    try:
                        list(self.prolog.query("retractall(observation(_,_,_,_,_,_))"))
                    except Exception as exc_clean:
                        print(f"Prolog cleanup(after) failed: {exc_clean!r}")
        except Exception as exc:
            print(f"Prolog decision failed: {exc!r}")
            return fallback_decision(d)


def fallback_decision(d: Detection) -> Decision:
    result = _fallback_decision_rules(d)
    # Record the priority of the rule that matched so the dashboard's decision
    # trace works identically on the fallback path. Python's fallback is
    # first-match (no conflict set), so the trace holds just the winner.
    try:
        rule_id = str(result[2]).removesuffix("_py")
        d.rule_priority = RULE_PRIORITIES.get(rule_id, 0)
        d.decision_trace = [{
            "rule_id": str(result[2]),
            "risk": str(result[0]),
            "priority": d.rule_priority,
            "explanation": str(result[3]),
            "winner": True,
        }]
    except Exception:
        pass
    return result


def _fallback_decision_rules(d: Detection) -> Decision:
    name = d.name.lower()
    # Return structured decision fields: (risk, action_display, rule_id, explanation, decision_source)
    if not d.in_lane:
        return ("SAFE", "OUTSIDE VEHICLE LANE", "object_outside_vehicle_lane", f"Object '{name}' is outside the vehicle lane and not an immediate collision threat.", "python_fallback")
    if d.confidence < 0.30:
        return ("SAFE", f"{name} AT SAFE DISTANCE", "object_at_safe_distance", "Detection confidence is below the rule threshold.", "python_fallback")
    if d.ttc_s <= 1.5 or d.distance_m <= 3.0 or d.box_height_ratio >= 0.52:
        if name in PERSON_CLASSES:
            return ("CRITICAL", "BRAKE IMMEDIATELY PERSON AHEAD", "brake_immediately_person_ahead", "Immediate braking required due to person in lane.", "python_fallback")
        if name in ROAD_DAMAGE_CLASSES:
            return ("CRITICAL", "BRAKE AND AVOID ROAD DAMAGE", "brake_and_avoid_road_damage", "Immediate maneuver to avoid road damage is required.", "python_fallback")
        if name in OBSTACLE_CLASSES:
            return ("CRITICAL", "BRAKE AND AVOID OBSTACLE", "brake_and_avoid_obstacle", f"Immediate maneuver to avoid obstacle '{name}' in lane is required.", "python_fallback")
        return ("CRITICAL", "BRAKE NOW OBJECT TOO CLOSE", "brake_now_object_too_close", f"Object '{name}' detected very close; brake immediately.", "python_fallback")
    if name in TRAFFIC_CONTROLS:
        return ("CAUTION", f"OBSERVE {name}", "observe_traffic_control", f"Traffic control '{name}' observed; monitor for signals or stops.", "python_fallback")
    if d.ttc_s <= 3.0 or d.distance_m <= 7.0 or d.box_height_ratio >= 0.32:
        if name in VEHICLE_CLASSES:
            return ("WARNING", "SLOW DOWN AND INCREASE FOLLOWING DISTANCE", "vehicle_following_distance", "Reduce speed and increase following distance to maintain a safe gap.", "python_fallback")
        if name in ANIMAL_CLASSES:
            return ("WARNING", "SLOW DOWN ANIMAL AHEAD", "animal_warning", "Animal detected near lane; slow down and proceed cautiously.", "python_fallback")
        if name in ROAD_DAMAGE_CLASSES:
            return ("WARNING", "SLOW DOWN AND PREPARE TO AVOID ROAD DAMAGE", "road_damage_warning", "Road damage ahead; slow down and prepare to avoid.", "python_fallback")
        if name in OBSTACLE_CLASSES:
            return ("WARNING", "SLOW DOWN AND PREPARE TO AVOID OBSTACLE", "obstacle_warning", f"Obstacle '{name}' ahead in lane; slow down and prepare to avoid.", "python_fallback")
        return ("WARNING", "SLOW DOWN HAZARD AHEAD", "generic_warning", f"Slow down: {name} detected ahead in the vehicle lane.", "python_fallback")
    if d.ttc_s <= 5.0 or d.distance_m <= 14.0 or d.box_height_ratio >= 0.17:
        return ("CAUTION", "CAUTION OBJECT IN VEHICLE LANE", "caution_object_in_vehicle_lane", f"Object '{name}' detected in vehicle lane; exercise caution.", "python_fallback")
    return ("SAFE", "OBJECT AT SAFE DISTANCE", "object_at_safe_distance", f"Object '{name}' is at a safe distance.", "python_fallback")


def is_valid_bbox(box: Any, frame_shape: tuple[int, int] | None = None) -> bool:
    """Structural validity of a RAW bounding box, checked BEFORE any clipping.

    Clipping first would turn a structurally invalid box into a drawable one,
    so this is the single gate every stage consults. A box is rejected when it
    is non-numeric, non-finite (NaN/inf), has zero or negative extent, or —
    when a frame shape is supplied — lies entirely outside the frame.
    """
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    if frame_shape is not None:
        h, w = frame_shape
        if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
            return False
    return True


def format_decision_trace(trace: list[dict]) -> str:
    """Compact one-cell rendering of a decision trace for the CSV, e.g.
    'critical:brake_and_avoid_road_damage:95* | critical:brake_now:90'
    (the winning rule is marked with a trailing asterisk)."""
    if not trace:
        return ""
    return " | ".join(
        f"{entry.get('risk', '').lower()}:{entry.get('rule_id', '')}:{entry.get('priority', 0)}"
        f"{'*' if entry.get('winner') else ''}"
        for entry in trace
    )


def estimate_distance(name: str, pixel_width: int) -> float:
    if pixel_width <= 0:
        return 999.0
    width_m = KNOWN_WIDTHS_M.get(name.lower(), DEFAULT_WIDTH_M)
    return round(min(999.0, max(0.1, width_m * FOCAL_LENGTH_PX / pixel_width)), 2)


class PerspectiveDistanceCalibrator:
    """Estimate ground-plane distance from bottom-center image point using a simple perspective model.

    Two modes:
      - reference_points: list of (y_px, distance_m) used to fit/interpolate a monotonic mapping
      - analytic: use camera_height and vertical_fov (radians) with a simple pinhole ground-plane formula

    If both are provided, reference_points take precedence.
    Falls back to width-based estimation via `estimate_distance` when bottom-center is invalid.
    """

    def __init__(
        self,
        image_height: int,
        horizon_y: float | None = None,
        horizon_ratio: float | None = None,
        camera_height_m: float | None = None,
        vertical_fov_deg: float | None = None,
        ref_points: list[tuple[int, float]] | None = None,
        min_distance: float = 0.5,
        max_distance: float = 999.0,
    ) -> None:
        if image_height <= 0:
            raise ValueError("image_height must be positive")
        self.image_height = int(image_height)
        # horizon can be absolute Y or ratio (0..1 from top)
        if horizon_y is None and horizon_ratio is None:
            self.horizon_y = int(self.image_height * 0.45)  # default guess
            self._uncalibrated = True
        elif horizon_y is not None:
            if horizon_y < 0 or horizon_y >= self.image_height:
                raise ValueError("horizon_y out of image range")
            self.horizon_y = int(horizon_y)
            self._uncalibrated = False
        else:
            assert horizon_ratio is not None
            if not (0.0 <= horizon_ratio <= 1.0):
                raise ValueError("horizon_ratio must be between 0 and 1")
            self.horizon_y = int(self.image_height * horizon_ratio)
            self._uncalibrated = False

        self.camera_height_m = camera_height_m
        self.vertical_fov_deg = vertical_fov_deg
        self.ref_points = None
        if ref_points:
            # validate ref points and sort by y (image y increases downwards)
            cleaned = []
            for y_px, dist in ref_points:
                if not (0 <= y_px < self.image_height):
                    raise ValueError(f"ref point y {y_px} out of range")
                if dist <= 0 or not math.isfinite(dist):
                    raise ValueError("ref point distance must be positive and finite")
                cleaned.append((int(y_px), float(dist)))
            cleaned.sort(key=lambda x: x[0])
            # ensure monotonic distances (decreasing y -> increasing distance)
            for i in range(1, len(cleaned)):
                if cleaned[i][1] <= cleaned[i - 1][1]:
                    raise ValueError("reference points must map increasing y to increasing distance")
            self.ref_points = cleaned

        if min_distance <= 0 or max_distance <= 0 or min_distance >= max_distance:
            raise ValueError("invalid min/max distances")
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)

        # per-track method selection to avoid mixing methods
        self._track_method: dict[str, str] = {}

    def _analytic_distance(self, bottom_y: float) -> float:
        # simple pinhole approximation: distance ~= camera_height * focal / (v - v_h)
        if self.camera_height_m is None or self.vertical_fov_deg is None:
            raise RuntimeError("analytic calibration requires camera_height_m and vertical_fov_deg")
        vfov = math.radians(self.vertical_fov_deg)
        focal = (self.image_height / 2.0) / math.tan(vfov / 2.0)
        denom = (bottom_y - self.horizon_y)
        if denom <= 0:
            raise ValueError("point at or above horizon")
        return float(self.camera_height_m * focal / denom)

    def _interpolated_distance(self, bottom_y: float) -> float:
        # piecewise-linear interpolation of reference points; monotonicity guaranteed at input validation
        pts = self.ref_points
        if not pts:
            raise RuntimeError("no reference points available")
        # if below last point, extrapolate linearly
        if bottom_y <= pts[0][0]:
            return pts[0][1]
        if bottom_y >= pts[-1][0]:
            return pts[-1][1]
        # find segment
        for i in range(1, len(pts)):
            y0, d0 = pts[i - 1]
            y1, d1 = pts[i]
            if y0 <= bottom_y <= y1:
                t = (bottom_y - y0) / (y1 - y0) if y1 != y0 else 0.0
                return d0 + t * (d1 - d0)
        return pts[-1][1]

    def distance(self, track_key: str, box: tuple[int, int, int, int], name: str | None = None) -> tuple[float, str]:
        x1, y1, x2, y2 = box
        bottom_y = float(y2)

        method = "unknown"
        dist = None
        # prefer reference points
        if self.ref_points:
            try:
                d = self._interpolated_distance(bottom_y)
                method = "perspective_ref"
                dist = d
            except Exception:
                dist = None

        # analytic fallback
        if dist is None and self.camera_height_m is not None and self.vertical_fov_deg is not None:
            try:
                d = self._analytic_distance(bottom_y)
                method = "perspective_analytic"
                dist = d
            except Exception:
                dist = None

        # width-based fallback
        if dist is None:
            w_px = max(1, int(x2 - x1))
            method = "width_fallback"
            dist = estimate_distance(name or "object", w_px)

        # clamp
        dist = max(self.min_distance, min(self.max_distance, float(dist)))

        # enforce per-track method consistency
        prev = self._track_method.get(track_key)
        if prev is None:
            self._track_method[track_key] = method
        elif prev != method:
            # prefer previous method to avoid mixing: if previous unavailable now, keep previous if possible
            method = prev

        return (round(dist, 2), method)


def default_lane_polygon(width: int, height: int) -> np.ndarray:
    return np.array([
        [int(width * 0.42), int(height * 0.56)],
        [int(width * 0.58), int(height * 0.56)],
        [int(width * 0.88), height - 1],
        [int(width * 0.12), height - 1],
    ], dtype=np.int32)


def detect_lane_polygon(frame: np.ndarray) -> tuple[np.ndarray, bool]:
    """Estimate lane boundaries with Canny + Hough; return fallback corridor if unavailable."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    mask = np.zeros_like(edges)
    roi = np.array([[
        (int(w * 0.06), h - 1), (int(w * 0.42), int(h * 0.55)),
        (int(w * 0.58), int(h * 0.55)), (int(w * 0.94), h - 1)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, [roi], 255)
    cropped = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, threshold=45,
                            minLineLength=max(35, w // 18), maxLineGap=max(40, w // 14))
    left_points: list[tuple[int, int]] = []
    right_points: list[tuple[int, int]] = []
    if lines is not None:
        # OpenCV HoughLinesP normally returns shape (N, 1, 4).
        # Reshape defensively so every iteration always has x1, y1, x2, y2.
        lines = np.asarray(lines, dtype=np.int32).reshape(-1, 4)
        for x1, y1, x2, y2 in lines:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.45:
                continue
            cx = (x1 + x2) / 2
            if slope < 0 and cx < w * 0.55:
                left_points += [(x1, y1), (x2, y2)]
            elif slope > 0 and cx > w * 0.45:
                right_points += [(x1, y1), (x2, y2)]
    if len(left_points) < 4 or len(right_points) < 4:
        return default_lane_polygon(w, h), False

    def fit(points: list[tuple[int, int]]) -> tuple[float, float]:
        """Fit x = a*y + b (x as a function of y, so near-vertical lines are fine)."""
        ys = np.array([p[1] for p in points], dtype=np.float32)
        xs = np.array([p[0] for p in points], dtype=np.float32)
        a, b = np.polyfit(ys, xs, 1)
        return float(a), float(b)

    try:
        a_left, b_left = fit(left_points)
        a_right, b_right = fit(right_points)
    except Exception:
        return default_lane_polygon(w, h), False

    y_bottom = h - 1

    # The two boundaries converge at the vanishing point and cross above it.
    # Sampling at a fixed fraction of the frame height put the polygon top past
    # that crossing, which inverted left/right and forced the fallback on
    # virtually every frame -- so derive the top from the actual intersection.
    denominator = a_left - a_right
    if abs(denominator) < 1e-6:
        return default_lane_polygon(w, h), False
    y_vanishing = (b_right - b_left) / denominator

    # Stay clearly below the vanishing point, and never start above the ROI.
    y_top = int(max(y_vanishing + h * 0.06, h * 0.56))
    if y_top >= y_bottom:
        return default_lane_polygon(w, h), False

    lx_top = int(a_left * y_top + b_left)
    lx_bottom = int(a_left * y_bottom + b_left)
    rx_top = int(a_right * y_top + b_right)
    rx_bottom = int(a_right * y_bottom + b_right)
    if lx_top >= rx_top or lx_bottom >= rx_bottom:
        return default_lane_polygon(w, h), False

    # Reject implausible corridors (a sliver, or nearly the whole frame).
    bottom_width = rx_bottom - lx_bottom
    if not (w * 0.15 <= bottom_width <= w * 1.6):
        return default_lane_polygon(w, h), False

    def clamp_x(x: float) -> int:
        return int(np.clip(x, -w, 2 * w))
    polygon = np.array([
        [clamp_x(lx_top), y_top], [clamp_x(rx_top), y_top],
        [clamp_x(rx_bottom), y_bottom], [clamp_x(lx_bottom), y_bottom],
    ], dtype=np.int32)
    return polygon, True



class BoundingBoxKalmanFilter:
    """Kalman filter for a bounding box tracking cx,cy,width,height and velocities."""

    def __init__(self, cx: float, cy: float, width: float, height: float, dt: float = 1.0) -> None:
        # state: [cx, cy, w, h, vx, vy, vw, vh]
        self.kf = cv2.KalmanFilter(8, 4, 0)
        # transition matrix will be updated with dt in predict
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        # measurement matrix maps state to [cx,cy,w,h]
        H = np.zeros((4, 8), dtype=np.float32)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        H[3, 3] = 1.0
        self.kf.measurementMatrix = H

        # process noise covariance (small for stable objects)
        q = np.eye(8, dtype=np.float32) * 1e-4
        q[4:, 4:] *= 1e-3
        self.kf.processNoiseCov = q

        # measurement noise covariance (trust measurements reasonably)
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-2

        # posterior error covariance (initial uncertainty)
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 1e-2

        # initial state
        state = np.zeros((8, 1), dtype=np.float32)
        state[0, 0] = float(cx)
        state[1, 0] = float(cy)
        state[2, 0] = float(width)
        state[3, 0] = float(height)
        self.kf.statePost = state

    def predict(self, dt: float) -> np.ndarray:
        # update transition matrix to include dt for position = pos + vel*dt
        A = np.eye(8, dtype=np.float32)
        A[0, 4] = dt
        A[1, 5] = dt
        A[2, 6] = dt
        A[3, 7] = dt
        self.kf.transitionMatrix = A
        pred = self.kf.predict()
        return pred.reshape(8)

    def correct(self, cx: float, cy: float, width: float, height: float) -> np.ndarray:
        meas = np.array([[float(cx)], [float(cy)], [float(width)], [float(height)]], dtype=np.float32)
        corrected = self.kf.correct(meas)
        return corrected.reshape(8)

    def state(self) -> np.ndarray:
        return self.kf.statePost.reshape(8)


class DetectionSmoother:
    """Per-track Kalman filters for smoothing bounding boxes and distance smoothing."""

    def __init__(self) -> None:
        self._filters: dict[str, BoundingBoxKalmanFilter] = {}
        self.distance_state: dict[str, float] = {}
        self.box_state: dict[str, tuple[float, float, float, float]] = {}
        self.last_seen: dict[str, int] = {}
        self.hits: dict[str, int] = {}
        self.track_meta: dict[str, tuple[str, str]] = {}
        self.cycle = 0
        self.label_side: dict[str, str] = {}
        # diagnostics
        self.deleted_tracks = 0
        self.rejected_associations = 0

    def smooth(self, detections: list[Detection], frame_shape: tuple[int, int], dt: float) -> list[Detection]:
        """Correct each detection's filter and return boxes for the current frame.

        Only objects detected in THIS frame are returned, so a prediction-only
        box is never produced. Road damage is handled differently from
        vehicles: it is stationary in the world but sweeps through image space
        as the camera advances, so constant-velocity prediction diverges from it
        quickly and its measured box is trusted directly.

        frame_shape: (height, width)
        dt: elapsed seconds between updates (e.g., 1/source_fps)
        """
        self.cycle += 1
        h, w = frame_shape
        max_jump = MAX_TRACK_JUMP_RATIO * math.hypot(w, h)

        # Structurally invalid boxes are dropped before anything else: they must
        # not create or update a track, seed a Kalman filter, or reach the
        # renderer. Validation happens on the raw coordinates, before clipping.
        detections = [d for d in detections if is_valid_bbox(d.box, (h, w))]

        # Only advance filters that were updated on the previous cycle. A filter
        # that already missed a frame would otherwise keep integrating a stale
        # velocity, and the accumulated drift is what dragged a freshly detected
        # box away from its object.
        for key, kf in list(self._filters.items()):
            if self.cycle - self.last_seen.get(key, -1) <= 1:
                try:
                    kf.predict(dt)
                except Exception:
                    pass

        used_keys: set[str] = set()
        for d in sorted(detections, key=lambda item: item.confidence, reverse=True):
            key = d.track_key
            is_damage = d.name.lower() in ROAD_DAMAGE_CLASSES
            forget_after = ROAD_DAMAGE_FORGET_AFTER if is_damage else TRACK_FORGET_AFTER

            # ByteTrack IDs can be lost when a detector skips a frame. Reattach
            # a new detector ID to the nearest live track before creating a new
            # filter. This prevents tiny gaps from resetting box and TTC state.
            if key not in self._filters:
                best_key = None
                best_score = -1.0
                x1, y1, x2, y2 = d.box
                dcx, dcy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                darea = max(1.0, float((x2 - x1) * (y2 - y1)))
                for candidate, kf in self._filters.items():
                    if candidate in used_keys or self.cycle - self.last_seen.get(candidate, -10**9) > forget_after:
                        continue
                    if self.track_meta.get(candidate) != (d.source, d.name):
                        continue
                    try:
                        st = kf.state()
                        pcx, pcy, pw, ph = map(float, st[:4])
                        px1, py1, px2, py2 = pcx - pw / 2, pcy - ph / 2, pcx + pw / 2, pcy + ph / 2
                        ix1, iy1, ix2, iy2 = max(x1, px1), max(y1, py1), min(x2, px2), min(y2, py2)
                        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                        union = darea + max(1.0, pw * ph) - inter
                        iou = inter / union
                        centre = math.hypot(dcx - pcx, dcy - pcy) / max(1.0, math.hypot(w, h))
                        size_ratio = max(darea, pw * ph) / max(1.0, min(darea, pw * ph))
                        score = iou - centre * 0.8 if size_ratio <= 2.4 else -1.0
                        if (iou >= 0.12 or centre <= 0.055) and score > best_score:
                            best_key, best_score = candidate, score
                    except Exception:
                        continue
                if best_key is not None:
                    key = best_key
                    d.track_key = key
            used_keys.add(key)
            self.track_meta.setdefault(key, (d.source, d.name))

            x1, y1, x2, y2 = d.box
            # clamp to the frame; a detector box may extend past the edge
            cx1 = max(0, min(int(x1), w - 1))
            cy1 = max(0, min(int(y1), h - 1))
            cx2 = max(0, min(int(x2), w - 1))
            cy2 = max(0, min(int(y2), h - 1))
            measured_box = (cx1, cy1, cx2, cy2) if (cx2 > cx1 and cy2 > cy1) else d.box
            meas_w = max(1.0, float(x2 - x1))
            meas_h = max(1.0, float(y2 - y1))
            meas_cx = float(x1) + meas_w / 2.0
            meas_cy = float(y1) + meas_h / 2.0

            gap = self.cycle - self.last_seen.get(key, -10 ** 9)
            restart = key not in self._filters or gap > forget_after

            # Association gate: a filter whose state sits implausibly far from
            # this measurement belongs to a different object, so start a new
            # track instead of teleporting the existing box across the road.
            if not restart:
                try:
                    st = self._filters[key].state()
                    if math.hypot(meas_cx - float(st[0]), meas_cy - float(st[1])) > max_jump:
                        self.rejected_associations += 1
                        # Keep the identity and history, but let the render
                        # gate below reject this one-frame innovation.
                except Exception:
                    pass

            if key not in self.label_side:
                # Prefer a label above the box; fall back to below only when the
                # box starts too close to the top edge. Decided once per track.
                self.label_side[key] = "top" if measured_box[1] >= LABEL_MIN_TOP_ROOM else "bottom"
            d.label_side = self.label_side[key]

            if restart:
                try:
                    self._filters[key] = BoundingBoxKalmanFilter(meas_cx, meas_cy, meas_w, meas_h, dt)
                except Exception:
                    self._filters.pop(key, None)
                self.distance_state[key] = d.distance_m
                self.box_state[key] = (meas_cx, meas_cy, meas_w, meas_h)
                self.hits[key] = 1
                self.last_seen[key] = self.cycle
                d.track_hits = 1
                d.box = measured_box  # trust the fresh detection outright
                d.tracking_source = "yolo"
                continue

            try:
                self._filters[key].correct(meas_cx, meas_cy, meas_w, meas_h)
            except Exception:
                pass

            # Distance smoothing; road damage is closing fast in image space so
            # it follows the measurement more closely.
            alpha = ROAD_DAMAGE_DISTANCE_ALPHA if is_damage else DISTANCE_SMOOTH_ALPHA
            previous_distance = self.distance_state.get(key)
            smoothed_distance = (
                d.distance_m if previous_distance is None
                else alpha * d.distance_m + (1.0 - alpha) * previous_distance
            )
            self.distance_state[key] = smoothed_distance
            d.distance_m = round(smoothed_distance, 2)
            self.last_seen[key] = self.cycle
            self.hits[key] = self.hits.get(key, 1) + 1
            d.track_hits = self.hits[key]

            try:
                previous = self.box_state.get(key)
                predicted = self._filters[key].state()
                prior_cx, prior_cy = float(predicted[0]), float(predicted[1])
                innovation = math.hypot(meas_cx - prior_cx, meas_cy - prior_cy)
                innovation_gate = max(0.12 * math.hypot(w, h), 2.0 * max(meas_w, meas_h))
                if innovation > innovation_gate and previous is not None:
                    # A single implausible measurement must not teleport the
                    # rendered box. The current detection still exists in the
                    # returned list, but its displayed geometry stays trusted.
                    rendered = previous
                    d.tracking_source = "kalman"
                else:
                    base = previous or (prior_cx, prior_cy, meas_w, meas_h)
                    alpha = BOX_SMOOTH_ALPHA
                    rendered = tuple(
                        alpha * current + (1.0 - alpha) * old
                        for current, old in zip((meas_cx, meas_cy, meas_w, meas_h), base)
                    )
                    d.tracking_source = "kalman"
                rcx, rcy, rw, rh = rendered
                render_box = (
                    max(0, min(int(round(rcx - rw / 2)), w - 1)),
                    max(0, min(int(round(rcy - rh / 2)), h - 1)),
                    max(0, min(int(round(rcx + rw / 2)), w - 1)),
                    max(0, min(int(round(rcy + rh / 2)), h - 1)),
                )
                if render_box[2] <= render_box[0] or render_box[3] <= render_box[1]:
                    render_box = measured_box
                self.box_state[key] = rendered
                d.box = render_box
            except Exception:
                d.box = measured_box
                d.tracking_source = "yolo"

        # Remove stale filters using the class-specific missing-frame budget.
        expired = [k for k, last in self.last_seen.items()
                   if self.cycle - last >= (
                       ROAD_DAMAGE_FORGET_AFTER
                       if self.track_meta.get(k, ("", ""))[1].lower() in ROAD_DAMAGE_CLASSES
                       else TRACK_MAX_AGE
                   )]
        for k in expired:
            self._filters.pop(k, None)
            self.distance_state.pop(k, None)
            self.box_state.pop(k, None)
            self.hits.pop(k, None)
            self.track_meta.pop(k, None)
            self.last_seen.pop(k, None)
            self.label_side.pop(k, None)
            self.deleted_tracks += 1

        return detections


class LaneTracker:
    """Temporal lane state.

    The visible shaking was not Hough noise: the overlay was flipping between a
    genuinely detected lane and the hardcoded fallback corridor (measured at 15
    flips per 300 frames on road_video_1 and 30 on road_video_3). Those are two
    different shapes, so every flip moved the overlay a long way -- raw vertex
    shifts reached 130px -- and an EMA cannot hide a model switch.

    This holds one lane estimate across frames, rejects implausible jumps,
    coasts through short detection gaps, and reports whether the estimate is
    trustworthy enough to draw. Risk reasoning always receives a polygon so
    in-lane semantics are unchanged; only the overlay is hidden when the lane
    is unreliable.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.fallback = default_lane_polygon(width, height)
        self.polygon: np.ndarray | None = None
        self.confidence = 0.0
        self.missed = 0
        self.last_valid_frame = -1
        self.rejected = 0

    @staticmethod
    def _centre_and_width(polygon: np.ndarray) -> tuple[float, float]:
        bottom = sorted(polygon.tolist(), key=lambda p: -p[1])[:2]
        xs = sorted(p[0] for p in bottom)
        return (xs[0] + xs[1]) / 2.0, float(xs[1] - xs[0])

    def _plausible(self, candidate: np.ndarray) -> bool:
        """Reject a candidate that moved or resized more than a real lane can
        between two frames."""
        if self.polygon is None:
            return True
        new_c, new_w = self._centre_and_width(candidate)
        old_c, old_w = self._centre_and_width(self.polygon)
        if abs(new_c - old_c) > LANE_MAX_CENTRE_SHIFT_RATIO * self.width:
            return False
        if old_w > 1 and not (1 / LANE_MAX_WIDTH_RATIO <= new_w / old_w <= LANE_MAX_WIDTH_RATIO):
            return False
        # per-vertex: reject if any vertex jumped more than 8 % of frame width
        max_vtx = 0.08 * self.width
        diffs = np.abs(candidate.astype(np.float32) - self.polygon.astype(np.float32))
        if np.any(diffs[:, 0] > max_vtx):
            return False
        return True

    def _clamp_vertices(self, candidate: np.ndarray) -> np.ndarray:
        """Soft-clamp each vertex toward the previous polygon so that small
        residual jitter is absorbed without rejecting the entire frame."""
        if self.polygon is None:
            return candidate
        max_vtx = 0.08 * self.width
        diff = candidate.astype(np.float32) - self.polygon.astype(np.float32)
        clamped = np.where(
            np.abs(diff) > max_vtx,
            self.polygon.astype(np.float32) + np.sign(diff) * max_vtx,
            candidate.astype(np.float32),
        )
        return np.rint(clamped).astype(np.int32)

    def update(self, raw_polygon: np.ndarray, detected: bool,
               frame_no: int) -> tuple[np.ndarray, bool, float]:
        """Returns (polygon_for_reasoning, draw_overlay, confidence)."""
        accepted = detected and self._plausible(raw_polygon)
        if detected and not accepted:
            self.rejected += 1

        if accepted:
            if self.polygon is None:
                self.polygon = raw_polygon.astype(np.int32)
            else:
                blended = (LANE_SMOOTH_ALPHA * raw_polygon.astype(np.float32)
                           + (1.0 - LANE_SMOOTH_ALPHA) * self.polygon.astype(np.float32))
                self.polygon = np.rint(blended).astype(np.int32)
            self.missed = 0
            self.last_valid_frame = frame_no
            self.confidence = min(1.0, self.confidence + 0.34)
        else:
            self.missed += 1
            self.confidence = max(0.0, self.confidence - 0.25)

        # Coast briefly through gaps, then stop drawing rather than let a stale
        # or fallback shape drift across the road.
        if self.polygon is not None and self.missed <= MAX_LANE_MISSED_FRAMES:
            return self.polygon, self.confidence >= LANE_MIN_DRAW_CONFIDENCE, self.confidence

        if self.missed > MAX_LANE_MISSED_FRAMES:
            self.polygon = None
        return self.fallback, False, self.confidence


def smooth_lane_polygon(
    current: np.ndarray,
    previous: np.ndarray | None,
    detected: bool,
) -> np.ndarray:
    """Smooth lane polygon movement to avoid visible left/right shaking."""
    if previous is None or previous.shape != current.shape:
        return current.astype(np.int32)

    alpha = LANE_SMOOTH_ALPHA if detected else min(0.12, LANE_SMOOTH_ALPHA)

    blended = (
        alpha * current.astype(np.float32)
        + (1.0 - alpha) * previous.astype(np.float32)
    )
    return np.rint(blended).astype(np.int32)

def lane_overlap(box: tuple[int, int, int, int], polygon: np.ndarray, shape: tuple[int, int]) -> float:
    h, w = shape
    mask_lane = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_lane, [polygon], 1)
    x1, y1, x2, y2 = box
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    # Bottom third is most relevant for road contact.
    contact_y = int(y1 + (y2 - y1) * 0.65)
    region = mask_lane[contact_y:y2, x1:x2]
    return float(region.mean()) if region.size else 0.0


def box_iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (a.x2 - a.x1) * (a.y2 - a.y1))
    area_b = max(1, (b.x2 - b.x1) * (b.y2 - b.y1))
    return inter / max(1, area_a + area_b - inter)


def deduplicate(items: list[Detection]) -> list[Detection]:
    # prefer specialized models (best.pt, any extra model) over the common
    # COCO model when their boxes overlap
    def score_key(d: Detection):
        source_priority = 0 if d.source == "yolo11n" else 1
        return (source_priority, d.confidence)

    kept: list[Detection] = []
    for d in sorted(items, key=score_key, reverse=True):
        if any(d.name == k.name and box_iou(d, k) >= 0.65 for k in kept):
            continue
        kept.append(d)
    return kept


def visible_tracks(items: list[Detection], confirmed_only: bool = False) -> list[Detection]:
    """Return one current detection per track, never prediction-only ghosts."""
    best: dict[str, Detection] = {}
    for d in items:
        if confirmed_only and d.track_hits < TRACK_MIN_HITS:
            continue
        current = best.get(d.track_key)
        if current is None or (LEVEL_PRIORITY[d.risk], d.confidence) > (LEVEL_PRIORITY[current.risk], current.confidence):
            best[d.track_key] = d
    return list(best.values())


def is_ego_vehicle(d: Detection, frame_h: int, frame_w: int) -> bool:
    """Return True if *d* is almost certainly the camera car's own hood/bumper.

    The ego vehicle appears as a large vehicle box anchored at the
    bottom-centre of the frame. All criteria are required:

    1. Box width ≥ ``EGO_MIN_BOX_WIDTH_RATIO`` of the frame width.
    2. Box top edge below ``EGO_TOP_MIN_RATIO`` of the frame height.
    3. Box bottom is anchored within ``EGO_MIN_BOTTOM_RATIO`` of frame height.
    4. ≥ ``EGO_OVERLAP_MIN_RATIO`` of the box area overlaps a narrow
       centred strip in the bottom 30 % of the frame (the ego ROI).

    A non-vehicle or a real nearby vehicle that merely enters the bottom-centre
    region therefore remains a valid detection.
    """
    if d.name.lower() not in VEHICLE_CLASSES or frame_h <= 0 or frame_w <= 0:
        return False
    x1, y1, x2, y2 = d.box
    bw = x2 - x1
    bh = y2 - y1
    if bw / frame_w < EGO_MIN_WIDTH_RATIO or y1 / frame_h < EGO_MIN_TOP_RATIO:
        return False
    if y2 / frame_h < EGO_MIN_BOTTOM_RATIO:
        return False
    roi_x1 = int(EGO_ROI_X1 * frame_w)
    roi_x2 = int(EGO_ROI_X2 * frame_w)
    roi_y1 = int(EGO_ROI_Y1 * frame_h)
    roi_y2 = frame_h
    centre_x = (x1 + x2) / 2.0
    ox1 = max(x1, roi_x1)
    oy1 = max(y1, roi_y1)
    ox2 = min(x2, roi_x2)
    oy2 = min(y2, roi_y2)
    if ox1 >= ox2 or oy1 >= oy2:
        return False
    overlap_area = (ox2 - ox1) * (oy2 - oy1)
    box_area = max(1, bw * bh)
    return overlap_area / box_area >= EGO_OVERLAP_MIN_RATIO


def select_device(requested: str | None = None) -> str:
    """Pick the inference device: explicit request, else CUDA, else Apple MPS,
    else CPU. Verified with a real forward pass so an unusable backend falls
    back instead of failing mid-run."""
    if requested:
        return requested
    try:
        import torch
    except Exception:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        if torch.backends.mps.is_available():
            # Confirm a real tensor operation and model forward pass work.
            probe = torch.nn.Conv2d(3, 4, kernel_size=3).to("mps")
            probe(torch.zeros((1, 3, 16, 16), device="mps"))
            return "mps"
    except Exception as exc:
        print(f"MPS unavailable ({exc!r}); using CPU")
    return "cpu"


def reset_tracker_state(*models: YOLO | None) -> None:
    """Clear ByteTrack state between videos.

    extract() calls model.track(persist=True), which keeps tracker state on the
    model object. Without this reset the next video inherits the previous
    video's live tracks, so ids collide and stale boxes carry over.
    """
    for model in models:
        if model is None:
            continue
        try:
            predictor = getattr(model, "predictor", None)
            for tracker in getattr(predictor, "trackers", []) or []:
                tracker.reset()
        except Exception as exc:
            print(f"Tracker reset skipped: {exc!r}")


def frame_brightness(frame: np.ndarray) -> float:
    """Mean grayscale brightness (0-255) used to flag low-light/night frames."""
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())


def enhance_low_light(frame: np.ndarray) -> np.ndarray:
    """Boost contrast on dark frames via CLAHE on the LAB lightness channel."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def extract(model: YOLO, frame: np.ndarray, source: str, polygon: np.ndarray, stream_id: str, calibrator: PerspectiveDistanceCalibrator | None = None) -> list[Detection]:
    h, w = frame.shape[:2]
    # per-model confidence
    is_damage_model = source != "yolo11n"
    conf = COMMON_MODEL_CONFIDENCE if not is_damage_model else CUSTOM_MODEL_CONFIDENCE
    imgsz = ROAD_DAMAGE_IMAGE_SIZE if is_damage_model else IMAGE_SIZE
    infer_kwargs: dict[str, Any] = dict(conf=conf, iou=IOU, imgsz=imgsz, verbose=False)
    if INFERENCE_DEVICE:
        infer_kwargs["device"] = INFERENCE_DEVICE
    # Track-capable YOLO models must use ByteTrack. This preserves IDs and lets
    # the smoother associate detections across frames; predict() is only a
    # compatibility fallback for detector stubs without track().
    results: Any = model.track(
        frame, persist=True, tracker=TRACKER, **infer_kwargs
    ) if hasattr(model, "track") else model.predict(frame, **infer_kwargs)
    output: list[Detection] = []
    # common model class whitelist
    COMMON_FILTER = {"person","bicycle","car","motorcycle","bus","truck","traffic light","stop sign","dog","cat","horse","cow","sheep"}
    fallback_index = 0
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0].item())
            name = str(model.names[cls_id]).strip().lower()
            # filter common model to road-relevant classes
            if source == "yolo11n" and name not in COMMON_FILTER:
                continue
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # clip measured box to frame boundaries
            x1 = max(0, x1)
            x2 = min(w - 1, x2)
            y1 = max(0, y1)
            y2 = min(h - 1, y2)
            # reject structurally invalid boxes at the source
            if not is_valid_bbox((x1, y1, x2, y2), (h, w)):
                continue
            # This is deliberately before IDs, distance, TTC, and risk fields
            # are created. A resized inference frame is already represented by
            # (h, w), so the normalized ROI remains resolution independent.
            probe = Detection(
                name=name, confidence=confidence, box=(x1, y1, x2, y2),
                source=source, track_key="", distance_m=0.0,
                in_lane=False, lane_overlap=0.0, box_height_ratio=0.0,
            )
            if is_ego_vehicle(probe, h, w):
                continue
            tid = int(box.id[0].item()) if box.id is not None else -1
            overlap = lane_overlap((x1, y1, x2, y2), polygon, (h, w))
            # A tracker id identifies the object on its own. Only fall back to a
            # coarse spatial grid when the tracker gave us nothing -- mixing the
            # grid into a tracked key would change identity every time the box
            # moved 40px, resetting its Kalman filter, distance smoothing and
            # TTC history each time (boxes appeared to jump).
            if tid >= 0:
                key = f"{stream_id}:{source}:{name}:{tid}"
            else:
                # A grid is useful as an association hint, but cannot be an
                # identity: two same-class detections may share one cell.
                key = f"{stream_id}:{source}:{name}:tmp{fallback_index}"
                fallback_index += 1
            # compute distance using calibrator if available
            if calibrator is not None:
                try:
                    dist, method = calibrator.distance(key, (x1, y1, x2, y2), name)
                except Exception:
                    dist = estimate_distance(name, x2 - x1)
                    method = "width_fallback"
            else:
                dist = estimate_distance(name, x2 - x1)
                method = "width_fallback"

            d = Detection(
                name=name, confidence=confidence, box=(x1, y1, x2, y2), source=source,
                track_key=key, distance_m=dist, distance_method=method,
                in_lane=overlap >= 0.18, lane_overlap=overlap,
                box_height_ratio=(y2 - y1) / max(1, h),
            )
            # measured box used for drawing
            d.measured_box = (x1, y1, x2, y2)
            # ensure visible box uses measured box
            d.box = d.measured_box
            output.append(d)
    return output


def update_ttc(d: Detection, history: dict[str, deque[tuple[float, float]]], video_time: float,
               ttc_state: dict[str, float] | None = None) -> None:
    q = history[d.track_key]
    q.append((video_time, d.distance_m))
    while len(q) > 8:
        q.popleft()
    if len(q) < 3:
        return
    t0, dist0 = q[0]
    t1, dist1 = q[-1]
    dt = t1 - t0
    if dt <= 0.08:
        return
    closing = (dist0 - dist1) / dt
    # Reject unstable values caused by box jitter.
    if 0.15 <= closing <= 45.0:
        d.closing_speed_mps = closing
        raw_ttc = d.distance_m / closing
        if ttc_state is not None:
            previous = ttc_state.get(d.track_key)
            d.ttc_s = raw_ttc if previous is None else TTC_SMOOTH_ALPHA * raw_ttc + (1.0 - TTC_SMOOTH_ALPHA) * previous
            ttc_state[d.track_key] = d.ttc_s
        else:
            d.ttc_s = raw_ttc


def vis_scale(frame_w: int, frame_h: int) -> float:
    """Visualization scale factor relative to 1280x720 reference resolution.

    All overlay sizes are multiplied by this value so text, boxes, and panels
    remain proportional regardless of the actual frame dimensions.  The result
    is clamped to [VIS_SCALE_MIN, VIS_SCALE_MAX] to prevent excessively large
    or microscopic overlays on extreme resolutions.
    """
    return max(VIS_SCALE_MIN, min(VIS_SCALE_MAX,
              min(frame_w / VIS_REF_W, frame_h / VIS_REF_H)))


def draw_lane(frame: np.ndarray, polygon: np.ndarray, detected: bool,
              show_fallback: bool = False) -> None:
    """Draw the current lane corridor supplied by the lane-tracking stage.

    The tracker supplies its stable perspective fallback during short detector
    gaps. Keep that corridor visible so the processed camera feed retains lane
    guidance instead of silently dropping the lane layer.
    """
    if polygon is None or len(polygon) < 4 or (not detected and not show_fallback):
        return
    vs = vis_scale(frame.shape[1], frame.shape[0])
    overlay = frame.copy()
    color = (255, 220, 0)
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, 0.10 if not detected else 0.14, frame,
                    0.90 if not detected else 0.86, 0, frame)
    cv2.polylines(frame, [polygon], True, color, max(2, int(3 * vs)), cv2.LINE_AA)
    if detected:
        cv2.putText(frame, "LANE DETECTED", tuple(polygon[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48 * vs, color,
                    max(1, int(2 * vs)), cv2.LINE_AA)


def panel_height(frame_height: int, frame_width: int = 0) -> int:
    """Height of the status panel, scaled to frame dimensions.

    Designed for 142 px at 720p.  The panel never exceeds half the frame.
    """
    if frame_width <= 0:
        frame_width = int(frame_height * 16 / 9)
    vs = vis_scale(frame_width, frame_height)
    return int(max(40, min(int(142 * vs), frame_height // 2)))


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def draw_detection(frame: np.ndarray, d: Detection,
                   placed_labels: list[tuple[int, int, int, int]] | None = None,
                   top_margin: int = 0) -> None:
    h, w = frame.shape[:2]
    # Same gate as the tracker; nothing is drawn for an invalid box -- no
    # rectangle, no label plate, no text.
    if not is_valid_bbox(d.box, (h, w)):
        return
    if not (0 <= d.x1 < d.x2 <= w - 1 and 0 <= d.y1 < d.y2 <= h - 1):
        return
    vs = vis_scale(w, h)
    color = LEVEL_COLORS[d.risk]
    # Keep the reference's crisp two-pixel outline even on small source clips.
    box_thickness = 3
    cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, box_thickness)
    # Choose a compact variant that fits the final box when possible. Never
    # truncate the class name: tiny objects fall back to `CLASS CONF%`.
    name = d.name.replace("_", " ").upper()
    track_id = d.track_key.rsplit(":", 1)[-1]
    track_suffix = f" #{track_id}" if track_id.isdigit() else ""
    confidence = f"{d.confidence:.0%}"
    distance = "--" if not math.isfinite(d.distance_m) else f"{d.distance_m:.1f}m"
    ttc = "--" if not math.isfinite(d.ttc_s) else f"{d.ttc_s:.1f}s"
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.45 * vs, max(1, int(round(1.5 * vs)))
    pad_x = max(3, int(3 * vs))
    pad_y = max(2, int(2 * vs))
    max_text_w = max(1, d.x2 - d.x1 - pad_x * 2)
    label_options = [
        f"{name}{track_suffix} {confidence} | {distance} | TTC:{ttc} | {d.risk}",
        f"{name}{track_suffix} {confidence} | {distance} | {d.risk}",
        f"{name}{track_suffix} {confidence} | {d.risk}",
        f"{name} {confidence}",
    ]
    label = next(
        (text for text in label_options
         if cv2.getTextSize(text, font, scale, thickness)[0][0] <= max_text_w),
        label_options[-1],
    )
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    label_w, label_h = tw + pad_x * 2, th + baseline + pad_y * 2

    def candidate(x: int, y: int) -> tuple[int, int, int, int]:
        x = max(0, min(x, w - label_w))
        y = max(top_margin, min(y, h - label_h))
        return (x, y, x + label_w, y + label_h)

    if d.y1 >= top_margin + label_h:
        candidates = [
            candidate(d.x1, d.y1 - label_h),
            candidate(d.x2 - label_w, d.y1 - label_h),
            candidate(d.x1, d.y1),
        ]
    else:
        # No room above: attach the plate to the inside of the final box.
        candidates = [
            candidate(d.x1, d.y1),
            candidate(d.x2 - label_w, d.y1),
        ]
    chosen = candidates[0]
    if placed_labels is not None:
        for option in candidates:
            if not any(_rects_overlap(option, other) for other in placed_labels):
                chosen = option
                break
        placed_labels.append(chosen)

    lx, ly, _, _ = chosen
    plate = frame[ly:ly + label_h, lx:lx + label_w]
    if plate.size:
        dark = np.zeros_like(plate)
        cv2.addWeighted(dark, 0.65, plate, 0.35, 0, plate)
    cv2.rectangle(frame, (lx, ly), (lx + label_w - 1, ly + label_h - 1),
                  color, 1, cv2.LINE_AA)
    cv2.putText(frame, label, (lx + pad_x, ly + label_h - pad_y - baseline),
                font, scale, color, thickness, cv2.LINE_AA)


def draw_panel(frame: np.ndarray, level: str, action: str, count: int, fps: float, ttc: float, rule_id: str = "", explanation: str = "") -> None:
    h, w = frame.shape[:2]
    vs = vis_scale(w, h)
    ph = panel_height(h, w)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, ph), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    x0 = int(18 * vs)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"STATUS: {level}", (x0, int(34 * vs)), font, 0.88 * vs,
                LEVEL_COLORS[level], max(1, int(2 * vs)), cv2.LINE_AA)
    cv2.putText(frame, f"ACTION: {action[:90]}", (x0, int(64 * vs)), font, 0.56 * vs,
                (255, 255, 255), max(1, int(2 * vs)), cv2.LINE_AA)
    # One line per row. Each row gets its own baseline, scaled to frame size.
    if rule_id and ph >= int(92 * vs):
        cv2.putText(frame, f"Rule: {rule_id}"[:90], (x0, int(88 * vs)), font, 0.45 * vs,
                    (200, 200, 200), max(1, int(vs)), cv2.LINE_AA)
    if explanation and ph >= int(112 * vs):
        cv2.putText(frame, f"{explanation[:76]}", (x0, int(108 * vs)), font, 0.42 * vs,
                    (200, 200, 200), max(1, int(vs)), cv2.LINE_AA)
    ttc_text = "--" if not math.isfinite(ttc) else f"{ttc:.1f} sec"
    if ph >= int(136 * vs):
        cv2.putText(frame, f"Objects: {count} | FPS: {fps:.1f} | Minimum TTC: {ttc_text}",
                    (x0, int(132 * vs)), font, 0.52 * vs, (255, 255, 255), max(1, int(2 * vs)), cv2.LINE_AA)


def _scale_detection(d: Detection, sx: float, sy: float) -> Detection:
    """Return a shallow copy of a detection with box coordinates scaled by sx/sy.

    Used to map detection coordinates from processing resolution to source
    resolution when writing the output video at original frame dimensions.
    Only the geometric fields are transformed; risk/classification/track fields
    are passed through unchanged.
    """
    if sx == 1.0 and sy == 1.0:
        return d
    def _s(v: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return (int(v[0] * sx), int(v[1] * sy), int(v[2] * sx), int(v[3] * sy))
    return Detection(
        name=d.name, confidence=d.confidence,
        box=_s(d.box), source=d.source, track_key=d.track_key,
        distance_m=d.distance_m, in_lane=d.in_lane,
        lane_overlap=d.lane_overlap, box_height_ratio=d.box_height_ratio,
        closing_speed_mps=d.closing_speed_mps, ttc_s=d.ttc_s,
        risk=d.risk, action=d.action, distance_method=d.distance_method,
        measured_box=_s(d.measured_box),
        rule_id=d.rule_id, explanation=d.explanation,
        decision_source=d.decision_source, rule_priority=d.rule_priority,
        decision_trace=d.decision_trace, tracking_source=d.tracking_source,
        label_side=d.label_side,
    )


def incident_folder_for(video_name: str) -> Path:
    """Each video gets its own incident folder so screenshots from different
    videos never mix: output/incidents/<video stem>/."""
    return INCIDENT_FOLDER / Path(video_name).stem


def save_incident(frame: np.ndarray, video_name: str, frame_no: int, d: Detection,
                  last_saved: dict[str, float], video_time: float) -> Path | None:
    key = f"{d.track_key}:{d.risk}"
    if video_time - last_saved.get(key, -999.0) < 3.0:
        return None
    last_saved[key] = video_time
    folder = incident_folder_for(video_name)
    folder.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() else "_" for c in d.name)
    # The video stem stays in the filename so an incident is still identifiable
    # if it is moved or shared outside its folder.
    path = folder / f"{Path(video_name).stem}_f{frame_no}_{d.risk}_{safe_name}.jpg"
    cv2.imwrite(str(path), frame)
    return path


def incident_reference(path: Path) -> str:
    """Return a portable public reference, never the local filesystem path."""
    try:
        relative = path.relative_to(INCIDENT_FOLDER)
    except ValueError:
        relative = Path(path.name)
    return str(Path("output/incidents") / relative).replace("\\", "/")


def print_performance_report(video_name: str, stats: RunStats, pipeline_stats: dict,
                             elapsed: float, source_fps: float) -> None:
    """Per-stage timing breakdown, printed when --debug-performance is set."""
    stage_times: dict[str, list[float]] = pipeline_stats.get("stage_times", {})
    print(f"\n{'=' * 58}\nPERFORMANCE  {video_name}   device={INFERENCE_DEVICE}\n{'=' * 58}")
    print(f"{'stage':20s} {'avg ms':>9s} {'p95 ms':>9s} {'max ms':>9s}")
    print("-" * 58)
    total = 0.0
    for name, samples in sorted(stage_times.items(), key=lambda kv: -sum(kv[1])):
        if not samples:
            continue
        ordered = sorted(samples)
        avg = sum(samples) / len(samples) * 1000.0
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] * 1000.0
        total += avg
        print(f"{name:20s} {avg:9.2f} {p95:9.2f} {max(samples) * 1000.0:9.2f}")
    print("-" * 58)
    processing_fps = stats.frames / max(1e-6, elapsed)
    print(f"{'sum of stages':20s} {total:9.2f} ms  -> {1000.0 / max(total, 1e-6):5.1f} FPS ceiling")
    print(f"{'source fps':20s} {source_fps:9.2f}")
    print(f"{'achieved fps':20s} {processing_fps:9.2f}   ({stats.frames} frames in {elapsed:.1f}s)")
    ages = pipeline_stats.get("detection_ages", [])
    lat = pipeline_stats.get("e2e_latency", [])
    if ages:
        print(f"{'avg detection age':20s} {sum(ages)/len(ages):9.2f} frames")
        print(f"{'max detection age':20s} {max(ages):9d} frames")
    if lat:
        ordered = sorted(lat)
        print(f"{'e2e latency avg':20s} {sum(lat)/len(lat)*1000:9.1f} ms")
        print(f"{'e2e latency p95':20s} {ordered[int(len(ordered)*0.95)]*1000:9.1f} ms")
    print(f"{'stale renders':20s} {pipeline_stats.get('stale_frames', 0):9d}")
    print(f"{'dropped frames':20s} {pipeline_stats.get('dropped_frames', 0):9d}")
    print(f"{'max read queue':20s} {pipeline_stats.get('max_read_q', 0):9d}")
    print(f"{'max result queue':20s} {pipeline_stats.get('max_result_q', 0):9d}")
    print("=" * 58 + "\n")


def build_decision_trace_panel(stats: RunStats) -> str:
    """'Prolog Decision Trace' dashboard panel: shows the rule-priority table
    and worked examples where a high-priority rule overrode competing ones."""
    priority_rows = "".join(
        f"<tr><td>{rid}</td><td class='num'>{pri}</td>"
        f"<td class='num'>{stats.triggered_rules.get(rid, 0)}</td>"
        f"<td class='num'>{stats.overridden_rules.get(rid, 0)}</td></tr>"
        for rid, pri in sorted(RULE_PRIORITIES.items(), key=lambda kv: kv[1], reverse=True)
    )

    if not stats.conflict_examples:
        examples_html = (
            "<p class='note'>No CRITICAL rule conflicts were recorded in this run.</p>"
        )
    else:
        blocks = []
        for ex in stats.conflict_examples:
            win = ex["winner"]
            losers = "".join(
                f"<li><span class='pill {lo['risk'].lower()}'>{lo['risk']}</span>"
                f"<code>{lo['rule_id']}</code> &mdash; priority {lo['priority']}</li>"
                for lo in ex["overridden"]
            )
            blocks.append(
                f"<div class='trace'>"
                f"<div class='trace-head'>frame {ex['frame']} &middot; {ex['object']}</div>"
                f"<div class='winner'><span class='pill {win['risk'].lower()}'>{win['risk']}</span>"
                f"<code>{win['rule_id']}</code>"
                f"<span class='prio'>priority {win['priority']}</span></div>"
                f"<div class='expl'>{win['explanation']}</div>"
                f"<div class='others-label'>Overridden ({len(ex['overridden'])})</div>"
                f"<ul class='others'>{losers}</ul>"
                f"</div>"
            )
        examples_html = f"<div class='trace-grid'>{''.join(blocks)}</div>"

    return f"""
    <h2>Prolog Decision Trace</h2>
    <p class='note'>Several rules can be true at once. Prolog collects every
    triggered rule and resolves the conflict by risk level first, then by
    explicit rule priority &mdash; it is not decided by clause order.
    Conflicts resolved this run: <strong>{stats.conflicts}</strong>.</p>
    <table><tr><th>Rule ID</th><th class='num'>Priority</th><th class='num'>Times won</th>
    <th class='num'>Times overridden</th></tr>{priority_rows}</table>
    <h3>Worked examples</h3>
    {examples_html}"""


def create_dashboard(video_name: str, stats: RunStats, elapsed: float, output_path: Path) -> None:
    avg_fps = sum(stats.processing_fps_samples) / max(1, len(stats.processing_fps_samples))
    min_ttc = "N/A" if not math.isfinite(stats.min_ttc) else f"{stats.min_ttc:.2f} seconds"
    top_objects = stats.object_counts.most_common(10)
    rows = "".join(f"<tr><td>{name}</td><td>{count}</td></tr>" for name, count in top_objects) or "<tr><td>None</td><td>0</td></tr>"
    level_cards = "".join(
        f'<div class="card"><h3>{level}</h3><p>{stats.level_counts[level]}</p></div>'
        for level in ["SAFE", "CAUTION", "WARNING", "CRITICAL"]
    )
    # Prolog reasoning summary
    top_rules = stats.triggered_rules.most_common(10)
    prolog_counts = " | ".join(f"{k}: {v}" for k, v in stats.decision_source_counts.items())
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Road Safety Report</title>
<style>body{{font-family:Arial;background:#10141b;color:#eee;margin:0;padding:28px}}h1{{margin-top:0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px}}.card{{background:#1d2530;border-radius:12px;padding:16px}}.card p{{font-size:28px;font-weight:bold;margin:8px 0}}table{{width:100%;border-collapse:collapse;background:#1d2530}}th,td{{padding:10px;border-bottom:1px solid #394555;text-align:left}}.note{{color:#b9c5d3}}
td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
code{{font-family:ui-monospace,Menlo,monospace;font-size:13px}}
.trace-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}}
.trace{{background:#1d2530;border-radius:12px;padding:14px 16px;border-left:4px solid #d03b3b}}
.trace-head{{color:#8b97a6;font-size:12px;margin-bottom:8px}}
.winner{{display:flex;align-items:center;gap:8px;flex-wrap:wrap}}
.prio{{color:#8b97a6;font-size:12px}}
.expl{{color:#b9c5d3;font-size:13px;margin:8px 0 10px}}
.others-label{{color:#8b97a6;font-size:11px;text-transform:uppercase;letter-spacing:.06em}}
ul.others{{list-style:none;margin:6px 0 0;padding:0}}
ul.others li{{display:flex;align-items:center;gap:8px;padding:4px 0;color:#b9c5d3;font-size:12px}}
.pill{{font-size:10px;font-weight:700;padding:2px 7px;border-radius:20px;color:#10141b}}
.pill.critical{{background:#e66767}}.pill.warning{{background:#ec835a}}
.pill.caution{{background:#e0a324}}.pill.safe{{background:#2fbf2f}}</style></head><body>
<h1>AI Road Hazard and Collision Warning Report</h1><p class='note'>Video: {video_name}</p>
<div class='grid'><div class='card'><h3>Frames</h3><p>{stats.frames}</p></div><div class='card'><h3>Detections</h3><p>{stats.total_detections}</p></div><div class='card'><h3>Incidents</h3><p>{stats.incidents}</p></div><div class='card'><h3>Average FPS</h3><p>{avg_fps:.1f}</p></div><div class='card'><h3>Minimum TTC</h3><p style='font-size:18px'>{min_ttc}</p></div><div class='card'><h3>Run Time</h3><p style='font-size:18px'>{elapsed:.1f}s</p></div></div>
<h2>Risk Levels</h2><div class='grid'>{level_cards}</div><h2>Most Detected Objects</h2><table><tr><th>Object</th><th>Count</th></tr>{rows}</table>
    <h2>Prolog Reasoning</h2>
    <p class='note'>Decision counts: {prolog_counts}</p>
    <table><tr><th>Rule ID</th><th>Count</th><th>Example Explanation</th></tr>{''.join(f"<tr><td>{rid}</td><td>{cnt}</td><td>{stats.triggered_rules[rid] and (stats.triggered_rules[rid] and '')}</td></tr>" for rid, cnt in top_rules)}</table>
    {build_decision_trace_panel(stats)}
    <p class='note'>Distance and TTC values are estimates. Calibrate the camera before real-world use.</p></body></html>"""
    output_path.write_text(html, encoding="utf-8")


def process_video(path: Path, common: YOLO, custom: YOLO, expert: PrologRiskEngine,
                  display: bool, voice_enabled: bool, next_video_name: str | None = None,
                  calibrator_config: dict | None = None,
                  drop_frames: bool = False,
                  extractor_callable: Any | None = None,
                  reader_queue_size: int = 8,
                  result_queue_size: int = 8,
                  extra_models: list[tuple[YOLO, str]] | None = None) -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    # Start each video with a clean tracker, otherwise the previous video's
    # tracks persist into this one (model.track uses persist=True).
    reset_tracker_state(common, custom, *[m for m, _ in (extra_models or [])])
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Cannot open {path}")
        cap.release()
        return

    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # 4K input is downscaled before processing. Measured on road_video_4:
    # running both models on native 3840px frames costs 77.5 ms/frame versus
    # 26.3 ms at 1920px for an identical detection count, because ultralytics
    # letterboxes the full-resolution frame on the CPU. It also keeps the
    # exported video a sane size. Smaller sources are untouched.
    scale = min(1.0, MAX_PROCESSING_WIDTH / max(1, source_w))
    needs_resize = scale < 1.0
    w = int(round(source_w * scale)) if needs_resize else source_w
    h = int(round(source_h * scale)) if needs_resize else source_h
    if needs_resize:
        print(f"Downscaling {source_w}x{source_h} -> {w}x{h} for processing")

    vs = vis_scale(w, h)
    print(f"  Input resolution:       {source_w}x{source_h}")
    print(f"  Processing resolution:  {w}x{h}")
    print(f"  Output-video resolution:{source_w}x{source_h}")
    print(f"  Visualization scale:    {vs:.2f}")

    video_out = OUTPUT_FOLDER / f"{path.stem}_advanced.mp4"
    csv_out = OUTPUT_FOLDER / f"{path.stem}_detections.csv"
    json_out = OUTPUT_FOLDER / f"{path.stem}_summary.json"
    dashboard_out = OUTPUT_FOLDER / f"{path.stem}_dashboard.html"

    # H.264 ('avc1') rather than MPEG-4 Part 2 ('mp4v'). QuickTime/AVFoundation
    # decodes mp4v output with heavy block artifacts and tearing, which looks
    # like the boxes are lagging the video even though the frames are correct.
    # Fall back to mp4v only if this build cannot open an H.264 writer.
    # The output video always uses the source resolution; frames are upscaled
    # back from the processing resolution before drawing overlays.
    fourcc = getattr(cv2, "VideoWriter_fourcc")
    writer = cv2.VideoWriter(str(video_out), fourcc(*"avc1"), source_fps, (source_w, source_h))
    # getattr keeps this tolerant of writer stand-ins that don't implement isOpened
    if not getattr(writer, "isOpened", lambda: True)():
        print("H.264 writer unavailable; falling back to mp4v")
        writer = cv2.VideoWriter(str(video_out), fourcc(*"mp4v"), source_fps, (source_w, source_h))

    # Pipeline items
    @dataclass
    class FrameItem:
        frame_no: int
        video_time: float
        frame: np.ndarray
        capture_timestamp: float = 0.0

        @property
        def frame_id(self) -> int:
            return self.frame_no

        @property
        def source_timestamp(self) -> float:
            return self.video_time

    @dataclass
    class ProcessedItem:
        frame_no: int
        video_time: float
        frame: np.ndarray
        detections: list[Detection]
        polygon: np.ndarray
        lane_detected: bool
        level: str
        action: str
        panel_rule_id: str = ""
        panel_explanation: str = ""
        # frame this result's detections were computed from, and when the frame
        # was captured -- lets the renderer assert it is never drawing a
        # detection that belongs to an older frame.
        detection_frame_no: int = 0
        capture_timestamp: float = 0.0
        detection_timestamp: float = 0.0

        @property
        def frame_id(self) -> int:
            return self.frame_no

        @property
        def source_timestamp(self) -> float:
            return self.video_time

    # Initialize resources and shared state
    voice = VoiceAlert(voice_enabled)
    history: dict[str, deque[tuple[float, float]]] = defaultdict(deque)
    ttc_state: dict[str, float] = {}
    last_incident: dict[str, float] = {}
    stats = RunStats()
    smoother = DetectionSmoother()
    methods_used: set[str] = set()

    # calibration
    calibrator = None
    try:
        calibrator = PerspectiveDistanceCalibrator(image_height=h, **(calibrator_config or {}))
    except Exception as exc:
        print(f"Calibration setup failed: {exc!r}; falling back to width-based estimation")
        calibrator = None

    # queues and control
    # Live preview favours low latency over throughput: a deep read queue buffers
    # frames ahead and adds capture->display delay (measured ~270 ms at depth 8).
    # File processing keeps the deeper queue, because every frame must reach the
    # writer -- dropping frames there would corrupt the exported video.
    if display and reader_queue_size > LIVE_READER_QUEUE_SIZE:
        reader_queue_size = LIVE_READER_QUEUE_SIZE
    read_q: "queue.Queue[FrameItem | object]" = queue.Queue(maxsize=reader_queue_size)
    result_q: "queue.Queue[ProcessedItem | object]" = queue.Queue(maxsize=result_queue_size)
    exception_q: "queue.Queue[BaseException]" = queue.Queue()
    stop_event = threading.Event()
    SENTINEL = object()

    # synchronization for shared mutation
    _methods_lock = threading.Lock()
    _rules_lock = threading.Lock()
    _pipeline_lock = threading.Lock()

    # stats for pipeline
    pipeline_stats: dict[str, Any] = {
        "max_read_q": 0,
        "max_result_q": 0,
        "read_put_times": [],
        "inference_times": [],
        "write_times": [],
        "dropped_frames": 0,
        # per-stage timings (seconds), populated when DEBUG_PERFORMANCE is on
        "stage_times": defaultdict(list),
        "detection_ages": [],      # display_frame - detection_frame, per rendered frame
        "e2e_latency": [],         # capture -> displayed, seconds
        "stale_frames": 0,         # renders where the detection was not this frame
    }
    sample_explanations: dict[str, str] = {}

    # reader thread: read frames and enqueue
    def reader():
        try:
            fno = 0
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                if needs_resize:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                fno += 1
                video_time = fno / source_fps
                item = FrameItem(fno, video_time, frame, time.perf_counter())
                # backpressure: block unless drop_frames mode
                if drop_frames:
                    try:
                        read_q.put(item, block=False)
                    except queue.Full:
                        pipeline_stats["dropped_frames"] += 1
                        continue
                else:
                    # block until space or stop
                    while not stop_event.is_set():
                        try:
                            read_q.put(item, timeout=0.2)
                            break
                        except queue.Full:
                            pipeline_stats["max_read_q"] = max(pipeline_stats["max_read_q"], read_q.qsize())
                            continue
            # signal EOF to workers
            try:
                read_q.put(SENTINEL, timeout=1.0)
            except Exception:
                pass
        except Exception as exc:
            try:
                exception_q.put(exc)
            except Exception:
                pass
            stop_event.set()

    # inference/processing worker: perform detection, smoothing, decision
    def worker():
        try:
            # cached detections for frames without fresh detection
            lane_tracker = LaneTracker(w, h)
            while not stop_event.is_set():
                item = read_q.get()
                if item is SENTINEL:
                    # propagate sentinel
                    try:
                        result_q.put(SENTINEL, timeout=1.0)
                    except Exception:
                        pass
                    break

                if not isinstance(item, FrameItem):
                    continue

                start_inf = time.perf_counter()
                frame_no = item.frame_no
                frame = item.frame
                video_time = item.video_time

                # lane detection runs on worker to avoid blocking main
                _t = time.perf_counter()
                raw_polygon, raw_detected = detect_lane_polygon(frame)
                # Temporal lane state: validates the candidate, coasts through
                # short gaps, and reports whether the estimate is good enough to
                # draw. `polygon` always drives in-lane reasoning; `lane_visible`
                # only controls the overlay.
                polygon, lane_visible, lane_conf = lane_tracker.update(
                    raw_polygon, raw_detected, frame_no)
                lane_detected = lane_visible
                if DEBUG_PERFORMANCE:
                    with _pipeline_lock:
                        pipeline_stats["stage_times"]["lane"].append(time.perf_counter() - _t)

                fresh_detection = (
                    frame_no == 1 or DETECTION_INTERVAL <= 1 or frame_no % DETECTION_INTERVAL == 0
                )

                detections: list[Detection] = []
                if fresh_detection:
                    try:
                        if extractor_callable is not None:
                            detections = extractor_callable(frame, polygon, path.stem, calibrator)
                        else:
                            # Night/low-light frames get a contrast boost before
                            # object detection only (not lane-finding or the
                            # drawn/written frame) so a dark clip doesn't lose
                            # detections to the detector, without touching the
                            # frame identity anything else depends on.
                            detect_frame = frame
                            if frame_brightness(frame) < LOW_LIGHT_THRESHOLD:
                                detect_frame = enhance_low_light(frame)
                                stats.low_light_frames += 1
                            parts = [
                                extract(common, detect_frame, "yolo11n", polygon, path.stem, calibrator),
                                extract(custom, detect_frame, "best.pt", polygon, path.stem, calibrator),
                            ]
                            for extra_model, extra_source in (extra_models or []):
                                parts.append(extract(extra_model, detect_frame, extra_source, polygon, path.stem, calibrator))
                            detections = deduplicate([d for part in parts for d in part])
                        if DEBUG_PERFORMANCE:
                            with _pipeline_lock:
                                pipeline_stats["stage_times"]["yolo"].append(time.perf_counter() - start_inf)
                    except Exception as exc:
                        # inference error: propagate and shutdown
                        exception_q.put(exc)
                        stop_event.set()
                        # push sentinel downstream
                        try:
                            result_q.put(SENTINEL, timeout=1.0)
                        except Exception:
                            pass
                        break

                    # suppress the camera car's own hood / bumper before
                    # any tracking, TTC, risk, Prolog, or logging
                    detections = [d for d in detections
                                  if not is_ego_vehicle(d, h, w)]

                    # smoothing and decisions
                    _t = time.perf_counter()
                    detections = smoother.smooth(detections, (h, w), 1.0 / max(1.0, source_fps))
                    # Only current, confirmed tracker results continue through
                    # TTC/risk and rendering. This also collapses duplicate
                    # source/model rows to one physical track per frame.
                    detections = visible_tracks(detections)
                    if DEBUG_PERFORMANCE:
                        with _pipeline_lock:
                            pipeline_stats["stage_times"]["kalman"].append(time.perf_counter() - _t)
                    _t_decide = time.perf_counter()
                    for d in detections:
                        d.lane_overlap = lane_overlap(d.box, polygon, (h, w))
                        d.in_lane = d.lane_overlap >= 0.18
                        d.box_height_ratio = (d.y2 - d.y1) / max(1, h)
                        update_ttc(d, history, video_time, ttc_state)
                        try:
                            res = expert.decide(d)
                        except Exception as exc:
                            # Prolog exceptions are handled inside expert.decide
                            res = fallback_decision(d)

                        # res: (risk, action_display, rule_id, explanation, decision_source)
                        try:
                            d.risk, d.action, d.rule_id, d.explanation, d.decision_source = res
                        except Exception:
                            # be defensive for older fallbacks
                            r = res if isinstance(res, tuple) else (res,)
                            d.risk = str(r[0]) if len(r) > 0 else "SAFE"
                            d.action = str(r[1]) if len(r) > 1 else "ROAD CLEAR"
                            d.rule_id = str(r[2]) if len(r) > 2 else ""
                            d.explanation = str(r[3]) if len(r) > 3 else ""
                            d.decision_source = str(r[4]) if len(r) > 4 else "python_fallback"

                        stats.total_detections += 1
                        stats.object_counts[d.name] += 1
                        stats.level_counts[d.risk] += 1
                        try:
                            with _methods_lock:
                                methods_used.add(d.distance_method)
                        except Exception:
                            pass
                        if math.isfinite(d.ttc_s):
                            stats.min_ttc = min(stats.min_ttc, d.ttc_s)

                        # record rule counts and example explanation
                        try:
                            with _rules_lock:
                                if d.rule_id:
                                    stats.triggered_rules[d.rule_id] += 1
                                    sample_explanations.setdefault(d.rule_id, d.explanation)
                                stats.decision_source_counts[d.decision_source] += 1
                                # rule-conflict bookkeeping: more than one rule
                                # fired, so priority resolution actually decided
                                # something
                                losers = [t for t in d.decision_trace if not t.get("winner")]
                                if losers:
                                    stats.conflicts += 1
                                    for loser in losers:
                                        stats.overridden_rules[loser["rule_id"]] += 1
                                    # keep a few worked examples for the dashboard
                                    if d.risk == "CRITICAL" and len(stats.conflict_examples) < 12:
                                        stats.conflict_examples.append({
                                            "object": d.name,
                                            "frame": frame_no,
                                            "winner": {
                                                "rule_id": d.rule_id,
                                                "risk": d.risk,
                                                "priority": d.rule_priority,
                                                "explanation": d.explanation,
                                            },
                                            "overridden": losers,
                                        })
                        except Exception:
                            pass

                    if DEBUG_PERFORMANCE:
                        with _pipeline_lock:
                            pipeline_stats["stage_times"]["risk+prolog"].append(time.perf_counter() - _t_decide)
                else:
                    # no inference performed on this frame -> no visible detections
                    detections = []

                # determine highest for panel/voice
                highest = max(
                    detections,
                    key=lambda d: (
                        LEVEL_PRIORITY[d.risk],
                        d.in_lane,
                        -d.ttc_s if math.isfinite(d.ttc_s) else -999,
                        d.box_height_ratio,
                    ),
                    default=None,
                )

                level = "SAFE"
                action = "ROAD CLEAR - CONTINUE CAREFULLY"
                if highest is not None:
                    level = highest.risk
                    action = highest.action
                    if fresh_detection and level in {"WARNING", "CRITICAL"}:
                        voice.speak(f"{highest.name}:{level}", action)

                inf_time = time.perf_counter() - start_inf
                with _pipeline_lock:
                    pipeline_stats["inference_times"].append(inf_time)
                    pipeline_stats["max_result_q"] = max(pipeline_stats["max_result_q"], result_q.qsize())

                panel_rule = highest.rule_id if highest is not None else ""
                panel_expl = (highest.explanation[:120] if highest is not None else "")
                processed = ProcessedItem(
                    frame_no=frame_no,
                    video_time=video_time,
                    frame=frame,
                    detections=detections,
                    polygon=polygon,
                    lane_detected=lane_detected,
                    level=level,
                    action=action,
                    panel_rule_id=panel_rule,
                    panel_explanation=panel_expl,
                    detection_frame_no=frame_no,
                    capture_timestamp=getattr(item, "capture_timestamp", 0.0),
                    detection_timestamp=time.perf_counter(),
                )

                # put processed result (may block for backpressure)
                while not stop_event.is_set():
                    try:
                        result_q.put(processed, timeout=0.2)
                        break
                    except queue.Full:
                        with _pipeline_lock:
                            pipeline_stats["max_result_q"] = max(pipeline_stats["max_result_q"], result_q.qsize())
                        continue
        except Exception as exc:
            try:
                exception_q.put(exc)
            except Exception:
                pass
            stop_event.set()

    # start threads
    reader_t = threading.Thread(target=reader, name="reader", daemon=True)
    worker_t = threading.Thread(target=worker, name="worker", daemon=True)
    reader_t.start()
    worker_t.start()

    # main writer loop: consume processed items and output in-order
    next_frame = 1
    buffer: dict[int, ProcessedItem] = {}
    started = time.time()
    fps_started = time.time()
    fps_frames = 0
    live_fps = 0.0
    target_frame_ms = max(1, int(round(1000.0 / max(1.0, source_fps))))

    # open CSV for writing
    try:
        with csv_out.open("w", newline="", encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                "frame", "frame_id", "video_time_s", "source_timestamp", "detection_timestamp", "object", "source", "confidence",
                "distance_m", "distance_method", "closing_speed_mps", "ttc_s", "lane_overlap",
                "x1", "y1", "x2", "y2",
                "in_lane", "risk", "action", "decision_source", "rule_id", "rule_priority",
                 "competing_rules", "decision_trace", "explanation", "incident_image", "track_key"
            ])

            while True:
                # check for exceptions
                try:
                    exc = exception_q.get(block=False)
                except Exception:
                    exc = None
                if exc is not None:
                    # re-raise worker exception
                    stop_event.set()
                    raise exc

                try:
                    item = result_q.get(timeout=0.2)
                except Exception:
                    item = None

                if item is SENTINEL:
                    # no more items incoming; flush any buffered frames in order
                    try:
                        exc = exception_q.get(block=False)
                    except Exception:
                        exc = None
                    if exc is not None:
                        stop_event.set()
                        raise exc
                    # flush available buffered frames in ascending order
                    while next_frame in buffer:
                        p = buffer.pop(next_frame)
                        # reuse existing per-frame output logic by pushing back to buffer then letting loop handle
                        # but since we're already in the writer loop, process directly
                        frame_started = time.perf_counter()
                        stats.frames += 1
                        fps_frames += 1
                        video_time = p.video_time

                        # Upscale to source resolution and map detection
                        # coordinates when the frame was downscaled for processing.
                        if needs_resize:
                            draw_frame = cv2.resize(p.frame, (source_w, source_h),
                                                    interpolation=cv2.INTER_LINEAR)
                            sx = source_w / w
                            sy = source_h / h
                            draw_polygon = (p.polygon.astype(np.float32)
                                            * np.array([sx, sy])).astype(np.int32)
                        else:
                            draw_frame = p.frame
                            sx = sy = 1.0
                            draw_polygon = p.polygon

                        draw_lane(draw_frame, draw_polygon, p.lane_detected,
                                  show_fallback=bool(p.detections))
                        placed_labels: list[tuple[int, int, int, int]] = []
                        for d in sorted(p.detections, key=lambda x: (-LEVEL_PRIORITY[x.risk], x.track_key)):
                            draw_d = _scale_detection(d, sx, sy)
                            draw_detection(draw_frame, draw_d, placed_labels, 0)
                        incident_paths: dict[str, str] = {}
                        if p.frame_no == 1 or DETECTION_INTERVAL <= 1 or p.frame_no % DETECTION_INTERVAL == 0:
                            for d in p.detections:
                                incident = None
                                if d.risk in {"WARNING", "CRITICAL"} and d.in_lane:
                                    incident = save_incident(draw_frame, path.name, p.frame_no, d, last_incident, video_time)
                                    if incident:
                                        stats.incidents += 1
                                        incident_paths[d.track_key] = incident_reference(incident)
                                csv_writer.writerow([
                                     p.frame_no,
                                     p.frame_id,
                                     round(video_time, 3),
                                     round(p.source_timestamp, 3),
                                     round(p.detection_timestamp, 6),
                                    d.name,
                                    d.source,
                                    round(d.confidence, 4),
                                    d.distance_m,
                                    d.distance_method,
                                    round(d.closing_speed_mps, 3),
                                    "" if not math.isfinite(d.ttc_s) else round(d.ttc_s, 3),
                                    round(d.lane_overlap, 3),
                                    d.x1, d.y1, d.x2, d.y2,
                                    d.in_lane,
                                    d.risk,
                                    d.action,
                                    d.decision_source,
                                    d.rule_id,
                                    d.rule_priority,
                                    max(0, len(d.decision_trace) - 1),
                                    format_decision_trace(d.decision_trace),
                                     d.explanation,
                                     incident_paths.get(d.track_key, ""),
                                     d.track_key,
                                ])
                        writer.write(draw_frame)
                        next_frame += 1
                    break

                if item is None:
                    # no item available yet; allow GUI responsiveness
                    if display:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            stop_event.set()
                            break
                        if key == ord(" "):
                            cv2.waitKey(0)
                    continue

                if not isinstance(item, ProcessedItem):
                    continue

                # buffer the item
                buffer[item.frame_no] = item

                # process any ready frames in order
                while next_frame in buffer:
                    p = buffer.pop(next_frame)
                    frame_started = time.perf_counter()
                    stats.frames += 1
                    fps_frames += 1
                    video_time = p.video_time

                    # Synchronisation check: the detections drawn below must
                    # belong to the frame being drawn. Any drift is recorded and
                    # warned about rather than silently rendered.
                    age = p.frame_no - p.detection_frame_no
                    with _pipeline_lock:
                        pipeline_stats["detection_ages"].append(age)
                        if p.capture_timestamp:
                            pipeline_stats["e2e_latency"].append(time.perf_counter() - p.capture_timestamp)
                        if age != 0:
                            pipeline_stats["stale_frames"] += 1
                    if age != 0 and DEBUG_PERFORMANCE:
                        print(f"WARNING stale render: display frame {p.frame_no}, "
                              f"detections from frame {p.detection_frame_no} (age {age})")

                    # overlay info and draw
                    # Upscale to source resolution and map detection
                    # coordinates when the frame was downscaled for processing.
                    if needs_resize:
                        draw_frame = cv2.resize(p.frame, (source_w, source_h),
                                                interpolation=cv2.INTER_LINEAR)
                        sx = source_w / w
                        sy = source_h / h
                        draw_polygon = (p.polygon.astype(np.float32)
                                        * np.array([sx, sy])).astype(np.int32)
                    else:
                        draw_frame = p.frame
                        sx = sy = 1.0
                        draw_polygon = p.polygon

                    draw_lane(draw_frame, draw_polygon, p.lane_detected,
                              show_fallback=bool(p.detections))
                    frame_labels: list[tuple[int, int, int, int]] = []
                    for d in sorted(p.detections, key=lambda x: (-LEVEL_PRIORITY[x.risk], x.track_key)):
                        draw_d = _scale_detection(d, sx, sy)
                        draw_detection(draw_frame, draw_d, frame_labels, 0)

                    # incidents and CSV (only write rows for frames where detections were run)
                    incident_paths: dict[str, str] = {}
                    if p.frame_no == 1 or DETECTION_INTERVAL <= 1 or p.frame_no % DETECTION_INTERVAL == 0:
                        for d in p.detections:
                            incident = None
                            if d.risk in {"WARNING", "CRITICAL"} and d.in_lane:
                                incident = save_incident(draw_frame, path.name, p.frame_no, d, last_incident, video_time)
                                if incident:
                                    stats.incidents += 1
                                    incident_paths[d.track_key] = incident_reference(incident)

                            csv_writer.writerow([
                                 p.frame_no,
                                 p.frame_id,
                                 round(video_time, 3),
                                 round(p.source_timestamp, 3),
                                 round(p.detection_timestamp, 6),
                                d.name,
                                d.source,
                                round(d.confidence, 4),
                                d.distance_m,
                                d.distance_method,
                                round(d.closing_speed_mps, 3),
                                "" if not math.isfinite(d.ttc_s) else round(d.ttc_s, 3),
                                round(d.lane_overlap, 3),
                                d.x1, d.y1, d.x2, d.y2,
                                d.in_lane,
                                d.risk,
                                d.action,
                                d.decision_source,
                                d.rule_id,
                                d.rule_priority,
                                max(0, len(d.decision_trace) - 1),
                                format_decision_trace(d.decision_trace),
                                     d.explanation,
                                     incident_paths.get(d.track_key, ""),
                                     d.track_key,
                            ])

                    # writer and display must run on main thread
                    writer.write(draw_frame)
                    if display:
                        # Preview: cap window at 1920x1080 for4K sources
                        MAX_PW, MAX_PH = 1920, 1080
                        if draw_frame.shape[1] > MAX_PW or draw_frame.shape[0] > MAX_PH:
                            _ps = min(MAX_PW / draw_frame.shape[1], MAX_PH / draw_frame.shape[0])
                            _preview = cv2.resize(draw_frame,
                                                  (int(draw_frame.shape[1] * _ps),
                                                   int(draw_frame.shape[0] * _ps)),
                                                  interpolation=cv2.INTER_AREA)
                        else:
                            _preview = draw_frame
                        cv2.imshow("Advanced AI Road Safety Assistant", _preview)
                        processing_ms = (time.perf_counter() - frame_started) * 1000.0
                        remaining_ms = max(1, int(target_frame_ms - processing_ms))
                        key = cv2.waitKey(remaining_ms) & 0xFF
                        if key == ord("q"):
                            stop_event.set()
                            break
                        if key == ord(" "):
                            cv2.waitKey(0)

                    # update fps samples
                    now = time.time()
                    if now - fps_started >= 1.0:
                        live_fps = fps_frames / max(0.001, now - fps_started)
                        stats.processing_fps_samples.append(live_fps)
                        fps_frames = 0
                        fps_started = now

                    next_frame += 1

                # update pipeline queue stats snapshot
                pipeline_stats["max_read_q"] = max(pipeline_stats["max_read_q"], read_q.qsize())

            # finished processing loop
    finally:
        # ensure voice worker is stopped no matter how processing ends
        try:
            voice.close()
        except Exception:
            pass

        # ensure threads are joined
        stop_event.set()
        try:
            reader_t.join(timeout=1.0)
        except Exception:
            pass
        try:
            worker_t.join(timeout=1.0)
        except Exception:
            pass

        # always release video resources so the MP4 moov atom is written
        try:
            cap.release()
        except Exception:
            pass
        try:
            writer.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

    elapsed = time.time() - started
    summary = {
        "video": path.name,
        "frames": stats.frames,
        "total_detections": stats.total_detections,
        "risk_counts": dict(stats.level_counts),
        "object_counts": dict(stats.object_counts),
        "incidents": stats.incidents,
        "minimum_ttc_s": (
            None
            if not math.isfinite(stats.min_ttc)
            else round(stats.min_ttc, 3)
        ),
        "processing_seconds": round(elapsed, 2),
        "average_processing_fps": round(
            sum(stats.processing_fps_samples)
            / max(1, len(stats.processing_fps_samples)),
            2,
        ),
        "detection_interval": DETECTION_INTERVAL,
        "low_light_frames": stats.low_light_frames,
        "night_video": stats.low_light_frames > (stats.frames / 2),
        "limitations": (
            "Distance and TTC are monocular estimates and require camera calibration."
        ),
    }
    # include calibration summary and distance methods used
    try:
        summary["distance_methods_used"] = sorted(list(methods_used))
        if calibrator is None:
            summary["calibration"] = {"mode": "none", "note": "width-based fallback"}
        else:
            summary["calibration"] = {
                "horizon_y": getattr(calibrator, "horizon_y", None),
                "camera_height_m": getattr(calibrator, "camera_height_m", None),
                "vertical_fov_deg": getattr(calibrator, "vertical_fov_deg", None),
                "reference_points": getattr(calibrator, "ref_points", None),
                "min_distance": getattr(calibrator, "min_distance", None),
                "max_distance": getattr(calibrator, "max_distance", None),
                "uncalibrated_defaults": getattr(calibrator, "_uncalibrated", False),
            }
    except Exception:
        pass

    # pipeline stats into summary
    try:
        summary["pipeline"] = {
            "max_read_queue": pipeline_stats.get("max_read_q", 0),
            "max_result_queue": pipeline_stats.get("max_result_q", 0),
            "inference_avg_s": round(sum(pipeline_stats.get("inference_times", []) or [0.0]) / max(1, len(pipeline_stats.get("inference_times", []))), 4),
            "dropped_frames": pipeline_stats.get("dropped_frames", 0),
        }
    except Exception:
        pass

    # include Prolog reasoning stats
    try:
        summary["triggered_rule_counts"] = dict(stats.triggered_rules)
        summary["decision_source_counts"] = dict(stats.decision_source_counts)
        # include example explanations for top rules
        examples = {}
        try:
            for rid in list(stats.triggered_rules.keys()):
                if rid in sample_explanations:
                    examples[rid] = sample_explanations[rid]
        except Exception:
            pass
        summary["triggered_rule_examples"] = examples
        # Rule-priority / conflict-resolution reporting
        summary["rule_conflicts"] = {
            "detections_with_conflicts": stats.conflicts,
            "overridden_rule_counts": dict(stats.overridden_rules),
            "rule_priorities": RULE_PRIORITIES,
            "examples": stats.conflict_examples,
        }
    except Exception:
        pass

    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    create_dashboard(path.name, stats, elapsed, dashboard_out)

    if DEBUG_PERFORMANCE:
        print_performance_report(path.name, stats, pipeline_stats, elapsed, source_fps)

    print(
        f"Finished {path.name}\n"
        f"  Video: {video_out}\n"
        f"  CSV: {csv_out}\n"
        f"  Dashboard: {dashboard_out}\n"
        f"  Incidents: {INCIDENT_FOLDER}"
    )


def choose_video(videos: list[Path]) -> int | None:
    """Ask the user to choose exactly one video from the videos folder."""
    print("\nAvailable videos:")
    for index, video in enumerate(videos, start=1):
        next_name = videos[index].name if index < len(videos) else "None"
        print(f"  {index}. {video.name}    -> next: {next_name}")
    print("  0. Exit")

    while True:
        choice = input("\nChoose ONE video number to detect: ").strip()
        if choice == "0":
            return None
        try:
            selected = int(choice) - 1
        except ValueError:
            print("Please enter a number from the list.")
            continue
        if 0 <= selected < len(videos):
            return selected
        print("Invalid selection. Please choose a listed video number.")


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


def natural_key(path: Path) -> list:
    """Sort key that orders road_video_2 before road_video_10."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.stem)]


def discover_videos(folder: Path = VIDEO_FOLDER) -> list[Path]:
    """Every supported video in the folder, naturally sorted.

    Generated files are excluded so a previous run's exports can never be fed
    back in as inputs.
    """
    if not folder.exists():
        return []
    found = [
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTENSIONS
        and not p.name.startswith(".")
        and "_advanced" not in p.stem          # our own exported videos
    ]
    return sorted(found, key=natural_key)


def display_name_for(path: Path) -> str:
    """'road_video_4.mp4' -> 'Road Video 4'. No scenario description is
    invented here; tags come from the metadata file if one exists."""
    stem = re.sub(r"\.(mp4|mov|avi|mkv|m4v|wmv)$", "", path.stem, flags=re.I)
    return re.sub(r"[_-]+", " ", stem).strip().title()


def save_thumbnail(path: Path, out_dir: Path) -> Path | None:
    """One representative frame per video, taken ~20% in to avoid the black or
    near-black opening frames many clips start with."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.stem}_thumb.jpg"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        best = None
        # sample a few candidates and keep the brightest, so we never ship a
        # black frame as the library thumbnail
        for frac in (0.2, 0.35, 0.5, 0.65):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac) if total else 0)
            ok, frame = cap.read()
            if not ok:
                continue
            brightness = float(frame.mean())
            if best is None or brightness > best[0]:
                best = (brightness, frame)
        if best is None:
            return None
        frame = best[1]
        scale = min(1.0, 640 / max(1, frame.shape[1]))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
        cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return out
    finally:
        cap.release()


def probe_video(path: Path) -> dict:
    """Container-level metadata, read without decoding the whole file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": "could not open"}
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return {
            "source_fps": round(fps, 2),
            "frame_count": frames,
            "resolution": f"{width}x{height}",
            "width": width,
            "height": height,
            "duration_s": round(frames / fps, 1) if fps else None,
        }
    finally:
        cap.release()


def build_video_metadata(paths: list[Path]) -> list[dict]:
    """One reusable record per video. Counts are filled in from each video's
    summary.json once it has been processed; unprocessed videos report
    processed_status 'not_processed' rather than fabricated numbers."""
    records = []
    for index, path in enumerate(paths, start=1):
        summary_path = OUTPUT_FOLDER / f"{path.stem}_summary.json"
        record = {
            "video_id": index,
            "filename": path.name,
            "display_name": display_name_for(path),
            "path": str(path),
            "processed_status": "not_processed",
            **probe_video(path),
        }
        thumb = save_thumbnail(path, OUTPUT_FOLDER / "thumbnails")
        record["thumbnail"] = str(thumb) if thumb else None

        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary = None
            if summary:
                objects = summary.get("object_counts", {})
                risks = summary.get("risk_counts", {})
                record.update({
                    "processed_status": "processed",
                    "output_path": str(OUTPUT_FOLDER / f"{path.stem}_advanced.mp4"),
                    "dashboard_path": str(OUTPUT_FOLDER / f"{path.stem}_dashboard.html"),
                    "incident_folder": str(incident_folder_for(path.name)),
                    "incident_count": summary.get("incidents", 0),
                    "total_detections": summary.get("total_detections", 0),
                    "frames_processed": summary.get("frames", 0),
                    "pothole_count": sum(v for k, v in objects.items() if k in ROAD_DAMAGE_CLASSES),
                    "vehicle_count": sum(v for k, v in objects.items() if k in VEHICLE_CLASSES),
                    "pedestrian_count": sum(v for k, v in objects.items() if k in PERSON_CLASSES),
                    "warning_count": risks.get("WARNING", 0),
                    "critical_count": risks.get("CRITICAL", 0),
                    "caution_count": risks.get("CAUTION", 0),
                    "safe_count": risks.get("SAFE", 0),
                    "average_fps": summary.get("average_processing_fps"),
                    "processing_seconds": summary.get("processing_seconds"),
                    "night_video": summary.get("night_video", False),
                    "object_counts": objects,
                })
        records.append(record)
    return records


def load_extra_models() -> list[tuple[YOLO, str]]:
    """Auto-load any *.pt models dropped in models/extra/ (e.g. a trained
    obstacle/tree detector). Absent directory or files -> no-op."""
    if not EXTRA_MODELS_DIR.exists():
        return []
    loaded: list[tuple[YOLO, str]] = []
    for model_path in sorted(EXTRA_MODELS_DIR.glob("*.pt")):
        try:
            model = YOLO(str(model_path))
            loaded.append((model, model_path.name))
            print(f"Extra model loaded: {model_path.name} classes: {model.names}")
        except Exception as exc:
            print(f"Failed to load extra model {model_path.name}: {exc!r}")
    return loaded


def aggregate_batch_summary(video_paths: list[Path]) -> dict:
    """Combine per-video *_summary.json files (already written by
    process_video) into one manifest for a batch run."""
    risk_counts: Counter = Counter()
    object_counts: Counter = Counter()
    triggered_rule_counts: Counter = Counter()
    total_frames = 0
    total_detections = 0
    total_incidents = 0
    total_low_light_frames = 0
    min_ttc: float | None = None
    videos_summary: list[dict] = []

    for path in video_paths:
        summary_path = OUTPUT_FOLDER / f"{path.stem}_summary.json"
        if not summary_path.exists():
            continue
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        risk_counts.update(data.get("risk_counts", {}))
        object_counts.update(data.get("object_counts", {}))
        triggered_rule_counts.update(data.get("triggered_rule_counts", {}))
        total_frames += data.get("frames", 0)
        total_detections += data.get("total_detections", 0)
        total_incidents += data.get("incidents", 0)
        total_low_light_frames += data.get("low_light_frames", 0)
        video_min_ttc = data.get("minimum_ttc_s")
        if video_min_ttc is not None:
            min_ttc = video_min_ttc if min_ttc is None else min(min_ttc, video_min_ttc)
        videos_summary.append({
            "video": data.get("video"),
            "frames": data.get("frames"),
            "total_detections": data.get("total_detections"),
            "incidents": data.get("incidents"),
            "minimum_ttc_s": video_min_ttc,
            "night_video": data.get("night_video", False),
        })

    return {
        "videos_processed": len(videos_summary),
        "videos": videos_summary,
        "total_frames": total_frames,
        "total_detections": total_detections,
        "total_incidents": total_incidents,
        "total_low_light_frames": total_low_light_frames,
        "minimum_ttc_s": min_ttc,
        "risk_counts": dict(risk_counts),
        "object_counts": dict(object_counts),
        "triggered_rule_counts": dict(triggered_rule_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO11 + Prolog road hazard and collision warning system")
    parser.add_argument("--source", help="Process one specific video file directly")
    parser.add_argument("--all", action="store_true", help="Batch-process every video in the videos/ folder, non-interactively")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-voice", action="store_true")
    parser.add_argument("--calib-horizon-y", type=int, help="Horizon line Y coordinate (pixels)")
    parser.add_argument("--calib-horizon-ratio", type=float, help="Horizon line Y as ratio of image height (0..1)")
    parser.add_argument("--camera-height", type=float, help="Camera height in metres for analytic model")
    parser.add_argument("--vertical-fov", type=float, help="Vertical field of view in degrees for analytic model")
    parser.add_argument("--calib-points", type=str, help="Reference points as y:distance,y:distance (image y in px)")
    parser.add_argument("--min-distance", type=float, default=0.5, help="Minimum valid distance (m)")
    parser.add_argument("--max-distance", type=float, default=999.0, help="Maximum valid distance (m)")
    parser.add_argument("--drop-frames", action="store_true", help="Drop frames when processing cannot keep up (live mode)")
    parser.add_argument("--device", help="Inference device: cpu, mps, cuda (default: auto-detect fastest available)")
    parser.add_argument("--debug-performance", action="store_true", help="Print a per-stage timing breakdown after each video")
    args = parser.parse_args()

    global INFERENCE_DEVICE, DEBUG_PERFORMANCE
    INFERENCE_DEVICE = select_device(args.device)
    DEBUG_PERFORMANCE = args.debug_performance
    print(f"Inference device: {INFERENCE_DEVICE}")

    # parse calibration points if provided
    calib_cfg: dict = {}
    if args.calib_horizon_y is not None:
        calib_cfg["horizon_y"] = args.calib_horizon_y
    if args.calib_horizon_ratio is not None:
        calib_cfg["horizon_ratio"] = args.calib_horizon_ratio
    if args.camera_height is not None:
        calib_cfg["camera_height_m"] = args.camera_height
    if args.vertical_fov is not None:
        calib_cfg["vertical_fov_deg"] = args.vertical_fov
    if args.calib_points:
        pts = []
        for item in args.calib_points.split("\n" if "\n" in args.calib_points else ","):
            item = item.strip()
            if not item:
                continue
            try:
                ystr, dstr = item.split(":")
                pts.append((int(ystr), float(dstr)))
            except Exception as exc:
                raise ValueError(f"Invalid calib point '{item}': {exc}")
        calib_cfg["ref_points"] = pts
    calib_cfg["min_distance"] = args.min_distance
    calib_cfg["max_distance"] = args.max_distance

    for required, label in [(COMMON_MODEL_PATH, "yolo11n.pt"), (CUSTOM_MODEL_PATH, "best.pt")]:
        if not required.exists():
            raise FileNotFoundError(f"{label} not found: {required}")

    common = YOLO(str(COMMON_MODEL_PATH))
    custom = YOLO(str(CUSTOM_MODEL_PATH))
    print("Common model classes:", common.names)
    print("Custom model classes:", custom.names)
    extra_models = load_extra_models()
    expert = PrologRiskEngine(PROLOG_FILE)

    # Direct mode: process only the file supplied with --source.
    if args.source is not None:
        source_path = Path(args.source)
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        process_video(
            source_path,
            common,
            custom,
            expert,
            not args.no_display,
            not args.no_voice,
            next_video_name=None,
            calibrator_config=calib_cfg,
            drop_frames=args.drop_frames,
            extra_models=extra_models,
        )
        return

    VIDEO_FOLDER.mkdir(exist_ok=True)
    videos = discover_videos(VIDEO_FOLDER)

    if not videos:
        raise FileNotFoundError(f"Put at least one video inside {VIDEO_FOLDER}")

    # Batch mode: process every video in videos/ back-to-back, no display/voice/prompts.
    if args.all:
        print(f"Batch mode: processing {len(videos)} video(s) from {VIDEO_FOLDER}")
        results: list[tuple[str, str, str]] = []   # (video, status, detail)
        for index, video in enumerate(videos, start=1):
            print(f"\n[{index}/{len(videos)}] {video.name}")
            started = time.time()
            try:
                process_video(
                    video,
                    common,
                    custom,
                    expert,
                    False,
                    False,
                    next_video_name=None,
                    calibrator_config=calib_cfg,
                    drop_frames=args.drop_frames,
                    extra_models=extra_models,
                )
            except KeyboardInterrupt:
                results.append((video.name, "SKIPPED", "interrupted by user"))
                print("  SKIPPED (interrupted)")
                break
            except Exception as exc:
                # One bad video must not abandon the rest of the batch.
                results.append((video.name, "FAILED", repr(exc)))
                print(f"  FAILED: {exc!r}")
                continue
            if (OUTPUT_FOLDER / f"{video.stem}_summary.json").exists():
                results.append((video.name, "PASSED", f"{time.time() - started:.1f}s"))
            else:
                results.append((video.name, "FAILED", "no summary written"))

        batch_summary = aggregate_batch_summary(videos)
        batch_summary["results"] = [
            {"video": name, "status": status, "detail": detail}
            for name, status, detail in results
        ]
        batch_summary["videos"] = build_video_metadata(videos)
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        batch_out = OUTPUT_FOLDER / "batch_summary.json"
        batch_out.write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")

        print(f"\n{'=' * 58}\nBATCH RESULTS\n{'=' * 58}")
        for name, status, detail in results:
            print(f"  {status:8s} {name:26s} {detail}")
        passed = sum(1 for _, s, _ in results if s == "PASSED")
        print(f"{'-' * 58}\n  {passed}/{len(results)} passed  ->  {batch_out}\n")
        return

    # Interactive mode: choose one video at a time.
    while True:
        selected_index = choose_video(videos)
        if selected_index is None:
            print("Program closed.")
            break

        current_video = videos[selected_index]
        next_video = videos[selected_index + 1] if selected_index + 1 < len(videos) else None
        next_name = next_video.name if next_video else None

        print("\n" + "=" * 60)
        print(f"NOW DETECTING : {current_video.name}")
        print(f"NEXT VIDEO    : {next_name if next_name else 'None - this is the last video'}")
        print("Press Q in the video window if you want to stop the current video early.")
        print("=" * 60 + "\n")

        process_video(
            current_video,
            common,
            custom,
            expert,
            not args.no_display,
            not args.no_voice,
            next_video_name=next_name,
            calibrator_config=calib_cfg,
            drop_frames=args.drop_frames,
            extra_models=extra_models,
        )

        if next_video is not None:
            answer = input(f"\nFinished {current_video.name}. Detect NEXT video ({next_video.name}) now? [Y/n]: ").strip().lower()
            if answer in {"", "y", "yes"}:
                print(f"\nStarting next video: {next_video.name}")
                following_name = videos[selected_index + 2].name if selected_index + 2 < len(videos) else None
                process_video(
                    next_video,
                    common,
                    custom,
                    expert,
                    not args.no_display,
                    not args.no_voice,
                    next_video_name=following_name,
                    calibrator_config=calib_cfg,
                    drop_frames=args.drop_frames,
                    extra_models=extra_models,
                )
            else:
                print("Returning to the video selection menu.")
        else:
            print("\nThat was the last video. Returning to the video selection menu.")


if __name__ == "__main__":
    main()
