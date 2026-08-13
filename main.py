from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import threading
import queue
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from pyswip import Prolog
except Exception:
    Prolog = None  # type: ignore[assignment]

try:
    import pyttsx3
except Exception:
    pyttsx3 = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parent
COMMON_MODEL_PATH = ROOT / "models" / "yolo11n.pt"
CUSTOM_MODEL_PATH = ROOT / "models" / "best.pt"
PROLOG_FILE = ROOT / "expert_system_advanced.pl"
VIDEO_FOLDER = ROOT / "videos"
OUTPUT_FOLDER = ROOT / "output"
INCIDENT_FOLDER = OUTPUT_FOLDER / "incidents"

CONFIDENCE = 0.30
IOU = 0.50
IMAGE_SIZE = 640
TRACKER = "bytetrack.yaml"

# Smoothness / performance settings
# 1 = detect on every frame (best accuracy, slower)
# 2 = detect every second frame (recommended for smoother playback)
DETECTION_INTERVAL = 2

DISTANCE_SMOOTH_ALPHA = 0.35
LANE_SMOOTH_ALPHA = 0.25

# Forget a track after it has not been seen for this many detection cycles.
TRACK_FORGET_AFTER = 20

FOCAL_LENGTH_PX = 700.0
DEFAULT_WIDTH_M = 0.8
KNOWN_WIDTHS_M = {
    "person": 0.50, "bicycle": 0.60, "car": 1.80, "motorcycle": 0.80,
    "bus": 2.50, "truck": 2.50, "dog": 0.45, "cat": 0.30,
    "traffic light": 0.30, "stop sign": 0.75, "pothole": 0.80,
    "road crack": 0.60, "road_crack": 0.60, "cone": 0.35,
    "barrier": 1.20, "debris": 0.50,
}

LEVEL_PRIORITY = {"SAFE": 0, "CAUTION": 1, "WARNING": 2, "CRITICAL": 3, "ERROR": 4}
LEVEL_COLORS = {
    "SAFE": (0, 220, 0), "CAUTION": (0, 255, 255),
    "WARNING": (0, 165, 255), "CRITICAL": (0, 0, 255),
    "ERROR": (255, 255, 255),
}

TRAFFIC_CONTROLS = {"traffic light", "stop sign", "parking meter"}
PERSON_CLASSES = {"person"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
ROAD_DAMAGE_CLASSES = {"pothole", "road crack", "road_crack", "crack"}
ANIMAL_CLASSES = {"dog", "cat", "cow", "horse", "sheep", "bird"}


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
    processing_fps_samples: list[float] = field(default_factory=list)


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
    def decide(self, d: Detection) -> tuple[str, str]:
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
                        result = list(self.prolog.query("decision(Level,Action)"))
                    except Exception as exc_query:
                        print(f"Prolog decision query failed: {exc_query!r}")
                        return fallback_decision(d)

                    if result:
                        try:
                            return (
                                str(result[0]["Level"]).upper(),
                                str(result[0]["Action"]).replace("_", " ").upper(),
                            )
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


def fallback_decision(d: Detection) -> tuple[str, str]:
    name = d.name.lower()
    if not d.in_lane:
        return "SAFE", f"{name} OUTSIDE VEHICLE LANE"
    if name in TRAFFIC_CONTROLS:
        return "CAUTION", f"OBSERVE {name}"
    if d.ttc_s <= 1.5 or d.distance_m <= 3.0 or d.box_height_ratio >= 0.52:
        if name in PERSON_CLASSES: return "CRITICAL", "BRAKE IMMEDIATELY - PERSON AHEAD"
        if name in ROAD_DAMAGE_CLASSES: return "CRITICAL", "BRAKE AND AVOID ROAD DAMAGE"
        return "CRITICAL", f"BRAKE NOW - {name} TOO CLOSE"
    if d.ttc_s <= 3.0 or d.distance_m <= 7.0 or d.box_height_ratio >= 0.32:
        if name in VEHICLE_CLASSES: return "WARNING", "SLOW DOWN AND INCREASE FOLLOWING DISTANCE"
        if name in ANIMAL_CLASSES: return "WARNING", "SLOW DOWN - ANIMAL AHEAD"
        if name in ROAD_DAMAGE_CLASSES: return "WARNING", "SLOW DOWN AND PREPARE TO AVOID ROAD DAMAGE"
        return "WARNING", f"SLOW DOWN - {name} AHEAD"
    if d.ttc_s <= 5.0 or d.distance_m <= 14.0 or d.box_height_ratio >= 0.17:
        return "CAUTION", f"CAUTION - {name} IN VEHICLE LANE"
    return "SAFE", f"{name} AT SAFE DISTANCE"


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
        bottom_x = (x1 + x2) / 2.0
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
    cv2.fillPoly(mask, roi, 255)
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

    def fit(points: list[tuple[int, int]], y_top: int, y_bottom: int) -> tuple[int, int]:
        ys = np.array([p[1] for p in points], dtype=np.float32)
        xs = np.array([p[0] for p in points], dtype=np.float32)
        a, b = np.polyfit(ys, xs, 1)
        return int(a * y_top + b), int(a * y_bottom + b)

    y_top, y_bottom = int(h * 0.56), h - 1
    lx_top, lx_bottom = fit(left_points, y_top, y_bottom)
    rx_top, rx_bottom = fit(right_points, y_top, y_bottom)
    if lx_top >= rx_top or lx_bottom >= rx_bottom:
        return default_lane_polygon(w, h), False
    polygon = np.array([[lx_top, y_top], [rx_top, y_top], [rx_bottom, y_bottom], [lx_bottom, y_bottom]], dtype=np.int32)
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
        self.last_seen: dict[str, int] = {}
        self.cycle = 0

    def smooth(self, detections: list[Detection], frame_shape: tuple[int, int], dt: float) -> list[Detection]:
        """Predict all filters with dt, then correct those with measurements.

        frame_shape: (height, width)
        dt: elapsed seconds between updates (e.g., 1/source_fps)
        """
        self.cycle += 1
        h, w = frame_shape

        # Predict all filters to advance their state
        for kf in list(self._filters.values()):
            try:
                kf.predict(dt)
            except Exception:
                pass

        for d in detections:
            key = d.track_key
            x1, y1, x2, y2 = d.box
            meas_w = max(1.0, float(x2 - x1))
            meas_h = max(1.0, float(y2 - y1))
            meas_cx = float(x1) + meas_w / 2.0
            meas_cy = float(y1) + meas_h / 2.0

            # Initialize filter for new tracks
            if key not in self._filters:
                try:
                    self._filters[key] = BoundingBoxKalmanFilter(meas_cx, meas_cy, meas_w, meas_h, dt)
                except Exception:
                    # fallback to raw box if Kalman cannot be created
                    self.last_seen[key] = self.cycle
                    continue

            # Correct with measurement
            try:
                self._filters[key].correct(meas_cx, meas_cy, meas_w, meas_h)
            except Exception:
                pass

            # Update distance smoothing (preserve behavior)
            previous_distance = self.distance_state.get(key)
            if previous_distance is None:
                smoothed_distance = d.distance_m
            else:
                smoothed_distance = (
                    DISTANCE_SMOOTH_ALPHA * d.distance_m
                    + (1.0 - DISTANCE_SMOOTH_ALPHA) * previous_distance
                )
            self.distance_state[key] = smoothed_distance
            d.distance_m = round(smoothed_distance, 2)

            # Pull current state and set box coordinates
            st = self._filters[key].state()
            cx, cy, bw, bh = float(st[0]), float(st[1]), float(st[2]), float(st[3])
            bw = max(1.0, bw)
            bh = max(1.0, bh)
            x1n = int(round(cx - bw / 2.0))
            y1n = int(round(cy - bh / 2.0))
            x2n = int(round(cx + bw / 2.0))
            y2n = int(round(cy + bh / 2.0))

            # Clip to frame
            x1n = max(0, min(w - 1, x1n))
            x2n = max(0, min(w - 1, x2n))
            y1n = max(0, min(h - 1, y1n))
            y2n = max(0, min(h - 1, y2n))

            # Ensure valid box
            if x2n <= x1n:
                x2n = min(w - 1, x1n + 1)
            if y2n <= y1n:
                y2n = min(h - 1, y1n + 1)

            d.box = (x1n, y1n, x2n, y2n)
            self.last_seen[key] = self.cycle

        # Remove stale filters
        stale = [k for k, last in self.last_seen.items() if self.cycle - last >= TRACK_FORGET_AFTER]
        for k in stale:
            self._filters.pop(k, None)
            self.distance_state.pop(k, None)
            self.last_seen.pop(k, None)

        return detections


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
    kept: list[Detection] = []
    for d in sorted(items, key=lambda x: x.confidence, reverse=True):
        if any(d.name == k.name and box_iou(d, k) >= 0.65 for k in kept):
            continue
        kept.append(d)
    return kept


def extract(model: YOLO, frame: np.ndarray, source: str, polygon: np.ndarray, stream_id: str, calibrator: PerspectiveDistanceCalibrator | None = None) -> list[Detection]:
    h, w = frame.shape[:2]
    results = model.track(frame, persist=True, tracker=TRACKER, conf=CONFIDENCE,
                          iou=IOU, imgsz=IMAGE_SIZE, verbose=False)
    output: list[Detection] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            name = str(model.names[cls_id]).strip().lower()
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, x2 = max(0, x1), min(w - 1, x2)
            y1, y2 = max(0, y1), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            tid = int(box.id[0].item()) if box.id is not None else -1
            overlap = lane_overlap((x1, y1, x2, y2), polygon, (h, w))
            key = f"{stream_id}:{source}:{name}:{tid if tid >= 0 else x1//40}:{y1//40}"
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

            output.append(Detection(
                name=name, confidence=confidence, box=(x1, y1, x2, y2), source=source,
                track_key=key, distance_m=dist, distance_method=method,
                in_lane=overlap >= 0.18, lane_overlap=overlap,
                box_height_ratio=(y2 - y1) / max(1, h),
            ))
    return output


def update_ttc(d: Detection, history: dict[str, deque[tuple[float, float]]], video_time: float) -> None:
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
        d.ttc_s = d.distance_m / closing


def draw_lane(frame: np.ndarray, polygon: np.ndarray, detected: bool) -> None:
    overlay = frame.copy()
    color = (255, 220, 0) if detected else (160, 160, 160)
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, 0.14, frame, 0.86, 0, frame)
    cv2.polylines(frame, [polygon], True, color, 2)
    cv2.putText(frame, "LANE DETECTED" if detected else "LANE FALLBACK", tuple(polygon[0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)


def draw_detection(frame: np.ndarray, d: Detection) -> None:
    color = LEVEL_COLORS[d.risk]
    cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 3 if d.in_lane else 2)
    ttc = "--" if not math.isfinite(d.ttc_s) else f"{d.ttc_s:.1f}s"
    label = f"{d.name} {d.confidence:.2f} | {d.distance_m:.1f}m | TTC:{ttc} | {d.risk}"
    cv2.putText(frame, label, (d.x1, max(22, d.y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, color, 2, cv2.LINE_AA)


def draw_panel(frame: np.ndarray, level: str, action: str, count: int, fps: float, ttc: float) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, min(142, h // 4)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.putText(frame, f"STATUS: {level}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.88,
                LEVEL_COLORS[level], 2, cv2.LINE_AA)
    cv2.putText(frame, f"ACTION: {action[:90]}", (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                (255, 255, 255), 2, cv2.LINE_AA)
    ttc_text = "--" if not math.isfinite(ttc) else f"{ttc:.1f} sec"
    cv2.putText(frame, f"Objects: {count} | FPS: {fps:.1f} | Minimum TTC: {ttc_text}",
                (18, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)


def save_incident(frame: np.ndarray, video_name: str, frame_no: int, d: Detection,
                  last_saved: dict[str, float], video_time: float) -> Path | None:
    key = f"{d.track_key}:{d.risk}"
    if video_time - last_saved.get(key, -999.0) < 3.0:
        return None
    last_saved[key] = video_time
    INCIDENT_FOLDER.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() else "_" for c in d.name)
    path = INCIDENT_FOLDER / f"{Path(video_name).stem}_f{frame_no}_{d.risk}_{safe_name}.jpg"
    cv2.imwrite(str(path), frame)
    return path


def create_dashboard(video_name: str, stats: RunStats, elapsed: float, output_path: Path) -> None:
    avg_fps = sum(stats.processing_fps_samples) / max(1, len(stats.processing_fps_samples))
    min_ttc = "N/A" if not math.isfinite(stats.min_ttc) else f"{stats.min_ttc:.2f} seconds"
    top_objects = stats.object_counts.most_common(10)
    rows = "".join(f"<tr><td>{name}</td><td>{count}</td></tr>" for name, count in top_objects) or "<tr><td>None</td><td>0</td></tr>"
    level_cards = "".join(
        f'<div class="card"><h3>{level}</h3><p>{stats.level_counts[level]}</p></div>'
        for level in ["SAFE", "CAUTION", "WARNING", "CRITICAL"]
    )
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Road Safety Report</title>
<style>body{{font-family:Arial;background:#10141b;color:#eee;margin:0;padding:28px}}h1{{margin-top:0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px}}.card{{background:#1d2530;border-radius:12px;padding:16px}}.card p{{font-size:28px;font-weight:bold;margin:8px 0}}table{{width:100%;border-collapse:collapse;background:#1d2530}}th,td{{padding:10px;border-bottom:1px solid #394555;text-align:left}}.note{{color:#b9c5d3}}</style></head><body>
<h1>AI Road Hazard and Collision Warning Report</h1><p class='note'>Video: {video_name}</p>
<div class='grid'><div class='card'><h3>Frames</h3><p>{stats.frames}</p></div><div class='card'><h3>Detections</h3><p>{stats.total_detections}</p></div><div class='card'><h3>Incidents</h3><p>{stats.incidents}</p></div><div class='card'><h3>Average FPS</h3><p>{avg_fps:.1f}</p></div><div class='card'><h3>Minimum TTC</h3><p style='font-size:18px'>{min_ttc}</p></div><div class='card'><h3>Run Time</h3><p style='font-size:18px'>{elapsed:.1f}s</p></div></div>
<h2>Risk Levels</h2><div class='grid'>{level_cards}</div><h2>Most Detected Objects</h2><table><tr><th>Object</th><th>Count</th></tr>{rows}</table>
<p class='note'>Distance and TTC values are estimates. Calibrate the camera before real-world use.</p></body></html>"""
    output_path.write_text(html, encoding="utf-8")


def process_video(path: Path, common: YOLO, custom: YOLO, expert: PrologRiskEngine,
                  display: bool, voice_enabled: bool, next_video_name: str | None = None,
                  calibrator_config: dict | None = None,
                  drop_frames: bool = False,
                  extractor_callable: Any | None = None,
                  reader_queue_size: int = 8,
                  result_queue_size: int = 8) -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Cannot open {path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    video_out = OUTPUT_FOLDER / f"{path.stem}_advanced.mp4"
    csv_out = OUTPUT_FOLDER / f"{path.stem}_detections.csv"
    json_out = OUTPUT_FOLDER / f"{path.stem}_summary.json"
    dashboard_out = OUTPUT_FOLDER / f"{path.stem}_dashboard.html"

    writer = cv2.VideoWriter(
        str(video_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (w, h),
    )

    # Pipeline items
    @dataclass
    class FrameItem:
        frame_no: int
        video_time: float
        frame: np.ndarray

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

    # Initialize resources and shared state
    voice = VoiceAlert(voice_enabled)
    history: dict[str, deque[tuple[float, float]]] = defaultdict(deque)
    last_incident: dict[str, float] = {}
    stats = RunStats()
    smoother = DetectionSmoother()
    previous_lane: np.ndarray | None = None
    methods_used: set[str] = set()

    # calibration
    calibrator = None
    try:
        calibrator = PerspectiveDistanceCalibrator(image_height=h, **(calibrator_config or {}))
    except Exception as exc:
        print(f"Calibration setup failed: {exc!r}; falling back to width-based estimation")
        calibrator = None

    # queues and control
    read_q: "queue.Queue[FrameItem | object]" = queue.Queue(maxsize=reader_queue_size)
    result_q: "queue.Queue[ProcessedItem | object]" = queue.Queue(maxsize=result_queue_size)
    exception_q: "queue.Queue[BaseException]" = queue.Queue()
    stop_event = threading.Event()
    SENTINEL = object()

    # synchronization for shared mutation
    _methods_lock = threading.Lock()
    _pipeline_lock = threading.Lock()

    # stats for pipeline
    pipeline_stats = {
        "max_read_q": 0,
        "max_result_q": 0,
        "read_put_times": [],
        "inference_times": [],
        "write_times": [],
        "dropped_frames": 0,
    }

    # reader thread: read frames and enqueue
    def reader():
        try:
            fno = 0
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                fno += 1
                video_time = fno / source_fps
                item = FrameItem(fno, video_time, frame)
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
            cached_detections: list[Detection] = []
            previous_lane_local: np.ndarray | None = None
            while not stop_event.is_set():
                item = read_q.get()
                if item is SENTINEL:
                    # propagate sentinel
                    try:
                        result_q.put(SENTINEL, timeout=1.0)
                    except Exception:
                        pass
                    break

                start_inf = time.perf_counter()
                frame_no = item.frame_no
                frame = item.frame
                video_time = item.video_time

                # lane detection runs on worker to avoid blocking main
                raw_polygon, lane_detected = detect_lane_polygon(frame)
                polygon = smooth_lane_polygon(raw_polygon, previous_lane_local, lane_detected)
                # update local previous lane for smoother continuity
                previous_lane_local = polygon.copy()

                fresh_detection = (
                    frame_no == 1 or DETECTION_INTERVAL <= 1 or frame_no % DETECTION_INTERVAL == 0
                )

                detections: list[Detection] = []
                if fresh_detection:
                    try:
                        if extractor_callable is not None:
                            detections = extractor_callable(frame, polygon, path.stem, calibrator)
                        else:
                            detections = deduplicate(
                                extract(common, frame, "yolo11n", polygon, path.stem, calibrator)
                                + extract(custom, frame, "best.pt", polygon, path.stem, calibrator)
                            )
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

                    # smoothing and decisions
                    detections = smoother.smooth(detections, (h, w), 1.0 / max(1.0, source_fps))
                    for d in detections:
                        d.lane_overlap = lane_overlap(d.box, polygon, (h, w))
                        d.in_lane = d.lane_overlap >= 0.18
                        d.box_height_ratio = (d.y2 - d.y1) / max(1, h)
                        update_ttc(d, history, video_time)
                        try:
                            d.risk, d.action = expert.decide(d)
                        except Exception as exc:
                            # Prolog exceptions are handled inside expert.decide
                            d.risk, d.action = fallback_decision(d)

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

                    cached_detections = detections
                else:
                    detections = cached_detections

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
                min_ttc = math.inf
                if highest is not None:
                    level = highest.risk
                    action = highest.action
                    min_ttc = highest.ttc_s
                    if fresh_detection and level in {"WARNING", "CRITICAL"}:
                        voice.speak(f"{highest.name}:{level}", action)

                inf_time = time.perf_counter() - start_inf
                with _pipeline_lock:
                    pipeline_stats["inference_times"].append(inf_time)
                    pipeline_stats["max_result_q"] = max(pipeline_stats["max_result_q"], result_q.qsize())

                processed = ProcessedItem(
                    frame_no=frame_no,
                    video_time=video_time,
                    frame=frame,
                    detections=detections,
                    polygon=polygon,
                    lane_detected=lane_detected,
                    level=level,
                    action=action,
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
                "frame", "video_time_s", "object", "source", "confidence",
                "distance_m", "distance_method", "closing_speed_mps", "ttc_s", "lane_overlap",
                "in_lane", "risk", "action", "incident_image"
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
                        draw_lane(p.frame, p.polygon, p.lane_detected)
                        for d in p.detections:
                            draw_detection(p.frame, d)
                        draw_panel(p.frame, p.level, p.action, len(p.detections), live_fps, math.inf)
                        try:
                            if calibrator is not None and getattr(calibrator, "_uncalibrated", False):
                                cv2.putText(p.frame, "WARNING: USING UNCALIBRATED PERSPECTIVE DEFAULTS", (18, 18),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                        except Exception:
                            pass
                        incident_paths: dict[str, str] = {}
                        if p.frame_no == 1 or DETECTION_INTERVAL <= 1 or p.frame_no % DETECTION_INTERVAL == 0:
                            for d in p.detections:
                                incident = None
                                if d.risk in {"WARNING", "CRITICAL"} and d.in_lane:
                                    incident = save_incident(p.frame, path.name, p.frame_no, d, last_incident, video_time)
                                    if incident:
                                        stats.incidents += 1
                                        incident_paths[d.track_key] = str(incident)
                                csv_writer.writerow([
                                    p.frame_no,
                                    round(video_time, 3),
                                    d.name,
                                    d.source,
                                    round(d.confidence, 4),
                                    d.distance_m,
                                    d.distance_method,
                                    round(d.closing_speed_mps, 3),
                                    "" if not math.isfinite(d.ttc_s) else round(d.ttc_s, 3),
                                    round(d.lane_overlap, 3),
                                    d.in_lane,
                                    d.risk,
                                    d.action,
                                    incident_paths.get(d.track_key, ""),
                                ])
                        writer.write(p.frame)
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

                # buffer the item
                buffer[item.frame_no] = item

                # process any ready frames in order
                while next_frame in buffer:
                    p = buffer.pop(next_frame)
                    frame_started = time.perf_counter()
                    stats.frames += 1
                    fps_frames += 1
                    video_time = p.video_time

                    # overlay info and draw
                    draw_lane(p.frame, p.polygon, p.lane_detected)
                    for d in p.detections:
                        draw_detection(p.frame, d)
                    draw_panel(p.frame, p.level, p.action, len(p.detections), live_fps, math.inf)

                    # calibration warning
                    try:
                        if calibrator is not None and getattr(calibrator, "_uncalibrated", False):
                            cv2.putText(p.frame, "WARNING: USING UNCALIBRATED PERSPECTIVE DEFAULTS", (18, 18),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                    except Exception:
                        pass

                    # incidents and CSV (only write rows for frames where detections were run)
                    incident_paths: dict[str, str] = {}
                    if p.frame_no == 1 or DETECTION_INTERVAL <= 1 or p.frame_no % DETECTION_INTERVAL == 0:
                        for d in p.detections:
                            incident = None
                            if d.risk in {"WARNING", "CRITICAL"} and d.in_lane:
                                incident = save_incident(p.frame, path.name, p.frame_no, d, last_incident, video_time)
                                if incident:
                                    stats.incidents += 1
                                    incident_paths[d.track_key] = str(incident)

                            csv_writer.writerow([
                                p.frame_no,
                                round(video_time, 3),
                                d.name,
                                d.source,
                                round(d.confidence, 4),
                                d.distance_m,
                                d.distance_method,
                                round(d.closing_speed_mps, 3),
                                "" if not math.isfinite(d.ttc_s) else round(d.ttc_s, 3),
                                round(d.lane_overlap, 3),
                                d.in_lane,
                                d.risk,
                                d.action,
                                incident_paths.get(d.track_key, ""),
                            ])

                    # writer and display must run on main thread
                    writer.write(p.frame)
                    if display:
                        cv2.imshow("Advanced AI Road Safety Assistant", p.frame)
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

    finally:
        # ensure voice worker is stopped no matter how processing ends
        try:
            voice.close()
        except Exception:
            pass

    cap.release()
    writer.release()
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

    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    create_dashboard(path.name, stats, elapsed, dashboard_out)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO11 + Prolog road hazard and collision warning system")
    parser.add_argument("--source", help="Process one specific video file directly")
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
    args = parser.parse_args()

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
        )
        return

    VIDEO_FOLDER.mkdir(exist_ok=True)
    videos = [
        p for p in sorted(VIDEO_FOLDER.iterdir())
        if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
    ]

    if not videos:
        raise FileNotFoundError(f"Put at least one video inside {VIDEO_FOLDER}")

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
                )
            else:
                print("Returning to the video selection menu.")
        else:
            print("\nThat was the last video. Returning to the video selection menu.")


if __name__ == "__main__":
    main()