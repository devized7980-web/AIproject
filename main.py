from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import threading
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

# Exponential moving-average weights.
# Higher values react faster; lower values look smoother.
BOX_SMOOTH_ALPHA = 0.55
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
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and pyttsx3 is not None
        self.last_spoken: dict[str, float] = {}
        self.lock = threading.Lock()

    def speak(self, key: str, message: str, cooldown: float = 3.0) -> None:
        if not self.enabled:
            return
        now = time.time()
        if now - self.last_spoken.get(key, 0.0) < cooldown:
            return
        self.last_spoken[key] = now
        threading.Thread(target=self._worker, args=(message,), daemon=True).start()

    def _worker(self, message: str) -> None:
        with self.lock:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 175)
                engine.say(message)
                engine.runAndWait()
                engine.stop()
            except Exception:
                pass


class PrologRiskEngine:
    def __init__(self, path: Path) -> None:
        self.available = False
        self.prolog: Any = None
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
            list(self.prolog.query("retractall(observation(_,_,_,_,_,_))"))
            ttc = 999.0 if not math.isfinite(d.ttc_s) else max(0.0, d.ttc_s)
            fact = (
                f"observation({self.atom(d.name)},{d.distance_m:.3f},{ttc:.3f},"
                f"{str(d.in_lane).lower()},{d.confidence:.3f},{d.box_height_ratio:.3f})"
            )
            list(self.prolog.query(f"assertz({fact})"))
            result = list(self.prolog.query("decision(Level,Action)"))
            if result:
                return str(result[0]["Level"]).upper(), str(result[0]["Action"]).replace("_", " ").upper()
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



class DetectionSmoother:
    """Reduce bounding-box and distance jitter between YOLO tracking frames."""

    def __init__(self) -> None:
        self.box_state: dict[str, np.ndarray] = {}
        self.distance_state: dict[str, float] = {}
        self.last_seen: dict[str, int] = {}
        self.cycle = 0

    def smooth(self, detections: list[Detection]) -> list[Detection]:
        self.cycle += 1

        for d in detections:
            key = d.track_key

            current_box = np.array(d.box, dtype=np.float32)
            previous_box = self.box_state.get(key)

            if previous_box is None:
                smoothed_box = current_box
            else:
                smoothed_box = (
                    BOX_SMOOTH_ALPHA * current_box
                    + (1.0 - BOX_SMOOTH_ALPHA) * previous_box
                )

            self.box_state[key] = smoothed_box
            d.box = tuple(int(round(v)) for v in smoothed_box)

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
            self.last_seen[key] = self.cycle

        stale = [
            key
            for key, last_cycle in self.last_seen.items()
            if self.cycle - last_cycle > TRACK_FORGET_AFTER
        ]
        for key in stale:
            self.box_state.pop(key, None)
            self.distance_state.pop(key, None)
            self.last_seen.pop(key, None)

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


def extract(model: YOLO, frame: np.ndarray, source: str, polygon: np.ndarray, stream_id: str) -> list[Detection]:
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
            output.append(Detection(
                name=name, confidence=confidence, box=(x1, y1, x2, y2), source=source,
                track_key=key, distance_m=estimate_distance(name, x2 - x1),
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
                  display: bool, voice_enabled: bool, next_video_name: str | None = None) -> None:
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

    voice = VoiceAlert(voice_enabled)
    history: dict[str, deque[tuple[float, float]]] = defaultdict(deque)
    last_incident: dict[str, float] = {}
    stats = RunStats()

    smoother = DetectionSmoother()
    previous_lane: np.ndarray | None = None
    cached_detections: list[Detection] = []

    started = time.time()
    fps_started = time.time()
    fps_frames = 0
    live_fps = 0.0

    target_frame_ms = max(1, int(round(1000.0 / max(1.0, source_fps))))

    with csv_out.open("w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            "frame", "video_time_s", "object", "source", "confidence",
            "distance_m", "closing_speed_mps", "ttc_s", "lane_overlap",
            "in_lane", "risk", "action", "incident_image"
        ])

        while True:
            frame_started = time.perf_counter()

            ok, frame = cap.read()
            if not ok:
                break

            stats.frames += 1
            fps_frames += 1
            video_time = stats.frames / source_fps

            raw_polygon, lane_detected = detect_lane_polygon(frame)
            polygon = smooth_lane_polygon(raw_polygon, previous_lane, lane_detected)
            previous_lane = polygon.copy()

            fresh_detection = (
                stats.frames == 1
                or DETECTION_INTERVAL <= 1
                or stats.frames % DETECTION_INTERVAL == 0
            )

            if fresh_detection:
                detections = deduplicate(
                    extract(common, frame, "yolo11n", polygon, path.stem)
                    + extract(custom, frame, "best.pt", polygon, path.stem)
                )

                detections = smoother.smooth(detections)

                for d in detections:
                    d.lane_overlap = lane_overlap(d.box, polygon, (h, w))
                    d.in_lane = d.lane_overlap >= 0.18
                    d.box_height_ratio = (d.y2 - d.y1) / max(1, h)

                    update_ttc(d, history, video_time)
                    d.risk, d.action = expert.decide(d)

                    stats.total_detections += 1
                    stats.object_counts[d.name] += 1
                    stats.level_counts[d.risk] += 1

                    if math.isfinite(d.ttc_s):
                        stats.min_ttc = min(stats.min_ttc, d.ttc_s)

                cached_detections = detections
            else:
                detections = cached_detections

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

            now = time.time()
            if now - fps_started >= 1.0:
                live_fps = fps_frames / max(0.001, now - fps_started)
                stats.processing_fps_samples.append(live_fps)
                fps_frames = 0
                fps_started = now

            draw_lane(frame, polygon, lane_detected)

            for d in detections:
                draw_detection(frame, d)

            draw_panel(
                frame,
                level,
                action,
                len(detections),
                live_fps,
                min_ttc,
            )

            current_text = f"CURRENT VIDEO: {path.name}"
            next_text = (
                f"NEXT VIDEO: {next_video_name}"
                if next_video_name
                else "NEXT VIDEO: None"
            )

            cv2.putText(
                frame,
                current_text,
                (18, max(24, h - 48)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                next_text,
                (18, max(24, h - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if fresh_detection:
                incident_paths: dict[str, str] = {}

                for d in detections:
                    incident = None

                    if d.risk in {"WARNING", "CRITICAL"} and d.in_lane:
                        incident = save_incident(
                            frame,
                            path.name,
                            stats.frames,
                            d,
                            last_incident,
                            video_time,
                        )

                        if incident:
                            stats.incidents += 1
                            incident_paths[d.track_key] = str(incident)

                    csv_writer.writerow([
                        stats.frames,
                        round(video_time, 3),
                        d.name,
                        d.source,
                        round(d.confidence, 4),
                        d.distance_m,
                        round(d.closing_speed_mps, 3),
                        "" if not math.isfinite(d.ttc_s) else round(d.ttc_s, 3),
                        round(d.lane_overlap, 3),
                        d.in_lane,
                        d.risk,
                        d.action,
                        incident_paths.get(d.track_key, ""),
                    ])

            writer.write(frame)

            if display:
                cv2.imshow("Advanced AI Road Safety Assistant", frame)

                processing_ms = (time.perf_counter() - frame_started) * 1000.0
                remaining_ms = max(1, int(target_frame_ms - processing_ms))

                key = cv2.waitKey(remaining_ms) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    cv2.waitKey(0)

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

    json_out.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

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
    args = parser.parse_args()

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
                )
            else:
                print("Returning to the video selection menu.")
        else:
            print("\nThat was the last video. Returning to the video selection menu.")


if __name__ == "__main__":
    main()