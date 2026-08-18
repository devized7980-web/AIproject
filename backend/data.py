"""Load the real pipeline output (output/*_summary.json + *_detections.csv)
into in-memory structures served by the API."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"

LEVELS = ["SAFE", "CAUTION", "WARNING", "CRITICAL"]
LEVEL_PRIORITY = {"SAFE": 0, "CAUTION": 1, "WARNING": 2, "CRITICAL": 3}
LEVEL_WEIGHT = {"SAFE": 0, "CAUTION": 1, "WARNING": 2, "CRITICAL": 3}

# Classes that matter for a road-safety dashboard (mirrors the frontend
# generator so counts stay consistent between static and live builds).
RELEVANT_CLASSES = {
    "car", "truck", "bus", "motorcycle", "bicycle", "person",
    "traffic light", "stop sign", "parking meter",
    "pothole", "road crack", "road_crack", "crack",
    "longitudinal", "transverse", "alligator",
}

DISPLAY_NAMES = {
    "road_crack": "road crack",
    "crack": "road crack",
    "longitudinal": "longitudinal crack",
    "transverse": "transverse crack",
    "alligator": "alligator cracking",
}

META = {
    "road_video_1.mp4.MP4": {
        "title": "Urban Traffic and Pedestrian Hazard Detection",
        "location": "City centre, 4-lane arterial",
        "weather": "Heavy rain",
        "raw": "road_video_1.mp4.MP4",
    },
    "road_video_2.mp4.MP4": {
        "title": "Multi-Vehicle Collision Risk and Safety Analysis",
        "location": "Two-lane residential",
        "weather": "Clear",
        "raw": "road_video_2.mp4.MP4",
    },
    "road_video_3.mp4.mp4": {
        "title": "Pothole and Road-Surface Hazard Detection",
        "location": "Dual carriageway",
        "weather": "Night, dry",
        "raw": "road_video_3.mp4.mp4",
    },
}


def mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _f(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _maybe_float(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def overall_risk(risk_counts: dict) -> str:
    total = sum(risk_counts.values()) or 1
    score = sum(LEVEL_WEIGHT[l] * risk_counts.get(l, 0) for l in LEVEL_WEIGHT) / total
    if score >= 0.60:
        return "CRITICAL"
    if score >= 0.35:
        return "WARNING"
    if score >= 0.15:
        return "CAUTION"
    return "SAFE"


def make_event(row: dict, t: float, duration: float, w: int, h: int) -> dict:
    x1, y1, x2, y2 = (_f(row.get(k)) for k in ("x1", "y1", "x2", "y2"))
    obj = DISPLAY_NAMES.get(row.get("object", ""), row.get("object", "object"))
    conf = _f(row.get("confidence"))
    distance = round(_f(row.get("distance_m")), 1)
    ttc = _maybe_float(row.get("ttc_s"))
    risk = row.get("risk", "SAFE")
    action = row.get("action", "CONTINUE CAREFULLY").replace("_", " ").title()
    return {
        "t": mmss(t),
        "pct": round(min(100.0, max(0.0, t / duration * 100.0 if duration > 0 else 0.0)), 1),
        "level": risk,
        "object": obj,
        "label": action,
        "action": action.upper(),
        "distance_m": distance,
        "ttc_s": None if ttc is None else round(ttc, 2),
        "confidence": round(conf, 2),
        "boxes": [
            {
                "x": round(x1 / w * 100.0, 1) if w else 0,
                "y": round(y1 / h * 100.0, 1) if h else 0,
                "w": round((x2 - x1) / w * 100.0, 1) if w else 0,
                "h": round((y2 - y1) / h * 100.0, 1) if h else 0,
                "tag": f"{obj} {conf:.0%}",
            }
        ],
    }


def build_events(rows: list[dict], duration: float, w: int, h: int) -> list[dict]:
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        try:
            by_frame[int(_f(row["frame"]))].append(row)
        except (KeyError, ValueError):
            continue
    frames = sorted(by_frame)
    events: list[dict] = []
    last_level_prio = -1
    last_time = -999.0
    for frame in frames:
        best = max(
            by_frame[frame],
            key=lambda r: (LEVEL_PRIORITY.get(r.get("risk", "SAFE"), 0), -_f(r.get("ttc_s"), -999), _f(r.get("confidence"))),
        )
        risk = best.get("risk", "SAFE")
        prio = LEVEL_PRIORITY.get(risk, 0)
        t = _f(best.get("video_time_s"))
        if prio > last_level_prio or (prio >= 2 and t - last_time >= 2.0):
            if prio >= 1 or not events:
                events.append(make_event(best, t, duration, w, h))
                last_level_prio = max(last_level_prio, prio)
                last_time = t
                if len(events) >= 12:
                    break
    events.sort(key=lambda e: e["pct"])
    return events


class DataStore:
    """Loads output/ once and exposes structured views for the API."""

    def __init__(self, output: Path = OUTPUT) -> None:
        self.output = output
        self.videos: list[dict] = []
        self.frames: dict[str, list[dict]] = {}
        self.alerts: list[dict] = []
        self.load()

    def load(self) -> None:
        for summary_path in sorted(self.output.glob("*_summary.json")):
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            stem = summary_path.name.replace("_summary.json", "")
            raw_name = summary.get("video", stem)
            meta = META.get(raw_name, {})
            csv_path = self.output / f"{stem}_detections.csv"
            if not csv_path.exists():
                continue

            with csv_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            clean = []
            for r in rows:
                obj = r.get("object", "")
                if obj in RELEVANT_CLASSES or DISPLAY_NAMES.get(obj):
                    r["object"] = DISPLAY_NAMES.get(obj, obj)
                    clean.append(r)

            duration = max((_f(r.get("video_time_s")) for r in clean), default=0.0)

            # Video resolution (used to normalise box coordinates).
            w, h = self._video_size(stem, int(_f(summary.get("width"), 1280)), 720)

            object_counts: dict[str, int] = {}
            risk_counts: dict[str, int] = {}
            for r in clean:
                object_counts[r["object"]] = object_counts.get(r["object"], 0) + 1
                rc = r.get("risk", "SAFE")
                risk_counts[rc] = risk_counts.get(rc, 0) + 1

            # Full counts from the summary include non-relevant ("noise")
            # classes — used as a real false-positive proxy.
            raw_counts = summary.get("object_counts", {})
            noise_counts = {k: v for k, v in raw_counts.items()
                            if k not in RELEVANT_CLASSES
                            and DISPLAY_NAMES.get(k, k) not in object_counts}

            events = build_events(clean, duration, w, h)
            vid = f"video_{len(self.videos) + 1}"
            video = {
                "id": vid,
                "title": meta.get("title", stem.replace("_", " ").title()),
                "file": f"{stem}_processed.mp4",
                "raw": meta.get("raw", ""),
                "thumb": f"{stem}_thumb.jpg",
                "location": meta.get("location", "Untagged route"),
                "weather": meta.get("weather", "Unknown"),
                "duration": mmss(duration),
                "frames": int(_f(summary.get("frames"))),
                "total_detections": sum(object_counts.values()),
                "incidents": int(_f(summary.get("incidents"))),
                "minimum_ttc_s": _maybe_float(summary.get("minimum_ttc_s")),
                "average_processing_fps": round(_f(summary.get("average_processing_fps")), 2),
                "processing_seconds": round(_f(summary.get("processing_seconds")), 1),
                "risk_counts": risk_counts,
                "object_counts": object_counts,
                "noise_counts": noise_counts,
                "overall_risk": overall_risk(risk_counts),
                "events": events,
                "raw_name": raw_name,
                "stem": stem,
            }
            self.videos.append(video)
            self.frames[vid] = [self._frame_row(r, w, h) for r in clean]

        # Alerts: the highest-risk transitions, newest first.
        for v in self.videos:
            for e in v["events"]:
                if LEVEL_PRIORITY.get(e["level"], 0) >= 2:
                    self.alerts.append({
                        "id": f"{v['id']}-{len(self.alerts)}",
                        "video_id": v["id"],
                        "video_title": v["title"],
                        "time": e["t"],
                        "pct": e["pct"],
                        "level": e["level"],
                        "object": e["object"],
                        "label": e["label"],
                        "distance_m": e["distance_m"],
                        "ttc_s": e["ttc_s"],
                        "confidence": e["confidence"],
                        "status": "open",
                        "assignee": None,
                    })
        self.alerts.sort(
            key=lambda a: (LEVEL_PRIORITY.get(a["level"], 0), a["video_id"]),
            reverse=True,
        )

    def _video_size(self, stem: str, default_w: int, default_h: int) -> tuple[int, int]:
        try:
            import cv2  # noqa: PLC0415
        except Exception:
            return default_w, default_h
        for name in (f"{stem}_advanced.mp4",):
            path = self.output / name
            if path.exists():
                cap = cv2.VideoCapture(str(path))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or default_w
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or default_h
                cap.release()
                return w, h
        return default_w, default_h

    def _frame_row(self, r: dict, w: int, h: int) -> dict:
        x1, y1, x2, y2 = (_f(r.get(k)) for k in ("x1", "y1", "x2", "y2"))
        ttc = _maybe_float(r.get("ttc_s"))
        return {
            "frame": int(_f(r.get("frame"))),
            "t": round(_f(r.get("video_time_s")), 3),
            "object": DISPLAY_NAMES.get(r.get("object", ""), r.get("object", "")),
            "source": r.get("source", ""),
            "conf": round(_f(r.get("confidence")), 3),
            "x": round(x1 / w * 100.0, 2) if w else 0,
            "y": round(y1 / h * 100.0, 2) if h else 0,
            "w": round((x2 - x1) / w * 100.0, 2) if w else 0,
            "h": round((y2 - y1) / h * 100.0, 2) if h else 0,
            "distance_m": round(_f(r.get("distance_m")), 2),
            "ttc_s": ttc,
            "in_lane": str(r.get("in_lane", "False")).lower() == "true",
            "risk": r.get("risk", "SAFE"),
            "action": r.get("action", "").replace("_", " "),
        }

    def counts(self) -> dict:
        total = {k: 0 for k in LEVELS}
        objects: dict[str, int] = {}
        for v in self.videos:
            for lv in LEVELS:
                total[lv] += v["risk_counts"].get(lv, 0)
            for obj, n in v["object_counts"].items():
                objects[obj] = objects.get(obj, 0) + n
        return {
            "clips": len(self.videos),
            "frames": sum(v["frames"] for v in self.videos),
            "detections": sum(v["total_detections"] for v in self.videos),
            "incidents": sum(v["incidents"] for v in self.videos),
            "min_ttc": min((v["minimum_ttc_s"] for v in self.videos if v["minimum_ttc_s"]), default=None),
            "avg_fps": round(sum(v["average_processing_fps"] for v in self.videos) / max(1, len(self.videos)), 2),
            "risk_counts": total,
            "object_counts": objects,
        }


STORE = DataStore()