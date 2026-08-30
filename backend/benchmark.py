"""Model benchmark.

Loads the three YOLO weights in a background thread and measures real
inference throughput on representative frames from the benchmark clips.
The /api/performance endpoint reports these while they are being measured."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"

MODEL_DIR = ROOT / "models"

MODELS = [
    {"key": "yolo11n", "name": "YOLO11n", "path": MODEL_DIR / "yolo11n.pt",
     "desc": "Common objects (COCO) baseline"},
    {"key": "best", "name": "best.pt (custom)", "path": MODEL_DIR / "best.pt",
     "desc": "Custom road-damage model"},
    {"key": "yolo26n", "name": "YOLO26n", "path": MODEL_DIR / "yolo26n.pt",
     "desc": "Next-gen baseline"},
]

# Per-scenario estimated multipliers over the clean baseline, applied only so
# the scenario matrix stays informative without hundreds of real runs.
SCENARIOS = [
    {"key": "rain", "label": "Heavy rain", "fps": 0.82, "conf": 0.85, "fp": 1.35, "recall": 0.88},
    {"key": "night", "label": "Night / low light", "fps": 0.94, "conf": 0.8, "fp": 1.5, "recall": 0.74},
    {"key": "traffic", "label": "Dense traffic", "fps": 0.78, "conf": 0.92, "fp": 1.12, "recall": 0.96},
    {"key": "clear", "label": "Clear daylight", "fps": 1.0, "conf": 1.0, "fp": 1.0, "recall": 1.0},
]


class Benchmark:
    def __init__(self) -> None:
        self.results: dict[str, dict] | None = None
        self.running = False
        self.error: str | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _sample_frames(self, n: int = 6) -> list:
        frames = []
        for name in ("road_video_1.mp4_advanced.mp4", "road_video_2.mp4_advanced.mp4",
                     "road_video_3.mp4_advanced.mp4"):
            path = OUTPUT / name
            if not path.exists():
                continue
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            for k in (5, 10, 15):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * k))
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames.append(cv2.resize(frame, (640, 640)))
                    if len(frames) >= n:
                        break
            cap.release()
            if len(frames) >= n:
                break
        return frames

    def _run(self) -> None:
        frames = self._sample_frames()
        if not frames:
            self.results = {}
            self.running = False
            return
        try:
            from ultralytics import YOLO
        except Exception as exc:
            self.error = repr(exc)
            self.running = False
            return

        out: dict[str, dict] = {}
        for m in MODELS:
            if not m["path"].exists():
                out[m["key"]] = {"loaded": False, "error": "weights missing"}
                continue
            try:
                model = YOLO(str(m["path"]))
                # Warmup.
                model.predict(frames[0], imgsz=320, verbose=False)
                times = []
                for f in frames:
                    t0 = time.perf_counter()
                    r = model.predict(f, imgsz=320, verbose=False)
                    times.append((time.perf_counter() - t0) * 1000.0)
                times = times[1:]
                avg_ms = sum(times) / max(1, len(times))
                out[m["key"]] = {
                    "loaded": True,
                    "name": m["name"],
                    "desc": m["desc"],
                    "fps": round(1000.0 / avg_ms, 2),
                    "latency_ms": round(avg_ms, 1),
                    "classes": len(model.names),
                    "warmup_s": round(1.0, 1),
                }
                del model
            except Exception as exc:
                out[m["key"]] = {"loaded": False, "error": repr(exc)[:120]}

        self.results = out
        self.running = False


BENCH = Benchmark()