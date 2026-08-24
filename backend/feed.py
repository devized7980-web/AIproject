"""Live feed simulator.

Replays the real per-frame detections recorded by the pipeline (from
output/*_detections.csv) as a continuous live stream, exactly as if the
camera were running right now. Every frame message carries telemetry
(fps, latency, safety state, counts) plus the detection boxes so the
command center can overlay them on the video."""

from __future__ import annotations

import asyncio
import json
import random
import threading
import time
from collections import defaultdict
from typing import Any

from .data import DataStore, LEVEL_PRIORITY, LEVELS

TICK_S = 0.21  # ~4.8 fps, matching the real pipeline's ~4.5 fps


class Broadcaster:
    """Fan-out a single message to every connected WebSocket."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[Any] = set()

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, ws: Any) -> None:
        self._clients.add(ws)

    async def disconnect(self, ws: Any) -> None:
        self._clients.discard(ws)

    async def _broadcast(self, msg: dict) -> None:
        if not self._clients:
            return
        text = json.dumps(msg)
        for ws in list(self._clients):
            try:
                await ws.send_text(text)
            except Exception:
                self._clients.discard(ws)

    def publish(self, msg: dict) -> None:
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._broadcast(msg), self._loop)


class LiveFeed:
    def __init__(self, store: DataStore, broadcaster: Broadcaster) -> None:
        self.store = store
        self.broadcaster = broadcaster
        self.lock = threading.Lock()
        self.video_id = store.videos[0]["id"] if store.videos else None
        self.snapshot: dict = self._blank_snapshot()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._alert_ring: dict[str, float] = {}
        self._ring: dict[str, list[int]] = defaultdict(list)

    def _blank_snapshot(self) -> dict:
        return {
            "type": "frame",
            "video_id": self.video_id,
            "video_title": "",
            "frame": 0,
            "video_time": 0.0,
            "duration": 0.0,
            "fps": 0.0,
            "latency_ms": 0.0,
            "state": {"level": "SAFE", "action": "INITIALISING"},
            "counts": {"potholes": 0, "vehicles": 0, "persons": 0, "total": 0},
            "detections": [],
            "alert": None,
            "cumulative": {"frames": 0, "detections": 0, "alerts": 0},
            "ring": {"potholes": [0] * 24, "vehicles": [0] * 24, "persons": [0] * 24},
            "ts": 0,
        }

    def select(self, video_id: str) -> bool:
        with self.lock:
            if not any(v["id"] == video_id for v in self.store.videos):
                return False
            self.video_id = video_id
            self.snapshot = self._blank_snapshot()
            self._ring.clear()
            self._alert_ring.clear()
            return True

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def get_snapshot(self) -> dict:
        with self.lock:
            return dict(self.snapshot)

    def _frames_of(self, video_id: str) -> list[tuple[int, list[dict]]]:
        """Group per-frame detections into (frame_no, rows)."""
        by_frame: dict[int, list[dict]] = defaultdict(list)
        for row in self.store.frames.get(video_id, []):
            by_frame[row["frame"]].append(row)
        frames = sorted(by_frame)
        if not frames:
            return [(0, [])]
        # Pad so the feed feels continuous even through empty stretches.
        seq: list[tuple[int, list[dict]]] = []
        last = 0
        for f in frames:
            if f - last > 2:
                for gap in range(last + 1, f):
                    seq.append((gap, []))
            seq.append((f, by_frame[f]))
            last = f
        return seq

    def _run(self) -> None:
        while not self._stop.is_set():
            with self.lock:
                video_id = self.video_id
            video = next((v for v in self.store.videos if v["id"] == video_id), None)
            if video is None:
                time.sleep(TICK_S)
                continue
            frames = self._frames_of(video_id)
            i = 0
            while not self._stop.is_set() and i < len(frames):
                frame_no, rows = frames[i]
                msg = self._build(video, frame_no, rows)
                self.broadcaster.publish(msg)
                i += 1
                time.sleep(TICK_S)

    def _build(self, video: dict, frame_no: int, rows: list[dict]) -> dict:
        now = time.time()
        with self.lock:
            base = dict(self.snapshot)
            base.pop("type", None)
        fps = round(random.uniform(4.2, 5.3), 2)
        latency = round(1000.0 / fps + random.uniform(2, 14), 1)

        detections = []
        counts = {"potholes": 0, "vehicles": 0, "persons": 0, "total": 0}
        worst = None
        worst_prio = -1
        for r in rows:
            name = r["object"]
            if name in {"pothole", "road crack", "longitudinal crack", "transverse crack", "alligator cracking"}:
                counts["potholes"] += 1
            elif name in {"car", "truck", "bus", "motorcycle", "bicycle"}:
                counts["vehicles"] += 1
            elif name == "person":
                counts["persons"] += 1
            counts["total"] += 1
            prio = LEVEL_PRIORITY.get(r["risk"], 0)
            if prio > worst_prio:
                worst_prio = prio
                worst = r
            detections.append({
                "x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"],
                "conf": r["conf"], "name": name, "risk": r["risk"],
                "distance_m": r["distance_m"], "ttc_s": r["ttc_s"],
                "in_lane": r["in_lane"], "source": r["source"],
                "track_id": r.get("track_id"),
            })

        state = {"level": "SAFE", "action": "ROAD CLEAR — CONTINUE CAREFULLY"}
        if worst is not None:
            state = {"level": worst["risk"], "action": worst["action"].upper()}

        # Rolling window for the route schematic.
        self._ring["potholes"].append(counts["potholes"])
        self._ring["vehicles"].append(counts["vehicles"])
        self._ring["persons"].append(counts["persons"])
        for k in self._ring:
            if len(self._ring[k]) > 24:
                self._ring[k] = self._ring[k][-24:]

        # Emit an alert the first time a WARNING/CRITICAL object appears.
        alert = None
        if worst is not None and worst_prio >= 2:
            key = f"{video['id']}:{worst['object']}:{worst['risk']}"
            if now - self._alert_ring.get(key, -999.0) > 4.0:
                self._alert_ring[key] = now
                alert = {
                    "id": f"live-{int(now * 1000)}",
                    "video_id": video["id"],
                    "video_title": video["title"],
                    "time": f"{int(worst.get('t', 0) // 60):02d}:{int(worst.get('t', 0) % 60):02d}",
                    "level": worst["risk"],
                    "object": worst["object"],
                    "label": worst["action"],
                    "distance_m": worst["distance_m"],
                    "ttc_s": worst["ttc_s"],
                    "confidence": worst["conf"],
                    "status": "open",
                    "assignee": None,
                }

        with self.lock:
            snap = self.snapshot
            snap["type"] = "frame"
            snap["video_id"] = video["id"]
            snap["video_title"] = video["title"]
            snap["frame"] = frame_no
            snap["video_time"] = frame_no / 30.0
            snap["duration"] = video["frames"] / 30.0
            snap["fps"] = fps
            snap["latency_ms"] = latency
            snap["state"] = state
            snap["counts"] = counts
            snap["detections"] = detections
            snap["alert"] = alert
            snap["cumulative"]["frames"] += 1
            snap["cumulative"]["detections"] += counts["total"]
            if alert is not None:
                snap["cumulative"]["alerts"] += 1
            snap["ring"]["potholes"] = list(self._ring["potholes"])
            snap["ring"]["vehicles"] = list(self._ring["vehicles"])
            snap["ring"]["persons"] = list(self._ring["persons"])
            snap["ts"] = int(now * 1000)
            return dict(snap)


FEED = None  # set in server.py


def build_feed(store: DataStore, broadcaster: Broadcaster) -> LiveFeed:
    global FEED
    FEED = LiveFeed(store, broadcaster)
    return FEED
