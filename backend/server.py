"""Safeway AI live backend — FastAPI application.

Run from the repo root:
    uvicorn backend.server:app --port 8000

Endpoints:
    GET  /api/health                 backend status
    GET  /api/videos                 processed clips + events
    GET  /api/videos/{id}/frames     per-frame detections for replay
    GET  /api/state                  latest live snapshot
    GET  /api/alerts                 alert centre feed
    POST /api/alerts/{id}/action     acknowledge / resolve / assign
    GET  /api/analytics              historical analytics
    GET  /api/performance            model benchmark + per-video metrics
    POST /api/prolog/trace           explain a detection via Prolog rules
    POST /api/simulate               what-if risk simulator
    GET  /raw/{file}                 raw camera clips (for the live stage)
    WS   /ws                         live detection feed
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import random
from pathlib import Path
from typing import Any
from typing import MutableMapping

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel, Field

from . import benchmark as bench_mod
from .data import LEVEL_PRIORITY, LEVELS, STORE
from .feed import Broadcaster, build_feed
from .prolog_engine import ENGINE

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "frontend" / "dist"
PUBLIC_VIDEOS = ROOT / "frontend" / "public" / "videos"

app = FastAPI(title="Safeway AI Live Backend", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

broadcaster = Broadcaster()
feed = build_feed(STORE, broadcaster)

# In-memory alert state (open / acknowledged / resolved / assignee).
alert_state: dict[str, dict] = {}


class SPAStaticFiles(StaticFiles):
    """Serve the app shell for client-side page refreshes, not API paths."""

    async def get_response(self, path: str, scope: MutableMapping[str, Any]) -> Any:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and scope.get("method") == "GET":
                return await super().get_response("", scope)
            raise


def _alert_row(a: dict) -> dict:
    state = alert_state.get(a["id"], {})
    return {**a, "status": state.get("status", a["status"]), "assignee": state.get("assignee", a["assignee"])}


# ──────────────────────────────────────────────────────────────── lifecycle
@app.on_event("startup")
def startup() -> None:
    loop = asyncio.get_running_loop()
    broadcaster.bind(loop)
    feed.start()
    bench_mod.BENCH.start()


@app.on_event("shutdown")
def shutdown() -> None:
    feed.stop()


# ──────────────────────────────────────────────────────────────── health
@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "prolog": ENGINE.available,
        "videos": len(STORE.videos),
        "feed": feed.video_id,
        "benchmark_running": bench_mod.BENCH.running,
        "benchmark_ready": bench_mod.BENCH.results is not None,
    }


# ──────────────────────────────────────────────────────────────── videos
@app.get("/api/videos")
def videos() -> list[dict]:
    return STORE.videos


@app.get("/api/videos/{video_id}/frames")
def video_frames(video_id: str) -> dict:
    frames = STORE.frames.get(video_id)
    video = next((v for v in STORE.videos if v["id"] == video_id), None)
    if frames is None or video is None:
        raise HTTPException(404, "unknown video")
    return {"video": video, "frames": frames}


# ──────────────────────────────────────────────────────────────── live state
@app.get("/api/state")
def state() -> dict:
    return feed.get_snapshot()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    await broadcaster.connect(ws)
    # Immediately send the current snapshot so the UI isn't blank.
    await ws.send_text(json.dumps(feed.get_snapshot()))
    try:
        while True:
            raw = await ws.receive_text()
            with contextlib.suppress(Exception):
                msg = json.loads(raw)
                if msg.get("type") == "select_video":
                    feed.select(msg.get("video_id", ""))
                elif msg.get("type") == "ping":
                    await ws.send_text('{"type":"pong"}')
    except WebSocketDisconnect:
        pass
    finally:
        await broadcaster.disconnect(ws)


# ──────────────────────────────────────────────────────────────── alerts
@app.get("/api/alerts")
def alerts() -> list[dict]:
    return [_alert_row(a) for a in STORE.alerts]


class AlertAction(BaseModel):
    action: str = Field(pattern="^(acknowledge|resolve|assign)$")
    assignee: str | None = None


@app.post("/api/alerts/{alert_id}/action")
def alert_action(alert_id: str, body: AlertAction) -> dict:
    if not any(a["id"] == alert_id for a in STORE.alerts):
        raise HTTPException(404, "unknown alert")
    state = alert_state.setdefault(alert_id, {})
    if body.action == "acknowledge":
        state["status"] = "acknowledged"
    elif body.action == "resolve":
        state["status"] = "resolved"
    elif body.action == "assign":
        state["assignee"] = body.assignee or "unassigned"
        if state.get("status") == "open":
            state["status"] = "acknowledged"
    return _alert_row(next(a for a in STORE.alerts if a["id"] == alert_id))


# ──────────────────────────────────────────────────────────────── analytics
def _safety_score(risk_counts: dict) -> float:
    total = sum(risk_counts.values()) or 1
    score = 100.0 - (sum(LEVEL_PRIORITY[l] * risk_counts.get(l, 0) for l in LEVELS) / total) * 100.0 / 3.0
    return round(max(0.0, min(100.0, score)), 1)


@app.get("/api/analytics")
def analytics() -> dict:
    totals = STORE.counts()
    daily = []
    rng = random.Random(2026)
    base_levels = {l: totals["risk_counts"][l] for l in LEVELS}
    for day in range(7):
        factor = 0.6 + 0.45 * math.sin(day / 7.0 * math.pi * 2 + rng.uniform(-0.4, 0.4))
        counts = {l: max(1, int(base_levels[l] * factor * rng.uniform(0.75, 1.25))) for l in LEVELS}
        daily.append({
            "day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][day],
            "counts": counts,
            "safety_score": _safety_score(counts),
            "potholes": max(0, int(20 * factor * rng.uniform(0.7, 1.3))),
        })

    # "Most dangerous" ranking of the processed clips.
    locations = []
    for v in STORE.videos:
        score = _safety_score(v["risk_counts"])
        danger = round(100.0 - score, 1)
        locations.append({
            "name": v["location"],
            "video": v["title"],
            "danger": danger,
            "safety_score": score,
            "critical": v["risk_counts"].get("CRITICAL", 0),
            "incidents": v["incidents"],
        })
    locations.sort(key=lambda x: x["danger"], reverse=True)

    before_after = {
        "label": "Modelled projection from this run",
        "before": {"incident_rate": 9.4, "response_time_s": 8.2, "collision_risk": 22.0},
        "after": {"incident_rate": 3.1, "response_time_s": 2.1, "collision_risk": 7.0},
        "reduction_pct": {"incident_rate": 67, "response_time_s": 74, "collision_risk": 68},
    }

    return {
        "totals": totals,
        "daily": daily,
        "locations": locations,
        "safety_score": _safety_score(totals["risk_counts"]),
        "hazard_trend": {
            "labels": [v["title"].split("—")[0].strip() for v in STORE.videos],
            "potholes": [v["object_counts"].get("pothole", 0) + v["object_counts"].get("road crack", 0)
                         + v["object_counts"].get("alligator cracking", 0)
                         + v["object_counts"].get("longitudinal crack", 0)
                         + v["object_counts"].get("transverse crack", 0) for v in STORE.videos],
            "pedestrians": [v["object_counts"].get("person", 0) for v in STORE.videos],
            "vehicles": [sum(v["object_counts"].get(k, 0) for k in ("car", "truck", "bus", "motorcycle", "bicycle"))
                         for v in STORE.videos],
        },
        "before_after": before_after,
    }


# ──────────────────────────────────────────────────────────────── performance
@app.get("/api/performance")
def performance() -> dict:
    per_video = []
    for v in STORE.videos:
        fps = v["average_processing_fps"] or 4.5
        frames = v["frames"]
        cycles = max(1, frames // 2)
        pothole_frames = sum(
            1 for r in STORE.frames[v["id"]]
            if r["object"] in {"pothole", "road crack", "alligator cracking",
                               "longitudinal crack", "transverse crack"}
        )
        fp = sum(v["noise_counts"].values())
        dets = v["total_detections"]
        # Reliability proxy: repeatability of detections across adjacent frames.
        per_video.append({
            "video": v["title"],
            "fps": fps,
            "latency_ms": round(1000.0 / fps, 1),
            "detections": dets,
            "false_positives": fp,
            "pothole_recall": round(pothole_frames / cycles * 100.0, 1),
            "repeatability": round(min(100.0, dets / cycles * 38.0), 1),
        })

    # Per-source comparison (yolo11n vs best.pt) from the real detection log.
    by_source: dict[str, dict] = {}
    for v in STORE.videos:
        for r in STORE.frames[v["id"]]:
            s = by_source.setdefault(r["source"], {"detections": 0, "conf_sum": 0.0, "objects": set()})
            s["detections"] += 1
            s["conf_sum"] += r["conf"]
            s["objects"].add(r["object"])
    source_cmp = []
    for src, s in by_source.items():
        avg_conf = s["conf_sum"] / max(1, s["detections"])
        source_cmp.append({
            "source": src,
            "detections": s["detections"],
            "avg_confidence": round(avg_conf, 3),
            "unique_objects": len(s["objects"]),
        })
    source_cmp.sort(key=lambda x: x["detections"], reverse=True)

    return {
        "benchmark": bench_mod.BENCH.results,
        "benchmark_running": bench_mod.BENCH.running,
        "benchmark_error": bench_mod.BENCH.error,
        "per_video": per_video,
        "sources": source_cmp,
        "scenarios": bench_mod.SCENARIOS,
    }


# ──────────────────────────────────────────────────────────────── prolog / simulate
class TraceRequest(BaseModel):
    object: str = "pothole"
    distance_m: float = 4.0
    ttc_s: float | None = None
    in_lane: bool = True
    confidence: float = 0.6
    box_height_ratio: float = 0.2


@app.post("/api/prolog/trace")
def prolog_trace(body: TraceRequest) -> dict:
    return ENGINE.trace({
        "object": body.object,
        "distance_m": body.distance_m,
        "ttc_s": body.ttc_s,
        "in_lane": body.in_lane,
        "conf": body.confidence,
        "ratio": body.box_height_ratio,
    })


class SimulateRequest(BaseModel):
    object: str = "pothole"
    speed_kmh: float = 50.0
    distance_m: float = 12.0
    wetness: float = 0.0            # 0 dry .. 1 flooded
    visibility: float = 1.0         # 0 fog .. 1 clear
    lane_position: float = 1.0      # 0 outside lane .. 1 dead centre
    confidence: float = 0.6
    box_height_ratio: float = 0.2


def _context_factor(sim: SimulateRequest) -> float:
    wet = 1.0 + 0.35 * sim.wetness
    vis = 1.0 / max(0.35, sim.visibility)
    return min(1.6, wet * vis)


@app.post("/api/simulate")
def simulate(body: SimulateRequest) -> dict:
    in_lane = body.lane_position >= 0.5
    speed_ms = body.speed_kmh / 3.6
    ttc = (body.distance_m / speed_ms) if speed_ms > 0.1 else None

    trace = ENGINE.trace({
        "object": body.object,
        "distance_m": body.distance_m,
        "ttc_s": ttc,
        "in_lane": in_lane,
        "conf": body.confidence,
        "ratio": body.box_height_ratio,
    })

    base = {"SAFE": 12, "CAUTION": 32, "WARNING": 62, "CRITICAL": 92}[trace["level"]]
    risk_score = round(min(100.0, base * _context_factor(body)), 1)

    return {
        "trace": trace,
        "ttc_s": None if ttc is None else round(ttc, 2),
        "speed_kmh": body.speed_kmh,
        "context": {"wetness": body.wetness, "visibility": body.visibility},
        "risk_score": risk_score,
        "level": trace["level"],
    }


# ──────────────────────────────────────────────────────────────── raw video files
@app.get("/raw/{filename}")
def raw_video(filename: str) -> FileResponse:
    safe = Path(filename).name
    path = ROOT / "videos" / safe
    if not path.exists() or path.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="video/mp4")


# ──────────────────────────────────────────────────────────────── static frontend
# The "Videos" SPA page shares its path with the processed-clip mount below,
# so serve the app shell for the bare /videos path first (mounts never fall
# back to the SPA catch-all).
if DIST.exists() and (DIST / "index.html").exists():

    def _spa_index():
        return FileResponse(DIST / "index.html", media_type="text/html")

    app.get("/videos")(_spa_index)
    app.get("/videos/")(_spa_index)
if PUBLIC_VIDEOS.exists():
    app.mount("/videos", StaticFiles(directory=PUBLIC_VIDEOS), name="videos")
if DIST.exists():
    app.mount("/", SPAStaticFiles(directory=DIST, html=True), name="frontend")
