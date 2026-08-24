"""Regenerate frontend/src/videos.generated.js from the output/ folder.

Reads output/<name>_summary.json + output/<name>_detections.csv produced by
main.py, copies the processed ("trained") videos into frontend/public/videos/
and writes a JS module exporting the VIDEOS array used by the gallery.

Run from the repo root:  python generate_frontend_data.py
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
FRONTEND = ROOT / "frontend"
PUBLIC_VIDEOS = FRONTEND / "public" / "videos"
GEN_MODULE = FRONTEND / "src" / "videos.generated.js"

# Thumbnails are written next to the processed videos and served by the dev
# server; keep them small so the gallery stays snappy.
THUMB_W = 640
THUMB_Q = 78

LEVEL_PRIORITY = {"SAFE": 0, "CAUTION": 1, "WARNING": 2, "CRITICAL": 3}

# Classes that matter for a road-safety dashboard. Anything else (giraffes,
# potted plants, frisbees, ...) is YOLO noise and is dropped from the stats.
RELEVANT_CLASSES = {
    "car", "truck", "bus", "motorcycle", "bicycle", "person",
    "traffic light", "stop sign", "parking meter",
    "pothole", "road crack", "road_crack", "crack",
    "longitudinal", "transverse", "alligator",
}

# Nicer names for classes that are technically correct but confusing as-is.
DISPLAY_NAMES = {
    "road_crack": "road crack",
    "crack": "road crack",
    "longitudinal": "longitudinal crack",
    "transverse": "transverse crack",
    "alligator": "alligator cracking",
}

# Hand-written presentation metadata for the benchmark clips.
META = {
    "road_video_1.mp4.MP4": {
        "title": "Urban Traffic and Pedestrian Hazard Detection",
        "location": "City centre, 4-lane arterial",
        "weather": "Heavy rain",
    },
    "road_video_2.mp4.MP4": {
        "title": "Multi-Vehicle Collision Risk and Safety Analysis",
        "location": "Two-lane residential",
        "weather": "Clear",
    },
    "road_video_3.mp4.mp4": {
        "title": "Pothole and Road-Surface Hazard Detection",
        "location": "Dual carriageway",
        "weather": "Night, dry",
    },
}


def mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def to_browser_playable(src: Path, dest: Path) -> None:
    """Copy + transcode the processed video to H.264 so browsers can play it."""
    if imageio_ffmpeg is None:
        shutil.copy2(str(src), str(dest))
        return
    try:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        shutil.copy2(str(src), str(dest))
        return
    subprocess.run(
        [
            ffmpeg, "-y", "-i", str(src),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(dest),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not dest.exists() or dest.stat().st_size == 0:
        shutil.copy2(str(src), str(dest))


def detect_key(row: dict) -> tuple:
    ttc = row.get("ttc_s")
    try:
        ttc = float(ttc) if ttc not in ("", None) else -999.0
    except (TypeError, ValueError):
        ttc = -999.0
    return (
        LEVEL_PRIORITY.get(row.get("risk", "SAFE"), 0),
        str(row.get("in_lane", "False")) == "True",
        -ttc,
        float(row.get("confidence", 0.0)),
    )


def build_events(rows: list[dict], duration: float, w: int, h: int) -> list[dict]:
    """Turn the per-frame detection log into a compact risk timeline."""
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        try:
            by_frame[int(row["frame"])].append(row)
        except (KeyError, ValueError):
            continue

    frames = sorted(by_frame)
    events: list[dict] = []
    last_level_prio = -1
    last_time = -999.0
    min_ttc_time = None
    min_ttc = 999.0

    for frame in frames:
        best = max(by_frame[frame], key=detect_key)
        risk = best.get("risk", "SAFE")
        prio = LEVEL_PRIORITY.get(risk, 0)
        try:
            t = float(best["video_time_s"])
        except (TypeError, ValueError):
            t = 0.0

        ttc = best.get("ttc_s")
        try:
            ttc = float(ttc) if ttc not in ("", None) else None
        except (TypeError, ValueError):
            ttc = None
        if ttc is not None and ttc < min_ttc:
            min_ttc = ttc
            min_ttc_time = t

        # Record a new event when risk rises, or after a gap with risk >= WARNING.
        if prio > last_level_prio or (prio >= 2 and t - last_time >= 2.0):
            if prio >= 1 or not events:
                events.append(make_event(best, t, duration, w, h))
                last_level_prio = max(last_level_prio, prio)
                last_time = t
                if len(events) >= 12:
                    break

    # Always surface the single closest call, even if it fell between samples.
    if min_ttc_time is not None and not any(abs(e.get("_t", -999.0) - min_ttc_time) < 1.0 for e in events):
        by_t = {float(r["video_time_s"]): r for rows_ in by_frame.values() for r in rows_}
        near = min(by_t, key=lambda k: abs(k - min_ttc_time))
        events.append(make_event(by_t[near], near, duration, w, h))

    events.sort(key=lambda e: e["t"])
    for e in events:
        e.pop("_t", None)
    return events


def make_event(row: dict, t: float, duration: float, w: int, h: int) -> dict:
    try:
        x1, y1, x2, y2 = (int(float(row[k])) for k in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        x1 = y1 = x2 = y2 = 0
    obj = row.get("object", "object")
    conf = float(row.get("confidence", 0.0))
    try:
        distance = float(row.get("distance_m", 0.0))
    except (TypeError, ValueError):
        distance = 0.0
    ttc = row.get("ttc_s")
    try:
        ttc = float(ttc) if ttc not in ("", None) else None
    except (TypeError, ValueError):
        ttc = None

    return {
        "t": mmss(t),
        "_t": round(t, 2),
        "pct": round(min(100.0, max(0.0, t / duration * 100.0 if duration > 0 else 0.0)), 1),
        "level": row.get("risk", "SAFE"),
        "object": obj,
        "label": row.get("action", "DETECTED OBJECT").replace("_", " ").title(),
        "action": row.get("action", "CONTINUE CAREFULLY").replace("_", " "),
        "distance_m": round(distance, 1),
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


def overall_risk(risk_counts: dict) -> str:
    """Weighted risk score -> level, so a clip with one CRITICAL frame is not
    automatically labelled CRITICAL when almost everything else is SAFE."""
    weights = {"SAFE": 0, "CAUTION": 1, "WARNING": 2, "CRITICAL": 3}
    total = sum(risk_counts.values()) or 1
    score = sum(weights[l] * risk_counts.get(l, 0) for l in weights) / total
    if score >= 0.60:
        return "CRITICAL"
    if score >= 0.35:
        return "WARNING"
    if score >= 0.15:
        return "CAUTION"
    return "SAFE"


def make_thumb(src: Path, dest: Path, events: list[dict], duration: float) -> None:
    """Save a representative frame as a JPEG for the gallery card.

    Picks the frame of the highest-risk event when available, otherwise a frame
    about a third of the way into the clip."""
    target = None
    for e in events:
        if e.get("level") == "CRITICAL":
            target = e.get("_t")
            break
    if target is None and events:
        mid = duration / 3.0
        target = min(events, key=lambda e: abs(e.get("_t", 0.0) - mid)).get("_t")
    if target is None:
        target = duration / 3.0

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(target * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return

    h, w = frame.shape[:2]
    if w > THUMB_W:
        scale = THUMB_W / w
        frame = cv2.resize(frame, (THUMB_W, int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dest), frame, [cv2.IMWRITE_JPEG_QUALITY, THUMB_Q])


def main() -> None:
    PUBLIC_VIDEOS.mkdir(parents=True, exist_ok=True)
    videos = []

    for summary_path in sorted(OUTPUT.glob("*_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        raw_name = summary.get("video", summary_path.name.replace("_summary.json", ""))
        stem = summary_path.name.replace("_summary.json", "")
        csv_path = OUTPUT / f"{stem}_detections.csv"
        src_video = OUTPUT / f"{stem}_advanced.mp4"

        if not csv_path.exists() or not src_video.exists():
            print(f"Skipping {stem}: missing detections.csv or advanced video")
            continue

        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        cap = cv2.VideoCapture(str(src_video))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        cap.release()

        clean = [r for r in rows if r.get("object", "") in RELEVANT_CLASSES]
        for r in clean:
            r["object"] = DISPLAY_NAMES.get(r["object"], r["object"])

        object_counts: dict[str, int] = {}
        risk_counts: dict[str, int] = {}
        for r in clean:
            object_counts[r["object"]] = object_counts.get(r["object"], 0) + 1
            risk_counts[r.get("risk", "SAFE")] = risk_counts.get(r.get("risk", "SAFE"), 0) + 1

        duration = max((float(r["video_time_s"]) for r in clean if r.get("video_time_s")), default=0.0)
        events = build_events(clean, duration, w, h)

        meta = META.get(raw_name, {})
        dest_name = f"{stem}_processed.mp4"
        thumb_name = f"{stem}_thumb.jpg"
        to_browser_playable(src_video, PUBLIC_VIDEOS / dest_name)
        make_thumb(src_video, PUBLIC_VIDEOS / thumb_name, events, duration)

        videos.append({
            "id": f"video_{len(videos) + 1}",
            "title": meta.get("title", stem.replace("_", " ").title()),
            "file": dest_name,
            "thumb": thumb_name,
            "location": meta.get("location", "Untagged route"),
            "weather": meta.get("weather", "Unknown"),
            "duration": mmss(duration),
            "frames": summary.get("frames", 0),
            "total_detections": sum(object_counts.values()),
            "incidents": summary.get("incidents", 0),
            "minimum_ttc_s": summary.get("minimum_ttc_s"),
            "average_processing_fps": summary.get("average_processing_fps", 0.0),
            "risk_counts": risk_counts,
            "object_counts": object_counts,
            "overall_risk": overall_risk(risk_counts),
            "events": events,
        })
        print(f"Built {stem}: {len(events)} events, {sum(object_counts.values())} rows, {w}x{h}")

    if not videos:
        print("No output data found. Run main.py first.")
        return

    module = (
        "// AUTO-GENERATED by generate_frontend_data.py — do not edit by hand.\n"
        "// Run `python generate_frontend_data.py` to rebuild from output/.\n\n"
        "export const VIDEOS = "
        + json.dumps(videos, indent=2, ensure_ascii=False)
        + ";\n"
    )
    GEN_MODULE.write_text(module, encoding="utf-8")
    print(f"\nWrote {GEN_MODULE} ({len(videos)} videos)")


if __name__ == "__main__":
    main()
