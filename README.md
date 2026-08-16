# RoadSafetyAI

An AI road-hazard and collision-warning system that analyses dashcam video in real time, flags risks, and presents results in a web dashboard.

## Pipeline

```
dashcam video → YOLO11 object detection + tracking → lane estimation
             → distance / time-to-collision (TTC) estimates → risk rules
             → annotated video + CSV + JSON summary + HTML dashboard
```

- **Object detection & tracking** — Ultralytics YOLO11 (`yolo11n.pt` common COCO classes + a custom `best.pt` model for road damage such as potholes, cracks and alligatoring). ByteTrack keeps stable identities across frames.
- **Lane awareness** — Canny + Hough line detection builds a drivable-corridor polygon (with a fallback) so objects beside the vehicle path are treated differently from objects in the lane.
- **Distance & TTC** — Monocular distance estimation from known object widths and focal length; closing speed and time-to-collision are computed from tracking history.
- **Risk rules** — A Prolog expert system (`expert_system.pl`) converts observations into `SAFE / CAUTION / WARNING / CRITICAL` levels with plain-language actions. A Python fallback applies identical rules when SWI-Prolog is not installed.
- **Outputs** — Every run produces an annotated `*_advanced.mp4`, a per-frame `*_detections.csv`, a `*_summary.json`, and a self-contained `*_dashboard.html`. Incident frames for `WARNING`/`CRITICAL` lane threats are saved under `output/incidents/`.

## Requirements

- Python 3.10+
- [SWI-Prolog](https://www.swi-prolog.org/) (optional — enables the Prolog expert system)
- Node.js 18+ (for the frontend)

```bash
pip install ultralytics opencv-python numpy pyswip pyttsx3
```

## Getting started

```bash
# 1. Make sure the detection models exist (downloads yolo11n.pt if missing)
python download_yolo.py

# 2. Put dashcam clips (.mp4/.avi/.mov/.mkv/.m4v/.wmv) in the repo root

# 3. Run the detector on one video at a time
python main.py                 # interactive video picker
python main.py --source clip.mp4
python main.py --no-display    # headless (no preview window)
python main.py --no-voice      # disable spoken alerts
```

Results are written to `output/` and incident snapshots to `output/incidents/`.

### Frontend (Safeway AI dashboard)

A React/Vite app (`frontend/`) renders the processed clips as a benchmark gallery with risk timelines, event boxes, object stats and emergency/quick-action pages.

```bash
cd frontend
npm install
npm run dev      # dev server
npm run build    # production build
```

## Regenerating the gallery data

After running `main.py`, rebuild the frontend dataset from the `output/` folder. The generator transcodes the annotated clips to browser-playable H.264, creates thumbnails, and rewrites `frontend/src/videos.generated.js`:

```bash
python generate_frontend_data.py
```

## Project layout

```
main.py                     detection pipeline + CLI
expert_system.pl            Prolog risk rules
download_yolo.py            downloads the common YOLO11 model
generate_frontend_data.py   rebuilds frontend gallery data from output/
yolo11n.pt                  common COCO model
best.pt                     custom road-damage model
frontend/                   React/Vite dashboard (Safeway AI)
  src/data.js               editable constants (emergency numbers, levels)
  src/videos.generated.js   auto-generated gallery data (do not edit)
  public/videos/            processed clips + thumbnails
output/                     annotated videos, CSV, JSON, HTML, incidents
```

## Configuration

All tuning knobs live at the top of `main.py`:

| Setting | Default | Purpose |
| --- | --- | --- |
| `CONFIDENCE` / `IOU` | `0.30` / `0.50` | Detection thresholds |
| `DETECTION_INTERVAL` | `2` | Run detection every N frames (1 = most accurate) |
| `FOCAL_LENGTH_PX` | `700.0` | Camera calibration for distance estimates |
| `KNOWN_WIDTHS_M` | per-class | Real-world object widths in metres |
| `TRACK_FORGET_AFTER` | `20` | Drop stale tracks after N cycles |

## Limitations

Distance and TTC values are **monocular estimates** and require camera calibration before real-world use. Detection quality depends on the model weights and the camera position.