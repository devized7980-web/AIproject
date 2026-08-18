"""Safeway AI live backend.

FastAPI + WebSocket server that replays the real pipeline output
(output/*_summary.json + *_detections.csv) as a live detection feed and
exposes the Prolog expert system, analytics and model benchmarks as API.
"""