Drop any additional trained YOLO models here (e.g. `tree.pt` for a fallen-tree/obstacle
detector). Every `*.pt` file in this folder is auto-loaded at startup and fused into
the detection pipeline alongside the common (COCO) and custom (road-damage) models —
no code changes needed. See `OBSTACLE_CLASSES` in `main.py` and the `obstacle/1` facts
in `expert_system.pl` for the hazard category these feed into.
