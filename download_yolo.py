from pathlib import Path
import shutil

from ultralytics import YOLO

models_dir = Path(__file__).resolve().parent / "models"
models_dir.mkdir(exist_ok=True)
target = models_dir / "yolo11n.pt"
model = YOLO(str(target) if target.exists() else "yolo11n.pt")
downloaded = Path("yolo11n.pt")
if not target.exists() and downloaded.exists():
    shutil.move(str(downloaded), str(target))

print(f"YOLO11n available at {target}")
