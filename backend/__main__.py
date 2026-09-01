"""Run the reproducible model benchmark with ``python -m backend``."""

from .benchmark import BENCH


BENCH.start()
if BENCH._thread is not None:
    BENCH._thread.join()
for key, result in (BENCH.results or {}).items():
    print(f"{key}: {result}")
print("Previous known MPS result: approximately 26.47 FPS (comparison only).")
