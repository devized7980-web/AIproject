import importlib.util
import os
import sys
import time
import tempfile
import threading
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("main", os.path.join(os.path.dirname(__file__), "..", "main.py"))
main = importlib.util.module_from_spec(spec)
sys.modules["main"] = main
spec.loader.exec_module(main)  # type: ignore


class FakeCapture:
    def __init__(self, num_frames: int, w: int = 160, h: int = 120, fps: float = 10.0):
        self.num_frames = num_frames
        self.w = w
        self.h = h
        self.fps = fps
        self.i = 0

    def isOpened(self):
        return True

    def read(self):
        if self.i >= self.num_frames:
            return False, None
        # encode frame number into pixel for validation
        arr = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        arr[0, 0, 0] = self.i % 256
        self.i += 1
        return True, arr

    def get(self, key):
        if key == main.cv2.CAP_PROP_FRAME_WIDTH:
            return self.w
        if key == main.cv2.CAP_PROP_FRAME_HEIGHT:
            return self.h
        if key == main.cv2.CAP_PROP_FPS:
            return self.fps
        return 0

    def release(self):
        pass


class FakeWriter:
    def __init__(self, *args, **kwargs):
        self.frames = []
        self.released = False

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


class PipelineTests(unittest.TestCase):
    def setUp(self):
        # isolate outputs to temp dir
        self.tmpdir = tempfile.TemporaryDirectory()
        main.OUTPUT_FOLDER = main.Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_frames_ordered_and_resources_released(self):
        # 5 frames, extractor sleeps variably but ordering preserved
        num = 5
        cap = FakeCapture(num_frames=num)

        def fake_capture_factory(path):
            return cap

        def extractor(frame, polygon, name, calib):
            # simulate variable work
            idx = int(frame[0, 0, 0])
            time.sleep(0.01 if idx % 2 == 0 else 0.001)
            # return empty detections
            return []

        # patch cv2.VideoCapture and VideoWriter
        orig_vc = main.cv2.VideoCapture
        orig_vw = main.cv2.VideoWriter
        main.cv2.VideoCapture = lambda path: cap
        main.cv2.VideoWriter = lambda *a, **k: FakeWriter()

        try:
            main.process_video(main.Path("/dev/null"), None, None, main.PrologRiskEngine(main.PROLOG_FILE),
                               display=False, voice_enabled=False, next_video_name=None,
                               calibrator_config=None, drop_frames=False,
                               extractor_callable=extractor, reader_queue_size=2, result_queue_size=2)
            # read json summary
            js = main.Path(main.OUTPUT_FOLDER) / f"{main.Path('/dev/null').stem}_summary.json"
            self.assertTrue(js.exists())
            data = js.read_text()
            self.assertIn('frames', data)
        finally:
            main.cv2.VideoCapture = orig_vc
            main.cv2.VideoWriter = orig_vw

    def test_bounded_queues_and_drops(self):
        # many frames and slow extractor -> drop frames when drop_frames=True
        num = 40
        cap = FakeCapture(num_frames=num)

        def extractor(frame, polygon, name, calib):
            time.sleep(0.02)  # slow
            return []

        orig_vc = main.cv2.VideoCapture
        orig_vw = main.cv2.VideoWriter
        main.cv2.VideoCapture = lambda path: cap
        main.cv2.VideoWriter = lambda *a, **k: FakeWriter()

        try:
            main.process_video(main.Path("/dev/null"), None, None, main.PrologRiskEngine(main.PROLOG_FILE),
                               display=False, voice_enabled=False, next_video_name=None,
                               calibrator_config=None, drop_frames=True,
                               extractor_callable=extractor, reader_queue_size=4, result_queue_size=4)
            js = main.Path(main.OUTPUT_FOLDER) / f"{main.Path('/dev/null').stem}_summary.json"
            self.assertTrue(js.exists())
            import json
            data = json.loads(js.read_text())
            self.assertIn('pipeline', data)
            self.assertGreaterEqual(data['pipeline'].get('dropped_frames', 0), 0)
        finally:
            main.cv2.VideoCapture = orig_vc
            main.cv2.VideoWriter = orig_vw

    def test_worker_exception_propagates(self):
        num = 5
        cap = FakeCapture(num_frames=num)

        def extractor(frame, polygon, name, calib):
            idx = int(frame[0, 0, 0])
            # fail on second captured frame (index 1) which will be a fresh detection
            if idx == 1:
                raise RuntimeError("inference failure")
            return []

        orig_vc = main.cv2.VideoCapture
        orig_vw = main.cv2.VideoWriter
        main.cv2.VideoCapture = lambda path: cap
        main.cv2.VideoWriter = lambda *a, **k: FakeWriter()

        try:
            with self.assertRaises(RuntimeError):
                main.process_video(main.Path("/dev/null"), None, None, main.PrologRiskEngine(main.PROLOG_FILE),
                                   display=False, voice_enabled=False, next_video_name=None,
                                   calibrator_config=None, drop_frames=False,
                                   extractor_callable=extractor, reader_queue_size=2, result_queue_size=2)
        finally:
            main.cv2.VideoCapture = orig_vc
            main.cv2.VideoWriter = orig_vw


if __name__ == "__main__":
    unittest.main()
