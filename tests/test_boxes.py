import importlib.util
import os
import sys
import tempfile
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("main", os.path.join(os.path.dirname(__file__), "..", "main.py"))
assert spec is not None
main = importlib.util.module_from_spec(spec)
sys.modules["main"] = main
spec.loader.exec_module(main)  # type: ignore


class FakeCapture:
    def __init__(self, frames):
        self.frames = frames
        self.i = 0

    def isOpened(self):
        return True

    def read(self):
        if self.i >= len(self.frames):
            return False, None
        f = self.frames[self.i]
        # encode frame index
        f[0, 0, 0] = self.i % 256
        self.i += 1
        return True, f

    def get(self, key):
        if key == main.cv2.CAP_PROP_FRAME_WIDTH:
            return self.frames[0].shape[1]
        if key == main.cv2.CAP_PROP_FRAME_HEIGHT:
            return self.frames[0].shape[0]
        if key == main.cv2.CAP_PROP_FPS:
            return 10.0
        return 0

    def release(self):
        pass


class FakeWriter:
    def __init__(self, *a, **k):
        self.frames = []
        self.released = False

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


class BoxesTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        setattr(main, "OUTPUT_FOLDER", main.Path(self.tmpdir.name))

    def tearDown(self):
        self.tmpdir.cleanup()

    def _run_with_extractor(self, frames, extractor):
        cap = FakeCapture(frames)
        orig_vc = main.cv2.VideoCapture
        orig_vw = main.cv2.VideoWriter
        writer = FakeWriter()
        main.cv2.VideoCapture = lambda p: cap
        main.cv2.VideoWriter = lambda *a, **k: writer
        try:
            main.process_video(main.Path('/dev/null'), None, None, main.PrologRiskEngine(main.PROLOG_FILE),
                               display=False, voice_enabled=False, next_video_name=None,
                               calibrator_config=None, drop_frames=False,
                               extractor_callable=extractor, reader_queue_size=4, result_queue_size=4)
            return writer.frames
        finally:
            main.cv2.VideoCapture = orig_vc
            main.cv2.VideoWriter = orig_vw

    def test_no_box_without_current_detection(self):
        h, w = 480, 640
        frames = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(2)]

        def extractor(frame, polygon, name, calib):
            idx = int(frame[0, 0, 0])
            if idx == 0:
                d = main.Detection(name='car', confidence=0.9, box=(200, 300, 350, 400), source='best.pt',
                                   track_key='t0', distance_m=5.0, distance_method='width_fallback',
                                   in_lane=True, lane_overlap=1.0, box_height_ratio=0.3)
                d.measured_box = (200, 300, 350, 400)
                d.box = d.measured_box
                return [d]
            return []

        out = self._run_with_extractor(frames, extractor)
        self.assertEqual(len(out), 2)
        # first frame should have non-zero pixel on box border; second frame unchanged
        self.assertGreater(out[0][300, 250].sum(), 0)
        self.assertEqual(out[1][300, 250].sum(), 0)

    def test_invalid_boxes_rejected(self):
        h, w = 120, 160
        frames = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(1)]

        def extractor(frame, polygon, name, calib):
            # invalid box x2 <= x1
            d = main.Detection(name='car', confidence=0.9, box=(30, 30, 20, 40), source='best.pt',
                               track_key='t0', distance_m=5.0, distance_method='width_fallback',
                               in_lane=True, lane_overlap=1.0, box_height_ratio=0.3)
            d.measured_box = (30, 30, 20, 40)
            d.box = d.measured_box
            return [d]

        out = self._run_with_extractor(frames, extractor)
        self.assertEqual(len(out), 1)
        # no rectangle drawn at invalid box top-left
        self.assertEqual(out[0][30, 30].sum(), 0)

    def test_duplicate_overlapping_boxes_removed_prefers_custom(self):
        h, w = 120, 160
        frames = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(1)]

        def extractor(frame, polygon, name, calib):
            # common box slightly larger; custom box inside
            d1 = main.Detection(name='car', confidence=0.8, box=(10, 10, 60, 60), source='yolo11n',
                                track_key='t1', distance_m=5.0, distance_method='width_fallback',
                                in_lane=True, lane_overlap=1.0, box_height_ratio=0.3)
            d1.measured_box = (10, 10, 60, 60)
            d1.box = d1.measured_box
            d2 = main.Detection(name='car', confidence=0.85, box=(12, 12, 58, 58), source='best.pt',
                                track_key='t1', distance_m=5.0, distance_method='width_fallback',
                                in_lane=True, lane_overlap=1.0, box_height_ratio=0.3)
            d2.measured_box = (12, 12, 58, 58)
            d2.box = d2.measured_box
            return [d1, d2]

        out = self._run_with_extractor(frames, extractor)
        self.assertEqual(len(out), 1)
        # ensure drawn box aligns with custom box (pixel at custom top-left changed)
        frame = out[0]
        self.assertGreater(frame[12, 12].sum(), 0)


if __name__ == "__main__":
    unittest.main()
