"""Regression tests for object-identity stability.

Two bugs made bounding boxes appear to jump between objects:

1. track_key mixed a coarse spatial grid (y1//40) into the key even when the
   tracker had supplied a stable id, so an object changed identity every time
   it moved 40px -- resetting its Kalman filter, distance smoothing and TTC.
2. model.track(persist=True) keeps tracker state on the model, and it was
   never reset between videos, so each video inherited the previous one's
   tracks.
"""

import importlib.util
import os
import sys
import unittest

spec = importlib.util.spec_from_file_location("main", os.path.join(os.path.dirname(__file__), "..", "main.py"))
main = importlib.util.module_from_spec(spec)
sys.modules["main"] = main
spec.loader.exec_module(main)  # type: ignore


class FakeBox:
    """Mimics one ultralytics result box."""

    def __init__(self, xyxy, cls_id=2, conf=0.9, track_id=None):
        self._xyxy = xyxy
        self.cls = [FakeScalar(cls_id)]
        self.conf = [FakeScalar(conf)]
        self.id = None if track_id is None else [FakeScalar(track_id)]
        self.xyxy = [FakeTensor(xyxy)]


class FakeScalar:
    def __init__(self, v):
        self._v = v

    def item(self):
        return self._v


class FakeTensor:
    def __init__(self, vals):
        self._vals = vals

    def tolist(self):
        return list(self._vals)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    """Stands in for a YOLO model; returns scripted boxes."""

    names = {2: "car"}

    def __init__(self, boxes_per_call):
        self._boxes_per_call = list(boxes_per_call)
        self.calls = 0

    def track(self, frame, **kwargs):
        boxes = self._boxes_per_call[min(self.calls, len(self._boxes_per_call) - 1)]
        self.calls += 1
        return [FakeResult(boxes)]


class TrackKeyStabilityTests(unittest.TestCase):
    def _keys_for_descending_car(self, track_id):
        """Same car, same track id, moving down the frame."""
        import numpy as np

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        polygon = np.array([[0, 479], [0, 0], [639, 0], [639, 479]], dtype=np.int32)

        keys = []
        for y1 in range(100, 340, 20):
            model = FakeModel([[FakeBox((300, y1, 380, y1 + 60), track_id=track_id)]])
            dets = main.extract(model, frame, "yolo11n", polygon, "vid1", None)
            self.assertEqual(len(dets), 1)
            keys.append(dets[0].track_key)
        return keys

    def test_tracked_object_keeps_one_key_while_moving(self):
        keys = self._keys_for_descending_car(track_id=7)
        self.assertEqual(
            len(set(keys)), 1,
            f"a tracked object must keep one identity while moving, got {sorted(set(keys))}",
        )

    def test_tracked_key_contains_track_id(self):
        keys = self._keys_for_descending_car(track_id=7)
        self.assertTrue(keys[0].endswith(":7"), keys[0])

    def test_untracked_objects_still_fall_back_to_spatial_grid(self):
        """Without a tracker id we must still separate distant objects."""
        import numpy as np

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        polygon = np.array([[0, 479], [0, 0], [639, 0], [639, 479]], dtype=np.int32)

        model = FakeModel([[
            FakeBox((100, 100, 180, 160), track_id=None),
            FakeBox((400, 300, 480, 360), track_id=None),
        ]])
        dets = main.extract(model, frame, "yolo11n", polygon, "vid1", None)
        self.assertEqual(len(dets), 2)
        self.assertNotEqual(dets[0].track_key, dets[1].track_key)

    def test_different_videos_do_not_share_keys(self):
        import numpy as np

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        polygon = np.array([[0, 479], [0, 0], [639, 0], [639, 479]], dtype=np.int32)

        box = [FakeBox((300, 200, 380, 260), track_id=3)]
        k1 = main.extract(FakeModel([box]), frame, "yolo11n", polygon, "vid1", None)[0].track_key
        k2 = main.extract(FakeModel([box]), frame, "yolo11n", polygon, "vid2", None)[0].track_key
        self.assertNotEqual(k1, k2)


class FakeTracker:
    def __init__(self):
        self.reset_called = 0

    def reset(self):
        self.reset_called += 1


class FakePredictor:
    def __init__(self):
        self.trackers = [FakeTracker()]


class ModelWithPredictor:
    def __init__(self):
        self.predictor = FakePredictor()


class TrackerResetTests(unittest.TestCase):
    def test_reset_clears_every_model(self):
        a, b = ModelWithPredictor(), ModelWithPredictor()
        main.reset_tracker_state(a, b)
        self.assertEqual(a.predictor.trackers[0].reset_called, 1)
        self.assertEqual(b.predictor.trackers[0].reset_called, 1)

    def test_reset_tolerates_none_and_unstarted_models(self):
        class NoPredictor:
            predictor = None

        # must not raise
        main.reset_tracker_state(None, NoPredictor())

    def test_reset_survives_a_failing_tracker(self):
        class Boom:
            def reset(self):
                raise RuntimeError("nope")

        class P:
            trackers = [Boom()]

        class M:
            predictor = P()

        good = ModelWithPredictor()
        main.reset_tracker_state(M(), good)
        # a failure in one model must not stop the others being reset
        self.assertEqual(good.predictor.trackers[0].reset_called, 1)


if __name__ == "__main__":
    unittest.main()
