"""Structural bounding-box validation.

An invalid box must be rejected on its RAW coordinates, before anything clips
them -- clipping first can turn a structurally invalid box into a drawable one.
A rejected box must not be drawn, must not create or update a track, and must
not crash the pipeline (NaN/inf reached int() and raised before this gate).
"""

import importlib.util
import os
import sys
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("main", os.path.join(os.path.dirname(__file__), "..", "main.py"))
assert spec is not None
main = importlib.util.module_from_spec(spec)
sys.modules["main"] = main
spec.loader.exec_module(main)  # type: ignore

FRAME = (120, 160)  # h, w


def detection(box):
    d = main.Detection(name="car", confidence=0.9, box=box, source="s", track_key="k",
                       distance_m=5.0, in_lane=True, lane_overlap=0.5, box_height_ratio=0.3)
    d.measured_box = box
    d.box = box
    return d


class IsValidBboxTests(unittest.TestCase):
    def test_accepts_a_normal_box(self):
        self.assertTrue(main.is_valid_bbox((10, 10, 50, 50), FRAME))

    def test_rejects_inverted_axes(self):
        self.assertFalse(main.is_valid_bbox((30, 30, 20, 40), FRAME))   # x2 < x1
        self.assertFalse(main.is_valid_bbox((10, 40, 50, 30), FRAME))   # y2 < y1

    def test_rejects_zero_area(self):
        self.assertFalse(main.is_valid_bbox((10, 10, 10, 10), FRAME))
        self.assertFalse(main.is_valid_bbox((10, 10, 10, 50), FRAME))

    def test_rejects_non_finite(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            self.assertFalse(main.is_valid_bbox((bad, 10, 50, 50), FRAME), bad)
            self.assertFalse(main.is_valid_bbox((10, 10, bad, 50), FRAME), bad)

    def test_rejects_numpy_non_finite(self):
        self.assertFalse(main.is_valid_bbox((np.nan, 10, 50, 50), FRAME))
        self.assertFalse(main.is_valid_bbox((np.inf, 10, 50, 50), FRAME))

    def test_rejects_fully_offscreen(self):
        self.assertFalse(main.is_valid_bbox((500, 500, 600, 600), FRAME))
        self.assertFalse(main.is_valid_bbox((-90, -90, -10, -10), FRAME))

    def test_accepts_partially_offscreen(self):
        """A box straddling the edge is valid; it gets clipped later."""
        self.assertTrue(main.is_valid_bbox((-20, -20, 40, 40), FRAME))
        self.assertTrue(main.is_valid_bbox((140, 100, 200, 160), FRAME))

    def test_rejects_malformed_input(self):
        self.assertFalse(main.is_valid_bbox(None))
        self.assertFalse(main.is_valid_bbox((1, 2, 3)))
        self.assertFalse(main.is_valid_bbox(("a", "b", "c", "d")))

    def test_validation_runs_before_clipping(self):
        """The regression this guards: clip-then-check would make this valid."""
        box = (30, 30, 20, 40)
        clipped = tuple(int(np.clip(v, 0, 159)) for v in box)
        self.assertFalse(main.is_valid_bbox(box, FRAME),
                         "raw box must be rejected even though clipping yields "
                         f"{clipped}")


class InvalidBoxesNeverTrackedTests(unittest.TestCase):
    def test_invalid_boxes_create_no_tracks(self):
        for box in [(30, 30, 20, 40), (10, 10, 10, 10), (500, 500, 600, 600),
                    (float("nan"), 30, 50, 60), (float("inf"), 30, 50, 60)]:
            smoother = main.DetectionSmoother()
            out = smoother.smooth([detection(box)], FRAME, 1 / 30)
            self.assertEqual(out, [], f"{box} should not survive tracking")
            self.assertEqual(len(smoother._filters), 0, f"{box} created a track")

    def test_valid_box_still_tracks(self):
        smoother = main.DetectionSmoother()
        out = smoother.smooth([detection((10, 10, 50, 50))], FRAME, 1 / 30)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(smoother._filters), 1)

    def test_non_finite_box_does_not_raise(self):
        """NaN/inf previously reached int() and raised ValueError/OverflowError."""
        smoother = main.DetectionSmoother()
        smoother.smooth([detection((float("nan"), float("inf"), 50, 60))], FRAME, 1 / 30)


class InvalidBoxesNeverDrawnTests(unittest.TestCase):
    def _blank(self):
        return np.zeros((120, 160, 3), dtype=np.uint8)

    def test_nothing_is_drawn_for_invalid_boxes(self):
        for box in [(30, 30, 20, 40), (10, 10, 10, 10), (500, 500, 600, 600)]:
            frame = self._blank()
            main.draw_detection(frame, detection(box), [], 0)
            self.assertEqual(int(frame.sum()), 0, f"{box} drew something")

    def test_valid_box_is_drawn(self):
        frame = self._blank()
        main.draw_detection(frame, detection((20, 40, 90, 90)), [], 0)
        self.assertGreater(int(frame.sum()), 0)


if __name__ == "__main__":
    unittest.main()
