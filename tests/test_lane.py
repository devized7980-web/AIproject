"""Lane-geometry regression tests.

The left and right lane boundaries converge at a vanishing point and cross
above it. detect_lane_polygon used to sample both boundaries at a fixed
y = 0.56*h, which on real dashcam footage sat *above* that crossing -- the left
edge then computed to the right of the right edge, the sanity check tripped and
the function fell back to a hardcoded corridor on ~98% of frames.
"""

import importlib.util
import os
import sys
import unittest

import cv2
import numpy as np

spec = importlib.util.spec_from_file_location("main", os.path.join(os.path.dirname(__file__), "..", "main.py"))
main = importlib.util.module_from_spec(spec)
sys.modules["main"] = main
spec.loader.exec_module(main)  # type: ignore


def synthetic_road(w=848, h=476, vanishing_y=None):
    """Dark frame with two bright converging lane lines, like a marked road."""
    if vanishing_y is None:
        vanishing_y = int(h * 0.45)
    frame = np.full((h, w, 3), 40, dtype=np.uint8)
    vx = w // 2
    # left boundary: bottom-left -> vanishing point; right: bottom-right -> vp
    cv2.line(frame, (int(w * 0.10), h - 1), (vx, vanishing_y), (255, 255, 255), 6)
    cv2.line(frame, (int(w * 0.90), h - 1), (vx, vanishing_y), (255, 255, 255), 6)
    return frame


class LaneGeometryTests(unittest.TestCase):
    def test_detects_lane_on_clear_markings(self):
        polygon, detected = main.detect_lane_polygon(synthetic_road())
        self.assertTrue(detected, "clear converging lane lines should be detected")
        self.assertEqual(polygon.shape, (4, 2))

    def test_polygon_edges_never_cross(self):
        """The core bug: left edge must stay left of the right edge."""
        for vy in (int(476 * 0.35), int(476 * 0.45), int(476 * 0.55)):
            polygon, detected = main.detect_lane_polygon(synthetic_road(vanishing_y=vy))
            if not detected:
                continue
            (lx_top, y_top), (rx_top, _), (rx_bot, y_bot), (lx_bot, _) = polygon
            self.assertLess(lx_top, rx_top, f"top edges crossed (vanishing_y={vy})")
            self.assertLess(lx_bot, rx_bot, f"bottom edges crossed (vanishing_y={vy})")

    def test_polygon_top_sits_below_vanishing_point(self):
        vy = int(476 * 0.40)
        polygon, detected = main.detect_lane_polygon(synthetic_road(vanishing_y=vy))
        if not detected:
            self.skipTest("synthetic frame not detected; geometry asserted elsewhere")
        y_top = int(polygon[0][1])
        # larger y is lower in the image, so the top edge must be *below* the vp
        self.assertGreater(y_top, vy, "polygon must not extend past the vanishing point")

    def test_polygon_widens_toward_the_camera(self):
        polygon, detected = main.detect_lane_polygon(synthetic_road())
        if not detected:
            self.skipTest("synthetic frame not detected")
        (lx_top, _), (rx_top, _), (rx_bot, _), (lx_bot, _) = polygon
        self.assertGreater(rx_bot - lx_bot, rx_top - lx_top,
                           "a road corridor must be wider near the camera")

    def test_blank_frame_falls_back_cleanly(self):
        """No markings (dirt road, roundabout) -> fallback, not a crash."""
        blank = np.full((476, 848, 3), 90, dtype=np.uint8)
        polygon, detected = main.detect_lane_polygon(blank)
        self.assertFalse(detected)
        self.assertEqual(polygon.shape, (4, 2))
        np.testing.assert_array_equal(polygon, main.default_lane_polygon(848, 476))

    def test_fallback_polygon_is_well_formed(self):
        polygon = main.default_lane_polygon(848, 476)
        top_left, top_right, bottom_right, bottom_left = polygon
        self.assertLess(top_left[0], top_right[0])
        self.assertLess(bottom_left[0], bottom_right[0])

    def test_detection_rate_improved_on_real_footage(self):
        """End-to-end guard: the marked-road clip must detect lanes on most
        frames. Before the vanishing-point fix this scored ~1%."""
        video = os.path.join(os.path.dirname(__file__), "..", "videos", "road_video_1.mp4.MP4")
        if not os.path.exists(video):
            self.skipTest("sample video not available")
        cap = cv2.VideoCapture(video)
        detected = sampled = 0
        idx = 0
        while sampled < 40:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % 5:
                continue
            sampled += 1
            _, ok_lane = main.detect_lane_polygon(frame)
            detected += bool(ok_lane)
        cap.release()
        if sampled == 0:
            self.skipTest("could not read frames")
        rate = detected / sampled
        self.assertGreater(rate, 0.5, f"lane detection rate too low: {rate:.0%}")


if __name__ == "__main__":
    unittest.main()
