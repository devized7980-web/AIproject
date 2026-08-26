import importlib.util
import os
import sys
import unittest

# Load main module
spec = importlib.util.spec_from_file_location("main", os.path.join(os.path.dirname(__file__), "..", "main.py"))
assert spec is not None
main = importlib.util.module_from_spec(spec)
sys.modules["main"] = main
spec.loader.exec_module(main)  # type: ignore


class KalmanTests(unittest.TestCase):
    def make_detection(self, x1, y1, x2, y2, tid="1"):
        return main.Detection(
            name="obj",
            confidence=0.9,
            box=(x1, y1, x2, y2),
            source="s",
            track_key=f"stream:src:obj:{tid}",
            distance_m=10.0,
            in_lane=True,
            lane_overlap=0.5,
            box_height_ratio=0.2,
        )

    def test_stationary_object(self):
        smoother = main.DetectionSmoother()
        w, h = 640, 480
        dt = 1.0 / 30.0

        measurements = [(100, 100, 140, 140) for _ in range(8)]
        prev_smooth = None
        for box in measurements:
            d = self.make_detection(*box)
            outs = smoother.smooth([d], (h, w), dt)
            sb = outs[0].box
            if prev_smooth is not None:
                # smoothed position should not jump much between frames
                self.assertLessEqual(abs(sb[0] - prev_smooth[0]), 8)
            prev_smooth = sb

    def test_constant_speed_movement(self):
        smoother = main.DetectionSmoother()
        w, h = 800, 600
        dt = 1.0 / 20.0

        measurements = [(100 + i * 8, 100, 140 + i * 8, 140) for i in range(10)]
        raw_step = []
        smooth_step = []
        prev_raw = None
        prev_smooth = None
        for box in measurements:
            d = self.make_detection(*box)
            if prev_raw is not None:
                raw_step.append(abs(box[0] - prev_raw[0]))
            prev_raw = box
            outs = smoother.smooth([d], (h, w), dt)
            sb = outs[0].box
            if prev_smooth is not None:
                smooth_step.append(abs(sb[0] - prev_smooth[0]))
            prev_smooth = sb

        # Smooth step sizes should be similar to raw (no big lag)
        if raw_step:
            avg_raw = sum(raw_step) / len(raw_step)
            avg_smooth = sum(smooth_step) / len(smooth_step)
            self.assertTrue(abs(avg_smooth - avg_raw) < 8.0)

    def test_sudden_measurement_noise(self):
        smoother = main.DetectionSmoother()
        w, h = 320, 240
        dt = 1.0 / 25.0

        measurements = [(150, 80, 190, 120) for _ in range(4)]
        # inject noisy spike
        measurements += [(300, 200, 340, 240)]
        measurements += [(150, 80, 190, 120) for _ in range(4)]

        diffs_raw = []
        diffs_smooth = []
        prev_raw = None
        prev_smooth = None
        for box in measurements:
            d = self.make_detection(*box)
            if prev_raw is not None:
                diffs_raw.append(abs(box[0] - prev_raw[0]))
            prev_raw = box
            outs = smoother.smooth([d], (h, w), dt)
            sb = outs[0].box
            if prev_smooth is not None:
                diffs_smooth.append(abs(sb[0] - prev_smooth[0]))
            prev_smooth = sb

        # smoother should reduce spike effect => total smooth diffs < raw diffs
        self.assertTrue(sum(diffs_smooth) < sum(diffs_raw))

    def test_missing_measurements_and_track_expiration(self):
        smoother = main.DetectionSmoother()
        w, h = 640, 480
        dt = 1.0 / 30.0

        # create initial track
        d = self.make_detection(50, 50, 90, 90, tid="A")
        smoother.smooth([d], (h, w), dt)

        # Now simulate missing measurement for TRACK_FORGET_AFTER - 1 cycles
        for _ in range(main.TRACK_FORGET_AFTER - 1):
            smoother.smooth([], (h, w), dt)

        # track should still exist internally
        self.assertIn("stream:src:obj:A", smoother.last_seen)

        # one more cycle -> expired
        smoother.smooth([], (h, w), dt)
        self.assertNotIn("stream:src:obj:A", smoother.last_seen)

    def test_frame_boundary_clipping(self):
        smoother = main.DetectionSmoother()
        w, h = 200, 100
        dt = 1.0 / 15.0

        # measurement outside frame
        d = self.make_detection(-50, -40, 300, 200, tid="B")
        outs = smoother.smooth([d], (h, w), dt)
        x1, y1, x2, y2 = outs[0].box
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, w - 1)
        self.assertLessEqual(y2, h - 1)


if __name__ == "__main__":
    unittest.main()
