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


class DistanceTests(unittest.TestCase):
    def test_nearer_bottom_shorter_distance(self):
        # image height 480, two boxes: one near bottom (y=470), one higher (y=300)
        calib = main.PerspectiveDistanceCalibrator(image_height=480, horizon_ratio=0.4,
                                                   camera_height_m=1.2, vertical_fov_deg=50.0)
        # bottom y values
        box_low = (100, 430, 140, 470)
        box_high = (100, 260, 140, 300)
        d_low, m_low = calib.distance("t1", box_low, "car")
        d_high, m_high = calib.distance("t2", box_high, "car")
        self.assertLess(d_low, d_high)

    def test_horizon_rejection_and_capping(self):
        calib = main.PerspectiveDistanceCalibrator(image_height=480, horizon_y=200,
                                                   camera_height_m=1.2, vertical_fov_deg=40.0,
                                                   min_distance=0.5, max_distance=100.0)
        # point at or above horizon should fall back to width-based or raise inside analytic
        box_on_horizon = (100, 0, 140, 200)
        # analytic should raise internally; calibrator should fallback to width method
        d, method = calib.distance("t3", box_on_horizon, "car")
        self.assertIn(method, {"width_fallback", "perspective_ref"})

    def test_reference_point_interpolation_monotonic(self):
        # reference points mapping deeper y to larger distances
        ref = [(300, 10.0), (400, 20.0), (450, 40.0)]
        calib = main.PerspectiveDistanceCalibrator(image_height=480, ref_points=ref)
        # sample y increasing should produce non-decreasing distances
        last = -1.0
        for y in range(300, 451, 10):
            d, _ = calib.distance("t", (0, 0, 10, y), None)
            self.assertGreaterEqual(d, last - 1e-6)
            last = d

    def test_invalid_calibration_values_raise(self):
        with self.assertRaises(ValueError):
            main.PerspectiveDistanceCalibrator(image_height=480, horizon_ratio=1.5)
        with self.assertRaises(ValueError):
            main.PerspectiveDistanceCalibrator(image_height=480, ref_points=[(500, 10.0)])
        with self.assertRaises(ValueError):
            main.PerspectiveDistanceCalibrator(image_height=480, min_distance=10.0, max_distance=5.0)

    def test_width_based_fallback_works(self):
        calib = main.PerspectiveDistanceCalibrator(image_height=480)
        d, method = calib.distance("t", (10, 10, 110, 110), "car")
        self.assertEqual(method, "width_fallback")


if __name__ == "__main__":
    unittest.main()
