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


class LowLightTests(unittest.TestCase):
    def _uniform_frame(self, value: int) -> np.ndarray:
        return np.full((100, 100, 3), value, dtype=np.uint8)

    def test_frame_brightness_matches_uniform_value(self):
        dark = self._uniform_frame(20)
        bright = self._uniform_frame(220)
        self.assertAlmostEqual(main.frame_brightness(dark), 20.0, delta=1.0)
        self.assertAlmostEqual(main.frame_brightness(bright), 220.0, delta=1.0)

    def test_low_light_threshold_classifies_night_vs_day(self):
        night_frame = self._uniform_frame(15)
        day_frame = self._uniform_frame(180)
        self.assertLess(main.frame_brightness(night_frame), main.LOW_LIGHT_THRESHOLD)
        self.assertGreaterEqual(main.frame_brightness(day_frame), main.LOW_LIGHT_THRESHOLD)

    def test_enhance_low_light_preserves_shape_and_dtype(self):
        frame = self._uniform_frame(25)
        enhanced = main.enhance_low_light(frame)
        self.assertEqual(enhanced.shape, frame.shape)
        self.assertEqual(enhanced.dtype, frame.dtype)

    def test_enhance_low_light_boosts_contrast(self):
        # low-contrast dark frame: two barely-different regions
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :50] = 20
        frame[:, 50:] = 35
        enhanced = main.enhance_low_light(frame)

        gray_in = np.mean(frame, axis=2)
        gray_out = np.mean(enhanced, axis=2)
        # CLAHE should spread the two regions' values further apart than the raw frame
        self.assertGreaterEqual(gray_out.std(), gray_in.std())


if __name__ == "__main__":
    unittest.main()
