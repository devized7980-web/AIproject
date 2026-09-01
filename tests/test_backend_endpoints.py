"""Backend endpoint contracts that do not require an HTTP test client."""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import server


class BackendEndpointTests(unittest.TestCase):
    def test_health_reports_real_catalogue_and_services(self):
        result = server.health()
        self.assertTrue(result["ok"])
        self.assertEqual(result["videos"], len(server.STORE.videos))
        self.assertIn(result["status"], {"Running", "Degraded", "Offline"})
        self.assertIn(result["feed_mode"], {"Recorded detection replay"})

    def test_catalogue_has_stable_ids_and_raw_media(self):
        items = server.videos()
        self.assertEqual(len(items), len(server.STORE.videos))
        self.assertEqual(len({v["id"] for v in items}), len(items))
        for video in items:
            self.assertTrue(video["raw_available"])
            self.assertTrue((Path(server.ROOT) / "videos" / video["raw"]).is_file())
            if video["processed_available"]:
                self.assertIsNotNone(video["file"])
            else:
                self.assertIsNone(video["file"])

    def test_unanalysed_video_has_no_statistics_or_frames(self):
        video = next((v for v in server.videos() if not v["analysis_available"]), None)
        if video is None:
            self.skipTest("fixture catalogue has no unanalysed video")
        self.assertEqual(video["processing_status"], "NOT_PROCESSED")
        self.assertEqual(video["events"], [])
        self.assertEqual(video["total_detections"], 0)
        self.assertEqual(server.video_frames(video["id"])["frames"], [])

    def test_simulator_rejects_invalid_values(self):
        for field, value in (("speed_kmh", -1), ("distance_m", 0), ("wetness", 2),
                             ("visibility", 0), ("lane_position", -1),
                             ("confidence", 2), ("box_height_ratio", -1)):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    server.SimulateRequest(**{field: value})

    def test_trace_uses_actual_engine(self):
        result = server.prolog_trace(server.TraceRequest())
        self.assertIn(result["engine"], {"SWI-Prolog", "Python fallback"})
        self.assertIn(result["decision_source"], {"prolog", "python_fallback"})

    def test_static_route_handlers_exist(self):
        routes = {getattr(route, "path", "") for route in server.app.routes}
        self.assertIn("/api/health", routes)
        self.assertIn("/api/videos", routes)
        self.assertIn("/ws", routes)


if __name__ == "__main__":
    unittest.main()
