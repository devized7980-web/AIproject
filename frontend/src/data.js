/* ═══════════════════════════════════════════════════════════════════
   THE ONLY FILE YOU NEED TO EDIT.

   The VIDEOS list is auto-generated from the real pipeline output.
   Run `python generate_frontend_data.py` (repo root) after each run of
   main.py to rebuild it from output/.
   ═══════════════════════════════════════════════════════════════════ */

// Where the .mp4 files live. Processed clips are copied here by the generator.
export const VIDEO_BASE = "/videos/";

export const LEVELS = ["SAFE", "CAUTION", "WARNING", "CRITICAL"];

// Real results generated from output/<name>_summary.json + _detections.csv.
export { VIDEOS } from "./videos.generated.js";

// Replace with the numbers for your region before you demo.
export const EMERGENCY = [
  { name: "Police", number: "199", note: "Crash with injury, obstruction, or hit-and-run" },
  { name: "Fire & Rescue", number: "191", note: "Fire, fuel leak, or trapped occupants" },
  { name: "Ambulance", number: "192", note: "Anyone hurt, unconscious, or bleeding" },
  { name: "Road assistance", number: "1899", note: "Breakdown, tyre, tow request" },
];

export const QUICK_ACTIONS = [
  { id: "location", title: "Share my position", body: "Copies your coordinates and the clip the last flagged event came from, ready to paste into a call or message." },
  { id: "snapshot", title: "Send incident snapshot", body: "Copies a formatted incident report for the highest-risk event — video, time, risk, distance, TTC and confidence." },
  { id: "hazard", title: "Report a road hazard", body: "Copies a hazard report with the detected object, distance and risk level for the council or other drivers." },
  { id: "log", title: "Copy detection log", body: "Copies the risk summary line so it can go into an insurance or council report." },
];

export const CRASH_STEPS = [
  "Stop the vehicle and switch on hazard lights.",
  "Check for injuries before moving anyone.",
  "Move to the roadside barrier if the vehicle is drivable.",
  "Call the number that matches the worst problem, not all of them.",
  "Photograph both vehicles, the road surface, and the plates.",
  "Save the incident frames from this run before closing the app.",
];
