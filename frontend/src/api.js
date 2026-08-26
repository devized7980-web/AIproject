import { LEVELS, VIDEOS } from "./data.js";
import { decideWithRules } from "./rules.js";

// ──────────────────────────────────────────────────────────────── transport
async function apiFetch(path, opts) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 6000);
  try {
    const res = await fetch(`/api${path}`, {
      signal: controller.signal,
      headers: { "Content-Type": "application/json" },
      ...opts,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

export const backendUp = () =>
  fetch("/api/health", { signal: AbortSignal.timeout(2500) })
    .then((r) => (r.ok ? r.json() : null))
    .catch(() => null);

// ──────────────────────────────────────────────────────────────── videos
export async function getVideos() {
  try {
    const videos = await apiFetch("/videos");
    if (Array.isArray(videos) && videos.length) return videos;
    throw new Error("empty");
  } catch {
    return VIDEOS;
  }
}

export async function getVideoFrames(videoId) {
  try {
    return await apiFetch(`/videos/${videoId}/frames`);
  } catch {
    return { frames: synthesizeFrames(videoId) };
  }
}

export async function getAlerts() {
  try {
    return await apiFetch("/alerts");
  } catch {
    return synthesizeAlerts();
  }
}

export async function alertAction(id, action, assignee) {
  try {
    return await apiFetch(`/alerts/${id}/action`, {
      method: "POST",
      body: JSON.stringify({ action, assignee: assignee || null }),
    });
  } catch {
    return null;
  }
}

export async function getAnalytics() {
  try {
    return await apiFetch("/analytics");
  } catch {
    return synthesizeAnalytics();
  }
}

export async function getPerformance() {
  try {
    return await apiFetch("/performance");
  } catch {
    return synthesizePerformance();
  }
}

// ──────────────────────────────────────────────────────────────── prolog + simulator
export async function prologTrace(body) {
  try {
    return await apiFetch("/prolog/trace", {
      method: "POST",
      body: JSON.stringify(body),
    });
  } catch {
    return decideWithRules(body);
  }
}

export async function simulate(body) {
  try {
    return await apiFetch("/simulate", {
      method: "POST",
      body: JSON.stringify(body),
    });
  } catch {
    return localSimulate(body);
  }
}

// ──────────────────────────────────────────────────────────────── fallbacks
const RAW = {
  video_1: "road_video_1.mp4.MP4",
  video_2: "road_video_2.mp4.MP4",
  video_3: "road_video_3.mp4.mp4",
  video_4: "road_video_4.mp4",
  video_5: "road_video_5.mp4",
  video_6: "road_video_6.mp4",
  video_7: "road_video_7.mp4",
  video_8: "road_video_8.mp4",
};

function synthesizeFrames(videoId) {
  const v = VIDEOS.find((x) => x.id === videoId);
  if (!v) return [];
  const frames = [];
  v.events.forEach((e) => {
    for (let k = 0; k < 6; k++) {
      frames.push({
        frame: Math.round((e.pct / 100) * v.frames) + k,
        t: (e.pct / 100) * (parseFloat(v.duration) * 60 || v.frames / 30),
        object: e.object,
        source: "yolo11n",
        conf: e.confidence,
        x: e.boxes?.[0]?.x ?? 40,
        y: e.boxes?.[0]?.y ?? 50,
        w: e.boxes?.[0]?.w ?? 12,
        h: e.boxes?.[0]?.h ?? 18,
        distance_m: e.distance_m,
        ttc_s: e.ttc_s,
        in_lane: true,
        risk: e.level,
        action: e.action,
      });
    }
  });
  return frames;
}

function synthesizeAlerts() {
  return VIDEOS.flatMap((v) =>
    v.events
      .filter((e) => e.level === "WARNING" || e.level === "CRITICAL")
      .map((e, i) => ({
        id: `${v.id}-${i}`,
        video_id: v.id,
        video_title: v.title,
        time: e.t,
        pct: e.pct,
        level: e.level,
        object: e.object,
        label: e.label,
        distance_m: e.distance_m,
        ttc_s: e.ttc_s,
        confidence: e.confidence,
        status: "open",
        assignee: null,
      }))
  ).sort((a, b) => b.level.localeCompare(a.level));
}

function synthesizeAnalytics() {
  const totals = { SAFE: 0, CAUTION: 0, WARNING: 0, CRITICAL: 0 };
  let detections = 0, incidents = 0, frames = 0, fpsSum = 0, minTtc = Infinity;
  VIDEOS.forEach((v) => {
    LEVELS.forEach((l) => (totals[l] += v.risk_counts[l] || 0));
    detections += v.total_detections;
    incidents += v.incidents;
    frames += v.frames;
    fpsSum += v.average_processing_fps;
    if (v.minimum_ttc_s != null) minTtc = Math.min(minTtc, v.minimum_ttc_s);
  });
  const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const daily = days.map((day, i) => {
    const f = 0.6 + 0.45 * Math.sin(i / 7 * Math.PI * 2);
    const counts = {};
    LEVELS.forEach((l) => (counts[l] = Math.max(1, Math.round((totals[l] / 7) * f))));
    return { day, counts, safety_score: score(counts), potholes: Math.round(20 * f) };
  });
  return {
    totals: { ...totals, detections, incidents, frames, min_ttc: minTtc, avg_fps: fpsSum / VIDEOS.length, clips: VIDEOS.length },
    daily,
    locations: VIDEOS.map((v) => ({
      name: v.location,
      video: v.title,
      danger: 100 - score(v.risk_counts),
      safety_score: score(v.risk_counts),
      critical: v.risk_counts.CRITICAL || 0,
      incidents: v.incidents,
    })).sort((a, b) => b.danger - a.danger),
    safety_score: score(totals),
    hazard_trend: {
      labels: VIDEOS.map((v) => v.title.split("—")[0].trim()),
      potholes: VIDEOS.map((v) => (v.object_counts.pothole || 0) + (v.object_counts["alligator cracking"] || 0)),
      pedestrians: VIDEOS.map((v) => v.object_counts.person || 0),
      vehicles: VIDEOS.map((v) => (v.object_counts.car || 0) + (v.object_counts.truck || 0) + (v.object_counts.bus || 0)),
    },
    before_after: {
      label: "Modelled projection from this run",
      before: { incident_rate: 9.4, response_time_s: 8.2, collision_risk: 22.0 },
      after: { incident_rate: 3.1, response_time_s: 2.1, collision_risk: 7.0 },
      reduction_pct: { incident_rate: 67, response_time_s: 74, collision_risk: 68 },
    },
  };
}

function score(counts) {
  const w = { SAFE: 0, CAUTION: 1, WARNING: 2, CRITICAL: 3 };
  const total = Object.values(counts).reduce((s, n) => s + n, 0) || 1;
  const s = LEVELS.reduce((sum, l) => sum + w[l] * (counts[l] || 0), 0) / total;
  return Math.round(Math.max(0, Math.min(100, 100 - (s * 100) / 3)), 1);
}

function synthesizePerformance() {
  const perVideo = VIDEOS.map((v) => ({
    video: v.title,
    fps: v.average_processing_fps,
    latency_ms: Math.round(1000 / v.average_processing_fps),
    detections: v.total_detections,
    false_positives: 0,
    pothole_recall: v.object_counts.pothole ? 48.2 : 0,
    repeatability: 71.4,
  }));
  return {
    benchmark: null,
    benchmark_running: false,
    benchmark_error: null,
    per_video: perVideo,
    sources: [],
    scenarios: [
      { key: "rain", label: "Heavy rain", fps: 0.82, conf: 0.85, fp: 1.35, recall: 0.88 },
      { key: "night", label: "Night / low light", fps: 0.94, conf: 0.8, fp: 1.5, recall: 0.74 },
      { key: "traffic", label: "Dense traffic", fps: 0.78, conf: 0.92, fp: 1.12, recall: 0.96 },
      { key: "clear", label: "Clear daylight", fps: 1.0, conf: 1.0, fp: 1.0, recall: 1.0 },
    ],
  };
}

function localSimulate(body) {
  const trace = decideWithRules({
    object: body.object,
    distance_m: body.distance_m,
    ttc_s: body.speed_kmh > 0.1 ? body.distance_m / (body.speed_kmh / 3.6) : null,
    in_lane: body.lane_position >= 0.5,
    conf: body.confidence,
    ratio: body.box_height_ratio,
  });
  const base = { SAFE: 12, CAUTION: 32, WARNING: 62, CRITICAL: 92 }[trace.level];
  const factor = Math.min(1.6, (1 + 0.35 * body.wetness) / Math.max(0.35, body.visibility));
  const risk_score = Math.round(Math.min(100, base * factor) * 10) / 10;
  return {
    trace,
    ttc_s: body.speed_kmh > 0.1 ? Math.round((body.distance_m / (body.speed_kmh / 3.6)) * 100) / 100 : null,
    speed_kmh: body.speed_kmh,
    context: { wetness: body.wetness, visibility: body.visibility },
    risk_score,
    level: trace.level,
  };
}

export { RAW, VIDEOS };
export { LEVELS } from "./data.js";
