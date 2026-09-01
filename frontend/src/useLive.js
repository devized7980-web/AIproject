import { useEffect, useRef, useState } from "react";
import { RAW } from "./api.js";
import { VIDEOS } from "./data.js";

const TICK_MS = 210;
const emptyVideo = (v) => ({
  id: v.id, title: v.title, frames: v.frames || 360,
  duration: v.duration, events: v.events || [],
});

function simTick(video, idx, prev) {
  const frames = Math.max(1, video.frames);
  const pct = ((idx % frames) / frames) * 100;
  let ev = null, best = Infinity;
  video.events.forEach((e) => {
    const d = Math.abs(e.pct - pct);
    if (d < best) { best = d; ev = e; }
  });
  const boxes = ev && ev.boxes ? ev.boxes : [];
  const detections = boxes.map((b, k) => ({
    x: b.x, y: b.y, w: b.w, h: b.h,
    track_id: `${ev.object || "object"}-${k + 1}`,
    conf: ev.confidence || 0.5,
    name: ev.object, risk: ev.level, distance_m: ev.distance_m,
    ttc_s: ev.ttc_s, in_lane: ev.level !== "SAFE", source: "yolo11n",
  }));
  const counts = {
    potholes: detections.filter((d) => /pothole|crack/i.test(d.name)).length,
    vehicles: detections.filter((d) => /car|truck|bus|motorcycle|bicycle/i.test(d.name)).length,
    persons: detections.filter((d) => d.name === "person").length,
    total: detections.length,
  };
  const state = ev && ev.level ? { level: ev.level, action: ev.action } : { level: "SAFE", action: "ROAD CLEAR — CONTINUE CAREFULLY" };

  const cum = prev.cumulative || { frames: 0, detections: 0, alerts: 0 };
  cum.frames += 1;
  cum.detections += counts.total;

  let alert = null;
  if (ev && (ev.level === "WARNING" || ev.level === "CRITICAL")) {
    const key = `${video.id}:${ev.level}`;
    if (prev._lastAlert !== key) {
      alert = {
        id: `sim-${idx}`, video_id: video.id, video_title: video.title,
        time: ev.t, level: ev.level, object: ev.object, label: ev.label,
        distance_m: ev.distance_m, ttc_s: ev.ttc_s, confidence: ev.confidence,
        status: "open", assignee: null,
      };
      cum.alerts += 1;
    }
  }
  const ring = prev.ring || { potholes: [], vehicles: [], persons: [] };
  ring.potholes.push(counts.potholes);
  ring.vehicles.push(counts.vehicles);
  ring.persons.push(counts.persons);
  Object.keys(ring).forEach((k) => { if (ring[k].length > 24) ring[k] = ring[k].slice(-24); });

  return {
    type: "frame",
    video_id: video.id,
    video_title: video.title,
    frame: idx % frames,
    video_time: (idx % frames) / 30,
    duration: frames / 30,
    fps: +(1000 / TICK_MS).toFixed(2),
    replay_speed_fps: +(1000 / TICK_MS).toFixed(2),
    latency_ms: null,
    feed_mode: "Recorded detection replay",
    state,
    counts,
    detections,
    alert,
    cumulative: cum,
    ring,
    ts: Date.now(),
    _lastAlert: ev && (ev.level === "WARNING" || ev.level === "CRITICAL") ? `${video.id}:${ev.level}` : null,
  };
}

function useSimFeed(videoId, onMessage, enabled) {
  useEffect(() => {
    if (!enabled) return;
    const video = emptyVideo(VIDEOS.find((v) => v.id === videoId) || VIDEOS[0]);
    let idx = Math.floor(video.frames * 0.25);
    let prev = {};
    const timer = setInterval(() => {
      idx += 1;
      const msg = simTick(video, idx, prev);
      prev = msg;
      onMessage(msg);
    }, TICK_MS);
    return () => clearInterval(timer);
  }, [videoId, enabled, onMessage]);
}

// Raw camera clip URL when the backend is up, otherwise the processed clip.
export function liveSource(video) {
  const raw = video?.raw || RAW[video?.id];
  if (raw) return `/raw/${raw}`;
  return `/videos/${video?.file}`;
}

export function useLiveFeed({ videoId, video: selectedVideo, enabled = true }) {
  const [feed, setFeed] = useState(null);
  const [mode, setMode] = useState("connecting");
  const videoIdRef = useRef(videoId);
  const playingRef = useRef(true);
  const controlRef = useRef(() => {});
  videoIdRef.current = videoId;

  const onMessage = useRef((msg) => {
    if (msg.type === "frame") setFeed((f) => ({ ...f, ...msg }));
  });

  useEffect(() => {
    setFeed(null);
    setMode("connecting");
    if (!enabled) { setMode("off"); return; }

    let ws = null;
    let alive = true;
    let simTimer = null;
    const setPlaying = (playing) => {
      playingRef.current = playing;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "playback", playing }));
      }
    };
    controlRef.current = setPlaying;

    const startSim = () => {
      if (!alive) return;
      setMode("sim");
      const video = emptyVideo(selectedVideo || VIDEOS.find((v) => v.id === videoIdRef.current) || VIDEOS[0]);
      let idx = Math.floor(video.frames * 0.25);
      let prev = {};
       simTimer = setInterval(() => {
         if (!playingRef.current) return;
         idx += 1;
        const msg = simTick(video, idx, prev);
        prev = msg;
        onMessage.current(msg);
      }, TICK_MS);
    };

    fetch("/api/health", { signal: AbortSignal.timeout(1500) })
      .then((r) => { if (!r.ok) throw new Error("backend unavailable"); return r.json(); })
      .then(() => {
        if (!alive) return;
        connect();
      })
      .catch(() => startSim());

    const connect = () => {
      ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
      ws.onopen = () => {
        if (!alive) return;
         setMode("live");
         ws.send(JSON.stringify({ type: "select_video", video_id: videoIdRef.current }));
         ws.send(JSON.stringify({ type: "playback", playing: playingRef.current }));
      };
      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          if (!msg.video_id || msg.video_id === videoIdRef.current) onMessage.current(msg);
        } catch { /* ignore */ }
      };
      ws.onclose = () => { if (alive) { setMode("sim"); startSim(); } };
      ws.onerror = () => { try { ws.close(); } catch { /* noop */ } };
    };

    const selectTimer = setInterval(() => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "select_video", video_id: videoIdRef.current }));
      }
    }, 1000);

    return () => {
      alive = false;
      controlRef.current = () => {};
      clearInterval(selectTimer);
      if (simTimer) clearInterval(simTimer);
      try { ws && ws.close(); } catch { /* noop */ }
    };
  }, [videoId, selectedVideo, enabled]);

  return { feed, mode, setPlaying: (playing) => controlRef.current(playing) };
}
