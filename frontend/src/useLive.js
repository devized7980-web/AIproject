import { useEffect, useRef, useState } from "react";
import { RAW } from "./api.js";

// Raw camera clip URL when the backend is up, otherwise the processed clip.
export function liveSource(video) {
  const raw = video?.raw || RAW[video?.id];
  if (raw) return `/raw/${raw}`;
  return `/videos/${video?.file}`;
}

export function useLiveFeed({ videoId, video: selectedVideo, enabled = true }) {
  const [feed, setFeed] = useState(null);
  const [mode, setMode] = useState("connecting");
  const playingRef = useRef(true);
  const controlRef = useRef(() => {});

  useEffect(() => {
    setFeed(null);
    setMode("connecting");
    if (!enabled) { setMode("off"); return; }

    let ws = null;
    let alive = true;
    let retryTimer = null;
    const sessionVideoId = videoId;
    console.info("Safeway replay selected video", { videoId: sessionVideoId, normalizedFilename: selectedVideo?.raw, sourceFps: selectedVideo?.source_fps });
    const setPlaying = (playing) => {
      playingRef.current = playing;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "playback", playing }));
      }
    };
    controlRef.current = setPlaying;

    const connect = () => {
      if (!alive) return;
      ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
      ws.onopen = () => {
        if (!alive) return;
        setMode("live");
        console.info("Safeway WebSocket connected", { videoId: sessionVideoId });
        ws.send(JSON.stringify({ type: "select_video", video_id: sessionVideoId }));
        ws.send(JSON.stringify({ type: "playback", playing: playingRef.current }));
      };
      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          if (msg.type === "frame" && msg.video_id === sessionVideoId) {
            console.debug("Safeway replay frame", { videoId: sessionVideoId, frame: msg.frame, boxes: msg.detections?.length || 0 });
            setFeed((f) => ({ ...f, ...msg }));
          }
        } catch { /* ignore */ }
      };
      ws.onclose = () => {
        if (!alive) return;
        setMode("disconnected");
        console.warn("Safeway WebSocket disconnected", { videoId: sessionVideoId });
        retryTimer = setTimeout(connect, 1500);
      };
      ws.onerror = () => { try { ws.close(); } catch { /* noop */ } };
    };

    const checkBackend = () => {
      if (!alive) return;
      fetch("/api/health", { signal: AbortSignal.timeout(1500) })
        .then((r) => { if (!r.ok) throw new Error("backend unavailable"); return r.json(); })
        .then(() => connect())
        .catch(() => {
          if (!alive) return;
          setMode("offline");
          retryTimer = setTimeout(checkBackend, 1500);
        });
    };
    checkBackend();

    const selectTimer = setInterval(() => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "select_video", video_id: sessionVideoId }));
      }
    }, 1000);

    return () => {
      alive = false;
      controlRef.current = () => {};
      clearInterval(selectTimer);
      if (retryTimer) clearTimeout(retryTimer);
      try { ws && ws.close(); } catch { /* noop */ }
    };
  }, [videoId, selectedVideo, enabled]);

  return { feed, mode, setPlaying: (playing) => controlRef.current(playing) };
}
