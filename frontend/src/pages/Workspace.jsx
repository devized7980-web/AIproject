import React, { useState, useEffect, useRef } from "react";
import { Video, Play, Pause, Car, Users, Bike, Construction } from "lucide-react";
import { VIDEO_BASE, LEVELS } from "../data.js";
import { Panel, RiskBar, levelColor, fmtTtc } from "../ui.jsx";

const GROUPS = [
  { label: "Vehicles", keys: ["car", "truck", "bus", "motorcycle"], Icon: Car },
  { label: "Pedestrians", keys: ["person"], Icon: Users },
  { label: "Cyclists", keys: ["bicycle"], Icon: Bike },
  { label: "Road anomalies", keys: ["pothole", "road crack", "longitudinal crack", "transverse crack", "alligator cracking"], Icon: Construction },
];

const mmss = (s) => {
  if (!Number.isFinite(s) || s < 0) return "00:00";
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
};

export default function Workspace({ video, onBack }) {
  const [active, setActive] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(video.events[0]?.pct ?? 0);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [hasFile, setHasFile] = useState(true);
  const videoRef = useRef(null);
  const ev = video.events[active];

  // Reset when a different clip is opened.
  useEffect(() => {
    setActive(0);
    setProgress(video.events[0]?.pct ?? 0);
    setPlaying(false);
    setDuration(0);
    setCurrentTime(0);
    setHasFile(true);
  }, [video.id]);

  const syncFromVideo = () => {
    const el = videoRef.current;
    if (!el || !el.duration) return;
    const pct = (el.currentTime / el.duration) * 100;
    setDuration(el.duration);
    setCurrentTime(el.currentTime);
    setProgress(pct);
    // Highlight the event nearest to the live position.
    let best = 0;
    let bestDist = Infinity;
    video.events.forEach((e, i) => {
      const d = Math.abs(e.pct - pct);
      if (d < bestDist) { bestDist = d; best = i; }
    });
    setActive(best);
  };

  const seekTo = (i) => {
    const el = videoRef.current;
    const e = video.events[i];
    if (!el || !e) return;
    setActive(i);
    if (el.duration) {
      el.currentTime = (e.pct / 100) * el.duration;
    } else {
      setProgress(e.pct);
    }
    setPlaying(false);
  };

  const togglePlay = () => {
    const el = videoRef.current;
    if (!el) return;
    if (el.paused) el.play().catch(() => {});
    else el.pause();
  };

  return (
    <>
      <button
        className="sw-eyebrow"
        onClick={onBack}
        style={{ marginBottom: 10, display: "inline-flex", alignItems: "center", gap: 6, cursor: "pointer", color: "var(--muted)" }}
      >
        <span style={{ color: "var(--gold)" }}>←</span> Back to gallery
      </button>
      <h1 className="sw-h1">Video Analysis Workspace</h1>
      <p className="sw-sub">
        {video.title} — {video.location}, {video.weather}. {video.frames.toLocaleString()} frames
        processed at {video.average_processing_fps} fps.
      </p>

      <div className="sw-2col">
        <Panel title="Hazard detection feed" pad={false} right={<span className="sw-eyebrow">{video.file}</span>}>
          <div className="sw-stage">
            {hasFile ? (
              <video
                ref={videoRef}
                src={VIDEO_BASE + video.file}
                muted
                playsInline
                loop
                onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
                onTimeUpdate={syncFromVideo}
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                onError={() => setHasFile(false)}
              />
            ) : (
              <div className="sw-empty">
                <Video size={26} />
                <div style={{ fontSize: 13, color: "var(--text)" }}>No video file loaded</div>
                <div style={{ fontSize: 12, maxWidth: "42ch" }}>
                  Copy the clip into <code>public/videos/</code> as <code>{video.file}</code>. The overlay
                  below still replays the saved detections.
                </div>
              </div>
            )}

            <div className="sw-hud">
              <div className="who" style={{ color: levelColor(ev.level) }}>
                {ev.level} — {ev.action}
              </div>
              <div className="who" style={{ color: "var(--muted)" }}>
                {ev.distance_m} m · TTC {fmtTtc(ev.ttc_s)}
              </div>
              <div className="time">{mmss(currentTime)} / {duration ? mmss(duration) : video.duration}</div>
            </div>
          </div>

          {/* risk ribbon — follows the real video position */}
          <div className="sw-ribbon">
            <div className="sw-track">
              <div className="sw-rail" />
              <div className="sw-fill" style={{ width: `${progress}%` }} />
              {video.events.map((e, i) => (
                <button
                  key={i}
                  className="sw-tick"
                  data-on={i === active ? "1" : "0"}
                  aria-label={`${e.t} ${e.level}: ${e.label}`}
                  onClick={() => seekTo(i)}
                  style={{ left: `${e.pct}%`, background: levelColor(e.level) }}
                />
              ))}
            </div>
            <div className="sw-scrub">
              <button className="play" onClick={togglePlay} aria-label={playing ? "Pause" : "Play"}>
                {playing ? <Pause size={15} /> : <Play size={15} />}
              </button>
              <span style={{ marginLeft: "auto" }}>
                {video.events.length} flagged events · click a marker to jump
              </span>
            </div>
          </div>
        </Panel>

        <div className="sw-stack">
          <Panel title="Event timeline">
            {video.events.map((e, i) => (
              <button key={i} className="sw-event" data-on={i === active ? "1" : "0"} onClick={() => seekTo(i)}>
                <div className="sw-chip" style={{ background: levelColor(e.level) }}>{e.t}</div>
                <div className="grow" style={{ minWidth: 0 }}>
                  <div className="t sw-truncate">{e.label}</div>
                  <div className="d sw-truncate">{e.action}</div>
                </div>
              </button>
            ))}
          </Panel>

          <Panel title="Hazard statistics">
            {GROUPS.map(({ label, keys, Icon }) => {
              const n = keys.reduce((s, k) => s + (video.object_counts[k] || 0), 0);
              return (
                <div className="sw-row" key={label}>
                  <Icon size={15} style={{ color: "var(--muted)", flexShrink: 0 }} />
                  <div className="grow">{label}</div>
                  <div className="num" style={{ color: "var(--gold)", fontSize: 17 }}>{n}</div>
                </div>
              );
            })}
            <div style={{ marginTop: 14 }}>
              <div className="sw-eyebrow" style={{ marginBottom: 8 }}>Overall scene risk</div>
              <div className="sw-gauge">
                <div className="arc">
                  {LEVELS.map((l) => (
                    <span
                      key={l}
                      style={{
                        background: levelColor(l),
                        opacity: LEVELS.indexOf(l) <= LEVELS.indexOf(video.overall_risk) ? 1 : 0.18,
                      }}
                    />
                  ))}
                </div>
                <span className="sw-needle" style={{ color: levelColor(video.overall_risk) }}>
                  {video.overall_risk}
                </span>
              </div>
            </div>
          </Panel>

          <Panel title="Run summary">
            <RiskBar counts={video.risk_counts} />
          </Panel>
        </div>
      </div>
    </>
  );
}
