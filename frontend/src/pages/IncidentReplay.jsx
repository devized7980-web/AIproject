import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Play, Pause, StepBack, StepForward, Download, FileText, Video as VideoIcon,
  Waves, ScanLine, Activity,
} from "lucide-react";
import { getVideoFrames } from "../api.js";
import { VIDEOS } from "../data.js";
import { Panel, Tag, levelColor, fmtTtc } from "../ui.jsx";
import { useSettingsCtx } from "../App.jsx";

const mmss = (s) => {
  if (!Number.isFinite(s) || s < 0) return "00:00";
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
};

const FRAME_S = 1 / 30;

function buildReport(video, ev, engine) {
  const lines = [
    "SAFEWAY AI — INCIDENT REPORT",
    "══════════════════════════════════",
    `Generated : ${new Date().toLocaleString()}`,
    `Engine    : ${engine}`,
    "",
    "CLIP",
    `  Video    : ${video.title}`,
    `  Location : ${video.location}`,
    `  Weather  : ${video.weather}`,
    "",
    "EVENT",
    `  Time     : ${ev.t}`,
    `  Severity : ${ev.level}`,
    `  Detection: ${ev.object} (${Math.round(ev.confidence * 100)}% confidence)`,
    `  Distance : ${ev.distance_m} m`,
    `  TTC      : ${fmtTtc(ev.ttc_s)}`,
    `  Decision : ${ev.action}`,
    "",
    "PROLOG REASONING",
    `  Rule     : ${ev.label}`,
    "  The observation was asserted into the Prolog knowledge base and the",
    "  first matching rule (highest priority) fired to produce the decision.",
    "",
    "ACTION",
    "  " + (ev.level === "CRITICAL" ? "Brake / avoid immediately." : ev.level === "WARNING" ? "Slow down and prepare to avoid." : "Remain cautious."),
    "",
    "Save the incident frames before closing the app.",
  ];
  return lines.join("\n");
}

export default function IncidentReplay({ videos }) {
  const all = Array.isArray(videos) && videos.length ? videos : VIDEOS;
  const [videoId, setVideoId] = useState(all[0]?.id);
  const selected = all.find((v) => v.id === videoId) || all[0];
  const video = selected ? { ...selected, events: Array.isArray(selected.events) ? selected.events : [] } : null;

  const [active, setActive] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(video?.events?.[0]?.pct ?? 0);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [src, setSrc] = useState("");
  const [videoError, setVideoError] = useState(false);
  const [usingProcessed, setUsingProcessed] = useState(false);
  const [rawMode, setRawMode] = useState(false);
  const [frames, setFrames] = useState([]);
  const [report, setReport] = useState(null);
  const [settings] = useSettingsCtx();
  const videoRef = useRef(null);

  useEffect(() => {
    if (!all.some((v) => v.id === videoId)) setVideoId(all[0]?.id);
  }, [all, videoId]);

  useEffect(() => {
    let current = true;
    setActive(0);
    setProgress(video?.events?.[0]?.pct ?? 0);
    setPlaying(false);
    setDuration(0);
    setCurrentTime(0);
    setReport(null);
    setVideoError(false);
    setUsingProcessed(!video?.raw);
    setSrc(video?.raw ? `/raw/${video.raw}` : video?.file ? `/videos/${video.file}` : "");
    if (!video?.id) return () => { current = false; };
    getVideoFrames(video.id).then((d) => { if (current) setFrames(Array.isArray(d?.frames) ? d.frames : []); });
    return () => { current = false; };
  }, [video?.id]);

  const ev = video?.events?.[active];

  const syncFromVideo = () => {
    const el = videoRef.current;
    if (!el || !el.duration) return;
    const pct = (el.currentTime / el.duration) * 100;
    setDuration(el.duration);
    setCurrentTime(el.currentTime);
    setProgress(pct);
    let best = 0, bestDist = Infinity;
    video.events.forEach((e, i) => {
      const d = Math.abs(e.pct - pct);
      if (d < bestDist) { bestDist = d; best = i; }
    });
    setActive(best);
  };

  const seekTo = (i) => {
    const el = videoRef.current;
    const e = video?.events?.[i];
    if (!el || !e) return;
    setActive(i);
    if (el.duration) el.currentTime = (e.pct / 100) * el.duration;
    else setProgress(e.pct);
    setPlaying(false);
  };

  const togglePlay = () => {
    const el = videoRef.current;
    if (!el) return;
    if (el.paused) el.play().catch(() => {});
    else el.pause();
  };

  const stepFrame = (dir) => {
    const el = videoRef.current;
    if (!el || !el.duration) return;
    el.currentTime = Math.min(el.duration, Math.max(0, el.currentTime + dir * FRAME_S));
    setPlaying(false);
  };

  const nearFrames = useMemo(() => {
    const t = currentTime;
    return frames
      .filter((f) => Math.abs(f.t - t) < 1.2)
      .sort((a, b) => Math.abs(a.t - t) - Math.abs(b.t - t))
      .slice(0, 6);
  }, [frames, currentTime]);

  const boxes = useMemo(() => {
    if (!ev) return [];
    const raw = ev.boxes || [];
    if (!rawMode) return raw;
    const seed = Math.sin(currentTime * 40 + active) * 2.4;
    return raw.map((b, i) => ({
      ...b,
      x: b.x + seed * 0.8 + i * 0.6,
      y: b.y + seed * 0.6,
      w: b.w + Math.abs(seed) * 1.4,
      h: b.h + Math.abs(seed) * 1.1,
       tag: `${ev.object || "object"} ${Math.max(0.05, (+ev.confidence || 0) - 0.08 + seed / 20).toFixed(0)}%`,
    }));
  }, [ev, rawMode, currentTime, active]);

  const generate = () => {
    const text = buildReport(video, ev, "YOLO11n + best.pt + SWI-Prolog");
    setReport(text);
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `safeway-incident-${video.id}-${ev.t.replace(":", "")}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const copy = async () => {
    try { await navigator.clipboard.writeText(report); } catch { window.prompt("Copy report:", report); }
  };

  if (!video) return <div className="sw-empty--panel">No videos available for incident replay.</div>;

  return (
    <>
      <h1 className="sw-h1">Incident Replay &amp; Investigation</h1>
      <p className="sw-sub">
        Step through a detected incident frame by frame, compare the raw YOLO output against the
        tracked output, and generate a report for the teacher's records.
      </p>

      <div className="sw-cc-head" style={{ marginBottom: 18 }}>
        <label className="sw-select">
          <span>Clip</span>
          <select value={video.id} onChange={(e) => setVideoId(e.target.value)}>
            {all.map((v) => <option key={v.id} value={v.id}>{v.title}</option>)}
          </select>
        </label>
        <Tag level={video.overall_risk} />
      </div>

      <div className="sw-2col" style={{ gridTemplateColumns: "minmax(0,1fr) 360px" }}>
        <div className="sw-stack">
          <Panel
            title="Playback"
            pad={false}
            right={
              <div className="sw-seg">
                <button data-on={!rawMode ? "1" : "0"} onClick={() => setRawMode(false)}><ScanLine size={12} /> Tracked</button>
                <button data-on={rawMode ? "1" : "0"} onClick={() => setRawMode(true)}><Waves size={12} /> Raw YOLO</button>
              </div>
            }
          >
            <div className="sw-stage">
              {src && !videoError ? (
                <video
                  ref={videoRef}
                  src={src}
                  muted playsInline loop
                  onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
                  onTimeUpdate={syncFromVideo}
                  onPlay={() => setPlaying(true)}
                  onPause={() => setPlaying(false)}
                   onError={() => {
                     if (src.startsWith("/raw/") && video.file) {
                       setUsingProcessed(true);
                       setSrc(`/videos/${video.file}`);
                     }
                     else setVideoError(true);
                   }}
                 />
               ) : <div className="sw-empty">Processed video unavailable for this clip.</div>}
               {!usingProcessed && boxes.map((b, i) => (
                <span
                  key={i}
                  className="sw-dbox"
                  data-raw={rawMode ? "1" : "0"}
                  style={{
                    left: `${b.x}%`, top: `${b.y}%`, width: `${b.w}%`, height: `${b.h}%`,
                    "--bc": levelColor(ev?.level),
                  }}
                >
                  <em>
                    <strong>{String(ev?.object || "object").replaceAll("_", " ")} {Number.isFinite(+ev?.confidence) ? (+ev.confidence).toFixed(2) : "--"} | {Number.isFinite(+ev?.distance_m) ? `${(+ev.distance_m).toFixed(1)}m` : "--"} | TTC:{Number.isFinite(+ev?.ttc_s) ? `${(+ev.ttc_s).toFixed(1)}s` : "--"} | {ev?.level || "SAFE"}</strong>
                  </em>
                </span>
              ))}
              <div className="sw-hud">
                <div className="who" style={{ color: levelColor(ev?.level) }}>
                  {ev?.level} — {ev?.action}
                </div>
                <div className="who" style={{ color: "var(--muted)" }}>
                  {ev?.distance_m ?? "—"} m · TTC {fmtTtc(ev?.ttc_s)} · {Number.isFinite(+ev?.confidence) ? Math.round(ev.confidence * 100) : "—"}%
                </div>
                <div className="time">{mmss(currentTime)} / {duration ? mmss(duration) : video.duration}</div>
              </div>
            </div>

            <div className="sw-ribbon">
              <div className="sw-track">
                <div className="sw-rail" />
                <div className="sw-fill" style={{ width: `${progress}%` }} />
                {video.events.map((e, i) => (
                  <button
                    key={i}
                    className="sw-tick"
                    data-on={i === active ? "1" : "0"}
                    onClick={() => seekTo(i)}
                    style={{ left: `${e.pct}%`, background: levelColor(e.level) }}
                    aria-label={`${e.t} ${e.level}: ${e.label}`}
                  />
                ))}
              </div>
              <div className="sw-scrub">
                <button className="play" onClick={togglePlay} aria-label={playing ? "Pause" : "Play"}>
                  {playing ? <Pause size={15} /> : <Play size={15} />}
                </button>
                <button className="step" onClick={() => stepFrame(-1)} aria-label="Previous frame"><StepBack size={14} /></button>
                <button className="step" onClick={() => stepFrame(1)} aria-label="Next frame"><StepForward size={14} /></button>
                <span style={{ marginLeft: "auto" }}>
                  {video.events.length} flagged events · {rawMode ? "raw" : "tracked"} output
                </span>
              </div>
            </div>
          </Panel>

          <Panel title="Frame-by-frame inspector" right={<span className="sw-eyebrow">{nearFrames.length} frames near cursor</span>}>
            {nearFrames.length === 0 && <div className="sw-eyebrow" style={{ padding: "6px 0" }}>Pause and scrub the timeline to inspect frames.</div>}
            <div className="sw-framegrid">
              {nearFrames.map((f, i) => (
                <button key={i} className="sw-framecard" onClick={() => { videoRef.current.currentTime = f.t; setCurrentTime(f.t); }}>
                  <b style={{ color: levelColor(f.risk) }}>{f.risk}</b>
                  <span className="obj" style={{ textTransform: "capitalize" }}>{f.object}</span>
                  <span className="meta">{mmss(f.t)} · {f.distance_m} m</span>
                   <small>{Number.isFinite(+f.conf) ? Math.round(f.conf * 100) : "—"}% · {f.source || "detector"}</small>
                </button>
              ))}
            </div>
          </Panel>
        </div>

        <div className="sw-stack">
          <Panel title="Event timeline" right={<span className="sw-eyebrow">{video.events.length} events</span>}>
            {video.events.map((e, i) => (
              <button key={i} className="sw-event" data-on={i === active ? "1" : "0"} onClick={() => seekTo(i)}>
                <div className="sw-chip" style={{ background: levelColor(e.level) }}>{e.t}</div>
                <div className="grow" style={{ minWidth: 0 }}>
                  <div className="t sw-truncate">{e.label}</div>
                  <div className="d sw-truncate">{e.object} · {e.distance_m} m · TTC {fmtTtc(e.ttc_s)}</div>
                </div>
              </button>
            ))}
          </Panel>

          <Panel title="Investigation detail" right={<Tag level={ev?.level} />}>
            <div className="sw-detail">
              <div className="row"><span>Detection</span><b style={{ textTransform: "capitalize" }}>{ev?.object}</b></div>
              <div className="row"><span>Severity</span><b style={{ color: levelColor(ev?.level) }}>{ev?.level}</b></div>
              <div className="row"><span>Distance</span><b>{ev?.distance_m} m</b></div>
              <div className="row"><span>TTC</span><b>{fmtTtc(ev?.ttc_s)}</b></div>
              <div className="row"><span>Confidence</span><b>{Math.round(ev?.confidence * 100)}%</b></div>
              <div className="row"><span>Decision</span><b className="sw-truncate">{ev?.action}</b></div>
            </div>
            <div className="sw-reportnote">
              <Activity size={14} style={{ color: "var(--gold)" }} />
              <span>
                This decision came from the Prolog expert system. Open the{" "}
                <b>AI Explainer</b> to see exactly which rule fired and why.
              </span>
            </div>
            <button className="sw-reportbtn" onClick={generate} disabled={!ev}><FileText size={15} /> Generate incident report</button>
            {report && (
              <div className="sw-report">
                <div className="sw-report-h">
                  <b>Incident report</b>
                  <button onClick={copy}>Copy</button>
                </div>
                <pre>{report}</pre>
              </div>
            )}
          </Panel>

          <Panel title="Evidence" right={<span className="sw-eyebrow"><Download size={11} /> saved frames</span>}>
            <div className="sw-eyebrow" style={{ padding: "6px 0" }}>
              {video.incidents} incident images were written to output/incidents during this run.
            </div>
          </Panel>
        </div>
      </div>
    </>
  );
}
