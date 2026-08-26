import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity, RadioTower, Gauge, Timer, Zap, Car, Users, Cone, BellRing,
  ChevronRight, ShieldAlert,
} from "lucide-react";
import { getVideoFrames, RAW } from "../api.js";
import { VIDEOS } from "../data.js";
import { useLiveFeed } from "../useLive.js";
import { playAlertSound } from "../sound.js";
import { Panel, Tag, levelColor, levelHue, fmtTtc } from "../ui.jsx";
import { useSettingsCtx } from "../App.jsx";

const mmss = (s) => {
  if (!Number.isFinite(s) || s < 0) return "00:00";
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
};

function uniqueFrameTracks(rows) {
  const unique = new Map();
  for (const row of rows) {
    const key = row.track_id || `${row.name}:${Math.round(row.x)}:${Math.round(row.y)}`;
    const current = unique.get(key);
    if (!current || (current.risk === "CRITICAL" ? 1 : 0) < (row.risk === "CRITICAL" ? 1 : 0) || row.conf > current.conf) {
      unique.set(key, row);
    }
  }
  return [...unique.values()];
}

const labelText = (d) => {
  const name = String(d.name || d.object || "object").replaceAll("_", " ").toUpperCase();
  const conf = Number.isFinite(+d.conf) ? `${Math.round(+d.conf * 100)}%` : "--";
  const ttc = Number.isFinite(+d.ttc_s) ? `${(+d.ttc_s).toFixed(1)}s` : "--";
  return { title: `${name} ${conf} | TTC:${ttc}` };
};

function placeLabels(rows) {
  const placed = [];
  return rows.map((d) => {
    const label = labelText(d);
    const width = Math.min(58, Math.max(8, label.title.length * 0.52 + 2));
    const height = 5.5;
    const x = Math.max(0, Math.min(100 - width, d.x));
    const above = d.y >= height;
    const candidates = above ? [
      { left: x, top: d.y - height, inside: false },
      { left: Math.max(0, Math.min(100 - width, d.x + d.w - width)), top: d.y - height, inside: false },
      { left: x, top: d.y, inside: true },
    ] : [{ left: x, top: d.y, inside: true }];
    const overlaps = (a, b) => a.left < b.left + b.width && a.left + width > b.left && a.top < b.top + b.height && a.top + height > b.top;
    let chosen = candidates.find((candidate) => !placed.some((other) => overlaps(candidate, other))) || candidates[0];
    for (let shift = 1; placed.some((other) => overlaps(chosen, other)) && shift < 5; shift += 1) {
      chosen = { ...chosen, left: Math.max(0, Math.min(100 - width, chosen.left + shift * 3)) };
    }
    placed.push({ ...chosen, width, height });
    return { ...d, label, labelStyle: {
      left: `${chosen.left - d.x}%`,
      ...(chosen.inside ? { top: "0" } : {}),
    } };
  });
}

function RouteSchematic({ ring, state }) {
  const max = Math.max(...(ring.vehicles || []), ...(ring.potholes || []), 1);
  const pos = Math.sin(Date.now() / 1200) * 0.5 + 0.5;
  return (
    <div className="sw-route">
      <svg viewBox="0 0 360 120" preserveAspectRatio="none">
        <defs>
          <linearGradient id="rtseg" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor={levelHue("SAFE")} stopOpacity="0.85" />
            <stop offset="55%" stopColor={levelHue("CAUTION")} stopOpacity="0.85" />
            <stop offset="100%" stopColor={state.level === "CRITICAL" ? levelHue("CRITICAL") : levelHue("WARNING")} stopOpacity="0.85" />
          </linearGradient>
        </defs>
        <rect x="30" y="46" width="300" height="18" rx="9" fill="url(#rtseg)" />
        <line x1="30" y1="55" x2="330" y2="55" stroke="#0B0F17" strokeWidth="1.5" strokeDasharray="6 6" />
        {(ring.potholes || []).slice(-6).map((n, i) =>
          n > 0 ? (
            <g key={i} className="sw-pulse" style={{ animationDelay: `${i * 0.18}s` }}>
              <circle cx={40 + i * 48} cy={46} r={9} fill="none" stroke={levelHue("CRITICAL")} strokeWidth="1.5" />
              <circle cx={40 + i * 48} cy={46} r={3.5} fill={levelHue("CRITICAL")} />
            </g>
          ) : null
        )}
        {(ring.persons || []).slice(-4).map((n, i) =>
          n > 0 ? (
            <g key={`p${i}`} className="sw-glow">
              <rect x={80 + i * 60} y={36} width={14} height={26} rx={4} fill={levelHue("CAUTION")} opacity="0.55" />
            </g>
          ) : null
        )}
        <g className="sw-drive" style={{ transform: `translateX(${pos * 260}px)` }}>
          <rect x="0" y="42" width="10" height="18" rx="3" fill="#EAF0F8" opacity="0.9" />
        </g>
      </svg>
      <div className="sw-route-key">
        <span><i style={{ background: levelHue("SAFE") }} />Safe</span>
        <span><i style={{ background: levelHue("CAUTION") }} />Pedestrian zone</span>
        <span><i style={{ background: levelHue("CRITICAL") }} />Pothole</span>
      </div>
    </div>
  );
}

export default function LiveCommandCenter({ videos }) {
  const all = Array.isArray(videos) && videos.length ? videos : VIDEOS;
  const [videoId, setVideoId] = useState(all[0]?.id);
  const video = all.find((v) => v.id === videoId) || all[0];
  const { feed, mode } = useLiveFeed({ videoId: video?.id, video, enabled: !!video });
  const [showOverlay, setShowOverlay] = useState(true);
  const [settings] = useSettingsCtx();
  const lastAlert = useRef(null);
  const videoRef = useRef(null);
  const [frames, setFrames] = useState([]);
  const [curTime, setCurTime] = useState(0);
  const [stageRatio, setStageRatio] = useState(null);
  const sourceFps = Number(video?.source_fps) > 0 ? Number(video.source_fps) : 30;

  useEffect(() => {
    if (!all.some((v) => v.id === videoId)) setVideoId(all[0]?.id);
  }, [all, videoId]);

  useEffect(() => {
    let on = true;
    setFrames([]);
    setCurTime(0);
    if (!video?.id) return () => { on = false; };
    getVideoFrames(video.id).then((d) => {
      const list = (d && Array.isArray(d.frames) ? d.frames : [])
        .filter((f) => Number.isFinite(f.t) && Number.isFinite(f.x));
      if (on) setFrames(list);
    });
    return () => { on = false; };
  }, [video?.id]);

  // onTimeUpdate is intentionally too sparse for a moving box. Sample the
  // presented video clock every animation frame instead of using processing FPS.
  useEffect(() => {
    const element = videoRef.current;
    if (!element) return undefined;
    let raf = 0;
    const sample = () => {
      setCurTime(element.currentTime || 0);
      raf = requestAnimationFrame(sample);
    };
    raf = requestAnimationFrame(sample);
    return () => cancelAnimationFrame(raf);
  }, [video?.id]);

  const synced = useMemo(() => {
    const empty = {
      rows: [],
      counts: { potholes: 0, vehicles: 0, persons: 0, total: 0 },
      state: { level: "SAFE", action: "ROAD CLEAR — CONTINUE CAREFULLY" },
      frame: null,
      time: 0,
    };
    if (!frames.length) return empty;
    // Pipeline timestamps are recorded as frame_no / source_fps (frame 1 is
    // the first displayed frame), so do not add a synthetic frame offset.
    const targetFrame = Math.max(1, Math.round(curTime * sourceFps));
    let best = frames[0];
    let bd = Infinity;
    for (const f of frames) {
      const d = Math.abs((Number.isFinite(+f.frame) ? +f.frame : Math.round(f.t * sourceFps) + 1) - targetFrame);
      if (d < bd) { bd = d; best = f; }
    }
    const rows = uniqueFrameTracks(frames.filter((f) => f.frame === best.frame)).sort((a, b) => b.conf - a.conf);
    const counts = { potholes: 0, vehicles: 0, persons: 0, total: rows.length };
    const prio = { SAFE: 0, CAUTION: 1, WARNING: 2, CRITICAL: 3 };
    let worst = null, wp = -1;
    for (const r of rows) {
      if (r.object === "pothole" || /crack/.test(r.object)) counts.potholes += 1;
      else if (["car", "truck", "bus", "motorcycle", "bicycle"].includes(r.object)) counts.vehicles += 1;
      else if (r.object === "person") counts.persons += 1;
      const p = prio[r.risk] ?? 0;
      if (p > wp) { wp = p; worst = r; }
    }
    const state = worst
       ? { level: worst.risk || "SAFE", action: String(worst.action || "CONTINUE CAREFULLY").toUpperCase() }
      : { level: "SAFE", action: "ROAD CLEAR — CONTINUE CAREFULLY" };
    return { rows, counts, state, frame: best.frame, time: best.t };
  }, [frames, curTime, sourceFps]);

  useEffect(() => {
    const a = feed?.alert;
    if (a && a.id !== lastAlert.current) {
      lastAlert.current = a.id;
      if (settings.sound && (a.level === "WARNING" || a.level === "CRITICAL")) {
        playAlertSound(a.level);
      }
    }
  }, [feed?.alert, settings.sound]);

  const kpi = useMemo(() => {
    const f = feed || {};
    const c = synced.counts;
    return [
       { label: "FPS", value: Number.isFinite(+f.fps) ? (+f.fps).toFixed(1) : "—", unit: "fps", Icon: Gauge },
       { label: "Latency", value: Number.isFinite(+f.latency_ms) ? (+f.latency_ms).toFixed(0) : "—", unit: "ms", Icon: Timer },
      { label: "Potholes", value: c.potholes ?? 0, Icon: Cone, accent: "var(--critical)" },
      { label: "Vehicles", value: c.vehicles ?? 0, Icon: Car, accent: "var(--gold)" },
      { label: "Pedestrians", value: c.persons ?? 0, Icon: Users, accent: "var(--caution)" },
      { label: "Live alerts", value: f.cumulative?.alerts ?? 0, Icon: BellRing, accent: "var(--warning)" },
    ];
  }, [feed, synced]);

  const overlayRows = useMemo(
    () => placeLabels([...synced.rows].sort((a, b) => (b.risk === "CRITICAL") - (a.risk === "CRITICAL"))),
    [synced.rows],
  );
  const dets = synced.rows.slice(0, 8);

  return (
    <>
      <div className="sw-cc-head">
        <div>
          <h1 className="sw-h1">Live Command Center</h1>
          <p className="sw-sub">
            The detection pipeline as a live feed. Replays the real recorded detections from the
            last pipeline run — this is what the driver sees in the moment.
          </p>
        </div>
        <div className="sw-cc-actions">
          <span className="sw-livebadge" data-mode={mode}>
            <RadioTower size={12} /> {mode === "live" ? "LIVE FEED" : mode === "sim" ? "LOCAL MIRROR" : "CONNECTING"}
          </span>
          <label className="sw-select">
            <span>Source</span>
            <select value={video?.id} onChange={(e) => setVideoId(e.target.value)}>
              {all.map((v) => <option key={v.id} value={v.id}>{String(v.title || v.id).split("—")[0].trim()}</option>)}
            </select>
          </label>
        </div>
      </div>

      <div className="sw-2col" style={{ gridTemplateColumns: "minmax(0,1fr) 360px" }}>
        <div className="sw-stack">
          <Panel
            title="Camera feed"
            pad={false}
            right={
              <button className="sw-linkbtn" onClick={() => setShowOverlay((s) => !s)}>
                <Activity size={12} /> {showOverlay ? "Hide" : "Show"} live boxes
              </button>
            }
          >
            <div className="sw-stage" style={stageRatio ? { aspectRatio: String(stageRatio) } : undefined}>
              <video
                ref={videoRef}
                key={video?.id}
                src={video?.raw || RAW[video?.id] ? `/raw/${video.raw || RAW[video.id]}` : ""}
                muted playsInline loop autoPlay
                onLoadedMetadata={(e) => {
                  const v = e.currentTarget;
                 if (v.videoWidth && v.videoHeight) setStageRatio(v.videoWidth / v.videoHeight);
                }}
              />
              {showOverlay && (video?.raw || RAW[video?.id]) && overlayRows.map((d, i) => (
                <span
                  key={`${d.track_id || d.name}-${d.frame || i}`}
                  className="sw-dbox"
                  data-soft={d.in_lane ? "0" : "1"}
                  data-risk={d.risk}
                  style={{
                    left: `${d.x}%`, top: `${d.y}%`, width: `${d.w}%`, height: `${d.h}%`,
                    "--bc": levelColor(d.risk),
                  }}
                >
                  <em style={d.labelStyle}>
                    <strong>{d.label.title}</strong>
                  </em>
                </span>
              ))}
              {showOverlay && <div className="sw-hud">
                <div className="who" style={{ color: levelColor(synced.state.level) }}>
                  {synced.state.level} — {synced.state.action}
                </div>
                <div className="who" style={{ color: "var(--muted)" }}>
                  {String(video?.title || video?.id || "").split("—")[0]} · {synced.frame != null ? `f${synced.frame}` : "…"} {mmss(synced.time)}
                </div>
                <div className="time" data-level={synced.state.level}>{feed ? `${Number.isFinite(+feed.fps) ? (+feed.fps).toFixed(1) : "—"} fps · ${Number.isFinite(+feed.latency_ms) ? (+feed.latency_ms).toFixed(0) : "—"} ms` : "—"}</div>
              </div>}
            </div>
          </Panel>

          <div className="sw-kpis" style={{ marginBottom: 0 }}>
            {kpi.map((k) => (
              <div className="sw-kpi" key={k.label}>
                <div className="l">
                  <k.Icon size={12} style={{ verticalAlign: -1, marginRight: 5, color: k.accent || "var(--gold)" }} />
                  {k.label}
                </div>
                <div className="v" style={k.accent ? { color: k.accent } : undefined}>
                  {k.value}{k.unit && <span style={{ fontSize: 13, color: "var(--muted)", marginLeft: 4 }}>{k.unit}</span>}
                </div>
              </div>
            ))}
          </div>

          <div className="sw-2col" style={{ gridTemplateColumns: "minmax(0,1fr) 320px" }}>
            <Panel title="Detection stream" right={<span className="sw-eyebrow">{dets.length} live objects</span>}>
              {dets.length === 0 && <div className="sw-eyebrow" style={{ padding: "6px 0" }}>No objects this frame — road clear.</div>}
              <div className="sw-detfeed">
                {dets.map((d, i) => (
                  <div className="row" key={i}>
                    <i className="dot" style={{ background: levelColor(d.risk) }} />
                    <div className="grow sw-truncate">
                      <span style={{ textTransform: "capitalize" }}>{d.name}</span>
                      <small>{d.in_lane ? "in lane" : "off lane"} · {d.distance_m} m · TTC {fmtTtc(d.ttc_s)}</small>
                    </div>
                    <b style={{ color: levelColor(d.risk), fontFamily: "var(--data)" }}>{Math.round(d.conf * 100)}%</b>
                  </div>
                ))}
              </div>
            </Panel>

            <div className="sw-stack">
              <Panel title="Route state" right={<Tag level={synced.state.level} />}>
                <RouteSchematic ring={feed?.ring || {}} state={synced.state} />
              </Panel>
              <Panel title="Cumulative">
                <div className="sw-statline">
                  <div><b>{(feed?.cumulative?.frames || 0).toLocaleString()}</b><span>frames</span></div>
                  <div><b>{(feed?.cumulative?.detections || 0).toLocaleString()}</b><span>detections</span></div>
                  <div><b>{feed?.cumulative?.alerts || 0}</b><span>alerts</span></div>
                </div>
              </Panel>
            </div>
          </div>
        </div>

        <div className="sw-stack">
          <Panel title="Current safety state" right={<ShieldAlert size={14} style={{ color: levelColor(synced.state.level) }} />}>
            <div className="sw-bigstate" style={{ "--c": levelColor(synced.state.level) }}>
              <div className="lvl">{synced.state.level}</div>
              <div className="act">{synced.state.action}</div>
            </div>
            <div style={{ marginTop: 12 }}>
              <div className="sw-eyebrow" style={{ marginBottom: 6 }}>Rolling counts (last 24 frames)</div>
              <div className="sw-sparkrow">
                <span>Potholes<SparkMini values={feed?.ring?.potholes || []} color={levelHue("CRITICAL")} /></span>
                <span>Vehicles<SparkMini values={feed?.ring?.vehicles || []} color={levelHue("CAUTION")} /></span>
                <span>Persons<SparkMini values={feed?.ring?.persons || []} color={levelHue("SAFE")} /></span>
              </div>
            </div>
          </Panel>

          <Panel title="Latest alerts">
            <div className="sw-livealerts">
              {!feed?.alert && <div className="sw-eyebrow" style={{ padding: "6px 0" }}>No recent alerts — all clear.</div>}
              {feed?.alert && (
                <button className="sw-alertrow" data-lvl={feed.alert.level}>
                  <i className="pulse" style={{ background: levelColor(feed.alert.level) }} />
                  <div className="grow" style={{ minWidth: 0 }}>
                    <div className="t sw-truncate">{feed.alert.label}</div>
                    <div className="d sw-truncate">{feed.alert.video_title} · {feed.alert.distance_m} m · {fmtTtc(feed.alert.ttc_s)}</div>
                  </div>
                  <ChevronRight size={15} style={{ color: "var(--muted)", flexShrink: 0 }} />
                </button>
              )}
               {all
                 .flatMap((v) => (Array.isArray(v.events) ? v.events : []).map((e) => ({ ...e, video: v.title || "" })))
                 .filter((e) => e.video === video?.title)
                 .filter((e) => e.level === "WARNING" || e.level === "CRITICAL")
                .slice(0, 4)
                .map((e, i) => (
                  <div className="sw-alertrow" data-lvl={e.level} key={`${e.video}-${e.t}-${i}`}>
                    <i className="pulse" style={{ background: levelColor(e.level) }} />
                    <div className="grow" style={{ minWidth: 0 }}>
                      <div className="t sw-truncate">{e.label}</div>
                      <div className="d sw-truncate">{e.video} · {e.distance_m} m</div>
                    </div>
                  </div>
                ))}
            </div>
          </Panel>

          <Panel title="Live feed engine" right={<span className="sw-eyebrow">v2.0</span>}>
            <div className="sw-engine">
              <div className="row"><span>Detector</span><b>YOLO11n + custom best.pt</b></div>
              <div className="row"><span>Reasoning</span><b>{feed ? (mode === "live" ? "SWI-Prolog live" : "Rule mirror") : "—"}</b></div>
              <div className="row"><span>Source clip</span><b>{String(video?.title || video?.id || "—").split("—")[0]}</b></div>
              <div className="row"><span>Weather</span><b>{video?.weather}</b></div>
            </div>
          </Panel>
        </div>
      </div>
    </>
  );
}

const SparkMini = ({ values, color }) => (
  <span className="sw-spark-mini">
    {(values.length ? values : [0]).map((v, i) => (
      <i key={i} style={{ height: `${Math.max(8, v * 18)}%`, background: color }} />
    ))}
  </span>
);
