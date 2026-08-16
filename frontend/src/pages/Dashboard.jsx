import React, { useMemo } from "react";
import { ChevronRight, Film, ScanLine, ShieldAlert, Timer, Gauge } from "lucide-react";
import { VIDEOS, LEVELS } from "../data.js";
import { Panel, Kpi, Tag, RiskBar, levelColor, fmtTtc } from "../ui.jsx";

export default function Dashboard({ onOpen }) {
  const totals = useMemo(() => {
    const t = { frames: 0, dets: 0, incidents: 0, fps: 0, minTtc: Infinity, counts: {}, objects: {} };
    VIDEOS.forEach((v) => {
      t.frames += v.frames;
      t.dets += v.total_detections;
      t.incidents += v.incidents;
      t.fps += v.average_processing_fps;
      t.minTtc = Math.min(t.minTtc, v.minimum_ttc_s);
      LEVELS.forEach((l) => (t.counts[l] = (t.counts[l] || 0) + (v.risk_counts[l] || 0)));
      Object.entries(v.object_counts).forEach(([k, n]) => (t.objects[k] = (t.objects[k] || 0) + n));
    });
    t.fps = t.fps / (VIDEOS.length || 1);
    return t;
  }, []);

  const objects = Object.entries(totals.objects).sort((a, b) => b[1] - a[1]).slice(0, 6);
  const maxObj = objects[0]?.[1] || 1;

  const incidents = VIDEOS
    .flatMap((v) => v.events.map((e) => ({ ...e, video: v.title, vid: v.id })))
    .sort((a, b) => LEVELS.indexOf(b.level) - LEVELS.indexOf(a.level));

  return (
    <>
      <h1 className="sw-h1">Detection Dashboard</h1>
      <p className="sw-sub">
        Combined results from every clip processed by the YOLO11 + Prolog pipeline. Distance and
        time-to-collision are monocular estimates, so treat them as relative, not surveyed.
      </p>

      <div className="sw-kpis">
        <Kpi label="Clips analysed" value={VIDEOS.length} Icon={Film} />
        <Kpi label="Frames" value={totals.frames.toLocaleString()} Icon={Film} />
        <Kpi label="Detections" value={totals.dets.toLocaleString()} Icon={ScanLine} />
        <Kpi label="Incidents saved" value={totals.incidents} accent="var(--critical)" Icon={ShieldAlert} />
        <Kpi label="Lowest TTC" value={totals.minTtc.toFixed(2)} unit="s" accent="var(--warning)" Icon={Timer} />
        <Kpi label="Avg processing" value={totals.fps.toFixed(1)} unit="fps" Icon={Gauge} />
      </div>

      <div className="sw-2col" style={{ marginBottom: 16 }}>
        <Panel title="Risk level distribution" right={<span className="sw-eyebrow">Prolog decisions</span>}>
          <RiskBar counts={totals.counts} />
          <div style={{ marginTop: 18 }}>
            <div className="sw-eyebrow" style={{ marginBottom: 8 }}>Objects detected</div>
            {objects.map(([name, n]) => (
              <div className="sw-row" key={name}>
                <div className="grow sw-truncate" style={{ textTransform: "capitalize" }}>{name}</div>
                <div className="sw-meter"><i style={{ width: `${(n / maxObj) * 100}%` }} /></div>
                <div className="num" style={{ width: 28, textAlign: "right" }}>{n}</div>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Highest-risk events">
          {incidents.slice(0, 5).map((e, i) => (
            <button key={i} className="sw-event" onClick={() => onOpen(e.vid)}>
              <div className="sw-chip" style={{ background: levelColor(e.level) }}>{e.t}</div>
              <div className="grow" style={{ minWidth: 0 }}>
                <div className="t sw-truncate">{e.label}</div>
                <div className="d sw-truncate">{e.video} · {e.distance_m} m · TTC {fmtTtc(e.ttc_s)}</div>
              </div>
              <ChevronRight size={15} style={{ color: "var(--muted)", flexShrink: 0, alignSelf: "center" }} />
            </button>
          ))}
        </Panel>
      </div>

      <Panel title="Per-clip results" right={<span className="sw-eyebrow">{VIDEOS.length} clips</span>}>
        {VIDEOS.map((v) => (
          <div className="sw-row" key={v.id}>
            <div className="grow" style={{ minWidth: 0 }}>
              <div className="sw-truncate" style={{ fontWeight: 500 }}>{v.title}</div>
              <div style={{ fontSize: 11.5, color: "var(--muted)" }}>
                {v.location} · {v.weather} · {v.duration}
              </div>
            </div>
            <div className="num" style={{ color: "var(--muted)", fontSize: 12 }}>
              {v.total_detections.toLocaleString()} det
            </div>
            <Tag level={v.overall_risk} />
            <button
              className="sw-tag"
              style={{ color: "var(--gold)", borderColor: "var(--gold-dim)" }}
              onClick={() => onOpen(v.id)}
            >
              Open
            </button>
          </div>
        ))}
      </Panel>
    </>
  );
}
