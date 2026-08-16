import React, { useState } from "react";
import { Phone, Copy, Check, AlertTriangle, MapPin, Clock, Activity, Download } from "lucide-react";
import { VIDEOS, LEVELS, EMERGENCY, QUICK_ACTIONS, CRASH_STEPS } from "../data.js";
import { Panel, Tag, levelColor, fmtTtc } from "../ui.jsx";

const LEVEL_PRIORITY = { SAFE: 0, CAUTION: 1, WARNING: 2, CRITICAL: 3 };

// The single most severe event across every processed clip.
// Highest risk level wins; ties go to the closest (lowest TTC) then highest confidence.
const worst = VIDEOS
  .flatMap((v) => v.events.map((e) => ({ ...e, video: v.title })))
  .sort((a, b) => {
    const lvl = LEVEL_PRIORITY[b.level] - LEVEL_PRIORITY[a.level];
    if (lvl !== 0) return lvl;
    const ttcA = a.ttc_s ?? Infinity;
    const ttcB = b.ttc_s ?? Infinity;
    if (ttcA !== ttcB) return ttcA - ttcB;
    return b.confidence - a.confidence;
  })[0];

function buildText(kind) {
  switch (kind) {
    case "location":
      return `Position: 16.8409, 96.1735 — ${worst.video.split(".")[0]}, nearest event at ${worst.t}.`;
    case "snapshot":
      return [
        `INCIDENT SNAPSHOT`,
        `Video: ${worst.video}`,
        `Time: ${worst.t}`,
        `Event: ${worst.label}`,
        `Risk: ${worst.level} — ${worst.action}`,
        `Distance: ${worst.distance_m} m`,
        `TTC: ${fmtTtc(worst.ttc_s)}`,
        `Confidence: ${Math.round(worst.confidence * 100)}%`,
      ].join("\n");
    case "hazard":
      return [
        `HAZARD REPORT`,
        `Video: ${worst.video}`,
        `Event: ${worst.label}`,
        `Object: ${worst.object}`,
        `Distance: ${worst.distance_m} m`,
        `Risk: ${worst.level}`,
      ].join("\n");
    case "log":
      return `${worst.level} at ${worst.t}: ${worst.label}. ${worst.distance_m} m, TTC ${fmtTtc(worst.ttc_s)}, ${Math.round(worst.confidence * 100)}% confidence. Advice: ${worst.action}.`;
    default:
      return "";
  }
}

export default function Emergency() {
  const [done, setDone] = useState({});

  const run = async (id) => {
    const text = buildText(id);
    let ok = false;
    try {
      await navigator.clipboard.writeText(text);
      ok = true;
    } catch {
      // Clipboard blocked (non-HTTPS). Fall back to a prompt so the text is still usable.
      window.prompt("Copy this to your clipboard:", text);
      ok = true;
    }
    setDone((d) => ({ ...d, [id]: ok }));
    setTimeout(() => setDone((d) => ({ ...d, [id]: null })), 2200);
  };

  return (
    <>
      <h1 className="sw-h1">Emergency Contact & Quick Action</h1>
      <p className="sw-sub">
        Call the number that matches the worst problem. Everything below works with one hand, because the
        driver using it has just stopped at the roadside.
      </p>

      <div className="sw-grid" style={{ marginBottom: 16 }}>
        {EMERGENCY.map((c) => (
          <a className="sw-call" key={c.number} href={`tel:${c.number}`}>
            <div className="sw-callicon"><Phone size={19} /></div>
            <div style={{ minWidth: 0 }}>
              <div className="n">{c.number}</div>
              <div className="name">{c.name}</div>
              <div className="note">{c.note}</div>
            </div>
          </a>
        ))}
      </div>

      <div className="sw-2col-even">
        <Panel title="Quick actions">
          <div style={{ display: "grid", gap: 10 }}>
            {QUICK_ACTIONS.map((a) => (
              <button className="sw-action" key={a.id} onClick={() => run(a.id)}>
                <h3>
                  {a.title}
                  {done[a.id] === true
                    ? <span className="sw-done"><Check size={12} /> COPIED</span>
                    : done[a.id] === false
                      ? <span className="sw-done" style={{ color: "var(--critical)" }}><AlertTriangle size={12} /> FAILED</span>
                      : <Copy size={13} style={{ color: "var(--muted)" }} />}
                </h3>
                <p>{a.body}</p>
              </button>
            ))}
          </div>
        </Panel>

        <div className="sw-stack">
          <Panel title="Last flagged event" right={<Tag level={worst.level} />}>
            <div style={{ fontFamily: "var(--data)", fontSize: 22, fontWeight: 600, color: levelColor(worst.level) }}>
              {worst.action}
            </div>
            <div style={{ color: "var(--muted)", fontSize: 12.5, marginTop: 6 }}>
              {worst.video} at {worst.t} — {worst.label}
            </div>
            <div className="sw-row" style={{ marginTop: 10 }}>
              <MapPin size={14} style={{ color: "var(--muted)" }} />
              <div className="grow">Distance</div>
              <div className="num">{worst.distance_m} m</div>
            </div>
            <div className="sw-row">
              <Clock size={14} style={{ color: "var(--muted)" }} />
              <div className="grow">Time to collision</div>
              <div className="num">{fmtTtc(worst.ttc_s)}</div>
            </div>
            <div className="sw-row">
              <Activity size={14} style={{ color: "var(--muted)" }} />
              <div className="grow">Confidence</div>
              <div className="num">{Math.round(worst.confidence * 100)}%</div>
            </div>
          </Panel>

          <Panel title="After a crash">
            <ol className="sw-steps">
              {CRASH_STEPS.map((s, i) => <li key={i}>{s}</li>)}
            </ol>
          </Panel>
        </div>
      </div>
    </>
  );
}