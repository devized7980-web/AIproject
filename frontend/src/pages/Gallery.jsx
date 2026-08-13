import React from "react";
import { Play, Activity, Save, Clock } from "lucide-react";
import { VIDEOS, VIDEO_BASE } from "../data.js";
import { Tag } from "../ui.jsx";

export default function Gallery({ onOpen }) {
  const empty = Math.max(0, 8 - VIDEOS.length);

  return (
    <>
      <h1 className="sw-h1">Video Gallery</h1>
      <p className="sw-sub">
        Every dashcam clip in the benchmark set. Open one to step through its detections frame by frame.
      </p>

      <div className="sw-grid">
        {VIDEOS.map((v) => (
          <button className="sw-card" key={v.id} onClick={() => onOpen(v.id)}>
            <div className="sw-thumb">
              {v.thumb && <img src={VIDEO_BASE + v.thumb} alt={v.title} loading="lazy" />}
              <div className="stamp"><Tag level={v.overall_risk} /></div>
              <div className="dur"><Clock size={10} style={{ verticalAlign: -1, marginRight: 3 }} />{v.duration}</div>
              <div className="play"><span><Play size={20} /></span></div>
            </div>
            <div className="sw-card-body">
              <div className="h">{v.title}</div>
              <div className="m">{v.location} · {v.weather}</div>
              <div className="meta">
                <span><Activity size={11} style={{ verticalAlign: -1, marginRight: 3, color: "var(--gold)" }} /><b>{v.events.length}</b> events</span>
                <span><Save size={11} style={{ verticalAlign: -1, marginRight: 3, color: "var(--gold)" }} /><b>{v.incidents}</b> saved</span>
                <span>TTC <b>{v.minimum_ttc_s}s</b></span>
              </div>
            </div>
          </button>
        ))}

        {Array.from({ length: empty }).map((_, i) => (
          <div className="sw-slot" key={i}>Slot {VIDEOS.length + i + 1} — empty</div>
        ))}
      </div>
    </>
  );
}
