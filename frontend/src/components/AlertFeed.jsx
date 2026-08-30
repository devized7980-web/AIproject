import React from "react";
import { levelColor, fmtTtc } from "../ui.jsx";

// Compact list of hazard alerts. Each item accepts:
//   { level, label, distance_m, ttc_s, video_title?, video? }
export default function AlertFeed({ alerts = [], empty = "No recent alerts — all clear.", limit = 8 }) {
  const list = Array.isArray(alerts) ? alerts.slice(0, limit) : [];
  return (
    <div className="sw-livealerts">
      {list.length === 0 && <div className="sw-eyebrow" style={{ padding: "6px 0" }}>{empty}</div>}
      {list.map((a, i) => (
        <div className="sw-alertrow" data-lvl={a.level} key={a.id || i}>
          <i className="pulse" style={{ background: levelColor(a.level), color: levelColor(a.level) }} aria-hidden="true" />
          <div className="grow" style={{ minWidth: 0 }}>
            <div className="t sw-truncate" title={a.label}>{a.label}</div>
            <div className="d sw-truncate">
              {(a.video_title || a.video || "")} {a.distance_m != null ? `· ${a.distance_m} m` : ""} {a.ttc_s != null ? `· TTC ${fmtTtc(a.ttc_s)}` : ""}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}