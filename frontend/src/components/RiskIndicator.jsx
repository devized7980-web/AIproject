import React from "react";
import { levelColor } from "../ui.jsx";

// Large current-risk readout. level: SAFE | CAUTION | WARNING | CRITICAL.
export default function RiskIndicator({ level = "SAFE", action, className = "" }) {
  return (
    <div className={`sw-bigstate rs-risk ${className}`} style={{ "--c": levelColor(level) }} aria-live="polite">
      <div className="lvl">{level}</div>
      <div className="act">{action || "—"}</div>
    </div>
  );
}