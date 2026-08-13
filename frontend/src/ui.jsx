import React from "react";
import { LEVELS } from "./data.js";

export const levelColor = (l) =>
  ({
    SAFE: "var(--safe)",
    CAUTION: "var(--caution)",
    WARNING: "var(--warning)",
    CRITICAL: "var(--critical)",
  }[l] || "var(--muted)");

export const fmtTtc = (ttc) => (ttc == null ? "—" : `${ttc} s`);

export const Tag = ({ level }) => (
  <span className="sw-tag" style={{ color: levelColor(level), borderColor: levelColor(level) }}>
    {level}
  </span>
);

export const Panel = ({ title, right, children, pad = true }) => (
  <section className="sw-panel">
    {title && (
      <header className="sw-panel-h">
        <h2>{title}</h2>
        {right}
      </header>
    )}
    {pad ? <div className="sw-panel-b">{children}</div> : children}
  </section>
);

export const Kpi = ({ label, value, unit, accent, Icon }) => (
  <div className="sw-kpi">
    <div className="l">
      {Icon && <Icon size={12} style={{ verticalAlign: -1, marginRight: 5, color: accent || "var(--gold)" }} />}
      {label}
    </div>
    <div className="v" style={accent ? { color: accent } : undefined}>
      {value}
      {unit && <span style={{ fontSize: 13, color: "var(--muted)", marginLeft: 4 }}>{unit}</span>}
    </div>
  </div>
);

export const RiskBar = ({ counts, hidePercent }) => {
  const total = LEVELS.reduce((s, l) => s + (counts[l] || 0), 0) || 1;
  return (
    <>
      <div className="sw-riskbar" role="img" aria-label="Risk level distribution">
        {LEVELS.map((l) => (
          <span key={l} style={{ width: `${((counts[l] || 0) / total) * 100}%`, background: levelColor(l) }} />
        ))}
      </div>
      <div className="sw-legend">
        {LEVELS.map((l) => (
          <div key={l}>
            <i className="sw-dot" style={{ background: levelColor(l) }} />
            {l}
            {!hidePercent && <b>{(((counts[l] || 0) / total) * 100).toFixed(1)}%</b>}
          </div>
        ))}
      </div>
    </>
  );
};
