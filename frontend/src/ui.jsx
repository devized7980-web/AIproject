import React from "react";
import { LEVELS } from "./data.js";

export const levelColor = (l) =>
  ({
    SAFE: "var(--safe)",
    CAUTION: "var(--caution)",
    WARNING: "var(--warning)",
    CRITICAL: "var(--critical)",
  }[l] || "var(--muted)");

export const Tag = ({ level }) => (
  <span className="sw-tag" data-level={level} style={{ color: levelColor(level), borderColor: levelColor(level) }}>
    {level}
  </span>
);

export const fmtTtc = (ttc) => (ttc == null ? "—" : `${ttc} s`);

const themeVar = (name) => {
  try {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "currentColor";
  } catch {
    return "currentColor";
  }
};

export const levelHue = (l) =>
  ({
    SAFE: themeVar("--safe"),
    CAUTION: themeVar("--caution"),
    WARNING: themeVar("--warning"),
    CRITICAL: themeVar("--critical"),
  }[l] || themeVar("--muted"));

export const SparkBars = ({ values, color = "var(--gold)", height = 40, title }) => (
  <div className="sw-spark" role="img" aria-label={title}>
    {values.map((v, i) => (
      <i key={i} style={{ height: `${Math.max(6, (v / (Math.max(...values, 1) || 1)) * 100)}%`, background: color }} />
    ))}
  </div>
);

export const ScoreGauge = ({ score, label }) => {
  const pct = Math.max(0, Math.min(100, score));
  const c = pct >= 80 ? "var(--safe)" : pct >= 55 ? "var(--caution)" : pct >= 35 ? "var(--warning)" : "var(--critical)";
  return (
    <div className="sw-score">
      <div className="ring" style={{ "--p": pct, "--c": c }}>
        <div className="hole">
          <b>{Math.round(pct)}</b>
          <span>/100</span>
        </div>
      </div>
      <div className="meta">
        <div style={{ color: c, fontWeight: 700 }}>{label || "Safety score"}</div>
        <small>{pct >= 80 ? "Excellent" : pct >= 55 ? "Moderate" : pct >= 35 ? "Elevated risk" : "Critical risk"}</small>
      </div>
    </div>
  );
};

export const HBars = ({ data, color }) => {
  const max = Math.max(...data.map((d) => d.value), 1);
  return (
    <div className="sw-hbars">
      {data.map((d, i) => (
        <div className="row" key={i}>
          <span className="l">{d.label}</span>
          <span className="t"><i style={{ width: `${(d.value / max) * 100}%`, background: d.color || color }} /></span>
          <b>{d.value}</b>
        </div>
      ))}
    </div>
  );
};

export const Toggle = ({ on, onChange, label, desc }) => (
  <button className={`sw-toggle ${on ? "on" : ""}`} onClick={() => onChange(!on)} role="switch" aria-checked={on}>
    <span className="k"><i /></span>
    {label && (
      <span className="txt">
        <b>{label}</b>
        {desc && <small>{desc}</small>}
      </span>
    )}
  </button>
);

export const Range = ({ value, min, max, step = 1, onChange, label, unit, marks }) => (
  <label className="sw-range">
    <span className="l"><b>{label}</b><em>{value}{unit || ""}</em></span>
    <input
      type="range" min={min} max={max} step={step} value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      style={{ "--val": `${((value - min) / (max - min)) * 100}%` }}
    />
    {marks && (
      <span className="m">
        {marks.map((mk) => <i key={mk}>{mk}</i>)}
      </span>
    )}
  </label>
);

export const Empty = ({ title, body }) => (
  <div className="sw-empty--panel">
    <div style={{ fontSize: 13, color: "var(--text)" }}>{title}</div>
    {body && <div style={{ fontSize: 12, maxWidth: "46ch", color: "var(--muted)" }}>{body}</div>}
  </div>
);

export const LevelBar = ({ counts }) => (
  <>
    <div className="sw-riskbar" role="img" aria-label="Risk level distribution">
      {LEVELS.map((l) => (
        <span key={l} style={{ width: `${((counts[l] || 0) / (Object.values(counts).reduce((s, n) => s + n, 0) || 1)) * 100}%`, background: levelColor(l) }} />
      ))}
    </div>
    <div className="sw-legend">
      {LEVELS.map((l) => (
        <div key={l}>
          <i className="sw-dot" style={{ background: levelColor(l) }} />
          {l} <b>{counts[l] || 0}</b>
        </div>
      ))}
    </div>
  </>
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
