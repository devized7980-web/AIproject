import React from "react";

// Single number highlight used across the dashboards (renders the .sw-kpi
// session so it participates in the shared KPI grid).
export default function MetricCard({ label, value, unit, accent, Icon, hint }) {
  return (
    <div className="sw-kpi">
      <div className="l">
        {Icon && <Icon size={12} style={{ verticalAlign: -1, marginRight: 5, color: accent || "var(--violet)" }} aria-hidden="true" />}
        {label}
      </div>
      <div className="v" style={accent ? { color: accent } : undefined}>
        {value}
        {unit && <span className="sw-unit">{unit}</span>}
      </div>
      {hint && <small className="sw-kpi-hint">{hint}</small>}
    </div>
  );
}