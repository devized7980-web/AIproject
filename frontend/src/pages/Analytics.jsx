import React, { useEffect, useState } from "react";
import { Film, ScanLine, ShieldAlert, Timer, Gauge } from "lucide-react";
import { getAnalytics } from "../api.js";
import { Panel, Kpi, ScoreGauge, HBars, levelColor, levelHue } from "../ui.jsx";

const W = 620, H = 210, PAD = { l: 34, r: 12, t: 16, b: 28 };

function LineChart({ labels, values, color = "var(--gold)" }) {
  labels = Array.isArray(labels) ? labels : [];
  values = Array.isArray(values) ? values.map((v) => Number.isFinite(+v) ? +v : 0) : [];
  if (!values.length) return <div className="sw-empty--panel">No daily safety data available.</div>;
  const max = Math.max(...values, 1) * 1.12;
  const min = Math.min(...values, 0);
  const x = (i) => PAD.l + (i / Math.max(1, values.length - 1)) * (W - PAD.l - PAD.r);
  const y = (v) => H - PAD.b - ((v - min) / (max - min)) * (H - PAD.t - PAD.b);
  const pts = values.map((v, i) => `${x(i)},${y(v)}`).join(" ");
  const area = `${PAD.l},${H - PAD.b} ${pts} ${x(values.length - 1)},${H - PAD.b}`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="sw-chart">
      {[0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={PAD.l} x2={W - PAD.r} y1={y(min + (max - min) * t)} y2={y(min + (max - min) * t)} stroke="rgba(148,163,184,0.12)" strokeDasharray="3 4" />
        </g>
      ))}
      <polygon points={area} fill={color} opacity="0.12" />
      <polyline points={pts} fill="none" stroke={color} strokeWidth="2.4" strokeLinejoin="round" strokeLinecap="round" />
      {values.map((v, i) => (
        <g key={i}>
          <circle cx={x(i)} cy={y(v)} r="3.4" fill={color} stroke="var(--bg)" strokeWidth="1.5" />
          <text x={x(i)} y={H - 8} textAnchor="middle" fontSize="10" fill="var(--muted)">{labels[i]}</text>
        </g>
      ))}
    </svg>
  );
}

function GroupedBars({ labels, series }) {
  labels = Array.isArray(labels) ? labels : [];
  series = Array.isArray(series) ? series.map((s) => ({ ...s, values: Array.isArray(s.values) ? s.values.map((v) => Number.isFinite(+v) ? +v : 0) : [] })) : [];
  if (!labels.length || !series.length) return <div className="sw-empty--panel">No hazard trend data available.</div>;
  const groups = labels.length;
  const bw = 34, gap = 6;
  const width = Math.min(W - PAD.l - PAD.r, groups * (bw * series.length + gap * (series.length - 1) + 24));
  const x0 = PAD.l;
  const max = Math.max(...series.flatMap((s) => s.values), 1) * 1.12;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="sw-chart">
      <text x={PAD.l} y={H - PAD.b + 16} fontSize="10" fill="var(--faint)">clip</text>
      {labels.map((l, g) => {
        let xx = x0 + g * ((width - PAD.l) / groups);
        return (
          <g key={l}>
            {series.map((s, i) => {
              const h = (s.values[g] / max) * (H - PAD.t - PAD.b);
              const bx = xx + i * (bw + gap);
              return (
                <g key={s.name}>
                  <rect x={bx} y={H - PAD.b - h} width={bw} height={h} rx="3" fill={s.color} opacity="0.9" />
                  {s.values[g] > 0 && <text x={bx + bw / 2} y={H - PAD.b - h - 5} textAnchor="middle" fontSize="9" fill={s.color}>{s.values[g]}</text>}
                </g>
              );
            })}
            <text x={xx + (bw * series.length + gap * (series.length - 1)) / 2} y={H - 8} textAnchor="middle" fontSize="9.5" fill="var(--muted)">{l}</text>
            xx += (width - PAD.l) / groups;
          </g>
        );
      })}
      <g>
        {series.map((s, i) => (
          <g key={s.name} transform={`translate(${x0 + 6 + i * 110}, 8)`}>
            <rect width="10" height="10" rx="2" fill={s.color} />
            <text x={16} y={9.5} fontSize="10" fill="var(--muted)">{s.name}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}

function DailyStack({ daily }) {
  daily = Array.isArray(daily) ? daily : [];
  if (!daily.length) return <div className="sw-empty--panel">No daily risk data available.</div>;
  const groups = daily.length;
  const bw = 44;
  const max = Math.max(...daily.map((d) => Object.values(d.counts || {}).reduce((s, n) => s + (+n || 0), 0)), 1) * 1.1;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="sw-chart">
      {daily.map((d, i) => {
        const x = PAD.l + i * (bw + 26);
        let y = H - PAD.b;
        return (
          <g key={d.day}>
            {["SAFE", "CAUTION", "WARNING", "CRITICAL"].map((l) => {
              const h = (d.counts?.[l] || 0) / max * (H - PAD.t - PAD.b);
              y -= h;
              return <rect key={l} x={x} y={y} width={bw} height={h} fill={levelHue(l)} opacity="0.9" />;
            })}
            <text x={x + bw / 2} y={H - 8} textAnchor="middle" fontSize="9.5" fill="var(--muted)">{d.day}</text>
          </g>
        );
      })}
    </svg>
  );
}

export default function Analytics({ videos }) {
  const [data, setData] = useState(null);
  const videoCount = Array.isArray(videos) ? videos.length : 0;

  useEffect(() => {
    getAnalytics().then(setData);
  }, []);

  if (!data) return <div className="sw-eyebrow">Loading analytics…</div>;

  const t = data.totals || {};
  const riskCounts = t.risk_counts || {};
  const daily = Array.isArray(data.daily) ? data.daily : [];
  const locations = Array.isArray(data.locations) ? data.locations : [];
  const trend = data.hazard_trend || {};
  const kpis = [
    { label: "Clips analysed", value: t.clips ?? videoCount, Icon: Film },
    { label: "Frames", value: (t.frames || 0).toLocaleString(), Icon: Film },
    { label: "Detections", value: (t.detections || 0).toLocaleString(), Icon: ScanLine },
    { label: "Incidents saved", value: t.incidents || 0, accent: "var(--critical)", Icon: ShieldAlert },
    { label: "Lowest TTC", value: Number.isFinite(+t.min_ttc) ? (+t.min_ttc).toFixed(2) : "—", unit: "s", accent: "var(--warning)", Icon: Timer },
    { label: "Avg processing", value: t.avg_fps ?? "—", unit: "fps", Icon: Gauge },
  ];
  const ba = data.before_after || { before: {}, after: {}, reduction_pct: {} };
  const baRows = Object.entries(ba.before || {}).filter(([k, v]) => Number.isFinite(+v) && +v > 0 && Number.isFinite(+(ba.after || {})[k]));
  const riskTotal = Object.values(riskCounts).reduce((s, n) => s + (+n || 0), 0) || 1;

  return (
    <>
      <h1 className="sw-h1">Safety Analytics</h1>
      <p className="sw-sub">
        Historical performance across every processed clip: risk distribution, trends, the most
        dangerous stretches, and the modelled impact of running the detection system.
      </p>

      <div className="sw-kpis">
        {kpis.map((k) => <Kpi key={k.label} {...k} />)}
      </div>

      <div className="sw-2col" style={{ gridTemplateColumns: "300px minmax(0,1fr)" }}>
        <div className="sw-stack">
          <Panel title="Overall safety score">
             <ScoreGauge score={Number.isFinite(+data.safety_score) ? +data.safety_score : 0} />
            <div style={{ marginTop: 16 }}>
              <div className="sw-eyebrow" style={{ marginBottom: 8 }}>Risk distribution (all runs)</div>
              <div className="sw-riskbar">
                {["SAFE", "CAUTION", "WARNING", "CRITICAL"].map((l) => (
                  <span key={l} style={{ width: `${(riskCounts[l] || 0) / riskTotal * 100}%`, background: levelColor(l) }} />
                ))}
              </div>
              <div className="sw-legend">
                {["SAFE", "CAUTION", "WARNING", "CRITICAL"].map((l) => (
                  <div key={l}><i className="sw-dot" style={{ background: levelColor(l) }} />{l} <b>{riskCounts[l] || 0}</b></div>
                ))}
              </div>
            </div>
          </Panel>

          <Panel title="Most dangerous locations">
            <HBars
              color="var(--critical)"
               data={locations.map((l) => ({
                 label: String(l.video || l.name || "Unknown route").split("—")[0].trim(),
                 value: Number.isFinite(+l.danger) ? +l.danger : 0,
                color: levelHue(l.danger >= 60 ? "CRITICAL" : l.danger >= 35 ? "WARNING" : "CAUTION"),
              }))}
            />
             {locations.length === 0 && <div className="sw-eyebrow">No location data available.</div>}
             {locations.map((l, i) => (
               <div className="sw-row" key={`${l.name || "route"}-${i}`}>
                 <div className="grow sw-truncate">{l.video || l.name || "Unknown route"}</div>
                <div className="num" style={{ color: levelColor(l.safety_score >= 80 ? "SAFE" : l.safety_score >= 55 ? "CAUTION" : "CRITICAL") }}>
                  {l.critical} critical
                </div>
              </div>
            ))}
          </Panel>
        </div>

        <div className="sw-stack">
          <Panel title="Safety score by day" right={<span className="sw-eyebrow">rolling 7-day</span>}>
             <LineChart labels={daily.map((d) => d.day || "")} values={daily.map((d) => d.safety_score)} color="var(--safe)" />
          </Panel>

          <Panel title="Daily risk mix" right={<span className="sw-eyebrow">stacked detections</span>}>
             <DailyStack daily={daily} />
          </Panel>

          <div className="sw-2col" style={{ gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)" }}>
            <Panel title="Hazards by clip">
              <GroupedBars
                labels={trend.labels}
                series={[
                  { name: "Vehicles", color: "#94A3B8", values: trend.vehicles },
                  { name: "Pedestrians", color: levelHue("CAUTION"), values: trend.pedestrians },
                  { name: "Potholes", color: levelHue("CRITICAL"), values: trend.potholes },
                ]}
              />
            </Panel>
            <Panel title="Before vs after detection system">
              <div className="sw-ba">
                {baRows.map(([k, v]) => (
                  <div className="ba-row" key={k}>
                    <span className="l" style={{ textTransform: "capitalize" }}>{k.replace(/_/g, " ")}</span>
                    <div className="bars">
                         <span className="before" style={{ width: `${Math.min(100, (ba.after[k] / v) * 100)}%` }}>
                        <i style={{ width: "100%" }} />
                        <em>{v}{k === "response_time_s" ? "s" : "%"}</em>
                      </span>
                    </div>
                    <div className="bars after">
                       <span style={{ width: `${Math.min(100, (ba.after[k] / v) * 100)}%` }}><i />
                        <em>{ba.after[k]}{k === "response_time_s" ? "s" : "%"}</em>
                      </span>
                    </div>
                    <b className="red" style={{ color: "var(--safe)" }}>−{ba.reduction_pct[k]}%</b>
                  </div>
                ))}
              </div>
              <div className="sw-note" style={{ marginTop: 10 }}>
                Before = unassisted driving, after = with Safeway alerts. Figures are a modelled
                projection for this benchmark, not a measured field trial.
              </div>
            </Panel>
          </div>
        </div>
      </div>
    </>
  );
}
