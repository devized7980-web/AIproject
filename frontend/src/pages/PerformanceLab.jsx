import React, { useEffect, useState } from "react";
import { Gauge, Timer, Cpu, Activity, ScanLine, RefreshCw } from "lucide-react";
import { getPerformance } from "../api.js";
import { Panel, HBars } from "../ui.jsx";
import { useSettingsCtx } from "../App.jsx";

const SCEN_COLOR = { rain: "#7DD3FC", night: "#818CF8", traffic: "#F5C33B", clear: "#34D399" };

export default function PerformanceLab({ videos }) {
  const [data, setData] = useState(null);
  const [settings] = useSettingsCtx();

  const load = () => { setData(null); getPerformance().then(setData); };
  useEffect(() => { load(); }, []);

  if (!data) return <div className="sw-eyebrow">Loading performance data…</div>;

  const bench = data.benchmark;
  const models = bench
    ? Object.values(bench)
    : [{ key: "yolo11n", name: "YOLO11n", loaded: true, fps: 12.4, latency_ms: 80.6, classes: 80 },
       { key: "best", name: "best.pt", loaded: true, fps: 10.1, latency_ms: 99.0, classes: 5 },
       { key: "yolo26n", name: "YOLO26n", loaded: true, fps: 9.3, latency_ms: 107.5, classes: 80 }];

  const srcBars = data.sources.map((s) => ({
    label: s.source,
    value: s.detections,
  }));

  return (
    <>
      <h1 className="sw-h1">AI Model Performance Lab</h1>
      <p className="sw-sub">
        The engineering side: real throughput measured on this machine plus detection reliability
        derived from the recorded log. Scenario multipliers show how weather would change recall.
      </p>

      <div className="sw-cc-head" style={{ marginBottom: 18 }}>
        <span className="sw-eyebrow">
          {data.benchmark_running ? "Benchmark running in background…" : data.benchmark ? "Benchmark: measured on this machine" : "Benchmark: static estimates (backend offline)"}
        </span>
        <button className="sw-linkbtn" onClick={load}><RefreshCw size={12} /> Re-run</button>
      </div>

      <div className="sw-kpis">
        {models.map((m) => (
          <div className="sw-kpi" key={m.name || m.key}>
            <div className="l"><Cpu size={12} style={{ verticalAlign: -1, marginRight: 5, color: "var(--gold)" }} />{m.name}</div>
            {m.loaded === false
              ? <div className="v" style={{ fontSize: 14, color: "var(--critical)" }}>not loaded</div>
              : <div className="v" style={{ fontSize: 22 }}>{m.fps}<span style={{ fontSize: 12, color: "var(--muted)", marginLeft: 4 }}>fps</span></div>}
            <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>
              {m.loaded === false ? m.error || "weights missing" : `${m.latency_ms} ms · ${m.classes} classes`}
            </div>
          </div>
        ))}
      </div>

      <div className="sw-2col" style={{ gridTemplateColumns: "minmax(0,1.2fr) minmax(0,1fr)" }}>
        <Panel title="Per-clip pipeline metrics" right={<span className="sw-eyebrow">from the run log</span>}>
          <div className="sw-table">
            <div className="th">
              <span>Clip</span><span>FPS</span><span>Latency</span><span>Dets</span><span>FP</span><span>Pothole recall</span><span>Repeatability</span>
            </div>
            {data.per_video.map((v, i) => (
              <div className="tr" key={i}>
                <span className="clip sw-truncate">{v.video.split("—")[0]}</span>
                <span>{v.fps}</span>
                <span>{v.latency_ms} ms</span>
                <span>{v.detections.toLocaleString()}</span>
                <span style={{ color: v.false_positives ? "var(--critical)" : "var(--muted)" }}>{v.false_positives}</span>
                <span>{v.pothole_recall}%</span>
                <span>{v.repeatability}%</span>
              </div>
            ))}
          </div>
          <div className="sw-eyebrow" style={{ marginTop: 10 }}>
            FP = non-road classes the model also produced (giraffe, potted plant…). Repeatability =
            detections with a temporal neighbour, a proxy for tracking reliability.
          </div>
        </Panel>

        <div className="sw-stack">
          <Panel title="Detector contribution (real log)" right={<ScanLine size={13} style={{ color: "var(--gold)" }} />}>
            <HBars color="var(--gold)" data={srcBars} />
            <div className="sw-engine" style={{ marginTop: 10 }}>
              {data.sources.map((s) => (
                <div className="row" key={s.source}>
                  <span>{s.source}</span>
                  <b className="sw-truncate">{s.detections.toLocaleString()} dets · {s.unique_objects} classes · avg conf {s.avg_confidence}</b>
                </div>
              ))}
            </div>
          </Panel>

          <Panel title="Scenario multipliers" right={<span className="sw-eyebrow">relative to clear</span>}>
            <div className="sw-scen">
              {data.scenarios.map((s) => (
                <div className="sc" key={s.key}>
                  <span className="l"><i style={{ background: SCEN_COLOR[s.key] }} />{s.label}</span>
                  <div className="bars">
                    <span title="fps"><i style={{ width: `${Math.min(100, s.fps * 100)}%`, background: SCEN_COLOR[s.key] }} /><em>FPS</em></span>
                    <span title="recall"><i style={{ width: `${Math.min(100, s.recall * 100)}%`, background: SCEN_COLOR[s.key] }} /><em>Recall</em></span>
                    <span title="false positives"><i style={{ width: `${Math.min(100, s.fp * 62)}%`, background: "var(--critical)" }} /><em>FPs</em></span>
                  </div>
                </div>
              ))}
            </div>
          </Panel>
        </div>
      </div>
    </>
  );
}