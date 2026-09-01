import React, { useEffect, useState } from "react";
import { SlidersHorizontal, Zap, Droplets, Eye, Car, ArrowRight } from "lucide-react";
import { simulate } from "../api.js";
import { Panel, ScoreGauge, Range, Tag, levelColor, fmtTtc } from "../ui.jsx";

const OBJECTS = ["pothole", "person", "car", "truck", "bus", "bicycle", "alligator cracking", "traffic light"];

const PRESETS = [
  { label: "School zone", speed: 30, distance: 14, wetness: 0.3, visibility: 0.9, object: "person" },
  { label: "City 35 km/h", speed: 35, distance: 20, wetness: 0.5, visibility: 0.8, object: "pothole" },
  { label: "City 70 km/h", speed: 70, distance: 20, wetness: 0.5, visibility: 0.8, object: "pothole" },
  { label: "Wet night", speed: 60, distance: 15, wetness: 0.9, visibility: 0.45, object: "pothole" },
  { label: "Heavy fog", speed: 50, distance: 18, wetness: 0.7, visibility: 0.3, object: "car" },
];

export default function WhatIfSimulator({ videos }) {
  const [params, setParams] = useState({
    object: "pothole", speed_kmh: 35, distance_m: 20,
    wetness: 0.5, visibility: 0.8, lane_position: 1.0,
    confidence: 0.7, box_height_ratio: 0.2,
  });
  const [result, setResult] = useState(null);
  const [thinking, setThinking] = useState(false);

  const set = (patch) => setParams((p) => ({ ...p, ...patch }));

  useEffect(() => {
    let alive = true;
    setThinking(true);
    const t = setTimeout(async () => {
      const r = await simulate(params);
      if (alive) { setResult(r); setThinking(false); }
    }, 250);
    return () => { alive = false; clearTimeout(t); };
  }, [params]);

  const speedRange = [10, 120];
  const distRange = [2, 60];

  return (
    <>
      <h1 className="sw-h1">What-If Safety Simulator</h1>
      <p className="sw-sub">
        Change the driving variables and watch the Prolog risk engine re-decide instantly.
        Same pothole, different speed: 35 km/h gives a warning, 70 km/h is critical.
      </p>

      <div className="sw-2col" style={{ gridTemplateColumns: "380px minmax(0,1fr)" }}>
        <Panel title="Simulation inputs" right={<SlidersHorizontal size={14} style={{ color: "var(--gold)" }} />}>
          <div className="sw-presets">
            {PRESETS.map((p) => (
              <button key={p.label} onClick={() => set(p)}>{p.label}</button>
            ))}
          </div>

          <div className="sw-paramfld">
            <label className="sw-eyebrow" style={{ marginBottom: 6, display: "block" }}>Hazard object</label>
            <select value={params.object} onChange={(e) => set({ object: e.target.value })}>
              {OBJECTS.map((o) => <option key={o}>{o}</option>)}
            </select>
          </div>

          <Range label="Vehicle speed" unit=" km/h" min={speedRange[0]} max={speedRange[1]} step={5}
            value={params.speed_kmh} onChange={(v) => set({ speed_kmh: v })}
            marks={["35", "70", "120"]} />

          <Range label="Hazard distance" unit=" m" min={distRange[0]} max={distRange[1]} step={1}
            value={params.distance_m} onChange={(v) => set({ distance_m: v })}
            marks={["3", "7", "60"]} />

          <Range label="Road wetness" unit="%" min={0} max={100} step={5}
            value={Math.round(params.wetness * 100)} onChange={(v) => set({ wetness: v / 100 })}
            marks={["dry", "wet", "flooded"]} />

          <Range label="Visibility" unit="%" min={10} max={100} step={5}
            value={Math.round(params.visibility * 100)} onChange={(v) => set({ visibility: v / 100 })}
            marks={["fog", "rain", "clear"]} />

          <Range label="Lane position" unit="%" min={0} max={100} step={5}
            value={Math.round(params.lane_position * 100)} onChange={(v) => set({ lane_position: v / 100 })}
            marks={["outside", "edge", "centre"]} />

          <Range label="Detection confidence" unit="%" min={10} max={99} step={1}
            value={Math.round(params.confidence * 100)} onChange={(v) => set({ confidence: v / 100 })}
            marks={["0.30 min", "0.90"]} />
        </Panel>

        <div className="sw-stack">
          <Panel title="Live risk evaluation" right={thinking ? <span className="sw-eyebrow">evaluating…</span> : <Tag level={result?.level || "SAFE"} />}>
            <div className="sw-2col" style={{ gridTemplateColumns: "220px minmax(0,1fr)" }}>
              <ScoreGauge score={result?.risk_score ?? 0} label={`Risk ${result?.risk_score ?? "—"} / 100`} />
              <div>
                <div className="sw-bigstate" style={{ "--c": levelColor(result?.level || "SAFE"), marginBottom: 12 }}>
                  <div className="lvl">{result?.level || "—"}</div>
                  <div className="act">{result?.trace?.advice || "Adjust the inputs…"}</div>
                </div>
                <div className="sw-engine">
                  <div className="row"><span>TTC</span><b>{fmtTtc(result?.ttc_s)}</b></div>
                  <div className="row"><span>Fired rule</span><b className="sw-truncate">{result?.trace?.rule_label}</b></div>
                  <div className="row"><span>Priority</span><b>P{result?.trace?.priority}</b></div>
                  <div className="row"><span>Engine</span><b className="sw-truncate">{result?.trace?.engine}</b></div>
                </div>
              </div>
            </div>

            <div className="sw-speedline">
              <div className="sw-eyebrow" style={{ marginBottom: 10 }}>Speed → risk (same pothole at {params.distance_m} m)</div>
              {[35, 70].map((s) => {
                const ttc = params.distance_m / (s / 3.6);
                const lvl = ttc <= 1.5 ? "CRITICAL" : ttc <= 3 ? "WARNING" : ttc <= 5 ? "CAUTION" : "SAFE";
                return (
                  <div className="row" key={s}>
                    <span><Car size={13} /> {s} km/h</span>
                    <span className="grow" style={{ textAlign: "right", color: "var(--muted)" }}>TTC {ttc.toFixed(1)} s</span>
                    <ArrowRight size={13} style={{ color: "var(--muted)" }} />
                    <b style={{ color: levelColor(lvl), width: 86, textAlign: "right" }}>{lvl}</b>
                  </div>
                );
              })}
            </div>
          </Panel>

          <Panel title="Why does it change?" right={<Zap size={14} style={{ color: "var(--gold)" }} />}>
            <div className="sw-why">
              <div className="row"><Droplets size={14} style={{ color: "var(--caution)" }} /><span>Wetness ×{((1 + 0.35 * params.wetness)).toFixed(2)}</span>
                <b>amplifies risk</b></div>
              <div className="row"><Eye size={14} style={{ color: "var(--warning)" }} /><span>Visibility ÷{Math.max(0.35, params.visibility).toFixed(2)}</span>
                <b>reduces reaction margin</b></div>
              <div className="row"><Zap size={14} style={{ color: "var(--critical)" }} /><span>Speed → TTC = distance ÷ speed</span>
                <b>{params.distance_m} m @ {params.speed_kmh} km/h</b></div>
            </div>
            <div className="sw-note" style={{ marginTop: 10 }}>
              The Prolog rule fires purely on distance, TTC, lane and confidence. The risk score
              additionally folds in road wetness and visibility so the presentation reflects
              conditions the camera cannot see.
            </div>
          </Panel>

          <Panel title="Demonstration script" right={<span className="sw-eyebrow">for the demo</span>}>
            <ol className="sw-steps">
              <li>Pick <b>City 35 km/h</b> → engine returns WARNING with TTC ≈ 2.1 s.</li>
              <li>Slide speed to <b>70 km/h</b> → TTC ≈ 1.0 s, decision becomes CRITICAL.</li>
              <li>Add <b>wetness</b> → the risk score climbs even at the same speed.</li>
              <li>Drop lane position below 50% → object is outside the lane, safe again.</li>
               <li>Read the inline <b>Why this alert?</b> rule trace above for the “why”.</li>
            </ol>
          </Panel>
        </div>
      </div>
    </>
  );
}
