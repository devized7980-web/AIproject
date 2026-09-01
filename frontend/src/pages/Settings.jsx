import React, { useEffect, useState } from "react";
import { Palette, Volume2, RotateCcw, Info, Shield, Activity } from "lucide-react";
import { Panel, Toggle } from "../ui.jsx";
import { useSettingsCtx } from "../App.jsx";
import { DEFAULTS } from "../settings.js";
import { backendUp } from "../api.js";

function Seg({ value, options, onChange }) {
  return <div className="sw-seg">{options.map((o) => <button key={o.value} data-on={value === o.value ? "1" : "0"} onClick={() => onChange(o.value)}>{o.icon}{o.label}</button>)}</div>;
}

const status = (value) => value === "Running" ? "Running" : value === "Offline" ? "Offline" : "Degraded";
function HealthRow({ label, value }) {
  const state = status(value);
  return <div className="row"><span>{label}</span><b style={{ color: state === "Running" ? "var(--safe)" : state === "Offline" ? "var(--critical)" : "var(--warning)" }}>{state}</b></div>;
}
function DataRow({ label, value }) { return <div className="row"><span>{label}</span><b>{value ?? "Unavailable"}</b></div>; }

export default function SettingsPage() {
  const [settings, update] = useSettingsCtx();
  const [health, setHealth] = useState(null);
  useEffect(() => { backendUp().then(setHealth); }, []);
  const h = health || {};
  const tracking = h.tracking || {};
  const reasoning = h.reasoning || {};
  const alerts = h.alerts_health || {};
  const services = h.services || {};
  return (
    <>
      <h1 className="sw-h1">System Health</h1>
      <p className="sw-sub">Measured service state for the RoadSafety project. The feed is a recorded detection replay, not live camera inference.</p>
      <div className="sw-2col-even">
        <div className="sw-stack">
          <Panel title="Detection" right={<Activity size={14} style={{ color: "var(--gold)" }} />}>
            <HealthRow label="Common YOLO model" value={h.models?.common} />
            <HealthRow label="Road-damage model" value={h.models?.road_damage} />
            <DataRow label="Confidence thresholds" value="Common 0.50 · damage 0.30" />
            <DataRow label="Last successful inference" value={h.last_inference ? new Date(h.last_inference).toLocaleString() : null} />
            <DataRow label="Measured processing FPS" value={h.processing_fps ? `${h.processing_fps} fps` : null} />
            <DataRow label="Current device" value={h.device} />
          </Panel>
          <Panel title="Tracking">
            <HealthRow label="ByteTrack" value={tracking.bytetrack} />
            <HealthRow label="Kalman filter" value={tracking.kalman} />
            <DataRow label="Active tracks" value={tracking.active_tracks} />
            <DataRow label="Rejected associations" value={tracking.rejected_associations} />
            <DataRow label="Deleted / stale tracks" value={tracking.deleted_tracks} />
            <HealthRow label="Frame synchronisation" value={tracking.frame_sync} />
          </Panel>
          <Panel title="Appearance" right={<Palette size={14} style={{ color: "var(--gold)" }} />}>
            <div className="sw-set"><label className="sw-setlabel">Theme</label><Seg value={settings.theme} onChange={(theme) => update({ theme })} options={[{ value: "dark", label: "Dark" }, { value: "light", label: "Light" }, { value: "colorblind", label: "Color blind" }]} /></div>
        </Panel>
        </div>
        <div className="sw-stack">
          <Panel title="Reasoning">
            <HealthRow label="SWI-Prolog" value={reasoning.prolog} />
            <HealthRow label="Python fallback" value={reasoning.python_fallback} />
            <DataRow label="Loaded rules" value={reasoning.loaded_rules} />
            <DataRow label="Last fired rule" value={reasoning.last_rule} />
            <DataRow label="Current decision source" value={reasoning.decision_source} />
          </Panel>
          <Panel title="Alerts" right={<Volume2 size={14} style={{ color: "var(--gold)" }} />}>
            <Toggle on={settings.sound} onChange={(sound) => update({ sound })} label="Alert sounds" desc="Short chime when a warning arrives" />
            <div className="sw-engine" style={{ marginTop: 12 }}>
              <HealthRow label="Voice output" value={alerts.voice_enabled ? "Running" : "Offline"} />
              <HealthRow label="TTS engine" value={alerts.tts_available ? "Running" : "Offline"} />
              <DataRow label="Latest warning level" value={alerts.latest_level} />
              <DataRow label="Queued alerts" value={alerts.queued_alerts} />
            </div>
          </Panel>
          <Panel title="Services">
            {Object.entries({ FastAPI: services.fastapi, WebSocket: services.websocket, "Raw-video service": services.raw_video, "Output-data service": services.output_data, "Model benchmark": services.model_benchmark }).map(([label, value]) => <HealthRow key={label} label={label} value={value} />)}
          </Panel>
          <Panel title="Organization Export Readiness" right={<Info size={14} style={{ color: "var(--gold)" }} />}>
            <HealthRow label="Incident CSV / JSON export" value={h.videos ? "Running" : "Degraded"} />
            <div className="sw-note" style={{ marginTop: 10 }}>Ready for a future road-safety organization integration. No external connection is configured.</div>
          </Panel>
          <Panel title="Reset & about" right={<Info size={14} style={{ color: "var(--gold)" }} />}>
            <button className="sw-reportbtn" onClick={() => update(DEFAULTS)}><RotateCcw size={14} /> Reset all settings</button>
            <div className="sw-note" style={{ marginTop: 12 }}><Shield size={14} style={{ color: "var(--gold)", verticalAlign: -2, marginRight: 6 }} />Research benchmark - not certified for on-road use.</div>
          </Panel>
        </div>
      </div>
    </>
  );
}
