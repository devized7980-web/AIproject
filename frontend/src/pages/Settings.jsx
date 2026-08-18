import React from "react";
import { Palette, Wind, Volume2, Monitor, Accessibility, Bug, RotateCcw, Info, Shield } from "lucide-react";
import { Panel, Toggle, Range } from "../ui.jsx";
import { useSettingsCtx } from "../App.jsx";
import { DEFAULTS } from "../settings.js";

function Seg({ value, options, onChange }) {
  return (
    <div className="sw-seg">
      {options.map((o) => (
        <button key={o.value} data-on={value === o.value ? "1" : "0"} onClick={() => onChange(o.value)}>
          {o.icon}{o.label}
        </button>
      ))}
    </div>
  );
}

export default function SettingsPage() {
  const [settings, update] = useSettingsCtx();

  return (
    <>
      <h1 className="sw-h1">System Settings &amp; Personalization</h1>
      <p className="sw-sub">
        Tune the look, motion, alerting and debug options for the demo. Everything saves to this
        browser automatically.
      </p>

      <div className="sw-2col-even">
        <div className="sw-stack">
          <Panel title="Appearance" right={<Palette size={14} style={{ color: "var(--gold)" }} />}>
            <div className="sw-set">
              <label className="sw-setlabel">Theme</label>
              <Seg
                value={settings.theme}
                onChange={(theme) => update({ theme })}
                options={[
                  { value: "dark", label: "Dark" },
                  { value: "light", label: "Light" },
                  { value: "colorblind", label: "Color blind" },
                ]}
              />
              <div className="sw-theme-prev" data-theme={settings.theme}>
                <i /><i /><i />
              </div>
            </div>
            <div className="sw-set">
              <label className="sw-setlabel">Layout density</label>
              <Seg
                value={settings.density}
                onChange={(density) => update({ density })}
                options={[
                  { value: "comfortable", label: "Comfortable" },
                  { value: "compact", label: "Compact" },
                ]}
              />
            </div>
          </Panel>

          <Panel title="Motion intensity" right={<Wind size={14} style={{ color: "var(--gold)" }} />}>
            <Seg
              value={settings.motion}
              onChange={(motion) => update({ motion })}
              options={[
                { value: "low", label: "Low" },
                { value: "normal", label: "Normal" },
                { value: "high", label: "High" },
              ]}
            />
            <p className="sw-set-note">
              Low disables pulsing/glowing and alert animations. High speeds up route animation and
              effect transitions.
            </p>
            <Toggle on={settings.reduced} onChange={(reduced) => update({ reduced })}
              label="Reduce animations & effects" desc="Accessibility: disable pulses, glows and sliding alerts" />
          </Panel>

          <Panel title="Alerts &amp; audio" right={<Volume2 size={14} style={{ color: "var(--gold)" }} />}>
            <Toggle on={settings.sound} onChange={(sound) => update({ sound })}
              label="Alert sounds" desc="Short chime when a critical alert arrives" />
            <div style={{ marginTop: 14 }}>
              <Range
                label="Detection confidence threshold" unit="%" min={5} max={99} step={1}
                value={Math.round(settings.threshold * 100)}
                onChange={(v) => update({ threshold: v / 100 })}
                marks={["0.10", "0.30 default", "0.90"]}
              />
            </div>
          </Panel>
        </div>

        <div className="sw-stack">
          <Panel title="Display" right={<Monitor size={14} style={{ color: "var(--gold)" }} />}>
            <Toggle on={settings.showFps} onChange={(showFps) => update({ showFps })}
              label="Show FPS & latency overlay" desc="Overlay telemetry on every video stage" />
            <Toggle on={settings.density === "compact"} onChange={(d) => update({ density: d ? "compact" : "comfortable" })}
              label="Compact dashboard layout" desc="Tighter spacing across all pages" />
          </Panel>

          <Panel title="Accessibility" right={<Accessibility size={14} style={{ color: "var(--gold)" }} />}>
            <Toggle on={settings.reduced} onChange={(reduced) => update({ reduced })}
              label="Respect reduced motion" desc="Stops pulsing boxes, glowing zones and rain effect" />
          </Panel>

          <Panel title="Performance &amp; debug" right={<Bug size={14} style={{ color: "var(--gold)" }} />}>
            <Toggle on={settings.debug} onChange={(debug) => update({ debug })}
              label="Debug mode" desc="Show data source badges and raw feed messages" />
            <Toggle on={settings.showFps} onChange={(showFps) => update({ showFps })}
              label="Telemetry overlay" desc="FPS / latency readout on video stages" />
          </Panel>

          <Panel title="Reset &amp; about" right={<Info size={14} style={{ color: "var(--gold)" }} />}>
            <button className="sw-reportbtn" onClick={() => { update(DEFAULTS); }}>
              <RotateCcw size={14} /> Reset all settings
            </button>
            <div className="sw-engine" style={{ marginTop: 14 }}>
              <div className="row"><span>System</span><b>Safeway AI v2.0</b></div>
              <div className="row"><span>Detection</span><b>YOLO11n + custom best.pt</b></div>
              <div className="row"><span>Reasoning</span><b>SWI-Prolog expert system</b></div>
              <div className="row"><span>Backend</span><b>FastAPI + WebSocket</b></div>
            </div>
            <div className="sw-note" style={{ marginTop: 12 }}>
              <Shield size={14} style={{ color: "var(--gold)", verticalAlign: -2, marginRight: 6 }} />
              Research benchmark — not certified for on-road use.
            </div>
          </Panel>
        </div>
      </div>
    </>
  );
}