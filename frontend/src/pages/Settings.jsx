import React from "react";
import { Palette, Volume2, RotateCcw, Info, Shield } from "lucide-react";
import { Panel, Toggle } from "../ui.jsx";
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
      <h1 className="sw-h1">System Settings</h1>

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
          </Panel>

          <Panel title="Alerts &amp; audio" right={<Volume2 size={14} style={{ color: "var(--gold)" }} />}>
            <Toggle on={settings.sound} onChange={(sound) => update({ sound })}
              label="Alert sounds" desc="Short chime when a critical alert arrives" />
          </Panel>
        </div>

        <div className="sw-stack">
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