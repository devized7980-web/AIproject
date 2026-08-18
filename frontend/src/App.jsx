import React, { createContext, useContext, useEffect, useState } from "react";
import {
  LayoutDashboard, Video, BarChart3, BrainCircuit, SlidersHorizontal,
  BellRing, Gauge, Settings, Shield, Cpu, Menu, X, Wifi, WifiOff,
} from "lucide-react";
import { getVideos, backendUp } from "./api.js";
import { useSettings } from "./settings.js";
import LiveCommandCenter from "./pages/LiveCommandCenter.jsx";
import IncidentReplay from "./pages/IncidentReplay.jsx";
import Analytics from "./pages/Analytics.jsx";
import AIExplainer from "./pages/AIExplainer.jsx";
import WhatIfSimulator from "./pages/WhatIfSimulator.jsx";
import AlertCenter from "./pages/AlertCenter.jsx";
import PerformanceLab from "./pages/PerformanceLab.jsx";
import SettingsPage from "./pages/Settings.jsx";

export const SettingsContext = createContext(null);
export const useSettingsCtx = () => useContext(SettingsContext);

const NAV = [
  { id: "command", label: "Live Command Center", icon: LayoutDashboard },
  { id: "replay", label: "Incident Replay", icon: Video },
  { id: "analytics", label: "Safety Analytics", icon: BarChart3 },
  { id: "explainer", label: "AI Explainer", icon: BrainCircuit },
  { id: "simulator", label: "What-If Simulator", icon: SlidersHorizontal },
  { id: "alerts", label: "Alert Center", icon: BellRing },
  { id: "performance", label: "Model Performance Lab", icon: Gauge },
  { id: "settings", label: "System Settings", icon: Settings },
];

export default function App() {
  const [page, setPage] = useState("command");
  const [menu, setMenu] = useState(false);
  const [videos, setVideos] = useState([]);
  const [backend, setBackend] = useState(null);
  const [settings, updateSettings] = useSettings();

  useEffect(() => {
    getVideos().then(setVideos);
    backendUp().then((h) => setBackend(h));
  }, []);

  const go = (p) => {
    setPage(p);
    setMenu(false);
    window.scrollTo(0, 0);
  };

  const crumb = NAV.find((n) => n.id === page)?.label || "";

  return (
    <SettingsContext.Provider value={[settings, updateSettings]}>
      <div className="sw">
        <aside className="sw-side" data-open={menu ? "1" : "0"}>
          <div className="sw-brand">
            <div className="sw-crest"><Shield size={18} /></div>
            <div>
              <b>Safeway AI</b>
              <small>Road Safety & Hazard Detection</small>
            </div>
          </div>

          <nav className="sw-nav">
            {NAV.map(({ id, label, icon: Icon }) => (
              <button key={id} data-on={page === id ? "1" : "0"} onClick={() => go(id)}>
                <Icon size={16} /> {label}
              </button>
            ))}
          </nav>

          <div className="sw-side-foot">
            <span className="sw-live" data-on={backend ? "1" : "0"} />{backend ? "BACKEND ONLINE" : "OFFLINE — LOCAL MIRROR"}
            <br />
            <span style={{ opacity: 0.6 }}>YOLO11 + PROLOG v2.0</span>
          </div>
        </aside>

        <div className="sw-main">
          <div className="sw-top">
            <button className="sw-burger" onClick={() => setMenu((m) => !m)} aria-label="Toggle menu">
              {menu ? <X size={16} /> : <Menu size={16} />}
            </button>
            <Cpu size={13} style={{ color: "var(--gold)" }} />
            <span className="sw-crumb">{crumb}</span>
            <span className="grow" />
            <span className="sw-status" title="Live backend status">
              {backend
                ? <><Wifi size={12} style={{ color: "var(--safe)" }} /> LIVE</>
                : <><WifiOff size={12} style={{ color: "var(--faint)" }} /> MIRROR</>}
            </span>
          </div>

          <main className="sw-body">
            {page === "command" && <LiveCommandCenter videos={videos} />}
            {page === "replay" && <IncidentReplay videos={videos} />}
            {page === "analytics" && <Analytics videos={videos} />}
            {page === "explainer" && <AIExplainer videos={videos} />}
            {page === "simulator" && <WhatIfSimulator videos={videos} />}
            {page === "alerts" && <AlertCenter videos={videos} />}
            {page === "performance" && <PerformanceLab videos={videos} />}
            {page === "settings" && <SettingsPage />}
          </main>
        </div>
      </div>
    </SettingsContext.Provider>
  );
}