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
import Videos from "./pages/Videos.jsx";

export const SettingsContext = createContext(null);
export const useSettingsCtx = () => useContext(SettingsContext);

const NAV = [
  { id: "videos", label: "Videos", icon: Video },
  { id: "command", label: "Live Command Center", icon: LayoutDashboard },
  { id: "replay", label: "Incident Replay", icon: Video },
  { id: "analytics", label: "Safety Analytics", icon: BarChart3 },
  { id: "explainer", label: "AI Explainer", icon: BrainCircuit },
  { id: "simulator", label: "What-If Simulator", icon: SlidersHorizontal },
  { id: "alerts", label: "Alert Center", icon: BellRing },
  { id: "performance", label: "Model Performance Lab", icon: Gauge },
  { id: "settings", label: "System Settings", icon: Settings },
];

const PAGE_IDS = new Set(NAV.map(({ id }) => id));
const PAGE_ALIASES = {
  "live-command-center": "command",
  "safety-analytics": "analytics",
  "incident-replay": "replay",
  "alert-center": "alerts",
};
const pageFromPath = () => {
  const id = window.location.pathname.replace(/^\/+|\/+$/g, "");
  const page = PAGE_ALIASES[id] || id;
  return PAGE_IDS.has(page) && page !== "videos" ? page : "command";
};

class SafetyErrorBoundary extends React.Component {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error, info) {
    console.error("Safety page rendering failed", error, info);
  }

  render() {
    if (this.state.failed) {
      return <div className="sw-empty--panel"><div>Safety page unavailable</div><small>Return to Live Command Center and try again.</small></div>;
    }
    return this.props.children;
  }
}

export default function App() {
  const [page, setPage] = useState(pageFromPath);
  const [menu, setMenu] = useState(false);
  const [videos, setVideos] = useState([]);
  const [backend, setBackend] = useState(null);
  const [settings, updateSettings] = useSettings();

  useEffect(() => {
    getVideos().then(setVideos);
    backendUp().then((h) => setBackend(h));
  }, []);

  useEffect(() => {
    const onPopState = () => setPage(pageFromPath());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  useEffect(() => {
    let frame = 0;
    const moveGlow = (event) => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        document.documentElement.style.setProperty("--pointer-x", `${event.clientX}px`);
        document.documentElement.style.setProperty("--pointer-y", `${event.clientY}px`);
        document.documentElement.style.setProperty("--pointer-opacity", "1");
      });
    };
    const hideGlow = () => document.documentElement.style.setProperty("--pointer-opacity", "0");
    window.addEventListener("pointermove", moveGlow, { passive: true });
    window.addEventListener("pointerleave", hideGlow, { passive: true });
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("pointermove", moveGlow);
      window.removeEventListener("pointerleave", hideGlow);
    };
  }, []);

  const go = (p) => {
    setPage(p);
    setMenu(false);
    if (p !== "videos") window.history.pushState({}, "", `/${p}`);
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
             <SafetyErrorBoundary key={page}>
               {page === "videos" && <Videos videos={videos} />}
               {page === "command" && <LiveCommandCenter videos={videos} />}
               {page === "replay" && <IncidentReplay videos={videos} />}
               {page === "analytics" && <Analytics videos={videos} />}
               {page === "explainer" && <AIExplainer videos={videos} />}
               {page === "simulator" && <WhatIfSimulator videos={videos} />}
               {page === "alerts" && <AlertCenter videos={videos} />}
               {page === "performance" && <PerformanceLab videos={videos} />}
               {page === "settings" && <SettingsPage />}
             </SafetyErrorBoundary>
           </main>
        </div>
      </div>
    </SettingsContext.Provider>
  );
}
