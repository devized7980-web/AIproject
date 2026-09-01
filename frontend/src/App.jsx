import React, { createContext, useContext, useEffect, useState } from "react";
import { getVideos, backendUp } from "./api.js";
import { useSettings } from "./settings.js";
import TopNav from "./components/TopNav.jsx";
import LiveCommandCenter from "./pages/LiveCommandCenter.jsx";
import IncidentReplay from "./pages/IncidentReplay.jsx";
import Analytics from "./pages/Analytics.jsx";
import WhatIfSimulator from "./pages/WhatIfSimulator.jsx";
import AlertCenter from "./pages/AlertCenter.jsx";
import SettingsPage from "./pages/Settings.jsx";
import Videos from "./pages/Videos.jsx";
import Home from "./pages/Home.jsx";

export const SettingsContext = createContext(null);
export const useSettingsCtx = () => useContext(SettingsContext);

const NAV = [
  {
    label: "Monitor",
    items: [
      { id: "home", label: "Home" },
      { id: "command", label: "Live Detection" },
      { id: "replay", label: "Incident Replay" },
      { id: "videos", label: "Videos" },
      { id: "analytics", label: "Analytics" },
    ],
  },
  {
    label: "Tools",
    items: [
      { id: "simulator", label: "What-If Simulator" },
      { id: "alerts", label: "Alert Center" },
      { id: "settings", label: "System Settings" },
    ],
  },
];

const PAGE_IDS = new Set(NAV.flatMap((g) => g.items).map(({ id }) => id));
const PAGE_ALIASES = {
  "live-command-center": "command",
  "live-detection": "command",
  "safety-analytics": "analytics",
  "incident-replay": "replay",
  "alert-center": "alerts",
};
const pageFromPath = () => {
  const id = window.location.pathname.replace(/^\/+|\/+$/g, "");
  const page = PAGE_ALIASES[id] || id;
  return PAGE_IDS.has(page) ? page : "home";
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
      return (
        <div className="sw-empty--panel">
          <div>This view hit an error</div>
          <small>Return to Home and try again.</small>
        </div>
      );
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

  const go = (p) => {
    setPage(p);
    setMenu(false);
    window.history.pushState({}, "", p === "videos" ? "/videos" : `/${p}`);
    window.scrollTo(0, 0);
  };

  return (
    <SettingsContext.Provider value={[settings, updateSettings]}>
      <div className="sw">
        <TopNav groups={NAV} page={page} onGo={go} open={menu} onToggleMenu={() => setMenu((m) => !m)} />

        <main className="sw-body">
          <div className="sw-page" key={page}>
            <SafetyErrorBoundary>
              {page === "home" && <Home videos={videos} onNavigate={go} />}
              {page === "command" && <LiveCommandCenter videos={videos} />}
              {page === "replay" && <IncidentReplay videos={videos} />}
              {page === "videos" && <Videos videos={videos} />}
              {page === "analytics" && <Analytics videos={videos} />}
              {page === "simulator" && <WhatIfSimulator videos={videos} />}
              {page === "alerts" && <AlertCenter videos={videos} />}
              {page === "settings" && <SettingsPage />}
            </SafetyErrorBoundary>
          </div>
        </main>
      </div>
    </SettingsContext.Provider>
  );
}
