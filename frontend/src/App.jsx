import React, { useState } from "react";
import { LayoutDashboard, Video, Phone, Info, Shield, Cpu, Menu, X } from "lucide-react";
import { VIDEOS } from "./data.js";
import Dashboard from "./pages/Dashboard.jsx";
import Gallery from "./pages/Gallery.jsx";
import Workspace from "./pages/Workspace.jsx";
import Emergency from "./pages/Emergency.jsx";
import About from "./pages/About.jsx";

const NAV = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "gallery", label: "Video Gallery", icon: Video },
  { id: "emergency", label: "Emergency & Actions", icon: Phone },
  { id: "about", label: "About", icon: Info },
];

export default function App() {
  const [page, setPage] = useState("dashboard");
  const [openId, setOpenId] = useState(null);
  const [menu, setMenu] = useState(false);

  const go = (p) => {
    setPage(p);
    setOpenId(null);
    setMenu(false);
    window.scrollTo(0, 0);
  };

  const openVideo = (id) => {
    setPage("gallery");
    setOpenId(id);
    setMenu(false);
    window.scrollTo(0, 0);
  };

  const current = VIDEOS.find((v) => v.id === openId);
  const crumb = current ? `Video Gallery / ${current.title}` : NAV.find((n) => n.id === page).label;

  return (
    <div className="sw">
      <aside className="sw-side" data-open={menu ? "1" : "0"}>
        <div className="sw-brand">
          <div className="sw-crest"><Shield size={18} /></div>
          <div>
            <b>Safeway AI</b>
            <small>Benchmark suite</small>
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
          <span className="sw-live" />PIPELINE IDLE<br />
          <span style={{ opacity: 0.6 }}>YOLO11 + PROLOG v1.0</span>
        </div>
      </aside>

      <div className="sw-main">
        <div className="sw-top">
          <button className="sw-burger" onClick={() => setMenu((m) => !m)} aria-label="Toggle menu">
            {menu ? <X size={16} /> : <Menu size={16} />}
          </button>
          <Cpu size={13} style={{ color: "var(--gold)" }} />
          <span className="sw-crumb">{crumb}</span>
        </div>

        <main className="sw-body">
          {page === "dashboard" && <Dashboard onOpen={openVideo} />}
          {page === "gallery" && (current
            ? <Workspace video={current} onBack={() => setOpenId(null)} />
            : <Gallery onOpen={openVideo} />)}
          {page === "emergency" && <Emergency />}
          {page === "about" && <About />}
        </main>
      </div>
    </div>
  );
}
