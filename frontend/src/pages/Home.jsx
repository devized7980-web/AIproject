import React from "react";
import { RadioTower, History, Clapperboard, BarChart3, SlidersHorizontal } from "lucide-react";
import HeroSection from "../components/HeroSection.jsx";

const FEATURES = [
  { id: "command", icon: RadioTower, title: "Live hazard detection", body: "Watch the YOLO + Prolog pipeline detect vehicles, pedestrians, potholes and collision risk in real time on each camera feed." },
  { id: "replay", icon: History, title: "Incident replay", body: "Seek back through the current footage and step through the exact detections, risk levels and decisions behind every alert." },
  { id: "videos", icon: Clapperboard, title: "Detection archive", body: "Annotated clips from every pipeline run — searchable by severity, with bounding boxes and risk tags burned in." },
  { id: "analytics", icon: BarChart3, title: "Safety analytics", body: "Best-safe, worst-risk and average-baseline scores per session, plus detection and risk distributions across every recording." },
  { id: "simulator", icon: SlidersHorizontal, title: "What-if simulator", body: "Adjust risk thresholds and see how the scores, alerts and emergency actions would change before you take the road." },
];

export default function Home({ onNavigate }) {
  return (
    <div>
      <HeroSection onLaunch={() => onNavigate("command")} onReplay={() => onNavigate("replay")} />

      <section className="home-feature-grid" aria-label="Features">
        {FEATURES.map(({ id, icon: Icon, title, body }) => (
          <button className="home-feature" key={id} onClick={() => onNavigate(id)}>
            <div className="ic"><Icon size={19} aria-hidden="true" /></div>
            <h3>{title}</h3>
            <p>{body}</p>
            <span className="rs-feature-go">Open →</span>
          </button>
        ))}
      </section>
    </div>
  );
}