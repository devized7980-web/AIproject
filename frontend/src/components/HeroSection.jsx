import React from "react";
import { Play, RotateCcw } from "lucide-react";

export default function HeroSection({ onLaunch, onReplay }) {
  return (
    <section className="hero">
      <div className="hero-inner">
        <h1 className="hero-title">
          See Road Hazards <em>Before</em> They Become Accidents
        </h1>
        <p className="hero-text">
          Real-time AI detection for vehicles, pedestrians, potholes, lane departures and
          collision risks — powered by YOLO detection and Prolog risk reasoning.
        </p>
        <div className="cta-row">
          <button className="cta cta--primary" onClick={onLaunch}>
            <Play size={16} aria-hidden="true" /> Launch Live Detection
          </button>
          <button className="cta cta--secondary" onClick={onReplay}>
            <RotateCcw size={16} aria-hidden="true" /> View Incident Replay
          </button>
        </div>
      </div>
    </section>
  );
}