import React, { useState } from "react";
import { Video as VideoIcon, Play } from "lucide-react";
import { Panel, Tag } from "../ui.jsx";

// Responsive video card used by the Videos library and landing page.
export default function VideoPanel({ video }) {
  const [playing, setPlaying] = useState(false);
  const analysed = video.analysis_available !== false;
  const source = video.processed_available && video.file
    ? `/videos/${video.file}`
    : video.raw_available !== false && video.raw
      ? `/raw/${video.raw}`
      : "";
  const poster = video.thumb ? `/videos/${video.thumb}` : undefined;

  return (
    <Panel pad={false}>
      <div className="sw-video-card">
        {playing && source ? (
          <video
            src={source}
            poster={poster}
            controls
            autoPlay
            playsInline
            onEnded={() => setPlaying(false)}
            aria-label={`Play ${video.title}`}
          />
        ) : (
          <div className="sw-thumb-wrap" onClick={() => source && setPlaying(true)} role={source ? "button" : undefined} tabIndex={source ? 0 : undefined} onKeyDown={(e) => { if (source && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); setPlaying(true); } }}>
            {poster ? <img src={poster} alt={video.title} loading="lazy" /> : <div className="sw-empty"><VideoIcon size={24} /></div>}
            {source && <span className="sw-play-btn"><Play size={28} fill="currentColor" /></span>}
          </div>
        )}
        <div className="sw-card-body">
          <div className="sw-video-title"><VideoIcon size={14} aria-hidden="true" /> <span className="h">{video.title}</span></div>
          <div className="m">{video.location} · {video.weather}</div>
          <div className="meta">
            <span><b>{video.duration || "—"}</b> duration</span>
            {analysed ? <><span><b>{video.total_detections}</b> detections</span><Tag level={video.overall_risk} /></> : <span className="sw-eyebrow">Not analysed yet</span>}
          </div>
        </div>
      </div>
    </Panel>
  );
}
