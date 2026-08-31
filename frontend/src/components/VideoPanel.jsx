import React, { useState } from "react";
import { Video as VideoIcon, Play } from "lucide-react";
import { Panel, Tag } from "../ui.jsx";

// Responsive video card used by the Videos library and landing page.
export default function VideoPanel({ video }) {
  const [playing, setPlaying] = useState(false);

  return (
    <Panel pad={false}>
      <div className="sw-video-card">
        {playing ? (
          <video
            src={`/videos/${video.file}`}
            poster={`/videos/${video.thumb}`}
            controls
            autoPlay
            playsInline
            onEnded={() => setPlaying(false)}
            aria-label={`Play ${video.title}`}
          />
        ) : (
          <div className="sw-thumb-wrap" onClick={() => setPlaying(true)} role="button" tabIndex={0} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setPlaying(true); } }}>
            <img
              src={`/videos/${video.thumb}`}
              alt={video.title}
              loading="lazy"
            />
            <span className="sw-play-btn"><Play size={28} fill="currentColor" /></span>
          </div>
        )}
        <div className="sw-card-body">
          <div className="sw-video-title"><VideoIcon size={14} aria-hidden="true" /> <span className="h">{video.title}</span></div>
          <div className="m">{video.location} · {video.weather}</div>
          <div className="meta">
            <span><b>{video.duration}</b> duration</span>
            <span><b>{video.total_detections}</b> detections</span>
            <Tag level={video.overall_risk} />
          </div>
        </div>
      </div>
    </Panel>
  );
}