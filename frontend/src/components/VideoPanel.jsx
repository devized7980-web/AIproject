import React from "react";
import { Video as VideoIcon } from "lucide-react";
import { Panel, Tag } from "../ui.jsx";

// Responsive video card used by the Videos library and landing page.
export default function VideoPanel({ video }) {
  return (
    <Panel pad={false}>
      <div className="sw-video-card">
        <video
          src={`/videos/${video.file}`}
          poster={`/videos/${video.thumb}`}
          controls
          preload="metadata"
          playsInline
          aria-label={`Play ${video.title}`}
        />
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