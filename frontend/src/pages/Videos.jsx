import React from "react";
import { Video as VideoIcon } from "lucide-react";
import { VIDEOS } from "../data.js";
import { Panel, Tag } from "../ui.jsx";

export default function Videos({ videos }) {
  const all = videos.length ? videos : VIDEOS;
  return (
    <>
      <h1 className="sw-h1">Videos</h1>
      <p className="sw-sub">The shared eight-clip library used by Live Command Center and Incident Replay.</p>
      <div className="sw-grid">
        {all.map((video) => (
          <Panel key={video.id} pad={false}>
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
                <div className="sw-video-title"><VideoIcon size={14} /> <span className="h">{video.title}</span></div>
                <div className="m">{video.location} · {video.weather}</div>
                <div className="meta"><span><b>{video.duration}</b> duration</span><span><b>{video.total_detections}</b> detections</span><Tag level={video.overall_risk} /></div>
              </div>
            </div>
          </Panel>
        ))}
      </div>
    </>
  );
}
