import React from "react";
import { VIDEOS } from "../data.js";
import VideoPanel from "../components/VideoPanel.jsx";

export default function Videos({ videos }) {
  const all = Array.isArray(videos) && videos.length ? videos : VIDEOS;
  return (
    <>
      <h1 className="sw-h1">Videos</h1>
      <p className="sw-sub">The shared video library used by Live Detection and Incident Replay.</p>
      <div className="sw-grid">
        {all.map((video) => <VideoPanel key={video.id} video={video} />)}
      </div>
    </>
  );
}
