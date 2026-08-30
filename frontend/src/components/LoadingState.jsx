import React from "react";

// Centered, muted loading placeholder.
export default function LoadingState({ title = "Loading…", body }) {
  return (
    <div className="rs-state" role="status">
      <div className="rs-state-icon"><span className="rs-spinner" aria-hidden="true" /></div>
      <b>{title}</b>
      {body && <small>{body}</small>}
    </div>
  );
}