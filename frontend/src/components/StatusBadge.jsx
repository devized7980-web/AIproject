import React from "react";

// Small semantic status pill. state: ok | warn | off | danger | idle.
export default function StatusBadge({ state = "idle", label, pulse, title, className = "" }) {
  return (
    <span className={`rs-badge ${className}`} data-state={state} title={title} role="status">
      <i className={pulse ? "is-pulse" : ""} aria-hidden="true" />
      {label}
    </span>
  );
}