import React from "react";
import { AlertTriangle } from "lucide-react";

// Reusable error-state block with optional retry action.
export default function ErrorState({ title = "Something went wrong", body, onRetry, retryLabel = "Retry" }) {
  return (
    <div className="rs-state rs-state--error">
      <div className="rs-state-icon"><AlertTriangle size={22} aria-hidden="true" /></div>
      <b style={{ color: "var(--text)" }}>{title}</b>
      {body && <small>{body}</small>}
      {onRetry && <button className="cta cta--secondary" onClick={onRetry}>{retryLabel}</button>}
    </div>
  );
}