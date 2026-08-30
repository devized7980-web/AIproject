import React from "react";
import { Inbox } from "lucide-react";

// Reusable empty-state block.
export default function EmptyState({ title = "Nothing here yet", body }) {
  return (
    <div className="rs-state">
      <div className="rs-state-icon"><Inbox size={22} aria-hidden="true" /></div>
      <b style={{ color: "var(--text)" }}>{title}</b>
      {body && <small>{body}</small>}
    </div>
  );
}