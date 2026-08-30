import React, { Fragment } from "react";
import { Menu, X, Radar } from "lucide-react";

// Top horizontal navigation bar — brand, grouped pill items (scrollable on
// narrow widths). On mobile the pills collapse into a dropdown opened by the
// burger. Item icons are intentionally omitted; the logo crest is the only
// icon in the bar.
export default function TopNav({ groups, page, onGo, open, onToggleMenu }) {
  const renderItem = ({ id, label }) => (
    <button key={id} data-on={page === id ? "1" : "0"} onClick={() => onGo(id)}>
      {label}
    </button>
  );

  return (
    <header className="sw-topbar" data-open={open ? "1" : "0"}>
      <button
        className="sw-burger"
        onClick={onToggleMenu}
        aria-label={open ? "Close navigation menu" : "Open navigation menu"}
        aria-expanded={open}
        aria-controls="rs-nav"
      >
        {open ? <X size={16} /> : <Menu size={16} />}
      </button>

      <div className="rs-brand">
        <div className="sw-crest" aria-hidden="true"><Radar size={18} /></div>
        <div className="rs-brand-name">
          <b>Safeway AI</b>
          <small>Road Safety AI</small>
        </div>
      </div>

      <nav className="rs-nav" id="rs-nav" aria-label="Primary navigation">
        {groups.map((g, gi) => (
          <Fragment key={g.label || gi}>
            {gi > 0 && <span className="rs-nav-sep" aria-hidden="true" />}
            {g.items.map(renderItem)}
          </Fragment>
        ))}
      </nav>
    </header>
  );
}