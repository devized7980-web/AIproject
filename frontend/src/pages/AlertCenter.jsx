import React, { useEffect, useMemo, useRef, useState } from "react";
import { BellRing, CheckCheck, UserCheck, Inbox, CircleDot, UserPlus, MailCheck } from "lucide-react";
import { getAlerts, alertAction } from "../api.js";
import { playAlertSound } from "../sound.js";
import { Panel, Tag, levelColor, fmtTtc } from "../ui.jsx";
import { useSettingsCtx } from "../App.jsx";

const GROUPS = [
  { key: "CRITICAL", label: "Critical", icon: CircleDot },
  { key: "WARNING", label: "High", icon: BellRing },
  { key: "CAUTION", label: "Medium", icon: Inbox },
  { key: "SAFE", label: "Low", icon: MailCheck },
];
const SEV = { CRITICAL: 0, WARNING: 1, CAUTION: 2, SAFE: 3 };
const ASSIGNEES = ["Roads Dept.", "Patrol A", "Patrol B", "Ops Centre"];

export default function AlertCenter({ videos }) {
  const [alerts, setAlerts] = useState([]);
  const [filter, setFilter] = useState("ALL");
  const [settings] = useSettingsCtx();
  const [newIds, setNewIds] = useState([]);
  const seen = useRef(new Set());

  useEffect(() => {
    getAlerts().then((list) => {
      const fresh = list.filter((a) => !seen.current.has(a.id));
      fresh.forEach((a) => seen.current.add(a.id));
      setNewIds(fresh.map((a) => a.id));
      setAlerts(list);
      if (settings.sound && fresh.some((a) => a.level === "CRITICAL")) playAlertSound("CRITICAL");
    });
  }, [settings.sound]);

  const act = async (id, action, assignee) => {
    const row = await alertAction(id, action, assignee);
    if (row) setAlerts((a) => a.map((x) => (x.id === id ? { ...x, ...row } : x)));
    else setAlerts((a) => a.map((x) => (x.id === id ? { ...x, status: action === "resolve" ? "resolved" : "acknowledged", assignee: assignee || x.assignee } : x)));
  };

  const simulateNew = () => {
    const id = `sim-${Date.now()}`;
    const a = {
      id, video_id: "video_3", video_title: "Pothole and Road-Surface Hazard Detection", time: "00:25",
      level: "CRITICAL", object: "pothole", label: "BRAKE AND AVOID ROAD DAMAGE",
      distance_m: 1.8, ttc_s: null, confidence: 0.71, status: "open", assignee: null,
    };
    setAlerts((prev) => [a, ...prev]);
    setNewIds((n) => [id, ...n]);
    if (settings.sound) playAlertSound("CRITICAL");
  };

  const filtered = useMemo(() => {
    const list = [...alerts].sort((a, b) => SEV[a.level] - SEV[b.level] || b.pct - a.pct);
    return filter === "ALL" ? list : list.filter((a) => a.level === filter);
  }, [alerts, filter]);

  const counts = useMemo(() => {
    const c = { CRITICAL: 0, WARNING: 0, CAUTION: 0, SAFE: 0, open: 0, resolved: 0, acknowledged: 0 };
    alerts.forEach((a) => { c[a.level] = (c[a.level] || 0) + 1; c[a.status] = (c[a.status] || 0) + 1; });
    return c;
  }, [alerts]);

  return (
    <>
      <h1 className="sw-h1">Alert Center</h1>
      <p className="sw-sub">
        Every WARNING and CRITICAL decision the pipeline made, plus advisory items. Acknowledge,
        assign, or resolve each one. Critical alerts pulse red; newly detected hazards slide in.
      </p>

      <div className="sw-cc-head" style={{ marginBottom: 18 }}>
        <div className="sw-seg">
          <button data-on={filter === "ALL" ? "1" : "0"} onClick={() => setFilter("ALL")}>All ({alerts.length})</button>
          {GROUPS.map((g) => (
            <button key={g.key} data-on={filter === g.key ? "1" : "0"} onClick={() => setFilter(g.key)}>
              {g.label} ({counts[g.key] || 0})
            </button>
          ))}
        </div>
        <button className="sw-linkbtn" onClick={simulateNew}><BellRing size={12} /> Simulate new hazard</button>
      </div>

      <div className="sw-kpis" style={{ marginBottom: 18 }}>
        {[
          { l: "Open", v: counts.open || 0, c: "var(--critical)" },
          { l: "Acknowledged", v: counts.acknowledged || 0, c: "var(--caution)" },
          { l: "Resolved", v: counts.resolved || 0, c: "var(--safe)" },
          { l: "Critical", v: counts.CRITICAL || 0, c: "var(--critical)" },
        ].map((k) => (
          <div className="sw-kpi" key={k.l}>
            <div className="l">{k.l}</div>
            <div className="v" style={{ color: k.c }}>{k.v}</div>
          </div>
        ))}
      </div>

      <Panel title={`${filter === "ALL" ? "All" : filter} alerts`} pad={false} right={<span className="sw-eyebrow">{filtered.length} shown</span>}>
        <div className="sw-alertlist">
          {filtered.length === 0 && (
            <div className="sw-empty--panel" style={{ minHeight: 120 }}><div style={{ fontSize: 13 }}>No {filter === "ALL" ? "" : filter.toLowerCase() + " "}alerts.</div></div>
          )}
          {filtered.map((a) => (
            <div className="sw-alertitem" data-lvl={a.level} data-new={newIds.includes(a.id) ? "1" : "0"} data-status={a.status} key={a.id}>
              <i className="pulse" style={{ background: levelColor(a.level) }} />
              <div className="body grow" style={{ minWidth: 0 }}>
                <div className="row1">
                  <Tag level={a.level} />
                  <span className="time">{a.video_title.split("—")[0]} · {a.time}</span>
                  {a.status !== "open" && <span className="st" data-s={a.status}>{a.status}</span>}
                </div>
                <div className="t sw-truncate">{a.label}</div>
                <div className="d sw-truncate">
                  {a.object} · {a.distance_m} m · TTC {fmtTtc(a.ttc_s)} · {Math.round(a.confidence * 100)}% conf
                  {a.assignee && <span className="ass"> → {a.assignee}</span>}
                </div>
              </div>
              <div className="acts">
                {a.status !== "acknowledged" && a.status !== "resolved" && (
                  <button onClick={() => act(a.id, "acknowledge")} title="Acknowledge"><CheckCheck size={14} /></button>
                )}
                {a.status !== "resolved" && (
                  <button onClick={() => act(a.id, "resolve")} title="Resolve"><MailCheck size={14} /></button>
                )}
                <button
                  onClick={() => {
                    const cur = a.assignee ? ASSIGNEES.indexOf(a.assignee) : -1;
                    const next = ASSIGNEES[(cur + 1) % ASSIGNEES.length];
                    act(a.id, "assign", next);
                  }}
                  title={a.assignee ? `Assigned to ${a.assignee}` : "Assign"}
                >
                  {a.assignee ? <UserCheck size={14} style={{ color: "var(--gold)" }} /> : <UserPlus size={14} />}
                </button>
              </div>
            </div>
          ))}
        </div>
      </Panel>
    </>
  );
}