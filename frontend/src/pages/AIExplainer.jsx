import React, { useEffect, useMemo, useState } from "react";
import { BrainCircuit, Braces, Check, X, ChevronDown, Sparkles } from "lucide-react";
import { VIDEOS } from "../api.js";
import { prologTrace } from "../api.js";
import { RULES } from "../rules.js";
import { Panel, Tag, levelColor } from "../ui.jsx";
import { useSettingsCtx } from "../App.jsx";

const OBJECTS = ["pothole", "person", "car", "bus", "truck", "bicycle", "traffic light", "alligator cracking"];

function chainFor(trace) {
  const close = trace.ttc_s != null && trace.ttc_s <= 1.5 || trace.distance_m <= 3.0 || trace.box_height_ratio >= 0.52;
  const medium = trace.ttc_s != null && trace.ttc_s <= 3.0 || trace.distance_m <= 7.0 || trace.box_height_ratio >= 0.32;
  const steps = [
    { icon: "obj", text: `${trace.object} detected at ${Math.round(trace.confidence * 100)}% confidence` },
    { icon: "lane", text: trace.in_lane ? "inside the driving lane" : "outside the driving lane" },
    { icon: "dist", text: `${trace.distance_m} m away — ${close ? "dangerously close" : medium ? "closing gap" : "still comfortable"}` },
  ];
  if (trace.ttc_s != null) {
    steps.push({ icon: "ttc", text: `time-to-collision ${trace.ttc_s} s — ${trace.ttc_s <= 1.5 ? "emergency" : trace.ttc_s <= 3 ? "warning" : "manageable"}` });
  }
  return steps;
}

export default function AIExplainer({ videos }) {
  const all = videos.length ? videos : VIDEOS;
  const [settings] = useSettingsCtx();

  const notable = useMemo(() =>
    all.flatMap((v) =>
      v.events.map((e) => ({
        ...e, video_title: v.title, video_id: v.id,
        box_height_ratio: e.boxes?.[0]?.h ? e.boxes[0].h / 100 : 0.2,
      }))
    ).sort((a, b) => b.level.localeCompare(a.level)),
  [all]);

  const [selected, setSelected] = useState(0);
  const [custom, setCustom] = useState(null);
  const [trace, setTrace] = useState(null);

  const pick = notable[selected];
  const baseObs = custom || {
    object: pick.object,
    distance_m: pick.distance_m,
    ttc_s: pick.ttc_s,
    in_lane: true,
    conf: pick.confidence,
    ratio: pick.box_height_ratio,
  };

  useEffect(() => {
    let alive = true;
    prologTrace({
      object: baseObs.object,
      distance_m: baseObs.distance_m,
      ttc_s: baseObs.ttc_s,
      in_lane: baseObs.in_lane,
      confidence: baseObs.conf,
      box_height_ratio: baseObs.ratio,
    }).then((t) => { if (alive) setTrace(t); });
    return () => { alive = false; };
  }, [selected, custom && JSON.stringify(custom)]);

  const obsForRules = { object: baseObs.object, distance_m: baseObs.distance_m, ttc_s: baseObs.ttc_s, in_lane: baseObs.in_lane, conf: baseObs.conf, ratio: baseObs.ratio };
  const fired = RULES.find((r) => r.match(obsForRules));

  const startCustom = () => setCustom({ object: "pothole", distance_m: 3.5, ttc_s: 1.4, in_lane: true, conf: 0.72, ratio: 0.3 });

  return (
    <>
      <h1 className="sw-h1">AI Explainer — Prolog Reasoning</h1>
      <p className="sw-sub">
        YOLO tells you <i>what</i> is ahead; the Prolog expert system tells you <i>what to do about it</i>.
        Pick a real detection and read exactly which rule fired, in what priority, and why.
      </p>

      <div className="sw-2col" style={{ gridTemplateColumns: "minmax(0,1fr) 340px" }}>
        <div className="sw-stack">
          <Panel title="Decision trace" right={<span className="sw-eyebrow">priority → first match wins</span>}>
            <div className="sw-chain">
              <div className="facts">
                <div className="sw-eyebrow" style={{ marginBottom: 10 }}>Observed facts (asserted into knowledge base)</div>
                <div className="fact-grid">
                  <div><span>object</span><b style={{ textTransform: "capitalize" }}>{baseObs.object}</b></div>
                  <div><span>distance</span><b>{baseObs.distance_m} m</b></div>
                  <div><span>ttc</span><b>{baseObs.ttc_s == null ? "—" : `${baseObs.ttc_s} s`}</b></div>
                  <div><span>in_lane</span><b>{baseObs.in_lane ? "true" : "false"}</b></div>
                  <div><span>confidence</span><b>{Math.round(baseObs.conf * 100)}%</b></div>
                  <div><span>box_ratio</span><b>{baseObs.ratio.toFixed(2)}</b></div>
                </div>
              </div>

              <div className="chain">
                {chainFor(trace || { object: baseObs.object, confidence: baseObs.conf, in_lane: baseObs.in_lane, distance_m: baseObs.distance_m, ttc_s: baseObs.ttc_s, box_height_ratio: baseObs.ratio }).map((s, i) => (
                  <div className="link" key={i}>
                    <div className="step"><span className="n">{i + 1}</span>{s.text}</div>
                    {i < 3 && <div className="arrow">↓</div>}
                  </div>
                ))}
                <div className="link">
                  <div className="step verdict" style={{ "--c": levelColor(trace?.level || "SAFE") }}>
                    <span className="n">✓</span>
                    <b>{trace ? `${trace.level} — ${trace.rule_label}` : "…"}</b>
                  </div>
                </div>
              </div>
            </div>

            <div className="sw-note" style={{ marginTop: 14 }}>
              <BrainCircuit size={14} style={{ color: "var(--gold)", verticalAlign: -2, marginRight: 6 }} />
              Evaluated by <b>{trace?.engine || "…"}</b>. Rules are ordered: the first rule whose
              conditions all hold wins, so pedestrians outrank vehicles outrank generic hazards.
            </div>
          </Panel>

          <Panel title="Rule that fired" right={trace && <Tag level={trace.level} />}>
            {trace && (
              <>
                <div className="sw-rule-head">
                  <span className="prio">P{trace.priority}</span>
                  <div className="grow">
                    <b>{trace.rule_label}</b>
                    <small>{trace.action.replace(/_/g, " ")}</small>
                  </div>
                  <Sparkles size={15} style={{ color: "var(--gold)" }} />
                </div>
                <div className="sw-rule-when">
                  <div className="sw-eyebrow" style={{ marginBottom: 6 }}>Rule fires when</div>
                  {trace.when}
                </div>
                <pre className="sw-prolog">{trace.code}</pre>
                <div className="sw-advice">
                  <b style={{ color: levelColor(trace.level) }}>{trace.advice}</b>
                </div>
              </>
            )}
          </Panel>

          <Panel title="Why not the other rules?" right={<span className="sw-eyebrow">evaluation</span>}>
            <div className="sw-rulecheck">
              {RULES.map((r) => {
                const ok = r.match(obsForRules);
                const isFired = fired && r.priority === fired.priority;
                return (
                  <div key={r.priority} data-fired={isFired ? "1" : "0"}>
                    <span className="p">P{r.priority}</span>
                    <span className="grow sw-truncate">{r.action.replace(/_/g, " ")}</span>
                    {isFired
                      ? <span className="st fired"><Check size={11} /> FIRED</span>
                      : <span className="st skip"><X size={11} /> skipped</span>}
                  </div>
                );
              })}
            </div>
            <div className="sw-eyebrow" style={{ marginTop: 12 }}>
              Rules after the fired one are never reached because the fired rule contains a cut (!).
            </div>
          </Panel>
        </div>

        <div className="sw-stack">
          <Panel title="Real detections" right={<span className="sw-eyebrow">{notable.length} events</span>}>
            <div className="sw-evlist">
              {notable.map((e, i) => (
                <button key={i} data-on={i === selected ? "1" : "0"} onClick={() => { setSelected(i); setCustom(null); }}>
                  <Tag level={e.level} />
                  <span className="grow sw-truncate">
                    <b style={{ textTransform: "capitalize" }}>{e.object}</b>
                    <small>{e.video_title.split("—")[0]} · {e.t}</small>
                  </span>
                  <ChevronDown size={13} style={{ color: "var(--muted)", transform: i === selected ? "rotate(180deg)" : "none" }} />
                </button>
              ))}
            </div>
          </Panel>

          <Panel title="Try a custom observation" right={<span className="sw-eyebrow">live</span>}>
            {!custom ? (
              <button className="sw-action" onClick={startCustom} style={{ marginBottom: 12 }}>
                <h3>Build one manually <Braces size={13} style={{ color: "var(--gold)" }} /></h3>
                <p>Set an object, distance, TTC and lane position and watch which Prolog rule fires.</p>
              </button>
            ) : (
              <div className="sw-custom">
                <div className="fld">
                  <label>Object</label>
                  <select value={custom.object} onChange={(e) => setCustom({ ...custom, object: e.target.value })}>
                    {OBJECTS.map((o) => <option key={o}>{o}</option>)}
                  </select>
                </div>
                <div className="fld"><label>Distance (m)</label><input type="number" min="0.5" max="90" step="0.5" value={custom.distance_m} onChange={(e) => setCustom({ ...custom, distance_m: +e.target.value })} /></div>
                <div className="fld"><label>TTC (s)</label><input type="number" min="0" max="20" step="0.1" value={custom.ttc_s ?? ""} onChange={(e) => setCustom({ ...custom, ttc_s: e.target.value === "" ? null : +e.target.value })} /></div>
                <div className="fld"><label>In lane</label>
                  <select value={custom.in_lane} onChange={(e) => setCustom({ ...custom, in_lane: e.target.value === "true" })}>
                    <option value="true">true</option><option value="false">false</option>
                  </select>
                </div>
                <div className="fld"><label>Confidence</label><input type="number" min="0.1" max="0.99" step="0.01" value={custom.conf} onChange={(e) => setCustom({ ...custom, conf: +e.target.value })} /></div>
                <button className="sw-reportbtn" onClick={() => setCustom(null)}>Reset to real events</button>
              </div>
            )}
          </Panel>
        </div>
      </div>
    </>
  );
}