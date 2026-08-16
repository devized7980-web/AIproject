import React from "react";
import { LEVELS } from "../data.js";
import { Panel, Tag, levelColor } from "../ui.jsx";

const PIPELINE = [
  { n: "01", h: "Capture", p: "Dashcam clips are read frame by frame with OpenCV. Detection runs every second frame; the boxes in between are smoothed so playback stays steady." },
  { n: "02", h: "Detect", p: "YOLO11n handles the common classes and a custom-trained model adds road damage: potholes and cracking types (longitudinal, transverse, alligator). ByteTrack keeps an ID on each object across frames." },
  { n: "03", h: "Measure", p: "Box width against a known object width gives a distance estimate. Change in distance over time gives closing speed and time-to-collision." },
  { n: "04", h: "Decide", p: "Each observation is asserted into a Prolog knowledge base. Rules fire in priority order and return one level and one action, so the advice is traceable to a rule." },
  { n: "05", h: "Alert", p: "The frame is annotated, spoken advice plays once per object, and any critical frame is written to the incidents folder with a CSV row and a summary." },
];

const MEANING = {
  SAFE: "Outside the lane or far enough away",
  CAUTION: "In lane, still comfortable",
  WARNING: "The gap is shrinking — slow down",
  CRITICAL: "Brake now",
};

const STACK = ["Python", "OpenCV", "Ultralytics YOLO11", "Custom best.pt", "ByteTrack",
  "SWI-Prolog", "pyswip", "pyttsx3", "React", "Vite"];

const SETTINGS = [
  ["Confidence", "0.30"],
  ["IoU", "0.50"],
  ["Image size", "640 px"],
  ["Detection interval", "every 2nd frame"],
  ["Focal length", "700 px assumed"],
];

export default function About() {
  return (
    <>
      <h1 className="sw-h1">About Safeway AI</h1>
      <p className="sw-sub">
        A driver-assistance benchmark built as a university final year project. It watches dashcam footage,
        finds what is in the vehicle's path, and explains what a driver should do about it.
      </p>

      <Panel title="How a frame becomes advice" right={<span className="sw-eyebrow">Pipeline</span>}>
        <div className="sw-pipe">
          {PIPELINE.map((s) => (
            <div className="sw-stage-card" key={s.n}>
              <div className="n">{s.n}</div>
              <h3>{s.h}</h3>
              <p>{s.p}</p>
            </div>
          ))}
        </div>
      </Panel>

      <div className="sw-2col-even" style={{ marginTop: 16 }}>
        <Panel title="Why a rule engine, not another network">
          <p className="sw-prose">
            A neural net can tell you a person is 3 metres ahead. It cannot tell you why it decided to brake.
            The Prolog layer turns measurements into a decision that can be read, argued with, and corrected
            by editing one line.
          </p>
          <div className="sw-note">
            A person in lane with TTC under 1.5 s, or closer than 3 m, or filling more than half the frame
            height, fires <code>brake_immediately_person_ahead</code>. Rules are ordered, so the first match
            wins and pedestrians outrank vehicles.
          </div>
          <div style={{ marginTop: 16 }}>
            <div className="sw-eyebrow" style={{ marginBottom: 8 }}>Risk levels</div>
            {LEVELS.map((l) => (
              <div className="sw-row" key={l}>
                <i className="sw-dot" style={{ background: levelColor(l) }} />
                <div className="grow" style={{ fontSize: 12.5 }}>{MEANING[l]}</div>
                <Tag level={l} />
              </div>
            ))}
          </div>
        </Panel>

        <div className="sw-stack">
          <Panel title="Built with">
            <div className="sw-chips">
              {STACK.map((t) => <span key={t}>{t}</span>)}
            </div>
            <div style={{ marginTop: 16 }}>
              <div className="sw-eyebrow" style={{ marginBottom: 8 }}>Run settings</div>
              {SETTINGS.map(([k, v]) => (
                <div className="sw-row" key={k}>
                  <div className="grow" style={{ fontSize: 12.5 }}>{k}</div>
                  <div className="num" style={{ fontSize: 12.5, color: "var(--muted)" }}>{v}</div>
                </div>
              ))}
            </div>
          </Panel>

          <Panel title="Known limits">
            <ul className="sw-limits">
              <li>Distance and TTC come from a single camera with an assumed focal length. Calibrate before quoting any figure as accurate.</li>
              <li>Heavy rain and night footage lower confidence, so small objects drop out between frames.</li>
              <li>The lane region is a fixed trapezoid, not a detected lane, so sharp curves misclassify in-lane objects.</li>
              <li>This is a research benchmark. It is not certified for use while driving.</li>
            </ul>
          </Panel>
        </div>
      </div>
    </>
  );
}
