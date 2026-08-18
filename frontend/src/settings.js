import { useEffect, useState } from "react";

const KEY = "safeway.settings.v1";

export const DEFAULTS = {
  theme: "dark",          // dark | light | colorblind
  motion: "normal",       // low | normal | high
  sound: true,            // alert sounds
  threshold: 0.3,         // detection confidence threshold
  density: "comfortable", // comfortable | compact
  showFps: false,
  debug: false,
  reduced: false,
};

export function loadSettings() {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return { ...DEFAULTS, ...JSON.parse(raw) };
  } catch {
    /* ignore */
  }
  return { ...DEFAULTS };
}

export function saveSettings(s) {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    /* ignore */
  }
}

export function applySettings(s) {
  const root = document.documentElement;
  root.dataset.theme = s.theme;
  root.dataset.motion = s.motion;
  root.dataset.density = s.density;
  root.dataset.debug = s.debug ? "1" : "0";
  root.dataset.reduced = s.reduced ? "1" : "0";
}

export function useSettings() {
  const [settings, setSettings] = useState(loadSettings);
  useEffect(() => {
    applySettings(settings);
    saveSettings(settings);
  }, [settings]);
  const update = (patch) => setSettings((s) => ({ ...s, ...patch }));
  return [settings, update];
}