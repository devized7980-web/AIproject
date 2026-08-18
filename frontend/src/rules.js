// Client-side mirror of backend/expert_system.pl (same rules, order and
// thresholds). Used to explain Prolog decisions and drive the what-if
// simulator when the live backend is unreachable.

const PERSON = new Set(["person"]);
const VEHICLE = new Set(["car", "truck", "bus", "motorcycle", "bicycle"]);
const ROAD_DAMAGE = new Set(["pothole", "road crack", "road_crack", "crack", "longitudinal", "transverse", "alligator"]);
const ANIMAL = new Set(["dog", "cat", "cow", "horse", "sheep", "bird"]);
const TRAFFIC = new Set(["traffic light", "stop sign", "parking meter"]);

const close = (d) =>
  (d.ttc_s != null && d.ttc_s <= 1.5) || d.distance_m <= 3.0 || d.ratio >= 0.52;
const medium = (d) =>
  (d.ttc_s != null && d.ttc_s <= 3.0) || d.distance_m <= 7.0 || d.ratio >= 0.32;
const early = (d) =>
  (d.ttc_s != null && d.ttc_s <= 5.0) || d.distance_m <= 14.0 || d.ratio >= 0.17;

export const RULES = [
  { priority: 1, level: "SAFE", action: "object_outside_vehicle_lane",
    label: "Object outside vehicle lane",
    when: "object is not inside the detected driving lane",
    advice: "Recorded but not treated as an immediate threat.",
    code: "decision(safe, object_outside_vehicle_lane) :-\n    observation(_, _, _, false, _, _), !.",
    match: (d) => !d.in_lane },
  { priority: 2, level: "CAUTION", action: "observe_traffic_control",
    label: "Observe traffic control",
    when: "a traffic control (traffic light / stop sign) is in lane with confidence >= 0.30",
    advice: "Requires attention rather than collision braking.",
    code: "decision(caution, observe_traffic_control) :-\n    observation(Object, _, _, true, Confidence, _),\n    traffic_control(Object), Confidence >= 0.30, !.",
    match: (d) => d.in_lane && TRAFFIC.has(d.object) && d.conf >= 0.30 },
  { priority: 3, level: "CRITICAL", action: "brake_immediately_person_ahead",
    label: "Brake immediately — person ahead",
    when: "person in lane with TTC <= 1.5 s, distance <= 3 m, or box filling >= 52% of frame height",
    advice: "BRAKE IMMEDIATELY. A pedestrian is directly in your path.",
    code: "decision(critical, brake_immediately_person_ahead) :-\n    observation(Object, Distance, TTC, true, Confidence, BoxRatio),\n    person(Object), Confidence >= 0.30,\n    (TTC =< 1.5 ; Distance =< 3.0 ; BoxRatio >= 0.52), !.",
    match: (d) => d.in_lane && PERSON.has(d.object) && d.conf >= 0.30 && close(d) },
  { priority: 4, level: "CRITICAL", action: "brake_and_avoid_road_damage",
    label: "Brake and avoid road damage",
    when: "road damage (pothole / crack) in lane with TTC <= 1.5 s, distance <= 3 m, or box >= 52% frame height",
    advice: "BRAKE AND AVOID. The pothole is close enough to damage the wheel or cause loss of control.",
    code: "decision(critical, brake_and_avoid_road_damage) :-\n    observation(Object, Distance, TTC, true, Confidence, BoxRatio),\n    road_damage(Object), Confidence >= 0.30,\n    (TTC =< 1.5 ; Distance =< 3.0 ; BoxRatio >= 0.52), !.",
    match: (d) => d.in_lane && ROAD_DAMAGE.has(d.object) && d.conf >= 0.30 && close(d) },
  { priority: 5, level: "CRITICAL", action: "brake_now_object_too_close",
    label: "Brake now — object too close",
    when: "any object in lane with TTC <= 1.5 s, distance <= 3 m, or box >= 52% frame height",
    advice: "BRAKE NOW. The object is too close to continue at current speed.",
    code: "decision(critical, brake_now_object_too_close) :-\n    observation(_, Distance, TTC, true, Confidence, BoxRatio),\n    Confidence >= 0.30,\n    (TTC =< 1.5 ; Distance =< 3.0 ; BoxRatio >= 0.52), !.",
    match: (d) => d.in_lane && d.conf >= 0.30 && close(d) },
  { priority: 6, level: "WARNING", action: "slow_down_and_increase_following_distance",
    label: "Slow down and increase following distance",
    when: "vehicle in lane with TTC <= 3.0 s, distance <= 7 m, or box >= 32% frame height",
    advice: "SLOW DOWN and increase your following distance.",
    code: "decision(warning, slow_down_and_increase_following_distance) :-\n    observation(Object, Distance, TTC, true, Confidence, BoxRatio),\n    vehicle(Object), Confidence >= 0.30,\n    (TTC =< 3.0 ; Distance =< 7.0 ; BoxRatio >= 0.32), !.",
    match: (d) => d.in_lane && VEHICLE.has(d.object) && d.conf >= 0.30 && medium(d) },
  { priority: 7, level: "WARNING", action: "slow_down_animal_ahead",
    label: "Slow down — animal ahead",
    when: "animal in lane with TTC <= 3.0 s, distance <= 7 m, or box >= 32% frame height",
    advice: "SLOW DOWN. An animal is ahead in your lane.",
    code: "decision(warning, slow_down_animal_ahead) :-\n    observation(Object, Distance, TTC, true, Confidence, BoxRatio),\n    animal(Object), Confidence >= 0.30,\n    (TTC =< 3.0 ; Distance =< 7.0 ; BoxRatio >= 0.32), !.",
    match: (d) => d.in_lane && ANIMAL.has(d.object) && d.conf >= 0.30 && medium(d) },
  { priority: 8, level: "WARNING", action: "slow_down_and_prepare_to_avoid_road_damage",
    label: "Slow down and prepare to avoid road damage",
    when: "road damage in lane with TTC <= 3.0 s, distance <= 7 m, or box >= 32% frame height",
    advice: "SLOW DOWN and prepare to avoid the pothole or crack.",
    code: "decision(warning, slow_down_and_prepare_to_avoid_road_damage) :-\n    observation(Object, Distance, TTC, true, Confidence, BoxRatio),\n    road_damage(Object), Confidence >= 0.30,\n    (TTC =< 3.0 ; Distance =< 7.0 ; BoxRatio >= 0.32), !.",
    match: (d) => d.in_lane && ROAD_DAMAGE.has(d.object) && d.conf >= 0.30 && medium(d) },
  { priority: 9, level: "WARNING", action: "slow_down_hazard_ahead",
    label: "Slow down — hazard ahead",
    when: "any object in lane with TTC <= 3.0 s, distance <= 7 m, or box >= 32% frame height",
    advice: "SLOW DOWN. A hazard is ahead in your lane.",
    code: "decision(warning, slow_down_hazard_ahead) :-\n    observation(_, Distance, TTC, true, Confidence, BoxRatio),\n    Confidence >= 0.30,\n    (TTC =< 3.0 ; Distance =< 7.0 ; BoxRatio >= 0.32), !.",
    match: (d) => d.in_lane && d.conf >= 0.30 && medium(d) },
  { priority: 10, level: "CAUTION", action: "caution_object_in_vehicle_lane",
    label: "Caution — object in vehicle lane",
    when: "object in lane with TTC <= 5.0 s, distance <= 14 m, or box >= 17% frame height",
    advice: "CAUTION. Stay alert and reduce speed slightly.",
    code: "decision(caution, caution_object_in_vehicle_lane) :-\n    observation(_, Distance, TTC, true, Confidence, BoxRatio),\n    Confidence >= 0.30,\n    (TTC =< 5.0 ; Distance =< 14.0 ; BoxRatio >= 0.32), !.",
    match: (d) => d.in_lane && d.conf >= 0.30 && early(d) },
  { priority: 11, level: "SAFE", action: "object_at_safe_distance",
    label: "Object at safe distance",
    when: "no higher-priority rule matched",
    advice: "Road clear. Continue carefully.",
    code: "decision(safe, object_at_safe_distance) :- observation(_, _, _, _, _, _), !.",
    match: () => true },
];

export function decideWithRules(d) {
  const rule = RULES.find((r) => r.match(d)) || RULES[RULES.length - 1];
  return {
    object: d.object,
    confidence: Math.round((d.conf || 0) * 1000) / 1000,
    distance_m: Math.round((d.distance_m || 0) * 100) / 100,
    ttc_s: d.ttc_s == null ? null : Math.round(d.ttc_s * 100) / 100,
    in_lane: d.in_lane,
    box_height_ratio: Math.round((d.ratio || 0) * 1000) / 1000,
    level: rule.level,
    action: rule.action,
    rule: rule.action,
    rule_label: rule.label,
    priority: rule.priority,
    when: rule.when,
    advice: rule.advice,
    code: rule.code,
    engine: "Client rule mirror (expert_system.pl)",
  };
}