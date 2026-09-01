// Client fallback mirror of expert_system.pl. Priorities and thresholds are
// intentionally identical to the Python and Prolog engines.
const PERSON = new Set(["person"]);
const VEHICLE = new Set(["car", "truck", "bus", "motorcycle", "bicycle"]);
const ROAD_DAMAGE = new Set(["pothole", "road crack", "road_crack", "crack", "longitudinal", "transverse", "alligator"]);
const ANIMAL = new Set(["dog", "cat", "cow", "horse", "sheep", "bird"]);
const TRAFFIC = new Set(["traffic light", "stop sign", "parking meter"]);
const OBSTACLE = new Set(["tree", "fallen_tree", "fallen tree", "obstacle", "cone", "barrier", "debris", "log", "branch"]);
const close = (d) => (d.ttc_s != null && d.ttc_s <= 1.5) || d.distance_m <= 3 || d.ratio >= 0.52;
const medium = (d) => (d.ttc_s != null && d.ttc_s <= 3) || d.distance_m <= 7 || d.ratio >= 0.32;
const early = (d) => (d.ttc_s != null && d.ttc_s <= 5) || d.distance_m <= 14 || d.ratio >= 0.17;
const ok = (d) => d.conf >= 0.30;

const R = (priority, level, rule, action, label, when, advice, match) => ({ priority, level, rule, action, label, when, advice, match });
export const RULES = [
  R(25, "SAFE", "object_outside_vehicle_lane", "OUTSIDE VEHICLE LANE", "Object outside vehicle lane", "object is not inside the detected driving lane", "Recorded but not treated as an immediate threat.", d => !d.in_lane),
  R(45, "CAUTION", "observe_traffic_control", "OBSERVE TRAFFIC CONTROL", "Observe traffic control", "a traffic control is in lane with confidence >= 0.30", "Requires attention rather than collision braking.", d => d.in_lane && TRAFFIC.has(d.object) && ok(d)),
  R(100, "CRITICAL", "brake_immediately_person_ahead", "BRAKE IMMEDIATELY PERSON AHEAD", "Brake immediately - person ahead", "person in lane with a close TTC, distance or box", "BRAKE IMMEDIATELY. A pedestrian is directly in your path.", d => d.in_lane && PERSON.has(d.object) && ok(d) && close(d)),
  R(95, "CRITICAL", "brake_and_avoid_road_damage", "BRAKE AND AVOID ROAD DAMAGE", "Brake and avoid road damage", "road damage in lane with a close TTC, distance or box", "BRAKE AND AVOID. Road damage is close enough to cause loss of control.", d => d.in_lane && ROAD_DAMAGE.has(d.object) && ok(d) && close(d)),
  R(94, "CRITICAL", "brake_and_avoid_obstacle", "BRAKE AND AVOID OBSTACLE", "Brake and avoid obstacle", "obstacle in lane with a close TTC, distance or box", "BRAKE AND AVOID. An obstacle is blocking the lane.", d => d.in_lane && OBSTACLE.has(d.object) && ok(d) && close(d)),
  R(90, "CRITICAL", "brake_now_object_too_close", "BRAKE NOW OBJECT TOO CLOSE", "Brake now - object too close", "any object in lane with a close TTC, distance or box", "BRAKE NOW. The object is too close to continue at current speed.", d => d.in_lane && ok(d) && close(d)),
  R(65, "WARNING", "animal_warning", "SLOW DOWN ANIMAL AHEAD", "Slow down - animal ahead", "animal in lane with a medium TTC, distance or box", "SLOW DOWN. An animal is ahead in your lane.", d => d.in_lane && ANIMAL.has(d.object) && ok(d) && medium(d)),
  R(62, "WARNING", "road_damage_warning", "SLOW DOWN AND PREPARE TO AVOID ROAD DAMAGE", "Slow down and prepare to avoid road damage", "road damage in lane with a medium TTC, distance or box", "SLOW DOWN and prepare to avoid road damage.", d => d.in_lane && ROAD_DAMAGE.has(d.object) && ok(d) && medium(d)),
  R(61, "WARNING", "obstacle_warning", "SLOW DOWN AND PREPARE TO AVOID OBSTACLE", "Slow down and prepare to avoid obstacle", "obstacle in lane with a medium TTC, distance or box", "SLOW DOWN and prepare to avoid the obstacle.", d => d.in_lane && OBSTACLE.has(d.object) && ok(d) && medium(d)),
  R(60, "WARNING", "vehicle_following_distance", "SLOW DOWN AND INCREASE FOLLOWING DISTANCE", "Slow down and increase following distance", "vehicle in lane with a medium TTC, distance or box", "SLOW DOWN and increase your following distance.", d => d.in_lane && VEHICLE.has(d.object) && ok(d) && medium(d)),
  R(55, "WARNING", "generic_warning", "SLOW DOWN HAZARD AHEAD", "Slow down - hazard ahead", "any object in lane with a medium TTC, distance or box", "SLOW DOWN. A hazard is ahead in your lane.", d => d.in_lane && ok(d) && medium(d)),
  R(40, "CAUTION", "caution_object_in_vehicle_lane", "CAUTION OBJECT IN VEHICLE LANE", "Caution - object in vehicle lane", "object in lane with an early TTC, distance or box", "CAUTION. Stay alert and reduce speed slightly.", d => d.in_lane && ok(d) && early(d)),
  R(20, "SAFE", "object_at_safe_distance", "OBJECT AT SAFE DISTANCE", "Object at safe distance", "no higher-priority rule matched", "Road clear. Continue carefully.", () => true),
];

export function decideWithRules(d) {
  const rule = RULES.find(r => r.match(d)) || RULES[RULES.length - 1];
  return {
    object: d.object, confidence: Math.round((d.conf || 0) * 1000) / 1000,
    distance_m: Math.round((d.distance_m || 0) * 100) / 100,
    ttc_s: d.ttc_s == null ? null : Math.round(d.ttc_s * 100) / 100,
    in_lane: d.in_lane, box_height_ratio: Math.round((d.ratio || 0) * 1000) / 1000,
    level: rule.level, action: rule.action, rule: rule.rule, rule_label: rule.label,
    priority: rule.priority, when: rule.when, advice: rule.advice,
    code: `${rule.level.toLowerCase()}:${rule.rule}`, engine: "Python fallback",
  };
}
