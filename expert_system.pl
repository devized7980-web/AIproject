:- dynamic observation/6.

% observation(Object, DistanceMetres, TTCSeconds, InLane, Confidence, BoxHeightRatio).

traffic_control(traffic_light).
traffic_control(stop_sign).
traffic_control(parking_meter).

person(person).
vehicle(car). vehicle(truck). vehicle(bus). vehicle(motorcycle). vehicle(bicycle).
road_damage(pothole). road_damage(road_crack). road_damage(crack).
road_damage(longitudinal). road_damage(transverse). road_damage(alligator).
animal(dog). animal(cat). animal(cow). animal(horse). animal(sheep). animal(bird).
obstacle(tree). obstacle(fallen_tree). obstacle(obstacle).
obstacle(cone). obstacle(barrier). obstacle(debris). obstacle(log). obstacle(branch).

% ---------------------------------------------------------------------------
% RULE PRIORITIES
%
% Several candidate rules can be true at once. Rather than letting clause
% order silently decide, every rule carries an explicit numeric priority and
% the winner is resolved by comparing them (see resolve_decision/5 below).
%
% Risk levels rank CRITICAL > WARNING > CAUTION > SAFE; priorities are laid
% out so that ordering holds, while still allowing rules within a level to be
% ranked against each other (a person in the lane outranks a generic object).
% ---------------------------------------------------------------------------
% Specific hazard rules outrank the generic collision rule: they prescribe a
% different action (maneuver/avoid) than a plain brake, so a pothole or fallen
% tree must not be reported as a nondescript "object too close".
rule_priority(brake_immediately_person_ahead, 100).
rule_priority(brake_and_avoid_road_damage,     95).
rule_priority(brake_and_avoid_obstacle,        94).
rule_priority(brake_now_object_too_close,      90).

rule_priority(animal_warning,                  65).
rule_priority(road_damage_warning,             62).
rule_priority(obstacle_warning,                61).
rule_priority(vehicle_following_distance,      60).
rule_priority(generic_warning,                 55).

rule_priority(observe_traffic_control,         45).
rule_priority(caution_object_in_vehicle_lane,  40).

rule_priority(object_outside_vehicle_lane,     25).
rule_priority(object_at_safe_distance,         20).

risk_rank(critical, 4).
risk_rank(warning,  3).
risk_rank(caution,  2).
risk_rank(safe,     1).

% ---------------------------------------------------------------------------
% CANDIDATE RULES
%
% These are deliberately written WITHOUT cuts so that every rule matching the
% current observation is collected. Conflict resolution happens in
% resolve_decision/5, not through clause ordering.
% ---------------------------------------------------------------------------

% Objects beside the vehicle path are recorded but not treated as immediate collision threats.
candidate(safe, object_outside_vehicle_lane, object_outside_vehicle_lane, 'Object is outside the vehicle lane and not an immediate collision threat.') :-
    observation(_, _, _, false, _, _).

% Traffic controls require attention rather than collision braking rules.
candidate(caution, observe_traffic_control, observe_traffic_control, 'Traffic control detected; monitor for instructions or stops.') :-
    observation(Object, _, _, true, Confidence, _),
    traffic_control(Object), Confidence >= 0.30.

% Person-specific emergency rule.
candidate(critical, brake_immediately_person_ahead, brake_immediately_person_ahead, 'Immediate braking required due to person in lane.') :-
    observation(Object, Distance, TTC, true, Confidence, BoxRatio),
    person(Object), Confidence >= 0.30,
    once((TTC =< 1.5 ; Distance =< 3.0 ; BoxRatio >= 0.52)).

% Road-damage emergency rule.
candidate(critical, brake_and_avoid_road_damage, brake_and_avoid_road_damage, 'Immediate maneuver to avoid road damage is required.') :-
    observation(Object, Distance, TTC, true, Confidence, BoxRatio),
    road_damage(Object), Confidence >= 0.30,
    once((TTC =< 1.5 ; Distance =< 3.0 ; BoxRatio >= 0.52)).

% Obstacle emergency rule (fallen tree, debris, barrier, etc. blocking the lane).
candidate(critical, brake_and_avoid_obstacle, brake_and_avoid_obstacle, 'Immediate maneuver to avoid obstacle in lane is required.') :-
    observation(Object, Distance, TTC, true, Confidence, BoxRatio),
    obstacle(Object), Confidence >= 0.30,
    once((TTC =< 1.5 ; Distance =< 3.0 ; BoxRatio >= 0.52)).

% Generic emergency collision rule.
candidate(critical, brake_now_object_too_close, brake_now_object_too_close, 'Object detected very close; immediate braking advised.') :-
    observation(_, Distance, TTC, true, Confidence, BoxRatio),
    Confidence >= 0.30,
    once((TTC =< 1.5 ; Distance =< 3.0 ; BoxRatio >= 0.52)).

% Vehicle following-distance warning.
candidate(warning, slow_down_and_increase_following_distance, vehicle_following_distance, 'Reduce speed and increase following distance to maintain a safe gap.') :-
    observation(Object, Distance, TTC, true, Confidence, BoxRatio),
    vehicle(Object), Confidence >= 0.30,
    once((TTC =< 3.0 ; Distance =< 7.0 ; BoxRatio >= 0.32)).

% Animal warning.
candidate(warning, slow_down_animal_ahead, animal_warning, 'Animal detected near lane; slow down and proceed cautiously.') :-
    observation(Object, Distance, TTC, true, Confidence, BoxRatio),
    animal(Object), Confidence >= 0.30,
    once((TTC =< 3.0 ; Distance =< 7.0 ; BoxRatio >= 0.32)).

% Road-damage warning.
candidate(warning, slow_down_and_prepare_to_avoid_road_damage, road_damage_warning, 'Road damage ahead; slow down and prepare to avoid.') :-
    observation(Object, Distance, TTC, true, Confidence, BoxRatio),
    road_damage(Object), Confidence >= 0.30,
    once((TTC =< 3.0 ; Distance =< 7.0 ; BoxRatio >= 0.32)).

% Obstacle warning.
candidate(warning, slow_down_and_prepare_to_avoid_obstacle, obstacle_warning, 'Obstacle ahead in lane; slow down and prepare to avoid.') :-
    observation(Object, Distance, TTC, true, Confidence, BoxRatio),
    obstacle(Object), Confidence >= 0.30,
    once((TTC =< 3.0 ; Distance =< 7.0 ; BoxRatio >= 0.32)).

% Generic warning.
candidate(warning, slow_down_hazard_ahead, generic_warning, 'Slow down: hazard detected ahead in vehicle lane.') :-
    observation(_, Distance, TTC, true, Confidence, BoxRatio),
    Confidence >= 0.30,
    once((TTC =< 3.0 ; Distance =< 7.0 ; BoxRatio >= 0.32)).

% Early caution.
candidate(caution, caution_object_in_vehicle_lane, caution_object_in_vehicle_lane, 'Object detected in vehicle lane; exercise caution.') :-
    observation(_, Distance, TTC, true, Confidence, BoxRatio),
    Confidence >= 0.30,
    once((TTC =< 5.0 ; Distance =< 14.0 ; BoxRatio >= 0.17)).

% Safe default.
candidate(safe, object_at_safe_distance, object_at_safe_distance, 'Object is at a safe distance.') :-
    observation(_, _, _, _, _, _).

% ---------------------------------------------------------------------------
% CONFLICT RESOLUTION
%
% triggered_rule/5 exposes every rule that fired, ranked. resolve_decision/5
% picks the single winner: highest risk level first, then highest rule
% priority within that level. decision/4 keeps the original 4-argument
% interface so existing callers are unaffected.
% ---------------------------------------------------------------------------

triggered_rule(Level, Action, RuleID, Explanation, Priority) :-
    candidate(Level, Action, RuleID, Explanation),
    rule_priority(RuleID, Priority).

% Rank key: risk level dominates, rule priority breaks ties within a level.
ranked_candidate(key(RiskRank, Priority), Level, Action, RuleID, Explanation) :-
    triggered_rule(Level, Action, RuleID, Explanation, Priority),
    risk_rank(Level, RiskRank).

resolve_decision(Level, Action, RuleID, Explanation, Priority) :-
    findall(
        Key-decision(L, A, R, E),
        ranked_candidate(Key, L, A, R, E),
        Pairs
    ),
    Pairs \= [],
    sort(1, @>=, Pairs, [key(_, Priority)-decision(Level, Action, RuleID, Explanation)|_]).

% Backwards-compatible entry point used by the Python pipeline.
decision(Level, Action, RuleID, Explanation) :-
    resolve_decision(Level, Action, RuleID, Explanation, _).

% Winning decision plus its priority, for the dashboard decision trace.
decision_with_priority(Level, Action, RuleID, Explanation, Priority) :-
    resolve_decision(Level, Action, RuleID, Explanation, Priority).

% Every rule that fired for the current observation, for inspection. The
% winner is included; callers identify it by matching the winning rule id.
all_triggered_rules(Rules) :-
    findall(
        rule(Priority, Level, RuleID, Explanation),
        triggered_rule(Level, _, RuleID, Explanation, Priority),
        Unsorted
    ),
    sort(1, @>=, Unsorted, Rules).
