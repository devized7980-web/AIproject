:- dynamic observation/6.

% observation(Object, DistanceMetres, TTCSeconds, InLane, Confidence, BoxHeightRatio).

traffic_control(traffic_light).
traffic_control(stop_sign).
traffic_control(parking_meter).

person(person).
vehicle(car). vehicle(truck). vehicle(bus). vehicle(motorcycle). vehicle(bicycle).
road_damage(pothole). road_damage(road_crack). road_damage(crack).
animal(dog). animal(cat). animal(cow). animal(horse). animal(sheep). animal(bird).

% Objects beside the vehicle path are recorded but not treated as immediate collision threats.
decision(safe, object_outside_vehicle_lane) :-
    observation(_, _, _, false, _, _), !.

% Traffic controls require attention rather than collision braking rules.
decision(caution, observe_traffic_control) :-
    observation(Object, _, _, true, Confidence, _),
    traffic_control(Object), Confidence >= 0.30, !.

% Person-specific emergency rule.
decision(critical, brake_immediately_person_ahead) :-
    observation(Object, Distance, TTC, true, Confidence, BoxRatio),
    person(Object), Confidence >= 0.30,
    (TTC =< 1.5 ; Distance =< 3.0 ; BoxRatio >= 0.52), !.

% Road-damage emergency rule.
decision(critical, brake_and_avoid_road_damage) :-
    observation(Object, Distance, TTC, true, Confidence, BoxRatio),
    road_damage(Object), Confidence >= 0.30,
    (TTC =< 1.5 ; Distance =< 3.0 ; BoxRatio >= 0.52), !.

% Generic emergency collision rule.
decision(critical, brake_now_object_too_close) :-
    observation(_, Distance, TTC, true, Confidence, BoxRatio),
    Confidence >= 0.30,
    (TTC =< 1.5 ; Distance =< 3.0 ; BoxRatio >= 0.52), !.

% Vehicle following-distance warning.
decision(warning, slow_down_and_increase_following_distance) :-
    observation(Object, Distance, TTC, true, Confidence, BoxRatio),
    vehicle(Object), Confidence >= 0.30,
    (TTC =< 3.0 ; Distance =< 7.0 ; BoxRatio >= 0.32), !.

% Animal warning.
decision(warning, slow_down_animal_ahead) :-
    observation(Object, Distance, TTC, true, Confidence, BoxRatio),
    animal(Object), Confidence >= 0.30,
    (TTC =< 3.0 ; Distance =< 7.0 ; BoxRatio >= 0.32), !.

% Road-damage warning.
decision(warning, slow_down_and_prepare_to_avoid_road_damage) :-
    observation(Object, Distance, TTC, true, Confidence, BoxRatio),
    road_damage(Object), Confidence >= 0.30,
    (TTC =< 3.0 ; Distance =< 7.0 ; BoxRatio >= 0.32), !.

% Generic warning.
decision(warning, slow_down_hazard_ahead) :-
    observation(_, Distance, TTC, true, Confidence, BoxRatio),
    Confidence >= 0.30,
    (TTC =< 3.0 ; Distance =< 7.0 ; BoxRatio >= 0.32), !.

% Early caution.
decision(caution, caution_object_in_vehicle_lane) :-
    observation(_, Distance, TTC, true, Confidence, BoxRatio),
    Confidence >= 0.30,
    (TTC =< 5.0 ; Distance =< 14.0 ; BoxRatio >= 0.17), !.

% Safe default.
decision(safe, object_at_safe_distance) :- observation(_, _, _, _, _, _), !.