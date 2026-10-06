"""Does the router produce drive times a driver would recognise?

The bug this file exists to prevent
-----------------------------------
Routing on ``length / posted_speed`` gave the 8 km UTC to Tustin Legacy trip a
travel time of 5.6 minutes -- an implied 78 km/h average along Jamboree Road,
through fifteen signalised crossings, without stopping once. Every number the
service reported downstream of that was wrong by the same factor, and none of
them looked wrong: the payload was well-formed, the path was sensible, and the
latency was excellent.

So the assertions here are about *magnitude*, which is the property no other
test in the suite pins down. They come in three layers:

* the delay model in the engine, on a synthetic intersection where every
  quantity is known exactly;
* the calibration constants and the classification rules, on tag fixtures;
* the end-to-end benchmark, on the real Irvine network -- an arterial trip
  across the city must land in the window a driver would actually experience,
  and the same trip on the old uncalibrated weights must not.

Retuning
--------
The constants live in ``src.data.calibration``. If a change there moves the
benchmark outside its window, one of the two is wrong -- and the window is the
one with ground truth behind it (Google/Apple typical off-peak times for this
pair are 13-16 minutes).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip(
    "traffic_engine", reason="native extension not built; run `pip install -e .`"
)

import traffic_engine  # noqa: E402

from src.data.calibration import (  # noqa: E402
    ARTERIAL_CRUISE_CAP_KPH,
    MINOR_CRUISE_CAP_KPH,
    SIGNAL_DELAY_SEC,
    TURN_PENALTY_SEC,
    YIELD_DELAY_SEC,
    RoadClass,
    cap_speeds_kph,
    classify_intersections,
    cruise_speed_kph,
    normalize_highway,
    polyline_bearings,
    road_class,
)

# ---------------------------------------------------------------------------
# The benchmark trip
# ---------------------------------------------------------------------------
# University Town Center, at the UCI campus edge, to Tustin Legacy at the north
# of the city: Campus/Carlson to Jamboree to Barranca, all arterial, and the
# trip the uncalibrated engine claimed to do in 5.6 minutes.
UTC = (33.6497, -117.8409)
TUSTIN_LEGACY = (33.6960, -117.8320)

# What the trip actually takes off-peak, with a minute of slack each side.
BENCHMARK_MIN_SEC = 12 * 60.0
BENCHMARK_MAX_SEC = 16 * 60.0

# The trip is ~8 km; the router may pick a slightly longer path if it saves
# signals, which is the point of the model, so this is a band and not a value.
BENCHMARK_MIN_M = 7_000.0
BENCHMARK_MAX_M = 10_000.0


# ---------------------------------------------------------------------------
# Layer 1: the delay model, on a synthetic intersection
# ---------------------------------------------------------------------------
# A plus junction: node 0 in the middle, four arms 200 m out. Bearings are
# unambiguous, so every turn classification has one right answer.
#
#            1 (N)
#            |
#   4 (W) -- 0 -- 2 (E)
#            |
#            3 (S)
CENTER = (33.6500, -117.8000)
ARM_OFFSET_DEG = 0.0018   # ~200 m
ARMS = {
    1: (CENTER[0] + ARM_OFFSET_DEG, CENTER[1]),   # north
    2: (CENTER[0], CENTER[1] + ARM_OFFSET_DEG),   # east
    3: (CENTER[0] - ARM_OFFSET_DEG, CENTER[1]),   # south
    4: (CENTER[0], CENTER[1] - ARM_OFFSET_DEG),   # west
}
ARM_TRAVEL_SEC = 12.0
ARM_LENGTH_M = 200.0


def _junction(
    priority_sec: float = 0.0,
    yield_sec: float = 0.0,
    turns: traffic_engine.TurnPenalties | None = None,
    road_class_of: dict[int, RoadClass] | None = None,
):
    """The plus junction, with the delay model configured as asked.

    Edge ids are deterministic: ``2 * (arm - 1)`` runs centre -> arm and
    ``2 * (arm - 1) + 1`` runs arm -> centre.
    """
    graph = traffic_engine.Graph(5)
    graph.add_node(0, *CENTER)
    for node, (lat, lon) in ARMS.items():
        graph.add_node(node, lat, lon)

    for arm in ARMS:
        out_id = 2 * (arm - 1)
        in_id = out_id + 1
        graph.add_edge(out_id, 0, arm, ARM_LENGTH_M, ARM_TRAVEL_SEC)
        graph.add_edge(in_id, arm, 0, ARM_LENGTH_M, ARM_TRAVEL_SEC)

    # Only the centre is an intersection; the arms are where trips start and
    # end, so a delay there would be charged to the wrong thing.
    graph.set_node_delays(
        [priority_sec if node == 0 else 0.0 for node in range(5)],
        [yield_sec if node == 0 else 0.0 for node in range(5)],
    )
    if road_class_of is not None:
        classes = [int(RoadClass.ARTERIAL)] * 8
        for edge_id, klass in road_class_of.items():
            classes[edge_id] = int(klass)
        graph.set_edge_road_classes(classes)
    if turns is not None:
        graph.set_turn_penalties(turns)
    graph.finalize()
    return graph


def _arm_to_arm(graph, start_arm: int, end_arm: int):
    return graph.find_shortest_path(start_arm, end_arm)


def test_a_bare_graph_charges_nothing_extra():
    """No calibration configured means the engine is a plain travel-time router.

    The default has to stay this way: the engine's own tests, the NetworkX
    cross-check in ``src/router/test_cpp_bridge.py`` and every synthetic graph
    in the suite assume a route costs exactly the sum of its edges.
    """
    graph = _junction()

    assert graph.has_intersection_costs is False
    result = _arm_to_arm(graph, 4, 2)
    assert result.total_duration_sec == pytest.approx(2 * ARM_TRAVEL_SEC)


def test_priority_delay_is_charged_once_per_interior_intersection():
    """A signal costs the driver who crosses it, and only them.

    Origin and destination are ends of the trip, not junctions driven through,
    so a two-edge route through one signal pays exactly one delay.
    """
    graph = _junction(priority_sec=SIGNAL_DELAY_SEC)

    assert graph.has_intersection_costs is True
    result = _arm_to_arm(graph, 4, 2)
    assert result.total_duration_sec == pytest.approx(
        2 * ARM_TRAVEL_SEC + SIGNAL_DELAY_SEC
    )


def test_no_delay_is_charged_at_the_origin_or_the_destination():
    """Pulling away from the kerb is not driving through an intersection."""
    # Delay on the *arm* nodes, which this route starts and ends at.
    graph = traffic_engine.Graph(5)
    graph.add_node(0, *CENTER)
    for node, (lat, lon) in ARMS.items():
        graph.add_node(node, lat, lon)
    for arm in ARMS:
        graph.add_edge(2 * (arm - 1), 0, arm, ARM_LENGTH_M, ARM_TRAVEL_SEC)
        graph.add_edge(2 * (arm - 1) + 1, arm, 0, ARM_LENGTH_M, ARM_TRAVEL_SEC)
    graph.set_node_delays([60.0] * 5, [])
    graph.finalize()

    result = _arm_to_arm(graph, 4, 2)
    # One interior node (the centre), so one delay -- not three.
    assert result.total_duration_sec == pytest.approx(2 * ARM_TRAVEL_SEC + 60.0)


def test_yield_delay_spares_the_arterial_through_movement():
    """A stop sign on a side street does not slow the road it meets.

    This is the distinction that decides whether the benchmark lands: Irvine's
    arterials pass hundreds of minor junctions, and charging every one of them
    would put a 14-minute trip at 19.
    """
    minor_out = 2 * (1 - 1)   # centre -> north, made residential
    minor_in = minor_out + 1  # north -> centre
    graph = _junction(
        yield_sec=YIELD_DELAY_SEC,
        road_class_of={minor_out: RoadClass.MINOR, minor_in: RoadClass.MINOR},
    )

    # West to east: arterial straight through, nothing to yield to.
    through = _arm_to_arm(graph, 4, 2)
    assert through.total_duration_sec == pytest.approx(2 * ARM_TRAVEL_SEC)

    # West to north: turning onto the residential street, so it is charged.
    turning = _arm_to_arm(graph, 4, 1)
    assert turning.total_duration_sec == pytest.approx(
        2 * ARM_TRAVEL_SEC + YIELD_DELAY_SEC
    )


def test_left_turns_cost_more_than_rights_and_u_turns_most():
    """Turn penalties key off the actual heading change, not the node."""
    penalties = traffic_engine.TurnPenalties(
        through_sec=TURN_PENALTY_SEC["through"],
        right_sec=TURN_PENALTY_SEC["right"],
        left_sec=TURN_PENALTY_SEC["left"],
        uturn_sec=TURN_PENALTY_SEC["uturn"],
    )
    graph = _junction(turns=penalties)

    # Arriving at the centre heading east (from the west arm):
    #   east  = straight on, north = left across oncoming, south = right.
    costs = {
        "through": _arm_to_arm(graph, 4, 2).total_duration_sec,
        "left": _arm_to_arm(graph, 4, 1).total_duration_sec,
        "right": _arm_to_arm(graph, 4, 3).total_duration_sec,
    }
    base = 2 * ARM_TRAVEL_SEC
    for movement, seconds in costs.items():
        assert seconds == pytest.approx(base + TURN_PENALTY_SEC[movement]), movement

    assert costs["left"] > costs["right"] > costs["through"]


def test_u_turn_is_charged_as_a_u_turn():
    penalties = traffic_engine.TurnPenalties(
        left_sec=TURN_PENALTY_SEC["left"], uturn_sec=TURN_PENALTY_SEC["uturn"]
    )
    graph = _junction(turns=penalties)

    # Out to the north arm and back: the only way home is a U-turn at node 1,
    # which is the route's one interior node.
    result = graph.find_shortest_path(0, 0)
    assert result.total_duration_sec == 0.0  # degenerate, no movement at all

    out_and_back = graph.score_path([0, 1])  # centre -> north -> centre
    assert out_and_back == pytest.approx(
        2 * ARM_TRAVEL_SEC + TURN_PENALTY_SEC["uturn"]
    )


def test_turn_penalties_reach_path_selection_not_just_the_total():
    """A cost the search cannot act on is a report, not a model.

    Made expensive enough, a left turn is worth driving around: the router
    should reach the north arm by turning right to the south arm, turning back,
    and crossing straight through. That is a real manoeuvre -- it is what a
    driver does at an intersection with no protected left -- and finding it is
    only possible because the search state carries the edge it arrived on.
    """
    graph = _junction(
        turns=traffic_engine.TurnPenalties(left_sec=90.0, right_sec=1.0, uturn_sec=2.0)
    )

    avoided = _arm_to_arm(graph, 4, 1)
    # west->centre, centre->south, south->centre, centre->north: four edges
    # rather than two, for 48 s of driving plus a right, a U-turn and a
    # straight-on, against 24 s plus a 90 s left.
    assert len(avoided.path_edge_ids) == 4
    assert avoided.total_duration_sec == pytest.approx(4 * ARM_TRAVEL_SEC + 1.0 + 2.0)

    # Price the U-turn above what the detour saves (the extra 24 s of driving
    # plus the right turn leave it 65 s of headroom) and the left wins again.
    graph.set_turn_penalties(
        traffic_engine.TurnPenalties(left_sec=90.0, right_sec=1.0, uturn_sec=75.0)
    )
    direct = _arm_to_arm(graph, 4, 1)
    assert len(direct.path_edge_ids) == 2
    assert direct.total_duration_sec == pytest.approx(2 * ARM_TRAVEL_SEC + 90.0)


def test_score_path_agrees_with_the_search():
    """The two must not drift: ``time_saved_sec`` is a difference of them.

    ``RoutingService`` scores the baseline path with ``score_path`` on the
    dynamic graph. If that undercounted delays the comparison would flatter
    the baseline, and if it overcounted, the AI route.
    """
    graph = _junction(
        priority_sec=SIGNAL_DELAY_SEC,
        turns=traffic_engine.TurnPenalties(
            left_sec=TURN_PENALTY_SEC["left"], right_sec=TURN_PENALTY_SEC["right"]
        ),
    )

    for start, goal in ((4, 1), (4, 2), (2, 3), (1, 3)):
        result = graph.find_shortest_path(start, goal)
        assert result.found
        assert graph.score_path(list(result.path_edge_ids)) == pytest.approx(
            result.total_duration_sec, abs=1e-4
        )


def test_score_path_rejects_a_disconnected_path():
    """Scoring edges that do not meet would silently invent a teleport."""
    graph = _junction(priority_sec=SIGNAL_DELAY_SEC)

    with pytest.raises(ValueError):
        graph.score_path([0, 2])   # centre->north then centre->east


def test_delays_must_be_non_negative():
    """A negative delay breaks the assumption A* is built on."""
    graph = _junction()
    with pytest.raises(ValueError):
        graph.set_node_delays([-5.0] * 5, [])
    with pytest.raises(ValueError):
        graph.set_turn_penalties(traffic_engine.TurnPenalties(left_sec=-1.0))


# ---------------------------------------------------------------------------
# Layer 2: the calibration constants and classification
# ---------------------------------------------------------------------------
def test_speed_caps_bind_on_arterials_and_spare_freeways():
    """Freeways keep their posted speed; the ST-GNN models their congestion."""
    assert cruise_speed_kph("motorway", 105.0) == pytest.approx(105.0)
    assert cruise_speed_kph("trunk", 88.0) == pytest.approx(88.0)
    # Jamboree, Culver, Alton: posted 72-80, cruised at the cap.
    assert cruise_speed_kph("primary", 72.0) == pytest.approx(ARTERIAL_CRUISE_CAP_KPH)
    assert cruise_speed_kph("secondary", 64.0) == pytest.approx(ARTERIAL_CRUISE_CAP_KPH)
    # Ramps and turn lanes are not arterials to cruise along.
    assert cruise_speed_kph("primary_link", 45.0) == pytest.approx(45.0)
    assert cruise_speed_kph("tertiary", 56.0) == pytest.approx(MINOR_CRUISE_CAP_KPH)
    # A cap is a ceiling, never a floor: a slow street stays slow.
    assert cruise_speed_kph("residential", 40.0) == pytest.approx(40.0)
    assert cruise_speed_kph("living_street", 25.0) == pytest.approx(25.0)


def test_capped_speeds_convert_to_seconds_in_metres_per_second():
    """The unit audit, as an assertion: metres / (km/h / 3.6) = seconds."""
    highways = ["primary", "motorway", "residential"]
    posted = np.array([72.0, 105.0, 40.0])
    length_m = np.array([1_000.0, 5_000.0, 250.0])

    cruise = cap_speeds_kph(highways, posted)
    seconds = length_m / (cruise / 3.6)

    assert cruise.tolist() == [ARTERIAL_CRUISE_CAP_KPH, 105.0, 40.0]
    # 1 km at 60 km/h is a minute, whatever the code says.
    assert seconds[0] == pytest.approx(60.0)
    assert seconds[1] == pytest.approx(5_000.0 / (105.0 / 3.6))
    assert seconds[2] == pytest.approx(250.0 / (40.0 / 3.6))
    # And the implied speed never exceeds the cap.
    assert (length_m / seconds * 3.6 <= np.maximum(cruise, posted) + 1e-9).all()


def test_highway_tags_map_to_road_classes():
    assert road_class("motorway") is RoadClass.FREEWAY
    assert road_class("trunk_link") is RoadClass.FREEWAY
    assert road_class("primary") is RoadClass.ARTERIAL
    assert road_class("secondary_link") is RoadClass.ARTERIAL
    assert road_class("tertiary") is RoadClass.MINOR
    assert road_class("residential") is RoadClass.MINOR
    # Unknown tags fall to minor, which can only add delay, never remove it.
    assert road_class("busway") is RoadClass.MINOR


def test_multi_valued_highway_tags_resolve_to_the_important_class():
    """OSMnx keeps a list where a way carries two tags; the GeoPackage a string."""
    assert normalize_highway("['tertiary', 'residential']") == "tertiary"
    assert normalize_highway(["residential", "secondary"]) == "secondary"
    assert normalize_highway("primary") == "primary"
    assert normalize_highway(None) == ""


def _linear_network():
    """Four nodes: an arterial crossing an arterial, then a residential stub.

        0 --Jamboree-- 1 --Jamboree-- 2 --Elm(residential)-- 3
                       |
                    Barranca (arterial, to node 4)
    """
    edge_u = np.array([0, 1, 1, 2, 1, 4])
    edge_v = np.array([1, 0, 2, 3, 4, 1])
    highways = ["primary", "primary", "primary", "residential", "secondary", "secondary"]
    names = ["Jamboree Road", "Jamboree Road", "Jamboree Road", "Elm Street",
             "Barranca Parkway", "Barranca Parkway"]
    return edge_u, edge_v, highways, names


def test_two_arterials_crossing_is_treated_as_signalised():
    """OSM's signal tagging in Irvine is partial; the crossing itself is not.

    Node 1 has Jamboree and Barranca meeting and carries no ``traffic_signals``
    tag. Irvine signalises that crossing, and so does this classifier.
    """
    edge_u, edge_v, highways, names = _linear_network()
    delays = classify_intersections(5, edge_u, edge_v, highways, names)

    assert delays.kind[1] == "signal"
    assert delays.priority_sec[1] == pytest.approx(SIGNAL_DELAY_SEC)
    assert delays.yield_sec[1] == pytest.approx(0.0)


def test_an_arterial_meeting_a_residential_street_is_a_yield():
    edge_u, edge_v, highways, names = _linear_network()
    # Give node 2 a third leg so it counts as a junction rather than a
    # geometry node: Jamboree continues past Elm Street.
    edge_u = np.append(edge_u, [2])
    edge_v = np.append(edge_v, [4])
    highways = highways + ["primary"]
    names = names + ["Jamboree Road"]

    delays = classify_intersections(5, edge_u, edge_v, highways, names)

    assert delays.kind[2] == "minor"
    assert delays.priority_sec[2] == pytest.approx(0.0)
    assert delays.yield_sec[2] == pytest.approx(YIELD_DELAY_SEC)


def test_a_tagged_signal_is_believed_even_without_a_crossing():
    edge_u, edge_v, highways, names = _linear_network()
    node_tags = ["", "", "traffic_signals", "", ""]

    delays = classify_intersections(
        5, edge_u, edge_v, highways, names, node_highway=node_tags
    )

    assert delays.kind[2] == "signal"
    assert delays.priority_sec[2] == pytest.approx(SIGNAL_DELAY_SEC)


def test_freeway_only_nodes_and_dead_ends_cost_nothing():
    """An interchange is grade separated and a cul-de-sac is not a junction."""
    edge_u = np.array([0, 1, 1, 2, 2, 3])
    edge_v = np.array([1, 0, 2, 1, 3, 2])
    highways = ["motorway", "motorway", "motorway_link", "motorway_link",
                "residential", "residential"]
    names = ["I-405"] * 4 + ["Cul de Sac"] * 2

    delays = classify_intersections(4, edge_u, edge_v, highways, names)

    assert delays.kind[1] == "freeway"
    assert delays.priority_sec[1] == 0.0
    assert delays.yield_sec[1] == 0.0
    # Node 3 is a dead end: two neighbours would already be too few, and it
    # has one.
    assert delays.kind[3] == "through"
    assert delays.priority_sec[3] == 0.0


def test_bearings_follow_the_road_rather_than_the_chord():
    """A curving arterial's heading at the junction is not its chord's.

    A quarter-circle sweeping from due north to due east has a chord bearing of
    about 45 degrees, which would classify a subsequent straight-on movement as
    a turn.
    """
    angles = np.linspace(0.0, np.pi / 2, 40)
    # ~1 km radius arc, starting at the origin heading north.
    curve = np.column_stack(
        [33.65 + 0.009 * np.sin(angles), -117.80 + 0.009 * (1 - np.cos(angles))]
    )
    (entry,), (exit_,) = polyline_bearings([curve])

    assert entry == pytest.approx(0.0, abs=12.0)    # sets off due north
    assert exit_ == pytest.approx(90.0, abs=12.0)   # arrives heading due east
    assert abs(exit_ - entry) > 45.0


# ---------------------------------------------------------------------------
# Layer 3: the end-to-end benchmark on the real network
# ---------------------------------------------------------------------------
pytest.importorskip("sklearn", reason="scikit-learn not installed; needed to snap")


@pytest.fixture(scope="module")
def network():
    """The real Irvine network, calibrated. No ONNX or dataset needed."""
    from app.config import Settings
    from app.network import RoadNetwork

    settings = Settings()
    for path in (
        settings.node_mapping_path,
        settings.graph_tensors_path,
        settings.geopackage_path,
    ):
        if not path.exists():
            pytest.skip(f"{path.name} missing; run `python -m src.data.graph_builder`")

    return RoadNetwork.load(
        mapping_path=settings.node_mapping_path,
        tensors_path=settings.graph_tensors_path,
        geopackage_path=settings.geopackage_path,
    )


@pytest.fixture(scope="module")
def baseline_graph(network):
    """Free-flow weights plus the intersection model: the baseline router."""
    return network.build_engine_graph()


@pytest.fixture(scope="module")
def benchmark_route(network, baseline_graph):
    origin = network.snap(*UTC)
    destination = network.snap(*TUSTIN_LEGACY)
    assert origin.distance_m < 500.0, "UTC did not snap to the network"
    assert destination.distance_m < 500.0, "Tustin Legacy did not snap to the network"

    result = baseline_graph.find_shortest_path(origin.node_id, destination.node_id)
    assert result.found, "no route from UTC to Tustin Legacy"
    return result


def test_utc_to_tustin_legacy_takes_a_realistic_baseline_time(benchmark_route):
    """The headline: ~8 km of Irvine arterial is 12-16 minutes, not 6.

    Both bounds matter. Under 12 the signals are missing again; over 16 the
    calibration has overshot and the router will start recommending freeway
    detours around trips that do not need them.
    """
    minutes = benchmark_route.total_duration_sec / 60.0
    km = benchmark_route.total_distance_meters / 1_000.0

    assert BENCHMARK_MIN_M <= benchmark_route.total_distance_meters <= BENCHMARK_MAX_M, (
        f"benchmark route is {km:.2f} km, outside the ~8 km trip this window "
        "was calibrated for"
    )
    assert BENCHMARK_MIN_SEC <= benchmark_route.total_duration_sec <= BENCHMARK_MAX_SEC, (
        f"UTC -> Tustin Legacy: {minutes:.1f} min over {km:.2f} km "
        f"({km / (minutes / 60):.0f} km/h). Expected 12-16 min."
    )


def test_the_benchmark_average_speed_is_arterial_not_freeway(benchmark_route):
    """An implied 78 km/h was the symptom; 30-40 km/h is the cure."""
    kph = (
        benchmark_route.total_distance_meters
        / benchmark_route.total_duration_sec
        * 3.6
    )
    assert 28.0 <= kph <= 45.0, f"implied average speed {kph:.1f} km/h"


def test_stopping_is_a_material_share_of_the_benchmark(benchmark_route, network):
    """Signals should account for roughly a third of the trip, not a rounding.

    Also checks the split adds up: rolling time plus intersection time is the
    whole duration, so nothing is being charged twice.
    """
    rolling_sec = float(
        network.edge_free_flow_s[
            np.asarray(list(benchmark_route.path_edge_ids), dtype=np.int64)
        ].sum()
    )
    stopped_sec = benchmark_route.total_duration_sec - rolling_sec

    assert stopped_sec > 0
    share = stopped_sec / benchmark_route.total_duration_sec
    assert 0.20 <= share <= 0.60, f"intersections are {share:.0%} of the trip"


def test_the_uncalibrated_graph_is_the_bug_this_window_rules_out(network):
    """Same trip, posted limits, no intersections: the 5-6 minute answer.

    Kept as a test rather than a comment because it is the regression that
    matters. If someone drops the calibration on the way through, the
    benchmark above fails -- and this one starts failing too, which says
    exactly *which* half went missing.
    """
    graph = traffic_engine.Graph(network.num_nodes)
    for node_id in range(network.num_nodes):
        graph.add_node(
            node_id,
            float(network.node_coords[node_id, 0]),
            float(network.node_coords[node_id, 1]),
        )
    # Posted-limit free-flow times, exactly as graph_tensors.pt carries them.
    posted_sec = network.edge_length_m / (network.edge_posted_kph / 3.6)
    for edge_id in range(network.num_edges):
        graph.add_edge(
            edge_id,
            int(network.edge_src[edge_id]),
            int(network.edge_dst[edge_id]),
            float(network.edge_length_m[edge_id]),
            float(posted_sec[edge_id]),
        )
    graph.finalize()
    assert graph.has_intersection_costs is False

    origin = network.snap(*UTC)
    destination = network.snap(*TUSTIN_LEGACY)
    result = graph.find_shortest_path(origin.node_id, destination.node_id)

    assert result.found
    assert result.total_duration_sec < BENCHMARK_MIN_SEC, (
        "the uncalibrated router now agrees with the calibrated one, which "
        "means the posted-limit weights are no longer what they were"
    )
    implied_kph = result.total_distance_meters / result.total_duration_sec * 3.6
    assert implied_kph > 60.0, "the bug was an implied 78 km/h along Jamboree"


def test_a_freeway_trip_is_not_penalised_like_an_arterial(network, baseline_graph):
    """Interchanges are grade separated, so I-405 must stay fast.

    The delay model has to be class-aware in both directions: adding signal
    time to a freeway would be as wrong as leaving it off an arterial.
    """
    # Irvine Business Complex to the Great Park: the I-405/I-5 run.
    origin = network.snap(33.6846, -117.8265)
    destination = network.snap(33.6595, -117.7467)
    result = baseline_graph.find_shortest_path(origin.node_id, destination.node_id)

    assert result.found
    kph = result.total_distance_meters / result.total_duration_sec * 3.6
    assert kph > 55.0, f"freeway trip averaged {kph:.0f} km/h"


def test_every_calibrated_delay_is_finite_and_non_negative(network):
    calibration = network.calibration
    assert calibration is not None

    for delays in (calibration.node_priority_delay_s, calibration.node_yield_delay_s):
        assert delays.shape == (network.num_nodes,)
        assert np.isfinite(delays).all()
        assert (delays >= 0).all()

    # Both forms at once would double-charge a movement onto a minor road.
    both = (calibration.node_priority_delay_s > 0) & (
        calibration.node_yield_delay_s > 0
    )
    assert not both.any()

    # Sanity on the classification: Irvine has hundreds of signalised
    # crossings, not a handful and not thousands.
    signals = int((calibration.node_priority_delay_s > 0).sum())
    assert 200 <= signals <= 1_500, f"{signals} signalised nodes"


def test_calibrated_free_flow_is_slower_than_the_posted_limit(network):
    """The cap has to actually reach the weights the engine is built with."""
    posted_sec = network.edge_length_m / (network.edge_posted_kph / 3.6)
    calibrated_sec = network.edge_free_flow_s

    # Never faster than the posted-limit time, anywhere.
    assert (calibrated_sec >= posted_sec - 1e-3).all()
    # And meaningfully slower across the arterial network.
    slower = calibrated_sec > posted_sec + 1e-3
    assert slower.sum() > 0.2 * network.num_edges
