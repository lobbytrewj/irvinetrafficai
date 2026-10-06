"""Turn geometry: bearings, angles, and the two movements worth flagging.

Why this is its own file
------------------------
A turn is the one property of a road network that lives on neither a node nor
an edge. "Left onto Jamboree" only means something once you know the road the
driver arrived on, so the pipeline builds a third table for it -- and a table
that no other artifact constrains is a table that can be silently wrong. A
sign flip in the angle convention would turn every left into a right, and
nothing else in the suite would notice: the counts would be identical, the
GeoPackage would open, the tensors would validate.

So the assertions here are geometric and absolute. Bearings are checked
against the compass (due north is 0, due east is 90), and the turn flags
against a synthetic four-way intersection built to the same layout a driver
would recognise:

                       N (secondary)
                       |
    W (primary) ------ 0 ------ E (primary)
                       |
                       S (secondary)

Arriving from the west, the driver is heading due east. North is then a left
across the oncoming westbound lanes, south is a right, and east is straight
on. Every expectation below follows from that picture rather than from what
the code happens to return.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch

from src.data.graph_builder import (
    UNPROTECTED_LEFT_MAX_DEG,
    UNPROTECTED_LEFT_MIN_DEG,
    annotate_road_types,
    annotate_signals,
    build_turn_table,
    calculate_bearing,
    extract_tensors,
    is_aggressive_merge,
    is_unprotected_left,
    turn_delta_degrees,
    validate_tensors,
)

# ---------------------------------------------------------------------------
# The synthetic intersection
# ---------------------------------------------------------------------------
CENTER = (33.6500, -117.8000)     # (lat, lon)
ARM_LAT_DEG = 0.0018              # ~200 m north/south
ARM_LON_DEG = 0.0022              # ~200 m east/west at this latitude

# node id -> (lat, lon), laid out by compass point.
NORTH, EAST, SOUTH, WEST = 1, 2, 3, 4
ARMS = {
    NORTH: (CENTER[0] + ARM_LAT_DEG, CENTER[1]),
    EAST: (CENTER[0], CENTER[1] + ARM_LON_DEG),
    SOUTH: (CENTER[0] - ARM_LAT_DEG, CENTER[1]),
    WEST: (CENTER[0], CENTER[1] - ARM_LON_DEG),
}

# What the driver arriving from the west does at the centre, by exit arm.
MOVEMENTS = {NORTH: "left", EAST: "through", SOUTH: "right", WEST: "u-turn"}


def _four_way(
    *,
    signalised: bool = False,
    ew_tag: str = "primary",
    ns_tag: str = "secondary",
    speed_kph: float = 60.0,
) -> nx.MultiDiGraph:
    """A four-way cross, OSMnx-shaped, with contiguous node ids.

    Node 0 is the intersection; 1-4 are the arms. Edges run both ways, so
    every movement including the U-turn is a legal transition.
    """
    graph = nx.MultiDiGraph(crs="epsg:4326")
    graph.add_node(
        0,
        x=CENTER[1],
        y=CENTER[0],
        street_count=4,
        highway="traffic_signals" if signalised else "",
    )
    for node, (lat, lon) in ARMS.items():
        graph.add_node(node, x=lon, y=lat, street_count=1, highway="")

    for node in ARMS:
        tag = ew_tag if node in (EAST, WEST) else ns_tag
        for u, v in ((0, node), (node, 0)):
            graph.add_edge(
                u,
                v,
                key=0,
                osmid=1000 + u * 10 + v,
                highway=tag,
                length=200.0,
                speed_kph=speed_kph,
                travel_time=200.0 / (speed_kph / 3.6),
            )
    return graph


def _turn_table(graph: nx.MultiDiGraph):
    """Run the pipeline stages a turn table depends on, then build it."""
    annotate_signals(graph)
    annotate_road_types(graph)
    return build_turn_table(graph)


def _edge_ids(graph: nx.MultiDiGraph) -> dict[tuple[int, int], int]:
    """``(u, v)`` to edge id, so no test has to hard-code an index."""
    from src.data.graph_builder import sorted_edges

    return {(u, v): index for index, (u, v, _) in enumerate(sorted_edges(graph))}


def _row(table, in_edge: int, out_edge: int) -> int:
    matches = np.flatnonzero(
        (table.in_edge_id == in_edge) & (table.out_edge_id == out_edge)
    )
    assert len(matches) == 1, f"expected one row for {in_edge}->{out_edge}"
    return int(matches[0])


# ---------------------------------------------------------------------------
# calculate_bearing
# ---------------------------------------------------------------------------
def test_bearing_points_at_the_compass():
    """Zero is north and the angle grows clockwise."""
    lat, lon = CENTER

    assert calculate_bearing(lat, lon, lat + 0.01, lon) == pytest.approx(0.0, abs=0.01)
    assert calculate_bearing(lat, lon, lat, lon + 0.01) == pytest.approx(90.0, abs=0.01)
    assert calculate_bearing(lat, lon, lat - 0.01, lon) == pytest.approx(180.0, abs=0.01)
    assert calculate_bearing(lat, lon, lat, lon - 0.01) == pytest.approx(270.0, abs=0.01)


def test_bearing_is_always_in_the_forward_range():
    """[0, 360), never negative -- the trap in wrapping atan2 output."""
    lat, lon = CENTER
    for dlat in (-0.02, -0.005, 0.0, 0.005, 0.02):
        for dlon in (-0.02, -0.005, 0.005, 0.02):
            bearing = calculate_bearing(lat, lon, lat + dlat, lon + dlon)
            assert 0.0 <= bearing < 360.0


def test_reverse_bearing_is_the_opposite_heading():
    """Driving back down a street points 180 degrees the other way.

    Only approximately, and deliberately asserted with slack: on a great
    circle the return bearing is the forward one reversed *plus* the
    convergence of the meridians, which over a city block is a hundredth of a
    degree but is not zero.
    """
    a_lat, a_lon = CENTER
    b_lat, b_lon = ARMS[EAST]

    forward = calculate_bearing(a_lat, a_lon, b_lat, b_lon)
    back = calculate_bearing(b_lat, b_lon, a_lat, a_lon)

    assert abs(((back - forward) % 360.0) - 180.0) < 0.1


def test_bearing_matches_a_known_diagonal():
    """A north-east step is 45 degrees, once the longitude is scaled by cos(lat).

    At 33.65 N a degree of longitude is cos(33.65) of a degree of latitude, so
    equal *distances* north and east need unequal degree steps. Getting this
    backwards is the classic lat/lon bug, and it would show up as every
    diagonal turn being misclassified.
    """
    lat, lon = CENTER
    step = 0.01
    east_step = step / np.cos(np.radians(lat))

    assert calculate_bearing(lat, lon, lat + step, lon + east_step) == pytest.approx(
        45.0, abs=0.2
    )


# ---------------------------------------------------------------------------
# turn_delta_degrees
# ---------------------------------------------------------------------------
def test_turn_delta_signs_left_negative_and_right_positive():
    assert turn_delta_degrees(90.0, 0.0) == pytest.approx(-90.0)    # east then north
    assert turn_delta_degrees(90.0, 180.0) == pytest.approx(90.0)   # east then south
    assert turn_delta_degrees(90.0, 90.0) == pytest.approx(0.0)     # straight on


def test_turn_delta_wraps_across_north():
    """350 to 10 is a 20 degree right, not a 340 degree left."""
    assert turn_delta_degrees(350.0, 10.0) == pytest.approx(20.0)
    assert turn_delta_degrees(10.0, 350.0) == pytest.approx(-20.0)


def test_turn_delta_stays_inside_the_half_circle():
    """[-180, 180), matching the native engine. An exact reversal is -180."""
    for bearing_in in range(0, 360, 17):
        for bearing_out in range(0, 360, 23):
            delta = turn_delta_degrees(float(bearing_in), float(bearing_out))
            assert -180.0 <= delta < 180.0


def test_a_u_turn_is_a_half_circle():
    """Sign is arbitrary for a reversal, so only the magnitude is asserted."""
    assert abs(turn_delta_degrees(90.0, 270.0)) == pytest.approx(180.0)
    assert abs(turn_delta_degrees(270.0, 90.0)) == pytest.approx(180.0)


# ---------------------------------------------------------------------------
# The four-way intersection
# ---------------------------------------------------------------------------
def test_every_movement_at_the_intersection_is_enumerated():
    """Four legs in, four out: sixteen movements at the centre, U-turns included.

    Plus one at each arm, where the only movement is to turn around.
    """
    graph = _four_way()
    table = _turn_table(graph)

    assert len(table) == 16 + 4
    assert int((table.node_id == 0).sum()) == 16
    # No movement is listed twice.
    pairs = list(zip(table.in_edge_id.tolist(), table.out_edge_id.tolist()))
    assert len(set(pairs)) == len(pairs)


def test_turn_angles_match_the_compass_at_a_four_way():
    """Arriving from the west: north is -90, south is +90, east is 0."""
    graph = _four_way()
    table = _turn_table(graph)
    ids = _edge_ids(graph)

    arriving = ids[(WEST, 0)]
    expected = {NORTH: -90.0, EAST: 0.0, SOUTH: 90.0, WEST: 180.0}

    for arm, angle in expected.items():
        row = _row(table, arriving, ids[(0, arm)])
        assert table.bearing_in_deg[row] == pytest.approx(90.0, abs=0.5), "heading east"
        assert abs(table.delta_deg[row]) == pytest.approx(abs(angle), abs=0.5), (
            f"{MOVEMENTS[arm]} to arm {arm}"
        )
        if angle not in (180.0,):
            assert table.delta_deg[row] == pytest.approx(angle, abs=0.5)


def test_left_turn_detected_at_an_unsignalised_arterial_crossing():
    """The headline: west to north is an unprotected left, and nothing else is."""
    graph = _four_way()
    table = _turn_table(graph)
    ids = _edge_ids(graph)
    arriving = ids[(WEST, 0)]

    flagged = {
        arm: bool(table.is_unprotected_left[_row(table, arriving, ids[(0, arm)])])
        for arm in ARMS
    }

    assert flagged == {NORTH: True, EAST: False, SOUTH: False, WEST: False}


def test_left_turns_are_flagged_from_every_approach():
    """The rule is geometric, so it must hold whichever way the driver came.

    Each approach has exactly one left: the arm 90 degrees counter-clockwise
    of the direction of travel.
    """
    graph = _four_way()
    table = _turn_table(graph)
    ids = _edge_ids(graph)

    # Approaching from X, the left turn exits at Y.
    lefts = {WEST: NORTH, NORTH: EAST, EAST: SOUTH, SOUTH: WEST}

    for approach, exit_arm in lefts.items():
        for arm in ARMS:
            row = _row(table, ids[(approach, 0)], ids[(0, arm)])
            is_left = bool(table.is_unprotected_left[row])
            assert is_left == (arm == exit_arm), (
                f"approach {approach} -> arm {arm}: "
                f"delta {table.delta_deg[row]:.1f} deg"
            )

    assert int(table.is_unprotected_left.sum()) == 4


def test_a_traffic_signal_protects_the_left_turn():
    """Same geometry, same road classes -- the signal is the only difference."""
    graph = _four_way(signalised=True)
    table = _turn_table(graph)
    ids = _edge_ids(graph)

    row = _row(table, ids[(WEST, 0)], ids[(0, NORTH)])
    assert table.delta_deg[row] == pytest.approx(-90.0, abs=0.5)  # still a left
    assert not table.is_unprotected_left[row]                      # but protected
    assert int(table.is_unprotected_left.sum()) == 0


def test_the_signal_tag_is_read_off_the_osm_node():
    graph = _four_way(signalised=True)
    signalled = annotate_signals(graph)

    assert signalled == {0}
    assert graph.nodes[0]["has_traffic_signal"] is True
    assert graph.nodes[NORTH]["has_traffic_signal"] is False


def test_residential_crossings_are_not_unprotected_lefts():
    """The flag is about crossing an arterial's oncoming traffic.

    A left between two residential streets is a left turn, and the angle says
    so, but it is not the hazard this flag names.
    """
    graph = _four_way(ew_tag="residential", ns_tag="residential")
    table = _turn_table(graph)
    ids = _edge_ids(graph)

    row = _row(table, ids[(WEST, 0)], ids[(0, NORTH)])
    assert table.delta_deg[row] == pytest.approx(-90.0, abs=0.5)
    assert not table.is_unprotected_left[row]
    assert int(table.is_unprotected_left.sum()) == 0


def test_turning_onto_an_arterial_counts_too():
    """Crossing an arterial from a side street is the same conflict.

    East-west stays primary, north-south becomes residential: a driver leaving
    the residential arm still crosses the arterial's oncoming lanes.
    """
    graph = _four_way(ew_tag="primary", ns_tag="residential")
    table = _turn_table(graph)
    ids = _edge_ids(graph)

    # From the north (residential), heading south, turning left onto the
    # eastbound primary.
    row = _row(table, ids[(NORTH, 0)], ids[(0, EAST)])
    assert table.delta_deg[row] == pytest.approx(-90.0, abs=0.5)
    assert table.is_unprotected_left[row]


# ---------------------------------------------------------------------------
# The angle band
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "delta, expected",
    [
        pytest.param(-90.0, True, id="clean-left"),
        pytest.param(-46.0, True, id="just-inside-shallow"),
        pytest.param(-134.0, True, id="just-inside-sharp"),
        pytest.param(-44.0, False, id="too-shallow-a-veer"),
        pytest.param(-136.0, False, id="sharp-enough-to-be-a-u-turn"),
        pytest.param(0.0, False, id="straight-on"),
        pytest.param(90.0, False, id="a-right-is-not-a-left"),
        pytest.param(180.0, False, id="u-turn"),
    ],
)
def test_the_unprotected_left_band(delta, expected):
    """A veer across a fork is not a left; a hook back on yourself is a U-turn."""
    assert (
        is_unprotected_left(delta, "primary", "secondary", node_has_signal=False)
        is expected
    )


def test_the_band_constants_describe_a_left_turn():
    """Guard rail on the constants themselves, which are the tunable part."""
    assert UNPROTECTED_LEFT_MIN_DEG < UNPROTECTED_LEFT_MAX_DEG < 0
    assert -180.0 < UNPROTECTED_LEFT_MIN_DEG


# ---------------------------------------------------------------------------
# Aggressive merges
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "in_tag, out_tag, in_kph, out_kph, expected",
    [
        pytest.param("motorway_link", "motorway", 65.0, 105.0, True, id="on-ramp"),
        pytest.param("motorway", "motorway_link", 105.0, 65.0, True, id="off-ramp"),
        pytest.param("trunk_link", "trunk", 55.0, 88.0, True, id="trunk-on-ramp"),
        pytest.param(
            "motorway_link", "motorway_link", 65.0, 65.0, False, id="ramp-to-ramp"
        ),
        pytest.param(
            "motorway_link", "primary", 65.0, 60.0, False, id="ramp-onto-arterial"
        ),
        pytest.param("primary", "secondary", 60.0, 60.0, False, id="two-arterials"),
        pytest.param(
            "motorway", "motorway", 105.0, 105.0, False, id="staying-on-the-freeway"
        ),
    ],
)
def test_aggressive_merge_needs_a_ramp_and_speed(
    in_tag, out_tag, in_kph, out_kph, expected
):
    """The hazard is the speed differential at a ramp, in either direction.

    A ramp that lets out onto a 60 km/h arterial is a give-way, not a merge
    into traffic already at freeway speed; two arterials meeting is neither.
    """
    assert is_aggressive_merge(in_tag, out_tag, in_kph, out_kph) is expected


def test_merges_are_flagged_through_the_pipeline():
    """A ramp joining a motorway, built and classified end to end."""
    graph = nx.MultiDiGraph(crs="epsg:4326")
    # A straight motorway west to east, with a ramp joining from the south.
    graph.add_node(0, x=-117.81, y=33.65, street_count=3, highway="")
    graph.add_node(1, x=-117.82, y=33.65, street_count=1, highway="")
    graph.add_node(2, x=-117.80, y=33.65, street_count=1, highway="")
    graph.add_node(3, x=-117.812, y=33.646, street_count=1, highway="")

    def link(u, v, tag, kph):
        graph.add_edge(u, v, key=0, osmid=u * 10 + v, highway=tag, length=400.0,
                       speed_kph=kph, travel_time=400.0 / (kph / 3.6))

    link(1, 0, "motorway", 105.0)
    link(0, 2, "motorway", 105.0)
    link(3, 0, "motorway_link", 65.0)   # on-ramp
    link(0, 3, "motorway_link", 65.0)   # off-ramp

    table = _turn_table(graph)
    ids = _edge_ids(graph)

    on_ramp = _row(table, ids[(3, 0)], ids[(0, 2)])
    off_ramp = _row(table, ids[(1, 0)], ids[(0, 3)])
    mainline = _row(table, ids[(1, 0)], ids[(0, 2)])

    assert table.is_aggressive_merge[on_ramp]
    assert table.is_aggressive_merge[off_ramp]
    assert not table.is_aggressive_merge[mainline]
    # And a freeway movement is never an unprotected left, whatever its angle.
    assert not table.is_unprotected_left.any()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def test_the_turn_table_reaches_the_tensor_bundle():
    graph = _four_way()
    table = _turn_table(graph)
    tensors = extract_tensors(graph, turns=table)

    validate_tensors(tensors, num_nodes=graph.number_of_nodes())

    assert tensors["turn_node_id"].numel() == len(table)
    assert tensors["turn_delta_deg"].dtype == torch.float32
    assert tensors["turn_is_unprotected_left"].dtype == torch.bool
    assert bool(tensors["turn_is_unprotected_left"].any())
    assert tensors["node_has_traffic_signal"].shape == (graph.number_of_nodes(),)


def test_extract_tensors_builds_the_turn_table_when_not_given_one():
    """The bundle must never go out without it, however it was called."""
    graph = _four_way()
    annotate_signals(graph)
    annotate_road_types(graph)

    tensors = extract_tensors(graph)

    assert tensors["turn_node_id"].numel() == 20


def test_validation_catches_a_corrupted_turn_table():
    """Every one of these would route fine and mean something wrong."""
    graph = _four_way()
    tensors = extract_tensors(graph, turns=_turn_table(graph))
    num_nodes = graph.number_of_nodes()

    dangling = dict(tensors)
    dangling["turn_out_edge_id"] = tensors["turn_out_edge_id"].clone()
    dangling["turn_out_edge_id"][0] = 9_999
    with pytest.raises(ValueError, match="out-of-range edge id"):
        validate_tensors(dangling, num_nodes=num_nodes)

    impossible = dict(tensors)
    impossible["turn_delta_deg"] = tensors["turn_delta_deg"].clone()
    impossible["turn_delta_deg"][0] = 400.0
    with pytest.raises(ValueError, match="signed angle"):
        validate_tensors(impossible, num_nodes=num_nodes)

    ragged = dict(tensors)
    ragged["turn_is_unprotected_left"] = tensors["turn_is_unprotected_left"][:-1]
    with pytest.raises(ValueError, match="to match turn_node_id"):
        validate_tensors(ragged, num_nodes=num_nodes)

    truncated = {k: v for k, v in tensors.items() if k != "turn_delta_deg"}
    with pytest.raises(ValueError, match="missing"):
        validate_tensors(truncated, num_nodes=num_nodes)


def test_the_turns_layer_lands_in_the_geopackage(tmp_path):
    """QGIS is how a rule like this gets checked against a city you can drive."""
    import geopandas as gpd

    from src.data.graph_builder import (
        GRAPH_FILENAME,
        GraphBuildConfig,
        build_mapping_payload,
        build_node_mapping,
        save_artifacts,
    )

    graph = _four_way()
    table = _turn_table(graph)
    tensors = extract_tensors(graph, turns=table)
    payload = build_mapping_payload(
        graph,
        build_node_mapping(graph),
        GraphBuildConfig(output_dir=tmp_path),
        num_edges=tensors["edge_index"].size(1),
        turns=table,
    )
    save_artifacts(graph, tensors, payload, tmp_path)

    turns = gpd.read_file(tmp_path / GRAPH_FILENAME, layer="turns")

    assert len(turns) == len(table)
    assert set(turns.columns) >= {
        "node_id",
        "in_edge_id",
        "out_edge_id",
        "delta_deg",
        "is_unprotected_left",
        "is_aggressive_merge",
        "geometry",
    }
    assert int(turns["is_unprotected_left"].sum()) == 4
    # Each row sits on its intersection, so the layer can be mapped.
    centre = turns[turns["node_id"] == 0].iloc[0].geometry
    assert centre.y == pytest.approx(CENTER[0], abs=1e-6)
    assert centre.x == pytest.approx(CENTER[1], abs=1e-6)

    # And the summary in the JSON mapping agrees with the table.
    assert payload["turn_summary"]["unprotected_left"] == 4
    assert payload["traffic_signal_nodes"] == []


def test_signalised_nodes_are_listed_in_the_mapping():
    from src.data.graph_builder import (
        GraphBuildConfig,
        build_mapping_payload,
        build_node_mapping,
    )

    graph = _four_way(signalised=True)
    table = _turn_table(graph)
    payload = build_mapping_payload(
        graph, build_node_mapping(graph), GraphBuildConfig(), num_edges=8, turns=table
    )

    assert payload["traffic_signal_nodes"] == [0]
    assert payload["turn_summary"]["transitions"] == 20
    assert payload["turn_summary"]["unprotected_left"] == 0
