"""Routing preferences: trading time to avoid a movement.

The manoeuvre under test
------------------------
Turning left across an arterial with no signal holding the oncoming traffic is
the one movement drivers reroute around without being asked. What they do
instead is universal enough to have a name -- go up to the light and turn
there -- and it is exactly what a preference-aware search should discover on
its own, from nothing but a penalty on the flagged movement.

So the fixture is that junction, twice over:

      N ------------- NE          W -> J -> N   is the direct left at J,
      |               |                         unsignalised: 40 s.
      W ---- J ------ E           W -> J -> E -> NE -> N  goes up to the
             ^        ^                         signal at E: 80 s.
      unsignalised   signalised

Every edge is 200 m and 20 s, so the detour costs exactly 40 s more. With the
default 45 s penalty the detour wins by five seconds; with a 20 s penalty it
does not. Both are asserted, because a preference that always reroutes is not
a preference -- it is a ban.

The fixture also prices U-turns, and has to. Left unpriced, the search finds a
third route -- drive past J, turn around at E, come back and enter N as a
*right* turn -- which costs the same 80 s as the signal detour and dodges the
flag just as well. It is a real manoeuvre and finding it is the search working
correctly, but it makes the intended route non-unique, so the U-turn carries
30 s here.

What the penalty is not
-----------------------
It is not time. The detour really does take 80 seconds and
``total_duration_sec`` says 80, not 125. The bias lives in
``preference_penalty_sec``, and that separation is what keeps an ETA honest
while still letting the search be steered.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip(
    "traffic_engine", reason="native extension not built; run `pip install -e .`"
)

import traffic_engine  # noqa: E402
from traffic_engine import (  # noqa: E402
    TURN_FLAG_AGGRESSIVE_MERGE,
    TURN_FLAG_UNPROTECTED_LEFT,
    RoutePreferences,
)

EDGE_M = 200.0
EDGE_SEC = 20.0

# The junction above, as (lat, lon). 0.0018 deg of latitude and 0.0022 of
# longitude are both ~200 m at this latitude.
COORDS = {
    "W": (33.6500, -117.8044),
    "J": (33.6500, -117.8022),
    "E": (33.6500, -117.8000),
    "N": (33.6518, -117.8022),
    "NE": (33.6518, -117.8000),
}
NODE_ID = {name: index for index, name in enumerate(COORDS)}


class Fixture:
    """The two-route network, plus the edge ids needed to talk about it."""

    def __init__(self, graph, edges: dict[tuple[str, str], int]):
        self.graph = graph
        self.edges = edges

    def edge(self, u: str, v: str) -> int:
        return self.edges[(u, v)]

    def path(self, *names: str) -> list[int]:
        return [self.edges[(a, b)] for a, b in zip(names, names[1:])]

    def route(self, start: str, goal: str, preferences=None):
        args = () if preferences is None else (preferences,)
        return self.graph.find_shortest_path(NODE_ID[start], NODE_ID[goal], *args)


# Enough to keep "drive past and turn around" from tying with the detour; see
# the module docstring. Every other movement stays free, so the only costs in
# play are the 20 s edges and the preference penalty under test.
UTURN_SEC = 30.0


def _build(flag_the_left: bool = True) -> Fixture:
    """Build the network and flag the left at J, but not the one at E."""
    graph = traffic_engine.Graph(len(COORDS))
    for name, (lat, lon) in COORDS.items():
        graph.add_node(NODE_ID[name], lat, lon)

    links = [("W", "J"), ("J", "E"), ("J", "N"), ("E", "NE"), ("N", "NE")]
    edges: dict[tuple[str, str], int] = {}
    next_id = 0
    for u, v in links:
        for a, b in ((u, v), (v, u)):  # both directions: real roads have two
            graph.add_edge(next_id, NODE_ID[a], NODE_ID[b], EDGE_M, EDGE_SEC)
            edges[(a, b)] = next_id
            next_id += 1

    graph.set_turn_penalties(traffic_engine.TurnPenalties(uturn_sec=UTURN_SEC))

    fixture = Fixture(graph, edges)
    if flag_the_left:
        # Only the movement W->J->N is unprotected. The mirror image of it at
        # the signalised junction, E->NE, is not -- that is the whole reason a
        # detour exists.
        graph.set_turn_flags(
            [edges[("W", "J")]], [edges[("J", "N")]], [TURN_FLAG_UNPROTECTED_LEFT]
        )
    graph.finalize()
    return fixture


@pytest.fixture
def network() -> Fixture:
    return _build()


DIRECT = ("W", "J", "N")
DETOUR = ("W", "J", "E", "NE", "N")


# ---------------------------------------------------------------------------
# The headline behaviour
# ---------------------------------------------------------------------------
def test_the_fastest_route_takes_the_unprotected_left(network):
    """With no preference expressed, the flags cost nothing."""
    result = network.route("W", "N")

    assert result.found
    assert list(result.path_edge_ids) == network.path(*DIRECT)
    assert result.total_duration_sec == pytest.approx(2 * EDGE_SEC)
    assert result.preference_penalty_sec == 0.0


def test_avoiding_unprotected_lefts_reroutes_to_the_signal(network):
    """The requirement: toggling the preference reroutes around the turn.

    Nothing about the graph changed -- same weights, same topology, same
    flags. The only input that moved is a boolean, and the path moved with it.
    """
    fastest = network.route("W", "N")
    avoided = network.route(
        "W", "N", RoutePreferences(avoid_unprotected_lefts=True)
    )

    assert list(fastest.path_edge_ids) == network.path(*DIRECT)
    assert list(avoided.path_edge_ids) == network.path(*DETOUR)
    # It went the long way round, and it is honest about how long that takes.
    assert avoided.total_duration_sec == pytest.approx(4 * EDGE_SEC)
    assert avoided.total_duration_sec > fastest.total_duration_sec


def test_the_rerouted_path_carries_no_remaining_penalty(network):
    """Having avoided the turn, the winning path is unflagged end to end."""
    avoided = network.route(
        "W", "N", RoutePreferences(avoid_unprotected_lefts=True)
    )

    assert avoided.preference_penalty_sec == 0.0
    for in_edge, out_edge in zip(avoided.path_edge_ids, avoided.path_edge_ids[1:]):
        assert network.graph.turn_flags(in_edge, out_edge) == 0


def test_a_penalty_smaller_than_the_detour_does_not_reroute(network):
    """A preference is a price, not a prohibition.

    The detour costs 40 s. At 20 s the driver would rather make the turn, and
    the search agrees -- while still reporting that it charged the 20.
    """
    result = network.route(
        "W",
        "N",
        RoutePreferences(avoid_unprotected_lefts=True, unprotected_left_penalty_sec=20.0),
    )

    assert list(result.path_edge_ids) == network.path(*DIRECT)
    assert result.total_duration_sec == pytest.approx(2 * EDGE_SEC)
    assert result.preference_penalty_sec == pytest.approx(20.0)


@pytest.mark.parametrize(
    "penalty, expected",
    [
        pytest.param(39.0, DIRECT, id="just-cheaper-than-the-detour"),
        pytest.param(41.0, DETOUR, id="just-dearer-than-the-detour"),
    ],
)
def test_the_switchover_is_where_the_arithmetic_says(network, penalty, expected):
    """The detour costs exactly 40 s, so the decision flips across 40."""
    result = network.route(
        "W",
        "N",
        RoutePreferences(
            avoid_unprotected_lefts=True, unprotected_left_penalty_sec=penalty
        ),
    )
    assert list(result.path_edge_ids) == network.path(*expected)


def test_the_preference_does_not_leak_into_other_queries(network):
    """Preferences are per query, not state on the graph."""
    avoided = network.route("W", "N", RoutePreferences(avoid_unprotected_lefts=True))
    again = network.route("W", "N")

    assert list(avoided.path_edge_ids) == network.path(*DETOUR)
    assert list(again.path_edge_ids) == network.path(*DIRECT)


# ---------------------------------------------------------------------------
# Flags are per movement, not per edge
# ---------------------------------------------------------------------------
def test_the_flag_follows_the_approach_not_the_edge(network):
    """The design point, as an assertion.

    ``J->N`` is a left when approached from W and a right when approached from
    E. Storing the flag on the edge would penalise both; storing it on the
    movement penalises the one that earned it. On the real Irvine network 94%
    of the edges reachable by an unprotected left are also reachable without
    one, so this is the difference between a working preference and one that
    quietly taxes going straight.
    """
    graph = network.graph
    from_west = network.edge("W", "J")
    from_east = network.edge("E", "J")
    onto_north = network.edge("J", "N")

    assert graph.turn_flags(from_west, onto_north) == TURN_FLAG_UNPROTECTED_LEFT
    assert graph.turn_flags(from_east, onto_north) == 0

    prefs = RoutePreferences(avoid_unprotected_lefts=True)
    # Arriving from the east, the same edge costs nothing extra.
    assert graph.transition_cost(from_east, onto_north, prefs) == 0.0
    assert graph.transition_cost(from_west, onto_north, prefs) == pytest.approx(45.0)


def test_an_approach_that_was_never_flagged_reads_as_clear(network):
    for pair in (("J", "E"), ("E", "NE"), ("N", "NE")):
        in_edge = network.edge("W", "J") if pair[0] == "J" else network.edge("J", pair[0])
        assert network.graph.turn_flags(in_edge, network.edge(*pair)) == 0


# ---------------------------------------------------------------------------
# Merges, and the two preferences together
# ---------------------------------------------------------------------------
def test_low_stress_merges_reroutes_around_a_ramp():
    """The same machinery, driven by the other flag."""
    fixture = _build(flag_the_left=False)
    fixture.graph.set_turn_flags(
        [fixture.edge("W", "J")],
        [fixture.edge("J", "N")],
        [TURN_FLAG_AGGRESSIVE_MERGE],
    )

    fastest = fixture.route("W", "N")
    calm = fixture.route("W", "N", RoutePreferences(low_stress_merges=True))

    assert list(fastest.path_edge_ids) == fixture.path(*DIRECT)
    assert list(calm.path_edge_ids) == fixture.path(*DETOUR)
    # The default merge penalty is 60 s, comfortably over the 40 s detour.
    assert calm.preference_penalty_sec == 0.0


def test_one_preference_ignores_the_other_flag():
    """Asking to avoid lefts must not quietly avoid merges as well."""
    fixture = _build(flag_the_left=False)
    fixture.graph.set_turn_flags(
        [fixture.edge("W", "J")],
        [fixture.edge("J", "N")],
        [TURN_FLAG_AGGRESSIVE_MERGE],
    )

    result = fixture.route(
        "W", "N", RoutePreferences(avoid_unprotected_lefts=True)
    )

    assert list(result.path_edge_ids) == fixture.path(*DIRECT)
    assert result.preference_penalty_sec == 0.0


def test_a_movement_carrying_both_flags_pays_both():
    fixture = _build(flag_the_left=False)
    fixture.graph.set_turn_flags(
        [fixture.edge("W", "J")],
        [fixture.edge("J", "N")],
        [TURN_FLAG_UNPROTECTED_LEFT | TURN_FLAG_AGGRESSIVE_MERGE],
    )

    both = RoutePreferences(
        avoid_unprotected_lefts=True,
        low_stress_merges=True,
        unprotected_left_penalty_sec=5.0,
        aggressive_merge_penalty_sec=7.0,
    )
    cost = fixture.graph.transition_cost(
        fixture.edge("W", "J"), fixture.edge("J", "N"), both
    )
    assert cost == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# Scoring stays consistent with the search
# ---------------------------------------------------------------------------
def test_score_path_agrees_with_the_search_under_preferences(network):
    """``time_saved_sec`` is a difference of these two; they must not drift."""
    prefs = RoutePreferences(avoid_unprotected_lefts=True)

    for preferences in (None, prefs):
        result = network.route("W", "N", preferences)
        edge_ids = list(result.path_edge_ids)
        search_cost = result.total_duration_sec + result.preference_penalty_sec
        assert network.graph.score_path(
            edge_ids, preferences or RoutePreferences()
        ) == pytest.approx(search_cost, abs=1e-4)


def test_the_direct_route_is_priced_by_score_path(network):
    """Scoring the road not taken is how a caller compares the two."""
    prefs = RoutePreferences(avoid_unprotected_lefts=True)
    direct = network.path(*DIRECT)

    assert network.graph.score_path(direct) == pytest.approx(2 * EDGE_SEC)
    assert network.graph.score_path(direct, prefs) == pytest.approx(2 * EDGE_SEC + 45.0)
    assert network.graph.preference_penalty(direct, prefs) == pytest.approx(45.0)


def test_preferences_leave_an_unflagged_graph_alone():
    """A graph with no flags routes identically however it is asked."""
    fixture = _build(flag_the_left=False)
    assert not fixture.graph.has_turn_flags

    plain = fixture.route("W", "N")
    fussy = fixture.route(
        "W",
        "N",
        RoutePreferences(avoid_unprotected_lefts=True, low_stress_merges=True),
    )

    assert list(plain.path_edge_ids) == list(fussy.path_edge_ids)
    assert plain.total_duration_sec == pytest.approx(fussy.total_duration_sec)
    assert fussy.preference_penalty_sec == 0.0


# ---------------------------------------------------------------------------
# The setter
# ---------------------------------------------------------------------------
def test_flags_accept_a_numpy_table():
    """graph_builder hands these over as numpy arrays."""
    fixture = _build(flag_the_left=False)
    in_ids = np.array([fixture.edge("W", "J")], dtype=np.int32)
    out_ids = np.array([fixture.edge("J", "N")], dtype=np.int32)
    flags = np.array([TURN_FLAG_UNPROTECTED_LEFT], dtype=np.uint8)

    fixture.graph.set_turn_flags(in_ids.tolist(), out_ids.tolist(), flags.tolist())

    assert fixture.graph.has_turn_flags
    assert fixture.graph.turn_flags(int(in_ids[0]), int(out_ids[0])) == 1


def test_flags_survive_a_weight_update(network):
    """The forecaster rewrites travel times every five minutes."""
    network.graph.update_edge_weights([EDGE_SEC * 2] * network.graph.num_edges)

    assert network.graph.has_turn_flags
    result = network.route("W", "N", RoutePreferences(avoid_unprotected_lefts=True))
    # Both routes doubled, so the 40 s detour is now 80 s and the 45 s penalty
    # no longer covers it: the search should take the turn after all.
    assert list(result.path_edge_ids) == network.path(*DIRECT)
    assert result.preference_penalty_sec == pytest.approx(45.0)


def test_changing_the_topology_clears_the_flags(network):
    """The table is indexed off the CSR, so a new edge invalidates it.

    Silently keeping it would map flags onto whichever movements happened to
    land in those slots afterwards -- wrong turns, penalised confidently.
    """
    assert network.graph.has_turn_flags

    network.graph.add_edge(
        network.graph.num_edges, NODE_ID["W"], NODE_ID["N"], 500.0, 50.0
    )

    assert not network.graph.has_turn_flags
    result = network.route("W", "N", RoutePreferences(avoid_unprotected_lefts=True))
    assert result.preference_penalty_sec == 0.0


@pytest.mark.parametrize(
    "in_ids, out_ids, flags, match",
    [
        pytest.param([0], [1, 2], [1], "same length", id="ragged"),
        pytest.param([0], [1], [255], "unknown turn flag", id="unknown-bits"),
        pytest.param([999], [1], [1], "out of range", id="bad-edge-id"),
    ],
)
def test_set_turn_flags_rejects_nonsense(network, in_ids, out_ids, flags, match):
    with pytest.raises((ValueError, IndexError), match=match):
        network.graph.set_turn_flags(in_ids, out_ids, flags)


def test_flags_for_edges_that_do_not_meet_are_rejected(network):
    """A movement between two edges that never touch is not a movement."""
    with pytest.raises(ValueError, match="do not meet"):
        network.graph.set_turn_flags(
            [network.edge("W", "J")], [network.edge("E", "NE")], [1]
        )


def test_querying_a_pair_that_does_not_meet_is_an_error(network):
    with pytest.raises(ValueError, match="do not meet"):
        network.graph.turn_flags(network.edge("W", "J"), network.edge("E", "NE"))


# ---------------------------------------------------------------------------
# The pipeline's turn table drives the engine
# ---------------------------------------------------------------------------
def test_the_builder_and_the_engine_agree_on_direction():
    """A transposed (in, out) pair would be invisible in either half alone.

    ``graph_builder`` decides which movements are unprotected lefts; the engine
    prices them. Both index by edge id and both care which edge comes first,
    so this drives one through the other and checks the flag lands on the turn
    the builder actually meant.
    """
    pytest.importorskip("networkx")
    import networkx as nx

    from src.data.graph_builder import (
        annotate_road_types,
        annotate_signals,
        build_turn_table,
        sorted_edges,
    )

    # The same junction, built the way the pipeline sees it.
    graph = nx.MultiDiGraph(crs="epsg:4326")
    for name, (lat, lon) in COORDS.items():
        graph.add_node(NODE_ID[name], x=lon, y=lat, highway="", street_count=3)
    for u, v in (("W", "J"), ("J", "E"), ("J", "N"), ("E", "NE"), ("N", "NE")):
        tag = "primary" if (u, v) in (("W", "J"), ("J", "E")) else "secondary"
        for a, b in ((u, v), (v, u)):
            graph.add_edge(
                NODE_ID[a], NODE_ID[b], key=0, highway=tag, length=EDGE_M,
                speed_kph=EDGE_M / EDGE_SEC * 3.6,
            )

    annotate_signals(graph)
    annotate_road_types(graph)
    table = build_turn_table(graph)
    assert table.is_unprotected_left.any(), "fixture should contain a left"

    engine = traffic_engine.Graph(len(COORDS))
    for name, (lat, lon) in COORDS.items():
        engine.add_node(NODE_ID[name], lat, lon)
    ids = {}
    for edge_id, (u, v, key) in enumerate(sorted_edges(graph)):
        engine.add_edge(edge_id, u, v, EDGE_M, EDGE_SEC)
        ids[(u, v)] = edge_id

    flags = (
        table.is_unprotected_left.astype(np.uint8) * TURN_FLAG_UNPROTECTED_LEFT
        + table.is_aggressive_merge.astype(np.uint8) * TURN_FLAG_AGGRESSIVE_MERGE
    )
    engine.set_turn_flags(
        table.in_edge_id.tolist(), table.out_edge_id.tolist(), flags.tolist()
    )

    # Every left the builder flagged is priced by the engine, on the same
    # movement and in the same direction.
    prefs = RoutePreferences(avoid_unprotected_lefts=True, unprotected_left_penalty_sec=45.0)
    for row in np.flatnonzero(table.is_unprotected_left):
        in_edge = int(table.in_edge_id[row])
        out_edge = int(table.out_edge_id[row])
        assert engine.turn_flags(in_edge, out_edge) & TURN_FLAG_UNPROTECTED_LEFT
        assert engine.transition_cost(in_edge, out_edge, prefs) == pytest.approx(45.0)
        # A flag on its own costs nothing: it is the preference that prices it.
        assert engine.transition_cost(in_edge, out_edge) == 0.0

    # The builder found lefts in only one direction of the arterial pair, so
    # the flags are not simply set on everything that meets at a junction.
    assert int(table.is_unprotected_left.sum()) < len(table)
    assert ids  # the id map is what tied the two halves together
