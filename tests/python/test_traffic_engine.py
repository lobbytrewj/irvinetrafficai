"""Unit tests for the traffic_engine pybind11 extension.

Skipped entirely if the extension is not built (``pip install -e .``).

The properties worth pinning here are the ones whose failure is silent. A* with
an inadmissible heuristic still returns *a* path, and a weight update that
lands in the canonical edge array but not the CSR mirror the search traverses
still leaves every route self-consistent. Both would pass a smoke test and be
wrong in production, so both are asserted against an independent brute-force
shortest path.
"""

from __future__ import annotations

import heapq
import math

import pytest

traffic_engine = pytest.importorskip(
    "traffic_engine", reason="native extension not built; run `pip install -e .`"
)


# A 4x4 lattice of intersections roughly the size of a few Irvine blocks.
GRID = 4
LAT0, LON0 = 33.6600, -117.8300
SPACING_DEG = 0.004


def _node_id(row: int, col: int) -> int:
    return row * GRID + col


def _coords(node: int) -> tuple[float, float]:
    row, col = divmod(node, GRID)
    return LAT0 + row * SPACING_DEG, LON0 + col * SPACING_DEG


def _lattice_edges() -> list[tuple[int, int, int]]:
    """``(edge_id, u, v)`` for a bidirectional 4-connected lattice."""
    edges: list[tuple[int, int, int]] = []
    for row in range(GRID):
        for col in range(GRID):
            here = _node_id(row, col)
            for drow, dcol in ((0, 1), (1, 0)):
                nrow, ncol = row + drow, col + dcol
                if nrow < GRID and ncol < GRID:
                    there = _node_id(nrow, ncol)
                    edges.append((len(edges), here, there))
                    edges.append((len(edges), there, here))
    return edges


def _build(travel_times: dict[int, float] | None = None):
    """Lattice graph; every edge is 400 m at 20 s unless overridden."""
    graph = traffic_engine.Graph(GRID * GRID)
    for node in range(GRID * GRID):
        lat, lon = _coords(node)
        graph.add_node(node, lat, lon)

    edges = _lattice_edges()
    for edge_id, u, v in edges:
        length = traffic_engine.haversine_meters(*_coords(u), *_coords(v))
        seconds = (travel_times or {}).get(edge_id, 20.0)
        graph.add_edge(edge_id, u, v, length, seconds)
    graph.finalize()
    return graph, edges


def _brute_force(edges, travel_times, num_nodes, start, goal) -> float:
    """Textbook Dijkstra, as an oracle independent of the C++ implementation."""
    adjacency: dict[int, list[tuple[int, float]]] = {n: [] for n in range(num_nodes)}
    for edge_id, u, v in edges:
        adjacency[u].append((v, travel_times[edge_id]))

    best = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        cost, node = heapq.heappop(heap)
        if node == goal:
            return cost
        if cost > best.get(node, math.inf):
            continue
        for neighbour, weight in adjacency[node]:
            candidate = cost + weight
            if candidate < best.get(neighbour, math.inf):
                best[neighbour] = candidate
                heapq.heappush(heap, (candidate, neighbour))
    return math.inf


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------
def test_module_exposes_the_documented_api() -> None:
    for name in ("Graph", "Node", "Edge", "RouteResult", "haversine_meters"):
        assert hasattr(traffic_engine, name)


def test_haversine_matches_a_known_distance() -> None:
    # One degree of latitude is ~111.2 km anywhere on the sphere.
    metres = traffic_engine.haversine_meters(33.0, -117.0, 34.0, -117.0)
    assert metres == pytest.approx(111195.0, rel=1e-3)
    assert traffic_engine.haversine_meters(33.0, -117.0, 33.0, -117.0) == 0.0


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def test_graph_reports_its_size() -> None:
    graph, edges = _build()
    assert graph.num_nodes == GRID * GRID
    assert graph.num_edges == len(edges)
    assert graph.is_finalized
    assert len(graph) == GRID * GRID


def test_edge_and_node_accessors_round_trip() -> None:
    graph, edges = _build()
    edge_id, u, v = edges[3]
    edge = graph.edge(edge_id)
    assert (edge.id, edge.src, edge.dst) == (edge_id, u, v)
    assert graph.node(u).lat == pytest.approx(_coords(u)[0])


def test_out_of_range_ids_raise() -> None:
    graph, _ = _build()
    with pytest.raises(IndexError):
        graph.node(999)
    with pytest.raises(IndexError):
        graph.edge(999)
    with pytest.raises(IndexError):
        graph.find_shortest_path(0, 999)


def test_negative_travel_time_is_rejected() -> None:
    """A negative arc would break A*'s optimality guarantee silently."""
    graph = traffic_engine.Graph(2)
    graph.add_node(0, 33.0, -117.0)
    graph.add_node(1, 33.1, -117.0)
    with pytest.raises(ValueError, match="negative"):
        graph.add_edge(0, 0, 1, 100.0, -1.0)
    with pytest.raises(ValueError):
        graph.add_edge(1, 0, 1, 100.0, float("nan"))


def test_sparse_edge_ids_are_rejected() -> None:
    graph = traffic_engine.Graph(2)
    graph.add_node(0, 33.0, -117.0)
    graph.add_node(1, 33.1, -117.0)
    graph.add_edge(3, 0, 1, 100.0, 10.0)  # leaves 0..2 unassigned
    with pytest.raises(ValueError, match="dense"):
        graph.finalize()


def test_missing_node_is_rejected() -> None:
    graph = traffic_engine.Graph(3)
    graph.add_node(0, 33.0, -117.0)
    graph.add_node(1, 33.1, -117.0)
    with pytest.raises(ValueError, match="never added"):
        graph.finalize()


# ---------------------------------------------------------------------------
# Pathfinding
# ---------------------------------------------------------------------------
def test_route_matches_brute_force_dijkstra() -> None:
    """A* must return the optimal cost, not merely a valid path."""
    weights = {edge_id: 5.0 + (edge_id % 7) * 9.0 for edge_id, _, _ in _lattice_edges()}
    graph, edges = _build(weights)

    for start in range(GRID * GRID):
        for goal in range(GRID * GRID):
            result = graph.find_shortest_path(start, goal)
            expected = _brute_force(edges, weights, GRID * GRID, start, goal)
            assert result.found
            assert result.total_duration_sec == pytest.approx(expected, abs=1e-4)


def test_path_is_connected_and_consistent() -> None:
    graph, _ = _build()
    result = graph.find_shortest_path(0, GRID * GRID - 1)

    assert result.path_node_ids[0] == 0
    assert result.path_node_ids[-1] == GRID * GRID - 1
    assert len(result.path_edge_ids) == len(result.path_node_ids) - 1

    # Every edge must actually join its two reported nodes.
    total_seconds = 0.0
    total_metres = 0.0
    for edge_id, u, v in zip(
        result.path_edge_ids, result.path_node_ids, result.path_node_ids[1:]
    ):
        edge = graph.edge(edge_id)
        assert (edge.src, edge.dst) == (u, v)
        total_seconds += edge.travel_time_sec
        total_metres += edge.length_meters

    assert result.total_duration_sec == pytest.approx(total_seconds, abs=1e-4)
    assert result.total_distance_meters == pytest.approx(total_metres, abs=1e-3)


def test_self_route_is_empty_but_found() -> None:
    graph, _ = _build()
    result = graph.find_shortest_path(5, 5)
    assert result.found
    assert result.path_node_ids == [5]
    assert result.path_edge_ids == []
    assert result.total_duration_sec == 0.0


def test_unreachable_target_reports_not_found() -> None:
    graph = traffic_engine.Graph(3)
    for node in range(3):
        graph.add_node(node, 33.0 + node * 0.01, -117.0)
    graph.add_edge(0, 0, 1, 100.0, 10.0)  # node 2 is isolated
    result = graph.find_shortest_path(0, 2)
    assert not result.found
    assert not bool(result)
    assert result.path_node_ids == []


def test_heuristic_does_not_change_the_answer() -> None:
    """A* and Dijkstra must agree; if they don't, the heuristic is inadmissible.

    Making one edge implausibly fast raises max_speed_mps and so shrinks the
    heuristic. An admissible bound can only lose search efficiency that way,
    never optimality -- so the costs must be identical either way.
    """
    weights = {edge_id: 12.0 + (edge_id % 5) * 6.0 for edge_id, _, _ in _lattice_edges()}
    slow_graph, edges = _build(weights)

    fast_weights = dict(weights)
    fast_weights[0] = 0.01  # an absurdly fast edge inflates max_speed_mps
    fast_graph, _ = _build(fast_weights)

    assert fast_graph.max_speed_mps > slow_graph.max_speed_mps
    for goal in range(1, GRID * GRID):
        expected = _brute_force(edges, fast_weights, GRID * GRID, 0, goal)
        assert fast_graph.find_shortest_path(0, goal).total_duration_sec == pytest.approx(
            expected, abs=1e-4
        )


def test_repeated_queries_do_not_leak_state() -> None:
    """Scratch is reused across queries via a visit stamp; it must reset."""
    graph, _ = _build()
    first = graph.find_shortest_path(0, GRID * GRID - 1)
    for _ in range(50):
        graph.find_shortest_path(1, 7)
        graph.find_shortest_path(11, 2)
    repeat = graph.find_shortest_path(0, GRID * GRID - 1)
    assert repeat.total_duration_sec == pytest.approx(first.total_duration_sec)
    assert list(repeat.path_node_ids) == list(first.path_node_ids)


# ---------------------------------------------------------------------------
# Dynamic weights
# ---------------------------------------------------------------------------
def test_update_edge_weights_reroutes() -> None:
    """The update has to reach the CSR the search traverses, not just edges_."""
    graph, edges = _build()
    detour = graph.find_shortest_path(0, _node_id(1, 1))
    assert detour.found

    # Make the first leg of the chosen route ruinously slow.
    weights = [20.0] * len(edges)
    weights[detour.path_edge_ids[0]] = 5000.0
    graph.update_edge_weights(weights)

    rerouted = graph.find_shortest_path(0, _node_id(1, 1))
    assert detour.path_edge_ids[0] not in rerouted.path_edge_ids
    assert rerouted.total_duration_sec == pytest.approx(
        _brute_force(edges, weights, GRID * GRID, 0, _node_id(1, 1)), abs=1e-4
    )


def test_update_edge_weights_preserves_topology() -> None:
    graph, edges = _build()
    before = [(graph.edge(i).src, graph.edge(i).dst) for i, _, _ in edges]
    graph.update_edge_weights([float(i + 1) for i in range(len(edges))])
    after = [(graph.edge(i).src, graph.edge(i).dst) for i, _, _ in edges]
    assert before == after
    assert graph.edge(3).travel_time_sec == pytest.approx(4.0)


def test_update_edge_weights_rejects_wrong_length() -> None:
    graph, edges = _build()
    with pytest.raises(ValueError, match="expected"):
        graph.update_edge_weights([1.0] * (len(edges) - 1))


def test_update_edge_weights_subset_matches_full_update() -> None:
    """The O(k) path must be equivalent to the O(E) one on the same edges."""
    graph, edges = _build()
    full = [20.0] * len(edges)
    for edge_id in (0, 5, 9):
        full[edge_id] = 300.0

    graph.update_edge_weights(full)
    reference = graph.find_shortest_path(0, GRID * GRID - 1)

    fresh, _ = _build()
    fresh.update_edge_weights_subset([0, 5, 9], [300.0, 300.0, 300.0])
    subset = fresh.find_shortest_path(0, GRID * GRID - 1)

    assert subset.total_duration_sec == pytest.approx(reference.total_duration_sec)
    assert list(subset.path_edge_ids) == list(reference.path_edge_ids)


def test_update_edge_weights_subset_validates_input() -> None:
    graph, _ = _build()
    with pytest.raises(ValueError, match="same length"):
        graph.update_edge_weights_subset([0, 1], [1.0])
    with pytest.raises(IndexError):
        graph.update_edge_weights_subset([9999], [1.0])
    with pytest.raises(ValueError, match="negative"):
        graph.update_edge_weights_subset([0], [-1.0])


def test_max_speed_tracks_weight_updates() -> None:
    """A stale speed bound would make the heuristic inadmissible."""
    graph, edges = _build()
    original = graph.max_speed_mps
    graph.update_edge_weights([1.0] * len(edges))  # everything much faster
    assert graph.max_speed_mps > original
