"""Verify and benchmark the native routing engine against NetworkX.

What this checks
----------------
1. **Topology round-trip** -- the C++ graph built from ``irvine_graph.gpkg``
   has the same nodes, edges and endpoints as the NetworkX mirror, and its
   edge ids line up with the ones ``graph_builder`` assigned. If they did not,
   the forecaster's predictions would land on the wrong roads.
2. **Correctness** -- over a sample of origin/destination pairs, the C++ A*
   returns the same route *cost* as ``networkx.astar_path``. Costs, not node
   sequences: equal-cost routes are common on a grid, and which one you get is
   a tie-break, not a result.
3. **Dynamic weights** -- STGNN predictions for the 444 corridor segments are
   fed through ``update_edge_weights`` and the routes are re-verified, so the
   in-place update is checked against NetworkX rather than just timed.
4. **Latency** -- per-query wall time for both engines, and for both forms of
   the weight update.

The comparison is deliberately like-for-like: NetworkX gets a MultiDiGraph
with the same parallel edges, the same travel times, and the same admissible
Haversine heuristic (metres divided by the fastest edge in the graph). The
remaining difference is the implementation, which is the point.

Usage
-----
    python -m src.router.test_cpp_bridge
    python -m src.router.test_cpp_bridge --pairs 200 --horizon 60
    python -m src.router.test_cpp_bridge --no-networkx     # latency only
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)

DEFAULT_PROCESSED_DIR = Path("data/processed")
GRAPH_FILENAME = "irvine_graph.gpkg"
MAPPING_FILENAME = "node_mapping.json"
DATASET_FILENAME = "traffic_dataset.pt"
ONNX_FILENAME = "traffic_stgnn.onnx"

EARTH_RADIUS_M = 6_371_008.8


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------
@dataclass
class RoadNetwork:
    """The processed Irvine network, in the id space ``graph_builder`` defined.

    Arrays are parallel and indexed by edge id, which is the edge's column in
    ``graph_tensors.pt`` and the index the forecaster's ``segment_edge_id``
    points into.
    """

    latitude: np.ndarray  # [N]
    longitude: np.ndarray  # [N]
    source: np.ndarray  # [E]
    target: np.ndarray  # [E]
    key: np.ndarray  # [E] parallel-edge discriminator
    length_m: np.ndarray  # [E]
    free_flow_sec: np.ndarray  # [E]

    @property
    def num_nodes(self) -> int:
        return int(self.latitude.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.source.shape[0])


def load_road_network(processed_dir: Path = DEFAULT_PROCESSED_DIR) -> RoadNetwork:
    """Read the GeoPackage and node mapping into flat arrays.

    Edges are re-sorted by ``(u, v, key)``. The GeoPackage preserves graph
    iteration order, while ``graph_builder`` assigns edge ids by that sort, so
    the sort is what makes column ``i`` here the same road as edge ``i``
    everywhere else in the project.
    """
    import geopandas as gpd  # imported lazily: heavy, only needed here

    graph_path = processed_dir / GRAPH_FILENAME
    mapping_path = processed_dir / MAPPING_FILENAME
    for path in (graph_path, mapping_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m src.data.graph_builder` first."
            )

    edges = gpd.read_file(graph_path, layer="edges")
    edges = edges.sort_values(["u", "v", "key"], kind="stable").reset_index(drop=True)

    with mapping_path.open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    coordinates = np.asarray(mapping["coordinates"], dtype=np.float64)

    network = RoadNetwork(
        latitude=coordinates[:, 0].copy(),
        longitude=coordinates[:, 1].copy(),
        source=edges["u"].to_numpy(dtype=np.int64).copy(),
        target=edges["v"].to_numpy(dtype=np.int64).copy(),
        key=edges["key"].to_numpy(dtype=np.int64).copy(),
        length_m=edges["length"].to_numpy(dtype=np.float64).copy(),
        free_flow_sec=edges["travel_time"].to_numpy(dtype=np.float64).copy(),
    )
    if network.num_nodes != int(mapping["num_nodes"]):
        raise ValueError("node_mapping.json disagrees with its own coordinate list")
    LOGGER.info(
        "Loaded %s: %d nodes, %d edges (%d parallel)",
        graph_path,
        network.num_nodes,
        network.num_edges,
        network.num_edges - len(set(zip(network.source, network.target))),
    )
    return network


def verify_edge_ids(
    network: RoadNetwork, processed_dir: Path = DEFAULT_PROCESSED_DIR
) -> None:
    """Cross-check edge ordering against ``graph_tensors.pt``.

    Silent misalignment here would attach every forecast to the wrong road, so
    it is worth a hard check rather than a comment asserting the invariant.
    """
    import torch

    tensor_path = processed_dir / "graph_tensors.pt"
    if not tensor_path.exists():
        LOGGER.warning("%s absent; skipping edge-id cross-check", tensor_path)
        return

    bundle = torch.load(tensor_path, weights_only=True)
    edge_index = bundle["edge_index"].numpy()
    if not (
        np.array_equal(edge_index[0], network.source)
        and np.array_equal(edge_index[1], network.target)
    ):
        raise ValueError(
            "GeoPackage edge order does not match graph_tensors.pt; regenerate "
            "both artifacts from the same graph_builder run."
        )
    LOGGER.info("Edge ids agree with graph_tensors.pt across %d edges", network.num_edges)


# ---------------------------------------------------------------------------
# Engine construction
# ---------------------------------------------------------------------------
def build_native_graph(network: RoadNetwork):
    """Populate a ``traffic_engine.Graph``. Returns it finalized."""
    import traffic_engine

    started = time.perf_counter()
    graph = traffic_engine.Graph(network.num_nodes)
    for node_id in range(network.num_nodes):
        graph.add_node(node_id, float(network.latitude[node_id]),
                       float(network.longitude[node_id]))
    for edge_id in range(network.num_edges):
        graph.add_edge(
            edge_id,
            int(network.source[edge_id]),
            int(network.target[edge_id]),
            float(network.length_m[edge_id]),
            float(network.free_flow_sec[edge_id]),
        )
    graph.finalize()
    LOGGER.info(
        "Built %r in %.0f ms (max speed %.2f m/s)",
        graph,
        (time.perf_counter() - started) * 1e3,
        graph.max_speed_mps,
    )
    return graph


def build_networkx_graph(network: RoadNetwork):
    """Mirror the same topology into a NetworkX MultiDiGraph.

    A MultiDiGraph, not a DiGraph: the network has 413 parallel edges, and
    collapsing them would hand NetworkX a smaller graph to search than the C++
    engine gets.
    """
    import networkx as nx

    started = time.perf_counter()
    graph = nx.MultiDiGraph()
    for node_id in range(network.num_nodes):
        graph.add_node(
            node_id,
            lat=float(network.latitude[node_id]),
            lon=float(network.longitude[node_id]),
        )
    for edge_id in range(network.num_edges):
        graph.add_edge(
            int(network.source[edge_id]),
            int(network.target[edge_id]),
            key=int(network.key[edge_id]),
            edge_id=edge_id,
            length=float(network.length_m[edge_id]),
            travel_time=float(network.free_flow_sec[edge_id]),
        )
    LOGGER.info(
        "Built NetworkX MultiDiGraph in %.0f ms (%d nodes, %d edges)",
        (time.perf_counter() - started) * 1e3,
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph


def apply_weights_to_networkx(graph, travel_times: np.ndarray) -> None:
    """Push the same travel times into the NetworkX mirror."""
    for _, _, data in graph.edges(data=True):
        data["travel_time"] = float(travel_times[data["edge_id"]])


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
def predict_travel_times(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    window: int = 0,
    horizon_minutes: int = 15,
    split: str = "test",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run one ONNX forecast. Returns ``(edge_ids, seconds, info)``.

    The exported graph takes raw features and returns seconds, so the scalers
    live inside the model rather than being re-applied here.
    """
    import onnxruntime as ort

    from src.data.generate_traffic_data import TrafficDataset, load_bundle

    onnx_path = processed_dir / ONNX_FILENAME
    dataset_path = processed_dir / DATASET_FILENAME
    for path in (onnx_path, dataset_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m src.pipelines.train` first."
            )

    bundle = load_bundle(dataset_path)
    horizons = [int(h) * int(bundle["step_minutes"]) for h in bundle["horizons"]]
    if horizon_minutes not in horizons:
        raise ValueError(
            f"horizon {horizon_minutes} min not in the model's {horizons}"
        )
    horizon_index = horizons.index(horizon_minutes)

    # normalise=False: the ONNX graph does its own scaling.
    dataset = TrafficDataset(bundle, split=split, normalise=False)
    features, truth = dataset[window % len(dataset)]

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    started = time.perf_counter()
    predicted = session.run(
        None, {"features": features.unsqueeze(0).numpy()}
    )[0][0]
    inference_ms = (time.perf_counter() - started) * 1e3

    edge_ids = bundle["segment_edge_id"].numpy().astype(np.int64)
    seconds = predicted[:, horizon_index].astype(np.float64)
    info = {
        "inference_ms": inference_ms,
        "num_segments": int(edge_ids.shape[0]),
        "horizon_minutes": horizon_minutes,
        "mae_vs_truth_s": float(
            np.abs(seconds - truth[:, horizon_index].numpy()).mean()
        ),
        "free_flow_s": bundle["segment_free_flow_time_s"].numpy().astype(np.float64),
    }
    return edge_ids, seconds, info


def scatter_predictions(
    network: RoadNetwork, edge_ids: np.ndarray, seconds: np.ndarray
) -> np.ndarray:
    """Full-length travel-time vector: predictions where we have them.

    The forecaster covers 444 corridor segments of 15,553 edges. Everything
    else keeps its free-flow time, which is the honest default -- a residential
    street with no sensor is not congested as far as we know.
    """
    travel_times = network.free_flow_sec.copy()
    travel_times[edge_ids] = seconds
    return travel_times


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------
def haversine_meters(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, a)))


def networkx_astar_cost(graph, network: RoadNetwork, max_speed_mps: float):
    """Return a ``(start, goal) -> (cost, path)`` callable using NetworkX A*.

    Uses the same admissible heuristic as the C++ engine: Haversine metres
    over the fastest edge in the graph. Anything else would be benchmarking
    two different searches.
    """
    import networkx as nx

    latitude, longitude = network.latitude, network.longitude

    def route(start: int, goal: int) -> tuple[float, list[int]]:
        goal_lat, goal_lon = float(latitude[goal]), float(longitude[goal])

        def heuristic(node: int, _target: int) -> float:
            if max_speed_mps <= 0.0:
                return 0.0
            return (
                haversine_meters(
                    float(latitude[node]), float(longitude[node]), goal_lat, goal_lon
                )
                / max_speed_mps
            )

        try:
            path = nx.astar_path(
                graph, start, goal, heuristic=heuristic, weight="travel_time"
            )
        except nx.NetworkXNoPath:
            return math.inf, []
        cost = sum(
            min(
                data["travel_time"]
                for data in graph.get_edge_data(u, v).values()
            )
            for u, v in zip(path, path[1:])
        )
        return cost, path

    return route


def sample_od_pairs(
    network: RoadNetwork, count: int, seed: int, min_separation_m: float = 3000.0
) -> list[tuple[int, int]]:
    """Random origin/destination pairs at least ``min_separation_m`` apart.

    The graph is the largest strongly connected component, so every pair is
    reachable. The separation floor keeps the sample off trivial neighbouring
    intersections, where any router looks fast.
    """
    rng = np.random.default_rng(seed)
    pairs: list[tuple[int, int]] = []
    attempts = 0
    while len(pairs) < count and attempts < count * 200:
        attempts += 1
        start, goal = (int(x) for x in rng.integers(0, network.num_nodes, 2))
        if start == goal:
            continue
        separation = haversine_meters(
            float(network.latitude[start]),
            float(network.longitude[start]),
            float(network.latitude[goal]),
            float(network.longitude[goal]),
        )
        if separation >= min_separation_m:
            pairs.append((start, goal))
    if len(pairs) < count:
        raise RuntimeError(
            f"only found {len(pairs)}/{count} pairs at least "
            f"{min_separation_m:.0f} m apart"
        )
    return pairs


# ---------------------------------------------------------------------------
# Verification and benchmarking
# ---------------------------------------------------------------------------
def verify_agreement(
    native_graph,
    route_networkx: Callable[[int, int], tuple[float, list[int]]],
    pairs: Sequence[tuple[int, int]],
    tolerance_s: float = 1e-2,
    label: str = "",
) -> dict[str, Any]:
    """Compare route costs pair by pair. Raises on any real disagreement."""
    mismatches: list[str] = []
    deltas: list[float] = []
    unreachable = 0

    for start, goal in pairs:
        native = native_graph.find_shortest_path(start, goal)
        expected_cost, expected_path = route_networkx(start, goal)

        if not native.found or not expected_path:
            if native.found != bool(expected_path):
                mismatches.append(
                    f"{start}->{goal}: reachability disagrees "
                    f"(native={native.found}, networkx={bool(expected_path)})"
                )
            unreachable += 1
            continue

        delta = abs(native.total_duration_sec - expected_cost)
        deltas.append(delta)
        if delta > tolerance_s:
            mismatches.append(
                f"{start}->{goal}: {native.total_duration_sec:.4f}s vs "
                f"{expected_cost:.4f}s (delta {delta:.4f}s)"
            )

    if mismatches:
        raise AssertionError(
            f"{len(mismatches)}/{len(pairs)} routes disagree with NetworkX"
            + (f" [{label}]" if label else "")
            + ":\n  "
            + "\n  ".join(mismatches[:10])
        )

    summary = {
        "pairs": len(pairs),
        "unreachable": unreachable,
        "max_delta_s": max(deltas) if deltas else 0.0,
        "mean_delta_s": statistics.fmean(deltas) if deltas else 0.0,
    }
    LOGGER.info(
        "%-22s %d/%d routes agree with NetworkX (max delta %.2e s)",
        (label + ":") if label else "agreement:",
        len(deltas),
        len(pairs),
        summary["max_delta_s"],
    )
    return summary


def measure_reroute_impact(
    native_graph, pairs: Sequence[tuple[int, int]], baseline: Sequence[Any]
) -> dict[str, Any]:
    """How much the new weights actually moved the routes.

    Route costs agreeing with NetworkX proves the search is right; it does not
    prove the update reached the search. If a weight write landed in ``edges_``
    but not the CSR mirror the router traverses, every route would still be
    self-consistent and every cost would still match -- and every forecast
    would be ignored. This is the assertion that catches that: real predictions
    on a congested corridor have to change some routes.
    """
    changed_duration = 0
    rerouted = 0
    deltas: list[float] = []

    for (start, goal), before in zip(pairs, baseline):
        after = native_graph.find_shortest_path(start, goal)
        if not (before.found and after.found):
            continue
        delta = after.total_duration_sec - before.total_duration_sec
        deltas.append(delta)
        if abs(delta) > 1e-3:
            changed_duration += 1
        if list(after.path_edge_ids) != list(before.path_edge_ids):
            rerouted += 1

    if changed_duration == 0:
        raise AssertionError(
            "no route cost changed after update_edge_weights; the new travel "
            "times are not reaching the search"
        )
    return {
        "routes_with_changed_duration": changed_duration,
        "routes_rerouted": rerouted,
        "pairs": len(deltas),
        "mean_duration_delta_s": statistics.fmean(deltas) if deltas else 0.0,
        "max_duration_delta_s": max(deltas) if deltas else 0.0,
    }


def _latency_stats(samples_ms: Iterable[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    if not ordered:
        return {}

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
        return ordered[index]

    return {
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": ordered[-1],
    }


def benchmark_routing(
    native_graph,
    route_networkx: Callable[[int, int], tuple[float, list[int]]] | None,
    pairs: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Time both engines over the same OD pairs."""
    native_ms: list[float] = []
    expanded: list[int] = []
    for start, goal in pairs:
        started = time.perf_counter()
        result = native_graph.find_shortest_path(start, goal)
        native_ms.append((time.perf_counter() - started) * 1e3)
        expanded.append(int(result.nodes_expanded))

    report: dict[str, Any] = {
        "native": _latency_stats(native_ms),
        "nodes_expanded_mean": statistics.fmean(expanded),
    }

    if route_networkx is not None:
        networkx_ms: list[float] = []
        for start, goal in pairs:
            started = time.perf_counter()
            route_networkx(start, goal)
            networkx_ms.append((time.perf_counter() - started) * 1e3)
        report["networkx"] = _latency_stats(networkx_ms)
        report["speedup_mean"] = (
            report["networkx"]["mean_ms"] / report["native"]["mean_ms"]
        )
        report["speedup_p99"] = (
            report["networkx"]["p99_ms"] / report["native"]["p99_ms"]
        )
    return report


def benchmark_updates(
    native_graph,
    travel_times: np.ndarray,
    edge_ids: np.ndarray,
    seconds: np.ndarray,
    repeats: int = 200,
) -> dict[str, Any]:
    """Time both weight-update paths: whole network and corridor subset."""
    full = travel_times.astype(np.float32).tolist()
    subset_ids = edge_ids.astype(np.int64).tolist()
    subset_values = seconds.astype(np.float32).tolist()

    for _ in range(5):  # warm up
        native_graph.update_edge_weights(full)
        native_graph.update_edge_weights_subset(subset_ids, subset_values)

    started = time.perf_counter()
    for _ in range(repeats):
        native_graph.update_edge_weights(full)
    full_ms = (time.perf_counter() - started) / repeats * 1e3

    started = time.perf_counter()
    for _ in range(repeats):
        native_graph.update_edge_weights_subset(subset_ids, subset_values)
    subset_ms = (time.perf_counter() - started) / repeats * 1e3

    return {
        "full_update_ms": full_ms,
        "full_update_edges": len(full),
        "subset_update_ms": subset_ms,
        "subset_update_edges": len(subset_ids),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _log_latency(name: str, stats: dict[str, float]) -> None:
    LOGGER.info(
        "  %-12s mean %8.3f ms | p50 %8.3f | p95 %8.3f | p99 %8.3f | max %8.3f",
        name,
        stats["mean_ms"],
        stats["p50_ms"],
        stats["p95_ms"],
        stats["p99_ms"],
        stats["max_ms"],
    )


def run(
    processed_dir: Path,
    pairs_count: int,
    seed: int,
    horizon_minutes: int,
    window: int,
    use_networkx: bool,
    target_ms: float,
) -> dict[str, Any]:
    """Full verification + benchmark run. Returns the report."""
    network = load_road_network(processed_dir)
    verify_edge_ids(network, processed_dir)

    native = build_native_graph(network)
    if native.num_nodes != network.num_nodes or native.num_edges != network.num_edges:
        raise AssertionError("native graph size does not match the source network")

    nx_graph = build_networkx_graph(network) if use_networkx else None
    pairs = sample_od_pairs(network, pairs_count, seed)
    LOGGER.info("Sampled %d origin/destination pairs (seed %d)", len(pairs), seed)

    report: dict[str, Any] = {"nodes": network.num_nodes, "edges": network.num_edges}

    # -- free-flow weights --------------------------------------------------
    route_nx = (
        networkx_astar_cost(nx_graph, network, native.max_speed_mps)
        if nx_graph is not None
        else None
    )
    if route_nx is not None:
        report["agreement_free_flow"] = verify_agreement(
            native, route_nx, pairs, label="free flow"
        )
    # Snapshot the free-flow routes so the forecast leg can show what moved.
    baseline = [native.find_shortest_path(start, goal) for start, goal in pairs]

    # -- forecast weights ---------------------------------------------------
    try:
        edge_ids, seconds, info = predict_travel_times(
            processed_dir, window=window, horizon_minutes=horizon_minutes
        )
    except FileNotFoundError as exc:
        LOGGER.warning("Skipping the forecast leg: %s", exc)
        edge_ids = seconds = None
        info = {}
    else:
        travel_times = scatter_predictions(network, edge_ids, seconds)
        changed = np.abs(travel_times - network.free_flow_sec) > 1e-6
        LOGGER.info(
            "STGNN %d-min forecast: %d segments in %.2f ms, MAE vs truth %.2f s; "
            "%d/%d edges re-weighted, corridor time %+.1f%% vs free flow",
            info["horizon_minutes"],
            info["num_segments"],
            info["inference_ms"],
            info["mae_vs_truth_s"],
            int(changed.sum()),
            network.num_edges,
            100.0 * (seconds.sum() / info["free_flow_s"].sum() - 1.0),
        )

        native.update_edge_weights(travel_times.astype(np.float32).tolist())

        impact = measure_reroute_impact(native, pairs, baseline)
        report["reroute_impact"] = impact
        LOGGER.info(
            "Forecast changed %d/%d route costs and rerouted %d "
            "(mean %+.1f s, max %+.1f s)",
            impact["routes_with_changed_duration"],
            impact["pairs"],
            impact["routes_rerouted"],
            impact["mean_duration_delta_s"],
            impact["max_duration_delta_s"],
        )

        if nx_graph is not None:
            apply_weights_to_networkx(nx_graph, travel_times)
            route_nx = networkx_astar_cost(nx_graph, network, native.max_speed_mps)
            report["agreement_forecast"] = verify_agreement(
                native, route_nx, pairs, label="forecast weights"
            )
        report["updates"] = benchmark_updates(native, travel_times, edge_ids, seconds)
        LOGGER.info(
            "update_edge_weights: %.3f ms for %d edges | subset: %.3f ms for %d",
            report["updates"]["full_update_ms"],
            report["updates"]["full_update_edges"],
            report["updates"]["subset_update_ms"],
            report["updates"]["subset_update_edges"],
        )

    # -- latency ------------------------------------------------------------
    LOGGER.info("Pathfinding latency over %d queries:", len(pairs))
    timings = benchmark_routing(native, route_nx, pairs)
    report["latency"] = timings
    _log_latency("traffic_engine", timings["native"])
    if "networkx" in timings:
        _log_latency("networkx", timings["networkx"])
        LOGGER.info(
            "  speedup: %.0fx mean, %.0fx p99",
            timings["speedup_mean"],
            timings["speedup_p99"],
        )
    LOGGER.info("  mean nodes expanded: %.0f", timings["nodes_expanded_mean"])

    p99 = timings["native"]["p99_ms"]
    report["meets_target"] = bool(p99 < target_ms)
    LOGGER.info(
        "Target < %.1f ms/query: %s (p99 %.3f ms)",
        target_ms,
        "PASS" if report["meets_target"] else "FAIL",
        p99,
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--horizon", type=int, default=15, help="forecast horizon in minutes"
    )
    parser.add_argument(
        "--window", type=int, default=0, help="index into the test split"
    )
    parser.add_argument(
        "--no-networkx",
        action="store_true",
        help="skip the NetworkX comparison (latency only)",
    )
    parser.add_argument(
        "--target-ms", type=float, default=2.0, help="per-query p99 budget"
    )
    parser.add_argument("--json", type=Path, default=None, help="write the report here")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    )
    report = run(
        processed_dir=args.processed_dir,
        pairs_count=args.pairs,
        seed=args.seed,
        horizon_minutes=args.horizon,
        window=args.window,
        use_networkx=not args.no_networkx,
        target_ms=args.target_ms,
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        LOGGER.info("Wrote %s", args.json)
    return 0 if report["meets_target"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
