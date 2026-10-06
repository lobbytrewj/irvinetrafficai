"""Static road network: artifacts on disk to a populated native graph.

Loads the three graph_builder outputs and turns them into

* the arrays the service needs in Python (coordinates, lengths, free-flow
  times, per-edge polylines for map rendering),
* a ``BallTree`` for snapping a lat/lon to the nearest intersection,
* a factory for ``traffic_engine.Graph`` instances, each populated with all
  6,864 intersections and 15,553 directed segments.

Edge ordering
-------------
Two artifacts disagree about edge order and the difference is silent:
``graph_tensors.pt`` sorts edges by ``(u, v, key)`` while the GeoPackage
preserves graph iteration order. Row 0 of the GeoPackage is a *different road*
from column 0 of ``edge_index``. Every downstream id -- including the 444
``segment_edge_id`` values the forecaster predicts on -- follows the sorted
convention, so the frame is re-sorted on load and then cross-checked against
``edge_index``. Getting this wrong attaches each prediction to the wrong road
while every response still looks perfectly well-formed.

Calibration
-----------
``graph_tensors.pt`` carries free-flow times computed as length over the
posted limit, which describes an Irvine with no traffic signals in it. The
travel times handed to the engine are therefore recomputed here from capped
cruise speeds, and each intersection is classified from the roads that meet
at it so the engine can charge for stopping there. Both come from
``src.data.calibration``, which is also what ``graph_builder`` bakes into new
artifacts -- so a graph rebuilt from OSM and a graph loaded off disk are
calibrated identically. Artifacts that already carry the calibration (a
``node_delay_priority_s`` tensor) are used as they are rather than
reclassified.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.data.calibration import (
    TURN_PENALTY_SEC,
    cap_speeds_kph,
    classify_intersections,
    polyline_bearings,
    road_classes,
)

__all__ = ["NetworkCalibration", "RoadNetwork", "SnapResult"]

LOGGER = logging.getLogger(__name__)

# Matches irvine::kEarthRadiusMeters in cpp_engine/include/router.hpp, so the
# snap distances reported here agree with the engine's A* heuristic.
EARTH_RADIUS_M = 6371008.8


@dataclass(frozen=True)
class SnapResult:
    """A coordinate matched to the nearest intersection."""

    node_id: int
    latitude: float
    longitude: float
    distance_m: float


@dataclass(frozen=True)
class NetworkCalibration:
    """Everything the engine needs beyond topology to produce real drive times.

    Free-flow times are recomputed from capped cruise speeds rather than posted
    limits, so ``free_flow_s`` here is slower than the ``travel_time_s`` column
    in ``graph_tensors.pt`` on every arterial. The delays are per node and the
    bearings per edge; see ``src.data.calibration`` for what each one means.
    """

    node_priority_delay_s: np.ndarray   # float32 [N], charged to every movement
    node_yield_delay_s: np.ndarray      # float32 [N], only to minor movements
    turn_in_edge_id: np.ndarray         # int32   [T], the movement table
    turn_out_edge_id: np.ndarray        # int32   [T]
    turn_flags: np.ndarray              # uint8   [T], engine turn_flag bits
    edge_road_class: np.ndarray         # uint8   [E]
    edge_cruise_kph: np.ndarray         # float64 [E]
    free_flow_s: np.ndarray             # float32 [E], length / capped speed
    entry_bearing_deg: np.ndarray       # float32 [E]
    exit_bearing_deg: np.ndarray        # float32 [E]
    kinds: list[str]                    # per-node label, for reporting

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for label in self.kinds:
            counts[label] = counts.get(label, 0) + 1
        return ", ".join(f"{n} {label}" for label, n in sorted(counts.items()))


class RoadNetwork:
    """Topology, geometry and spatial index for the Irvine drive network."""

    def __init__(
        self,
        node_coords: np.ndarray,
        edge_src: np.ndarray,
        edge_dst: np.ndarray,
        edge_length_m: np.ndarray,
        edge_speed_kph: np.ndarray,
        edge_free_flow_s: np.ndarray,
        edge_geometry: list[np.ndarray],
        edge_label: list[str],
        metadata: dict,
        calibration: NetworkCalibration | None = None,
    ) -> None:
        self.node_coords = node_coords
        self.edge_src = edge_src
        self.edge_dst = edge_dst
        self.edge_length_m = edge_length_m
        # Posted limit as imputed by the graph builder, kept for reference.
        # The speed actually routed on is calibration.edge_cruise_kph.
        self.edge_posted_kph = edge_speed_kph
        self.edge_geometry = edge_geometry
        self.edge_label = edge_label
        self.metadata = metadata
        self.calibration = calibration

        # The free-flow weights the engine is built with. With a calibration
        # loaded these are length over *capped* cruise speed; without one they
        # are the posted-limit times straight out of graph_tensors.pt.
        self.edge_speed_kph = (
            calibration.edge_cruise_kph if calibration else edge_speed_kph
        )
        self.edge_free_flow_s = (
            calibration.free_flow_s if calibration else edge_free_flow_s
        )

        from sklearn.neighbors import BallTree

        # Haversine BallTree over radians. A KD-tree would need a projected
        # CRS to keep its Euclidean splits meaningful; a haversine ball tree
        # works on lat/lon directly and returns great-circle distances.
        self._tree = BallTree(np.radians(node_coords), metric="haversine")

    # -- sizes ---------------------------------------------------------------
    @property
    def num_nodes(self) -> int:
        return int(self.node_coords.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_src.shape[0])

    # -- construction --------------------------------------------------------
    @classmethod
    def load(
        cls,
        mapping_path: Path,
        tensors_path: Path,
        geopackage_path: Path,
    ) -> "RoadNetwork":
        """Read the graph_builder artifacts and validate them against each other."""
        for path in (mapping_path, tensors_path, geopackage_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Run `python -m src.data.graph_builder` first."
                )

        with mapping_path.open(encoding="utf-8") as handle:
            mapping = json.load(handle)

        # weights_only=True: our own plain tensor dict, and it keeps torch.load
        # from being an arbitrary-code-execution surface.
        bundle = torch.load(tensors_path, weights_only=True)
        edge_index = bundle["edge_index"].numpy()
        edge_attr = bundle["edge_attr"].numpy()
        node_coords = np.ascontiguousarray(bundle["pos"].numpy(), dtype=np.float64)

        declared_nodes = int(mapping["num_nodes"])
        declared_edges = int(mapping["num_edges"])
        if node_coords.shape[0] != declared_nodes:
            raise ValueError(
                f"node_mapping.json declares {declared_nodes} nodes but "
                f"graph_tensors.pt carries {node_coords.shape[0]}"
            )
        if edge_index.shape[1] != declared_edges:
            raise ValueError(
                f"node_mapping.json declares {declared_edges} edges but "
                f"graph_tensors.pt carries {edge_index.shape[1]}"
            )

        tables = _load_geopackage(geopackage_path, edge_index, node_coords)

        # edge_attr columns are (length_m, speed_kph, travel_time_s); see
        # EDGE_FEATURE_NAMES in src/data/graph_builder.py.
        calibration = _calibrate(bundle, edge_attr, tables)
        LOGGER.info(
            "Calibrated intersections: %s; arterial cruise capped on %d edges",
            calibration.summary(),
            int((calibration.edge_cruise_kph < edge_attr[:, 1] - 1e-6).sum()),
        )

        return cls(
            node_coords=node_coords,
            edge_src=np.ascontiguousarray(edge_index[0], dtype=np.int64),
            edge_dst=np.ascontiguousarray(edge_index[1], dtype=np.int64),
            edge_length_m=np.ascontiguousarray(edge_attr[:, 0], dtype=np.float64),
            edge_speed_kph=np.ascontiguousarray(edge_attr[:, 1], dtype=np.float64),
            edge_free_flow_s=np.ascontiguousarray(edge_attr[:, 2], dtype=np.float32),
            edge_geometry=tables.geometry,
            edge_label=tables.labels,
            metadata=mapping,
            calibration=calibration,
        )

    def build_engine_graph(self):
        """A finalized ``traffic_engine.Graph`` at free-flow travel times.

        Populated node by node and edge by edge because that is the binding's
        surface. ~22k pybind calls, a few hundred milliseconds once at startup;
        after ``finalize`` the CSR adjacency is fixed and only the weights ever
        move.

        The intersection cost model is installed here too, before ``finalize``,
        so every graph the service holds -- baseline and all three horizons --
        charges the same delays. Without a calibration the graph is a plain
        travel-time router, which is what the engine's own tests expect.
        """
        import traffic_engine

        graph = traffic_engine.Graph(self.num_nodes)
        coords = self.node_coords
        for node_id in range(self.num_nodes):
            graph.add_node(node_id, float(coords[node_id, 0]), float(coords[node_id, 1]))

        src = self.edge_src
        dst = self.edge_dst
        length = self.edge_length_m
        free_flow = self.edge_free_flow_s
        for edge_id in range(self.num_edges):
            graph.add_edge(
                edge_id,
                int(src[edge_id]),
                int(dst[edge_id]),
                float(length[edge_id]),
                float(free_flow[edge_id]),
            )

        calibration = self.calibration
        if calibration is not None and calibration.turn_flags.size:
            # Hazard flags for the preference modes. Inert unless a query asks
            # for them: RoutePreferences defaults to the fastest route.
            graph.set_turn_flags(
                calibration.turn_in_edge_id.tolist(),
                calibration.turn_out_edge_id.tolist(),
                calibration.turn_flags.tolist(),
            )
        if calibration is not None:
            # Bearings before finalize(): finalize derives them from the
            # straight src -> dst chord only when they have not been supplied,
            # and the real geometry is strictly better.
            graph.set_edge_bearings(
                calibration.entry_bearing_deg.tolist(),
                calibration.exit_bearing_deg.tolist(),
            )
            graph.set_edge_road_classes(calibration.edge_road_class.tolist())
            graph.set_node_delays(
                calibration.node_priority_delay_s.tolist(),
                calibration.node_yield_delay_s.tolist(),
            )
            graph.set_turn_penalties(
                traffic_engine.TurnPenalties(
                    through_sec=TURN_PENALTY_SEC["through"],
                    right_sec=TURN_PENALTY_SEC["right"],
                    left_sec=TURN_PENALTY_SEC["left"],
                    uturn_sec=TURN_PENALTY_SEC["uturn"],
                )
            )

        # Explicit, so the build cost is paid here and not charged to the
        # first query that happens to arrive.
        graph.finalize()
        return graph

    # -- spatial queries -----------------------------------------------------
    def snap(self, latitude: float, longitude: float) -> SnapResult:
        """Nearest intersection to a coordinate, with its great-circle distance."""
        query = np.radians([[latitude, longitude]])
        distances, indices = self._tree.query(query, k=1)
        node_id = int(indices[0][0])
        return SnapResult(
            node_id=node_id,
            latitude=float(self.node_coords[node_id, 0]),
            longitude=float(self.node_coords[node_id, 1]),
            # BallTree returns haversine distance in radians.
            distance_m=float(distances[0][0]) * EARTH_RADIUS_M,
        )

    # -- geometry ------------------------------------------------------------
    def polyline(self, edge_ids: list[int]) -> list[list[float]]:
        """``[[lat, lon], ...]`` tracing the road centreline along a path.

        Each edge's geometry runs u -> v, so consecutive edges join end to
        start; the shared vertex is emitted once.
        """
        if not edge_ids:
            return []

        parts: list[np.ndarray] = [self.edge_geometry[edge_ids[0]]]
        parts.extend(self.edge_geometry[edge_id][1:] for edge_id in edge_ids[1:])
        return np.concatenate(parts).round(6).tolist()

    def node_polyline(self, node_ids: list[int]) -> list[list[float]]:
        """Straight-line fallback through a node path."""
        if not node_ids:
            return []
        return self.node_coords[np.asarray(node_ids, dtype=np.int64)].round(6).tolist()


def _turn_flags_from_bundle(
    bundle: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Movement hazard flags for the engine's preference modes.

    Read from ``graph_tensors.pt`` when the artifact carries a turn table, and
    otherwise absent -- which leaves ``RoutePreferences`` inert rather than
    guessed at. Deriving the table here instead would mean re-enumerating
    every movement in the network at startup to reproduce something the
    pipeline already computes, so an artifact built before ``graph_builder``
    grew its turn stage routes without the preference modes until it is
    rebuilt.
    """
    empty_ids = np.zeros(0, dtype=np.int32)
    if "turn_in_edge_id" not in bundle:
        LOGGER.info(
            "graph_tensors.pt has no turn table; routing preferences "
            "(avoid_unprotected_lefts, low_stress_merges) are unavailable "
            "until the graph is rebuilt"
        )
        return empty_ids, empty_ids, np.zeros(0, dtype=np.uint8)

    import traffic_engine

    left = bundle["turn_is_unprotected_left"].numpy().astype(np.uint8)
    merge = bundle["turn_is_aggressive_merge"].numpy().astype(np.uint8)
    flags = left * np.uint8(traffic_engine.TURN_FLAG_UNPROTECTED_LEFT) | (
        merge * np.uint8(traffic_engine.TURN_FLAG_AGGRESSIVE_MERGE)
    )
    LOGGER.info(
        "Turn table: %d movements, %d unprotected lefts, %d aggressive merges",
        flags.shape[0],
        int(left.sum()),
        int(merge.sum()),
    )
    return (
        bundle["turn_in_edge_id"].numpy().astype(np.int32),
        bundle["turn_out_edge_id"].numpy().astype(np.int32),
        flags,
    )


@dataclass(frozen=True)
class _GeoPackageTables:
    """The columns the service reads out of ``irvine_graph.gpkg``."""

    geometry: list[np.ndarray]   # [k, 2] lat/lon polyline per edge id
    labels: list[str]            # road name per edge id
    highways: list[str]          # OSM highway tag per edge id
    names: list[str]             # raw name per edge id, for crossing detection
    node_highways: list[str]     # OSM highway tag per node id


def _load_geopackage(
    geopackage_path: Path, edge_index: np.ndarray, node_coords: np.ndarray
) -> _GeoPackageTables:
    """Per-edge geometry and the tags the calibration classifies from."""
    import geopandas as gpd  # heavy; only needed at startup

    edges = gpd.read_file(geopackage_path, layer="edges")
    if len(edges) != edge_index.shape[1]:
        raise ValueError(
            f"{geopackage_path.name} has {len(edges)} edges but graph_tensors.pt "
            f"has {edge_index.shape[1]}; regenerate both from one graph_builder run."
        )

    # See the module docstring: the GeoPackage is in iteration order, every
    # edge id in the project is in (u, v, key) order.
    edges = edges.sort_values(["u", "v", "key"], kind="stable").reset_index(drop=True)
    if not (
        np.array_equal(edges["u"].to_numpy(), edge_index[0])
        and np.array_equal(edges["v"].to_numpy(), edge_index[1])
    ):
        raise ValueError(
            "edge ordering in the GeoPackage does not match edge_index even "
            "after sorting; regenerate both artifacts from the same "
            "graph_builder run."
        )

    geometry: list[np.ndarray] = []
    for line, u, v in zip(edges.geometry, edge_index[0], edge_index[1]):
        if line is None or line.is_empty:
            # No stored centreline: fall back to the straight u -> v chord, so
            # a path through this edge still renders as a connected polyline.
            geometry.append(np.ascontiguousarray(node_coords[[u, v]]))
            continue
        # Shapely stores (x, y) = (lon, lat); the API speaks [lat, lon].
        coords = np.asarray(line.coords, dtype=np.float64)[:, ::-1]
        geometry.append(np.ascontiguousarray(coords))

    labels = _edge_labels(edges)
    LOGGER.info("Loaded %d edge geometries from %s", len(geometry), geopackage_path.name)

    highways = [str(value) for value in edges["highway"]] if "highway" in edges else []
    names = [
        value.strip() if isinstance(value, str) else ""
        for value in (edges["name"] if "name" in edges else [""] * len(edges))
    ]

    # The node layer is indexed by the same contiguous ids as `pos`, and
    # carries the OSM control tags (traffic_signals, stop) where mappers have
    # recorded them.
    node_highways = [""] * node_coords.shape[0]
    try:
        nodes = gpd.read_file(geopackage_path, layer="nodes")
    except Exception as exc:  # noqa: BLE001 - the layer is optional
        LOGGER.warning("No nodes layer in %s (%s); classifying from road "
                       "class alone", geopackage_path.name, exc)
    else:
        if "highway" in nodes and "osmid" in nodes:
            for index, tag in zip(nodes["osmid"], nodes["highway"]):
                node_id = int(index)
                if 0 <= node_id < len(node_highways) and isinstance(tag, str):
                    node_highways[node_id] = tag

    return _GeoPackageTables(
        geometry=geometry,
        labels=labels,
        highways=highways or ["" for _ in range(edge_index.shape[1])],
        names=names,
        node_highways=node_highways,
    )


def _calibrate(
    bundle: dict, edge_attr: np.ndarray, tables: _GeoPackageTables
) -> NetworkCalibration:
    """Cruise speeds, intersection delays and bearings for this network.

    Artifacts built by a calibrated ``graph_builder`` already carry the node
    delays; anything older is classified here from the GeoPackage tags. Either
    way the constants come from ``src.data.calibration``, so the two paths
    cannot disagree.
    """
    length_m = np.ascontiguousarray(edge_attr[:, 0], dtype=np.float64)
    posted_kph = np.ascontiguousarray(edge_attr[:, 1], dtype=np.float64)

    cruise_kph = cap_speeds_kph(tables.highways, posted_kph)
    # Straight to m/s: length_m / (kph / 3.6). The engine has no notion of
    # kilometres or hours anywhere in it.
    free_flow_s = (length_m / (cruise_kph / 3.6)).astype(np.float32)

    entry_deg, exit_deg = polyline_bearings(tables.geometry)
    num_nodes = len(tables.node_highways)

    if "node_delay_priority_s" in bundle and "node_delay_yield_s" in bundle:
        priority = bundle["node_delay_priority_s"].numpy().astype(np.float32)
        yielding = bundle["node_delay_yield_s"].numpy().astype(np.float32)
        kinds = ["prebuilt"] * num_nodes
        LOGGER.info("Using intersection delays baked into graph_tensors.pt")
    else:
        delays = classify_intersections(
            num_nodes=num_nodes,
            edge_u=np.asarray(bundle["edge_index"][0]),
            edge_v=np.asarray(bundle["edge_index"][1]),
            edge_highway=tables.highways,
            edge_name=tables.names,
            node_highway=tables.node_highways,
        )
        priority, yielding, kinds = (
            delays.priority_sec,
            delays.yield_sec,
            delays.kind,
        )

    turn_in, turn_out, turn_flags = _turn_flags_from_bundle(bundle)

    return NetworkCalibration(
        node_priority_delay_s=priority,
        node_yield_delay_s=yielding,
        turn_in_edge_id=turn_in,
        turn_out_edge_id=turn_out,
        turn_flags=turn_flags,
        edge_road_class=road_classes(tables.highways),
        edge_cruise_kph=cruise_kph,
        free_flow_s=free_flow_s,
        entry_bearing_deg=entry_deg,
        exit_bearing_deg=exit_deg,
        kinds=kinds,
    )


def _edge_labels(edges) -> list[str]:
    """Human-readable road name per edge, preferring the route ref."""
    ref = edges["ref"] if "ref" in edges.columns else None
    name = edges["name"] if "name" in edges.columns else None

    labels: list[str] = []
    for i in range(len(edges)):
        chosen = ""
        for column in (ref, name):
            if column is None:
                continue
            value = column.iloc[i]
            if isinstance(value, str) and value.strip():
                chosen = value.strip()
                break
        labels.append(chosen or "unnamed")
    return labels
