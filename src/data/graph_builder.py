"""Build the Irvine, CA road network and convert it to PyTorch Geometric form.

Pipeline
--------
1. Download the drivable network for the configured place via OSMnx. (Used to visualize street networks)
2. Clean it: keep the largest *strongly* connected component, which drops
   isolated nodes and one-way dead-ends that a router could enter but never
   leave. Topology is simplified by OSMnx at download time; edge ``geometry``
   is retained so lat/lon shape is never lost.
3. Impute missing speed limits, cap them at a cruise speed the road actually
   delivers, and derive free-flow travel times from that.
4. Relabel nodes to contiguous integers ``0..N-1`` and assign edges contiguous
   integer ids ``0..E-1`` (an edge's id is its column in ``edge_index``).
5. Classify every intersection, so the router can charge for stopping at it.
6. Read the geometry of every turn: the angle between each pair of edges that
   meet at a node, flagged for the two movements that behave differently from
   the rest -- an unprotected left across oncoming traffic, and a ramp merging
   into freeway speed.
7. Emit three artifacts under ``data/processed/``:
     - ``irvine_graph.gpkg``     full attributed graph (GeoPackage), in three
                                 layers: nodes, edges and turns
     - ``node_mapping.json``     osmid <-> index mapping + coordinates
     - ``graph_tensors.pt``      edge_index, edge_attr, node positions,
                                 per-node intersection delays, turn table

Turn geometry
-------------
Nodes and edges cannot express a turn between them: "left onto Jamboree" is a
property of the *pair* of edges a driver traverses, so it needs a table of its
own. Step 6 builds one -- ~40k rows for Irvine, one per legal movement -- and
it is the same shape the native engine searches over, where a state is an edge
rather than a node. Nothing here prices a turn; that is
:mod:`src.data.calibration`'s job. This stage only measures.

Calibration
-----------
Steps 3 and 5 are what separate a drive time from a distance divided by a
speed limit; the constants and the classification rules both live in
:mod:`src.data.calibration`, and ``app.network`` applies the same ones to
artifacts already on disk. ``speed_kph`` and ``travel_time`` on every edge are
the *capped* values -- the posted limit is kept alongside as
``posted_speed_kph`` -- so anything trained on a rebuilt graph learns against
free-flow times a driver could actually achieve.

Rebuilding with a different calibration changes the free-flow baseline the
ST-GNN predicts deltas against, so ``generate_traffic_data`` and the training
pipeline have to be re-run after it. Nothing here mutates artifacts in place.

Node and edge ordering is deterministic (sorted by OSM id, then by
``(u, v, key)``), so repeated runs over the same OSM extract produce
identical tensors.

Usage
-----
    python -m src.data.graph_builder --output-dir data/processed

    from src.data.graph_builder import IrvineGraphData
    graph = IrvineGraphData.load("data/processed")
    lat, lon = graph.coordinates(0)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import osmnx as ox
import torch

from src.data.calibration import (
    ARTERIAL_CRUISE_CAP_KPH,
    IntersectionDelays,
    classify_intersections,
    cruise_speed_kph,
    normalize_highway,
    polyline_bearings,
    road_class,
    road_classes,
)

LOGGER = logging.getLogger(__name__)

# Order of the columns in edge_attr. Anything reading graph_tensors.pt should
# index via this list rather than hard-coding positions.
EDGE_FEATURE_NAMES: tuple[str, ...] = ("length_m", "speed_kph", "travel_time_s")

# Node attribute holding the pre-relabel OSM id. Deliberately *not* "osmid":
# OSMnx names the node index "osmid" when converting to GeoDataFrames, and a
# same-named attribute collides on reset_index() during GeoPackage export.
OSMID_ATTR = "osmid_original"

DEFAULT_PLACE = "Irvine, California, USA"
DEFAULT_OUTPUT_DIR = Path("data/processed")

GRAPH_FILENAME = "irvine_graph.gpkg"
MAPPING_FILENAME = "node_mapping.json"
TENSOR_FILENAME = "graph_tensors.pt"

# Fallback free-flow speeds (km/h) by OSM highway tag, applied where `maxspeed`
# is absent. Values reflect posted limits on Irvine's arterial grid; the city
# is dominated by 6-8 lane arterials, so the generic OSMnx defaults understate
# it. Tune these against loop-detector ground truth once it is available.
IRVINE_HWY_SPEEDS_KPH: dict[str, float] = {
    "motorway": 105.0,
    "motorway_link": 65.0,
    "trunk": 88.0,
    "trunk_link": 55.0,
    "primary": 72.0,
    "primary_link": 45.0,
    "secondary": 64.0,
    "secondary_link": 40.0,
    "tertiary": 56.0,
    "tertiary_link": 40.0,
    "residential": 40.0,
    "living_street": 25.0,
    "unclassified": 40.0,
    "service": 25.0,
}

# Applied where a road has neither `maxspeed` nor a tag in the table above.
FALLBACK_SPEED_KPH = 40.0

EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class GraphBuildConfig:
    """Inputs to :func:`build_graph_artifacts`."""

    place: str = DEFAULT_PLACE
    network_type: str = "drive"
    output_dir: Path = DEFAULT_OUTPUT_DIR
    hwy_speeds: dict[str, float] = field(
        default_factory=lambda: dict(IRVINE_HWY_SPEEDS_KPH)
    )
    fallback_speed_kph: float = FALLBACK_SPEED_KPH
    # Strong connectivity is the right default for routing: every retained node
    # must be reachable from, and able to reach, every other node. Set False to
    # keep weak connectivity (more nodes, but some are routing traps).
    strongly_connected: bool = True
    # Merge clustered intersection nodes (e.g. divided-arterial pairs) within
    # this many metres. Off by default: consolidation synthesises new nodes, so
    # the resulting ids no longer correspond to OSM node ids.
    consolidate_tolerance_m: float | None = None
    use_cache: bool = True
    # Cap imputed speeds at the cruise speed the class actually delivers, and
    # classify intersections for the router's delay model. Both are the point
    # of the calibration; turn them off only to reproduce a pre-calibration
    # artifact for comparison.
    calibrate: bool = True


# ---------------------------------------------------------------------------
# Stage 1-3: download, clean, annotate
# ---------------------------------------------------------------------------
    """ 
    Graph Extraction and Topology Cleaning:
    Downloads the Raw Data and simplifies into complex edges while perserving longitude/latitude path.
    Extracts largest strongly connected component using Tarjan's algorithm so deadends can't break routing
    """
def download_graph(config: GraphBuildConfig) -> nx.MultiDiGraph:
    """Download the drivable network for ``config.place``.

    OSMnx simplifies topology during construction (``simplify=True``),
    collapsing interstitial geometry nodes into edge ``geometry`` LineStrings.
    That preserves true lat/lon road shape while removing nodes that carry no
    routing decision.
    """
    ox.settings.use_cache = config.use_cache
    LOGGER.info("Downloading %s network for %r", config.network_type, config.place)
    graph = ox.graph_from_place(
        config.place,
        network_type=config.network_type,
        simplify=True,
        retain_all=False,
        truncate_by_edge=False,
    )
    LOGGER.info(
        "Downloaded raw graph: %d nodes, %d edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph


def clean_graph(graph: nx.MultiDiGraph, config: GraphBuildConfig) -> nx.MultiDiGraph:
    """Drop isolated nodes and unreachable dead-ends; optionally consolidate.

    Taking the largest strongly connected component subsumes isolated-node
    removal: a node with no edges cannot be in a component of size > 1.
    """
    before = graph.number_of_nodes()
    graph = ox.truncate.largest_component(graph, strongly=config.strongly_connected)
    LOGGER.info(
        "Largest %s component: %d/%d nodes retained",
        "strongly connected" if config.strongly_connected else "connected",
        graph.number_of_nodes(),
        before,
    )

    if config.consolidate_tolerance_m is not None:
        # Consolidation must run in a projected CRS so the tolerance is metres.
        projected = ox.project_graph(graph)
        projected = ox.consolidate_intersections(
            projected,
            tolerance=config.consolidate_tolerance_m,
            rebuild_graph=True,
            dead_ends=False,
        )
        graph = ox.project_graph(projected, to_latlong=True)
        LOGGER.info(
            "Consolidated intersections at %.1fm: %d nodes",
            config.consolidate_tolerance_m,
            graph.number_of_nodes(),
        )
        # Re-truncate: consolidation can strand small fragments.
        graph = ox.truncate.largest_component(
            graph, strongly=config.strongly_connected
        )

    return graph


def annotate_speeds(
    graph: nx.MultiDiGraph, config: GraphBuildConfig
) -> nx.MultiDiGraph:
    """Impute ``speed_kph`` and derive ``travel_time`` (seconds) on every edge.

    ``add_edge_speeds`` parses the OSM ``maxspeed`` tag where present and falls
    back to the per-highway-type table; ``add_edge_travel_times`` then computes
    ``length / speed`` as the free-flow baseline the predictor learns deltas
    against.
    """
    graph = ox.routing.add_edge_speeds(
        graph,
        hwy_speeds=config.hwy_speeds,
        fallback=config.fallback_speed_kph,
    )
    graph = ox.routing.add_edge_travel_times(graph)
    if config.calibrate:
        graph = cap_cruise_speeds(graph)
    return graph


def cap_cruise_speeds(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Replace posted limits with achievable cruise speeds, in place.

    The posted limit is the speed reached midway between signals, not the one
    held across a block, and routing on it is most of why an Irvine arterial
    trip comes out at 78 km/h. ``speed_kph`` and ``travel_time`` become the
    capped values -- they are what everything downstream reads -- and the
    original survives as ``posted_speed_kph`` so the cap can be audited or
    undone without another download.

    Units are pinned here: ``speed_kph`` is km/h, ``length`` is metres, and
    ``travel_time`` is seconds via an explicit ``kph / 3.6`` in m/s. OSMnx uses
    the same convention, and this recomputation deliberately does not depend on
    that staying true.
    """
    capped = 0
    for _, _, data in graph.edges(data=True):
        posted = float(data["speed_kph"])
        data["posted_speed_kph"] = posted
        cruise = cruise_speed_kph(data.get("highway"), posted)
        if cruise < posted - 1e-9:
            capped += 1
        data["speed_kph"] = cruise
        # metres / (metres per second) = seconds.
        data["travel_time"] = float(data["length"]) / (cruise / 3.6)

    LOGGER.info(
        "Capped cruise speed on %d/%d edges (arterials at %.0f km/h)",
        capped,
        graph.number_of_edges(),
        ARTERIAL_CRUISE_CAP_KPH,
    )
    return graph


def annotate_intersections(graph: nx.MultiDiGraph) -> IntersectionDelays:
    """Classify every node and write its delay onto the graph.

    Must run *after* relabelling, since the classification is indexed by
    contiguous node id. The delays are written back as node attributes so they
    reach the GeoPackage as well as the tensor bundle.
    """
    num_nodes = graph.number_of_nodes()
    edges = sorted_edges(graph)

    delays = classify_intersections(
        num_nodes=num_nodes,
        edge_u=np.array([u for u, _, _ in edges], dtype=np.int64),
        edge_v=np.array([v for _, v, _ in edges], dtype=np.int64),
        edge_highway=[graph.edges[e].get("highway") for e in edges],
        edge_name=[graph.edges[e].get("name") for e in edges],
        node_highway=[graph.nodes[n].get("highway") for n in range(num_nodes)],
    )

    for node in range(num_nodes):
        graph.nodes[node]["delay_priority_s"] = float(delays.priority_sec[node])
        graph.nodes[node]["delay_yield_s"] = float(delays.yield_sec[node])
        graph.nodes[node]["intersection_kind"] = delays.kind[node]

    LOGGER.info(
        "Classified %d intersections: %s",
        num_nodes,
        ", ".join(f"{n} {kind}" for kind, n in sorted(delays.counts().items())),
    )
    return delays


# ---------------------------------------------------------------------------
# Stage 3b: turn geometry
# ---------------------------------------------------------------------------
# A turn is a property of an edge *pair*, not of an edge: "left onto Jamboree"
# only means something once you know which road the driver arrived on. So this
# stage produces a third table alongside nodes and edges -- one row per legal
# (in_edge, out_edge) movement at a node -- carrying the angle between them and
# the two hazard flags the routing layer cares about.
#
# Sign convention, used consistently here and in the native engine: compass
# bearings grow clockwise, so a *negative* heading change is a left turn in
# right-hand traffic. -90 degrees is a clean left, +90 a clean right.

# An unprotected left is one taken across oncoming traffic with nothing holding
# it back. The angle band is deliberately wide enough to catch the skewed
# junctions Irvine's arterial grid is full of, where a "left" is 60 or 120
# degrees rather than a textbook 90.
UNPROTECTED_LEFT_MIN_DEG = -135.0
UNPROTECTED_LEFT_MAX_DEG = -45.0

# Arterial classes, for the unprotected-left rule. Links are excluded: a slip
# road is not the arterial it feeds.
ARTERIAL_TURN_TAGS = frozenset({"primary", "secondary", "trunk"})

# Freeway on/off ramps, and the roads they merge into.
FREEWAY_LINK_TAGS = frozenset({"motorway_link", "trunk_link"})
FREEWAY_TAGS = frozenset({"motorway", "trunk"})

# What counts as "fast-moving traffic" to merge into, where the tag alone does
# not settle it. Irvine's trunk sections are posted at 88 km/h and its
# motorways at 105.
MERGE_SPEED_THRESHOLD_KPH = 80.0

# OSM node tag marking a signalised junction.
SIGNAL_NODE_TAG = "traffic_signals"


@dataclass(frozen=True)
class TurnTable:
    """Every (in_edge -> out_edge) movement in the graph, with its geometry.

    Arrays are parallel and ordered by ``(node_id, in_edge_id, out_edge_id)``,
    so the table is reproducible across runs. Edge ids are the same contiguous
    ids as ``edge_index`` columns.
    """

    node_id: np.ndarray              # int32  [T]
    in_edge_id: np.ndarray           # int32  [T]
    out_edge_id: np.ndarray          # int32  [T]
    bearing_in_deg: np.ndarray       # float32[T] heading arriving at the node
    bearing_out_deg: np.ndarray      # float32[T] heading leaving it
    delta_deg: np.ndarray            # float32[T] signed, negative = left
    is_unprotected_left: np.ndarray  # bool   [T]
    is_aggressive_merge: np.ndarray  # bool   [T]

    def __len__(self) -> int:
        return int(self.node_id.shape[0])

    def counts(self) -> dict[str, int]:
        return {
            "transitions": len(self),
            "unprotected_left": int(self.is_unprotected_left.sum()),
            "aggressive_merge": int(self.is_aggressive_merge.sum()),
        }


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Forward compass azimuth from point 1 to point 2, in degrees [0, 360).

    Zero is due north and the angle grows clockwise, which is the convention
    every other bearing in this project uses -- including
    ``irvine::bearing_degrees`` in the native engine, so a turn classified here
    and a turn priced there agree.

    This is the *initial* bearing of the great circle. Over an Irvine city
    block the difference from the final bearing is far below a degree, but the
    distinction matters enough to name: a long great circle changes heading as
    it goes.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)

    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(
        dlambda
    )
    return math.degrees(math.atan2(y, x)) % 360.0


def turn_delta_degrees(bearing_in: float, bearing_out: float) -> float:
    """Signed heading change in [-180, 180). Negative is a left turn.

    Straight on is near zero regardless of which way the two headings straddle
    north, which is the whole reason for wrapping rather than subtracting: a
    driver heading 350 degrees who leaves on 10 has turned 20 degrees right,
    not 340 left.

    An exact reversal lands on -180 rather than +180. The half-interval has to
    open at one end or the other, and a U-turn is neither a left nor a right,
    so which end costs nothing -- but it matches ``irvine::signed_turn_degrees``
    in the native engine, which is worth more than the aesthetics.
    """
    return (bearing_out - bearing_in + 180.0) % 360.0 - 180.0


def is_unprotected_left(
    delta_deg: float, in_tag: str, out_tag: str, node_has_signal: bool
) -> bool:
    """Is this movement a left turn across oncoming traffic, unsignalised?

    Three conditions, all of which have to hold:

    * the heading change is a left, in the band above;
    * the node carries no ``highway=traffic_signals``, so nothing is holding
      the conflicting stream -- a protected left phase would;
    * an arterial is involved. Turning left *from* one means crossing its
      oncoming lanes; turning left *onto* one means crossing them too, from a
      standing start with a worse sight line, so both directions count.
    """
    if node_has_signal:
        return False
    if not UNPROTECTED_LEFT_MIN_DEG <= delta_deg <= UNPROTECTED_LEFT_MAX_DEG:
        return False
    return in_tag in ARTERIAL_TURN_TAGS or out_tag in ARTERIAL_TURN_TAGS


def is_aggressive_merge(
    in_tag: str, out_tag: str, in_speed_kph: float, out_speed_kph: float
) -> bool:
    """Is this movement a ramp joining or leaving fast-moving traffic?

    True where exactly one side of the transition is a freeway ramp and the
    other is genuinely fast -- tagged as a motorway or trunk, or simply carrying
    a speed at or above :data:`MERGE_SPEED_THRESHOLD_KPH`. Both an on-ramp
    (link into the mainline) and an off-ramp (mainline into the link) qualify:
    the hazard is the speed differential, and it is there in both directions.

    Ramp-to-ramp movements are excluded. A freeway-to-freeway connector is
    high-speed driving, but the driver stays inside the ramp system rather than
    crossing a stream of traffic already at speed.
    """
    link_in = in_tag in FREEWAY_LINK_TAGS
    link_out = out_tag in FREEWAY_LINK_TAGS
    if link_in == link_out:  # neither side is a ramp, or both are
        return False

    other_tag = out_tag if link_in else in_tag
    other_speed = out_speed_kph if link_in else in_speed_kph
    return other_tag in FREEWAY_TAGS or other_speed >= MERGE_SPEED_THRESHOLD_KPH


def annotate_signals(graph: nx.MultiDiGraph) -> set[int]:
    """Mark every node carrying ``highway=traffic_signals``. Returns their ids.

    OSMnx keeps the raw OSM node tag, and its coverage in Irvine is partial --
    a few hundred of several thousand junctions -- so this is evidence that a
    junction *is* signalised and never evidence that it is not. The
    unprotected-left rule is written to fail safe in that direction: a missing
    tag makes a turn look unprotected, which is the cautious answer.
    """
    signalled: set[int] = set()
    for node, data in graph.nodes(data=True):
        tag = normalize_highway(data.get("highway"))
        has_signal = tag == SIGNAL_NODE_TAG
        data["has_traffic_signal"] = bool(has_signal)
        if has_signal:
            signalled.add(int(node))

    LOGGER.info(
        "Traffic signals: %d of %d nodes tagged in OSM",
        len(signalled),
        graph.number_of_nodes(),
    )
    return signalled


def annotate_road_types(graph: nx.MultiDiGraph) -> None:
    """Write the resolved ``road_type`` and ``road_class`` onto every edge.

    ``highway`` survives from OSM as-is, list values and all; ``road_type`` is
    the single tag this project routes on, and ``road_class`` its functional
    class as an integer, so downstream consumers never have to re-parse
    ``"['tertiary', 'residential']"``.
    """
    for _, _, data in graph.edges(data=True):
        tag = normalize_highway(data.get("highway"))
        data["road_type"] = tag
        data["road_class"] = int(road_class(tag))


def _edge_polyline_latlon(
    graph: nx.MultiDiGraph, u: int, v: int, key: int
) -> np.ndarray:
    """``[k, 2]`` lat/lon for one edge, from its geometry or its endpoints.

    OSMnx stores the simplified centreline as a Shapely LineString in (x, y) =
    (lon, lat) order; nodes carry ``x``/``y`` the same way. Everything past
    this function speaks (lat, lon).
    """
    line = graph.edges[u, v, key].get("geometry")
    if line is not None and not line.is_empty:
        coords = np.asarray(line.coords, dtype=np.float64)
        if coords.shape[0] >= 2:
            return coords[:, ::-1]  # (lon, lat) -> (lat, lon)

    start = graph.nodes[u]
    end = graph.nodes[v]
    return np.array(
        [[float(start["y"]), float(start["x"])], [float(end["y"]), float(end["x"])]],
        dtype=np.float64,
    )


def build_turn_table(graph: nx.MultiDiGraph) -> TurnTable:
    """Enumerate every movement at every node and classify its geometry.

    Runs after relabelling, so node keys are the contiguous ids and an edge's
    position in :func:`sorted_edges` is its id.

    Bearings come from the road's own shape rather than the straight line
    between its endpoints: measured over the first and last
    ``calibration.BEARING_BASELINE_M`` metres of the centreline. On Irvine's
    curving arterials the chord can sit tens of degrees off the direction the
    driver is actually pointing at the junction, which is enough to turn a
    left into a straight-on.
    """
    edges = sorted_edges(graph)
    edge_id_of = {edge: index for index, edge in enumerate(edges)}

    entry_deg, exit_deg = polyline_bearings(
        [_edge_polyline_latlon(graph, u, v, key) for u, v, key in edges]
    )

    tags = [normalize_highway(graph.edges[e].get("highway")) for e in edges]
    speeds = [float(graph.edges[e].get("speed_kph", 0.0)) for e in edges]

    # Incoming edges per node, so the pairing is a lookup rather than a scan.
    arriving: dict[int, list[int]] = {}
    for (u, v, key), edge_id in edge_id_of.items():
        arriving.setdefault(int(v), []).append(edge_id)

    rows: list[tuple] = []
    for node in sorted(graph.nodes):
        node = int(node)
        incoming = sorted(arriving.get(node, ()))
        if not incoming:
            continue
        has_signal = bool(graph.nodes[node].get("has_traffic_signal", False))

        outgoing = sorted(
            edge_id_of[(u, v, key)] for u, v, key in graph.out_edges(node, keys=True)
        )
        for in_edge in incoming:
            bearing_in = float(exit_deg[in_edge])
            for out_edge in outgoing:
                bearing_out = float(entry_deg[out_edge])
                delta = turn_delta_degrees(bearing_in, bearing_out)
                rows.append(
                    (
                        node,
                        in_edge,
                        out_edge,
                        bearing_in,
                        bearing_out,
                        delta,
                        is_unprotected_left(
                            delta, tags[in_edge], tags[out_edge], has_signal
                        ),
                        is_aggressive_merge(
                            tags[in_edge],
                            tags[out_edge],
                            speeds[in_edge],
                            speeds[out_edge],
                        ),
                    )
                )

    if not rows:
        empty_i = np.zeros(0, dtype=np.int32)
        empty_f = np.zeros(0, dtype=np.float32)
        empty_b = np.zeros(0, dtype=bool)
        return TurnTable(empty_i, empty_i, empty_i, empty_f, empty_f, empty_f,
                         empty_b, empty_b)

    columns = list(zip(*rows))
    table = TurnTable(
        node_id=np.array(columns[0], dtype=np.int32),
        in_edge_id=np.array(columns[1], dtype=np.int32),
        out_edge_id=np.array(columns[2], dtype=np.int32),
        bearing_in_deg=np.array(columns[3], dtype=np.float32),
        bearing_out_deg=np.array(columns[4], dtype=np.float32),
        delta_deg=np.array(columns[5], dtype=np.float32),
        is_unprotected_left=np.array(columns[6], dtype=bool),
        is_aggressive_merge=np.array(columns[7], dtype=bool),
    )

    counts = table.counts()
    LOGGER.info(
        "Turn table: %d transitions, %d unprotected lefts, %d aggressive merges",
        counts["transitions"],
        counts["unprotected_left"],
        counts["aggressive_merge"],
    )
    return table


# ---------------------------------------------------------------------------
# Stage 4: contiguous integer indexing
# ---------------------------------------------------------------------------
def build_node_mapping(graph: nx.MultiDiGraph) -> dict[int, int]:
    """Map each OSM node id to a contiguous index in ``0..N-1``.

    Sorted by OSM id so the assignment is reproducible across runs.
    """
    return {osmid: idx for idx, osmid in enumerate(sorted(graph.nodes))}


def relabel_graph(
    graph: nx.MultiDiGraph, node_mapping: dict[int, int]
) -> nx.MultiDiGraph:
    """Relabel nodes to contiguous indices, retaining the original id.

    The original OSM id is written to each node's :data:`OSMID_ATTR` attribute
    first, so it survives into the GeoPackage export where the node index
    becomes the primary key.
    """
    for osmid, data in graph.nodes(data=True):
        data[OSMID_ATTR] = osmid
        # Intersection consolidation can leave an "osmid" attribute behind,
        # which would collide with the GeoDataFrame index name on export.
        data.pop("osmid", None)
    relabelled = nx.relabel_nodes(graph, node_mapping, copy=True)
    # `graph_from_place` stamps the CRS here; nx.relabel_nodes preserves
    # graph-level attrs, but assert it since the GeoPackage writer needs it.
    relabelled.graph.setdefault("crs", graph.graph.get("crs", "epsg:4326"))
    return relabelled


def sorted_edges(graph: nx.MultiDiGraph) -> list[tuple[int, int, int]]:
    """Return ``(u, v, key)`` triples in deterministic order.

    Position in this list *is* the edge's integer id, so the ordering must be
    stable: it is what ties column ``i`` of ``edge_index`` to row ``i`` of
    ``edge_attr`` and to feature row ``i`` in every downstream tensor.
    """
    return sorted(graph.edges(keys=True))


# ---------------------------------------------------------------------------
# Stage 5: tensor extraction
# ---------------------------------------------------------------------------
def _edge_feature_row(data: dict[str, Any]) -> tuple[float, float, float]:
    """Pull (length_m, speed_kph, travel_time_s) out of one edge's attributes."""
    length = float(data["length"])
    speed = float(data["speed_kph"])
    travel_time = float(data["travel_time"])
    return length, speed, travel_time


def extract_tensors(
    graph: nx.MultiDiGraph, turns: "TurnTable | None" = None
) -> dict[str, torch.Tensor]:
    """Build the PyG-format tensors from a relabelled, annotated graph.

    Returns
    -------
    dict with:
        edge_index  : int64  [2, E]  COO connectivity, PyG convention
        edge_attr   : float32[E, 3]  EDGE_FEATURE_NAMES
        pos         : float64[N, 2]  (lat, lon) per node index
        edge_key    : int64  [E]     parallel-edge discriminator per edge id
        edge_road_class      : uint8 [E]  calibration.RoadClass per edge id
        node_delay_priority_s: float32[N] charged to every movement
        node_delay_yield_s   : float32[N] charged to minor-road movements
        node_has_traffic_signal    : bool [N]
        turn_node_id               : int32 [T]  one row per legal movement
        turn_in_edge_id            : int32 [T]
        turn_out_edge_id           : int32 [T]
        turn_bearing_in_deg        : float32[T]
        turn_bearing_out_deg       : float32[T]
        turn_delta_deg             : float32[T] signed, negative = left
        turn_is_unprotected_left   : bool  [T]
        turn_is_aggressive_merge   : bool  [T]

    ``turns`` is computed here when not supplied, so the bundle can never go
    out without it; ``build_graph_artifacts`` passes the table it already built
    for the GeoPackage rather than paying for it twice.

    ``edge_attr`` keeps its three columns: the model's input contract is fixed,
    and the calibration changes the *values* in the speed and travel-time
    columns rather than adding new ones.
    """
    edges = sorted_edges(graph)
    num_nodes = graph.number_of_nodes()

    edge_index = torch.empty((2, len(edges)), dtype=torch.int64)
    edge_attr = torch.empty((len(edges), len(EDGE_FEATURE_NAMES)), dtype=torch.float32)
    edge_key = torch.empty(len(edges), dtype=torch.int64)

    for edge_id, (u, v, key) in enumerate(edges):
        edge_index[0, edge_id] = u
        edge_index[1, edge_id] = v
        edge_key[edge_id] = key
        edge_attr[edge_id] = torch.tensor(
            _edge_feature_row(graph.edges[u, v, key]), dtype=torch.float32
        )

    # float64 is deliberate: float32 resolves lat/lon to roughly a metre, which
    # is coarse enough to snap probe points to the wrong side of an arterial.
    pos = torch.empty((num_nodes, 2), dtype=torch.float64)
    for idx, data in graph.nodes(data=True):
        pos[idx, 0] = float(data["y"])  # latitude
        pos[idx, 1] = float(data["x"])  # longitude

    # Written by annotate_intersections; zero (i.e. "no delay anywhere") when
    # the build was run uncalibrated.
    priority = torch.tensor(
        [float(graph.nodes[n].get("delay_priority_s", 0.0)) for n in range(num_nodes)],
        dtype=torch.float32,
    )
    yielding = torch.tensor(
        [float(graph.nodes[n].get("delay_yield_s", 0.0)) for n in range(num_nodes)],
        dtype=torch.float32,
    )
    edge_class = torch.from_numpy(
        road_classes(graph.edges[e].get("highway") for e in edges)
    )
    signals = torch.tensor(
        [
            bool(graph.nodes[n].get("has_traffic_signal", False))
            for n in range(num_nodes)
        ],
        dtype=torch.bool,
    )

    if turns is None:
        turns = build_turn_table(graph)

    return {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "pos": pos,
        "edge_key": edge_key,
        "edge_road_class": edge_class,
        "node_delay_priority_s": priority,
        "node_delay_yield_s": yielding,
        "node_has_traffic_signal": signals,
        "turn_node_id": torch.from_numpy(turns.node_id),
        "turn_in_edge_id": torch.from_numpy(turns.in_edge_id),
        "turn_out_edge_id": torch.from_numpy(turns.out_edge_id),
        "turn_bearing_in_deg": torch.from_numpy(turns.bearing_in_deg),
        "turn_bearing_out_deg": torch.from_numpy(turns.bearing_out_deg),
        "turn_delta_deg": torch.from_numpy(turns.delta_deg),
        "turn_is_unprotected_left": torch.from_numpy(turns.is_unprotected_left),
        "turn_is_aggressive_merge": torch.from_numpy(turns.is_aggressive_merge),
    }


def validate_tensors(tensors: dict[str, torch.Tensor], num_nodes: int) -> None:
    """Fail loudly on malformed tensors before they reach training.

    A silently mis-shaped ``edge_index`` produces a model that trains and
    predicts nonsense, so these are hard assertions rather than warnings.
    """
    edge_index = tensors["edge_index"]
    edge_attr = tensors["edge_attr"]
    pos = tensors["pos"]

    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must be [2, E], got {tuple(edge_index.shape)}")
    if edge_index.dtype != torch.int64:
        raise ValueError(f"edge_index must be int64, got {edge_index.dtype}")
    if edge_attr.size(0) != edge_index.size(1):
        raise ValueError(
            f"edge_attr rows ({edge_attr.size(0)}) != edges ({edge_index.size(1)})"
        )
    if edge_attr.size(1) != len(EDGE_FEATURE_NAMES):
        raise ValueError(
            f"edge_attr must have {len(EDGE_FEATURE_NAMES)} features, "
            f"got {edge_attr.size(1)}"
        )
    if pos.size(0) != num_nodes:
        raise ValueError(f"pos rows ({pos.size(0)}) != nodes ({num_nodes})")

    if edge_index.numel() and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes
    ):
        raise ValueError("edge_index contains out-of-range node indices")
    if not torch.isfinite(edge_attr).all():
        raise ValueError("edge_attr contains NaN or inf")
    if (edge_attr <= 0).any():
        bad = int((edge_attr <= 0).any(dim=1).sum())
        raise ValueError(f"{bad} edges have non-positive length/speed/travel_time")

    # Calibration tensors. A negative delay would make an intersection worth
    # driving through twice, which A* assumes is impossible.
    for name in ("node_delay_priority_s", "node_delay_yield_s"):
        delays = tensors.get(name)
        if delays is None:
            continue
        if delays.shape != (num_nodes,):
            raise ValueError(
                f"{name} must be [{num_nodes}], got {tuple(delays.shape)}"
            )
        if not torch.isfinite(delays).all() or bool((delays < 0).any()):
            raise ValueError(f"{name} contains a negative or non-finite delay")

    edge_class = tensors.get("edge_road_class")
    if edge_class is not None and edge_class.shape != (edge_index.size(1),):
        raise ValueError(
            f"edge_road_class must be [{edge_index.size(1)}], "
            f"got {tuple(edge_class.shape)}"
        )

    _validate_turn_tensors(tensors, num_nodes=num_nodes, num_edges=edge_index.size(1))


def _validate_turn_tensors(
    tensors: dict[str, torch.Tensor], num_nodes: int, num_edges: int
) -> None:
    """The turn table has to be internally consistent to be usable.

    A row pointing at an edge that does not exist, or an angle outside the
    circle, would survive every shape check downstream and quietly mislabel a
    movement -- which is the failure mode this whole table exists to remove.
    """
    if "turn_node_id" not in tensors:
        return

    columns = (
        "turn_node_id",
        "turn_in_edge_id",
        "turn_out_edge_id",
        "turn_bearing_in_deg",
        "turn_bearing_out_deg",
        "turn_delta_deg",
        "turn_is_unprotected_left",
        "turn_is_aggressive_merge",
    )
    missing = [name for name in columns if name not in tensors]
    if missing:
        raise ValueError(f"turn table is missing {missing}")

    rows = tensors["turn_node_id"].numel()
    for name in columns:
        if tensors[name].shape != (rows,):
            raise ValueError(
                f"{name} must be [{rows}] to match turn_node_id, "
                f"got {tuple(tensors[name].shape)}"
            )
    if rows == 0:
        return

    nodes = tensors["turn_node_id"]
    if int(nodes.min()) < 0 or int(nodes.max()) >= num_nodes:
        raise ValueError("turn_node_id contains an out-of-range node index")
    for name in ("turn_in_edge_id", "turn_out_edge_id"):
        ids = tensors[name]
        if int(ids.min()) < 0 or int(ids.max()) >= num_edges:
            raise ValueError(f"{name} contains an out-of-range edge id")

    for name in ("turn_bearing_in_deg", "turn_bearing_out_deg"):
        bearings = tensors[name]
        if not torch.isfinite(bearings).all():
            raise ValueError(f"{name} contains NaN or inf")
        if float(bearings.min()) < 0.0 or float(bearings.max()) >= 360.0:
            raise ValueError(f"{name} must be a compass bearing in [0, 360)")

    delta = tensors["turn_delta_deg"]
    if not torch.isfinite(delta).all():
        raise ValueError("turn_delta_deg contains NaN or inf")
    if float(delta.min()) < -180.0 or float(delta.max()) > 180.0:
        raise ValueError("turn_delta_deg must be a signed angle in [-180, 180]")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def build_mapping_payload(
    graph: nx.MultiDiGraph,
    node_mapping: dict[int, int],
    config: GraphBuildConfig,
    num_edges: int,
    turns: "TurnTable | None" = None,
) -> dict[str, Any]:
    """Assemble the JSON-serialisable contents of ``node_mapping.json``."""
    index_to_osmid = [0] * len(node_mapping)
    for osmid, idx in node_mapping.items():
        index_to_osmid[idx] = int(osmid)

    coordinates = [[0.0, 0.0] for _ in range(graph.number_of_nodes())]
    for idx, data in graph.nodes(data=True):
        coordinates[idx] = [float(data["y"]), float(data["x"])]

    return {
        "place": config.place,
        "network_type": config.network_type,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "num_nodes": len(index_to_osmid),
        "num_edges": num_edges,
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
        "crs": str(graph.graph.get("crs", "epsg:4326")),
        # JSON object keys are strings by spec; cast back to int on load.
        "osmid_to_index": {str(osmid): int(idx) for osmid, idx in node_mapping.items()},
        "index_to_osmid": index_to_osmid,
        "coordinates": coordinates,
        # Which junctions OSM says are signalised. A list rather than a
        # per-node flag: it is a few hundred ids out of several thousand
        # nodes, and the readers of this file want the set, not the mask.
        "traffic_signal_nodes": sorted(
            int(node)
            for node, data in graph.nodes(data=True)
            if data.get("has_traffic_signal", False)
        ),
        # The turn table itself lives in graph_tensors.pt -- 40k rows of it --
        # so what goes in the JSON is the shape of it, for a reader deciding
        # whether the bundle is the one they expect.
        "turn_summary": turns.counts() if turns is not None else None,
    }


def save_artifacts(
    graph: nx.MultiDiGraph,
    tensors: dict[str, torch.Tensor],
    mapping_payload: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Write the GeoPackage, JSON mapping, and tensor bundle. Returns paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = output_dir / GRAPH_FILENAME
    mapping_path = output_dir / MAPPING_FILENAME
    tensor_path = output_dir / TENSOR_FILENAME

    # Nodes and edges carry has_traffic_signal, road_type and road_class as
    # ordinary attributes, so OSMnx exports them with everything else.
    ox.io.save_graph_geopackage(graph, filepath=graph_path, directed=True)
    _write_turns_layer(tensors, graph_path)

    with mapping_path.open("w", encoding="utf-8") as handle:
        json.dump(mapping_payload, handle, indent=2)

    torch.save(
        {
            **tensors,
            "num_nodes": mapping_payload["num_nodes"],
            "edge_feature_names": list(EDGE_FEATURE_NAMES),
            "place": mapping_payload["place"],
        },
        tensor_path,
    )

    LOGGER.info("Wrote %s, %s, %s", graph_path, mapping_path, tensor_path)
    return {"graph": graph_path, "mapping": mapping_path, "tensors": tensor_path}


def _write_turns_layer(tensors: dict[str, torch.Tensor], graph_path: Path) -> None:
    """Append the turn table to the GeoPackage as a third layer.

    Written from the tensors rather than recomputed, so the layer and the
    bundle cannot disagree. Each row gets a Point at the intersection it
    happens at, which costs a little space and buys the ability to open the
    file in QGIS, filter to ``is_unprotected_left``, and look at where they
    are -- the only practical way to sanity-check a rule like this against a
    city you can drive around.
    """
    if "turn_node_id" not in tensors:
        return

    import geopandas as gpd  # heavy; already a hard dependency of OSMnx
    from shapely.geometry import Point

    positions = tensors["pos"].numpy()
    nodes = tensors["turn_node_id"].numpy()

    frame = gpd.GeoDataFrame(
        {
            "node_id": nodes,
            "in_edge_id": tensors["turn_in_edge_id"].numpy(),
            "out_edge_id": tensors["turn_out_edge_id"].numpy(),
            "bearing_in_deg": tensors["turn_bearing_in_deg"].numpy(),
            "bearing_out_deg": tensors["turn_bearing_out_deg"].numpy(),
            "delta_deg": tensors["turn_delta_deg"].numpy(),
            "is_unprotected_left": tensors["turn_is_unprotected_left"].numpy(),
            "is_aggressive_merge": tensors["turn_is_aggressive_merge"].numpy(),
        },
        # Shapely takes (x, y) = (lon, lat); pos is [lat, lon].
        geometry=[Point(positions[n, 1], positions[n, 0]) for n in nodes],
        crs="epsg:4326",
    )
    frame.to_file(graph_path, layer="turns", driver="GPKG", mode="a")
    LOGGER.info("Wrote %d turns to the %s layer", len(frame), "turns")


def build_graph_artifacts(
    config: GraphBuildConfig | None = None,
) -> dict[str, Path]:
    """Run the full pipeline and write all artifacts. Returns output paths."""
    config = config or GraphBuildConfig()

    graph = download_graph(config)
    graph = clean_graph(graph, config)
    graph = annotate_speeds(graph, config)

    node_mapping = build_node_mapping(graph)
    graph = relabel_graph(graph, node_mapping)
    # Node keys are now indices, so the mapping's values address `graph`
    # directly. Re-key by the original osmid stored on each node.
    node_mapping = {
        int(data[OSMID_ATTR]): idx for idx, data in graph.nodes(data=True)
    }

    if config.calibrate:
        # After relabelling: the classification is indexed by contiguous id.
        annotate_intersections(graph)

    # Turn geometry is descriptive rather than calibrated -- it says what the
    # junction *is*, not what it costs -- so it is extracted unconditionally.
    annotate_signals(graph)
    annotate_road_types(graph)
    turns = build_turn_table(graph)

    tensors = extract_tensors(graph, turns=turns)
    validate_tensors(tensors, num_nodes=graph.number_of_nodes())

    payload = build_mapping_payload(
        graph,
        node_mapping,
        config,
        num_edges=tensors["edge_index"].size(1),
        turns=turns,
    )
    return save_artifacts(graph, tensors, payload, config.output_dir)


# ---------------------------------------------------------------------------
# Consumer-facing accessor
# ---------------------------------------------------------------------------
class IrvineGraphData:
    """Read-only view over the processed graph artifacts.

    Wraps ``graph_tensors.pt`` + ``node_mapping.json`` and exposes coordinate
    and id lookups. Construct via :meth:`load`.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        pos: torch.Tensor,
        index_to_osmid: list[int],
        osmid_to_index: dict[int, int],
        metadata: dict[str, Any] | None = None,
        turns: TurnTable | None = None,
    ) -> None:
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.pos = pos
        self.turns = turns
        self.index_to_osmid = index_to_osmid
        self.osmid_to_index = osmid_to_index
        self.metadata = metadata or {}
        self.edge_feature_names: tuple[str, ...] = tuple(
            self.metadata.get("edge_feature_names", EDGE_FEATURE_NAMES)
        )
        # Cached float64 view for vectorised nearest-neighbour queries.
        self._coords = pos.numpy().astype(np.float64, copy=False)

    # -- construction -------------------------------------------------------
    @classmethod
    def load(cls, processed_dir: Path | str = DEFAULT_OUTPUT_DIR) -> "IrvineGraphData":
        """Load artifacts from a directory produced by the pipeline."""
        processed_dir = Path(processed_dir)
        tensor_path = processed_dir / TENSOR_FILENAME
        mapping_path = processed_dir / MAPPING_FILENAME

        for path in (tensor_path, mapping_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Run `python -m src.data.graph_builder` first."
                )

        # weights_only=True: these are our own plain tensor dicts, and it keeps
        # torch.load from being an arbitrary-code-execution surface.
        bundle = torch.load(tensor_path, weights_only=True)
        with mapping_path.open(encoding="utf-8") as handle:
            mapping = json.load(handle)

        return cls(
            edge_index=bundle["edge_index"],
            edge_attr=bundle["edge_attr"],
            pos=bundle["pos"],
            index_to_osmid=[int(x) for x in mapping["index_to_osmid"]],
            osmid_to_index={
                int(k): int(v) for k, v in mapping["osmid_to_index"].items()
            },
            metadata=mapping,
            turns=_turn_table_from_bundle(bundle),
        )

    # -- size ---------------------------------------------------------------
    @property
    def num_nodes(self) -> int:
        return self.pos.size(0)

    @property
    def num_edges(self) -> int:
        return self.edge_index.size(1)

    def __len__(self) -> int:
        return self.num_nodes

    # -- node lookups -------------------------------------------------------
    def _check_index(self, index: int) -> int:
        index = int(index)
        if not 0 <= index < self.num_nodes:
            raise IndexError(
                f"node index {index} out of range for {self.num_nodes} nodes"
            )
        return index

    def coordinates(self, index: int) -> tuple[float, float]:
        """Return ``(latitude, longitude)`` for a contiguous node index."""
        index = self._check_index(index)
        return float(self.pos[index, 0]), float(self.pos[index, 1])

    def latitude(self, index: int) -> float:
        return self.coordinates(index)[0]

    def longitude(self, index: int) -> float:
        return self.coordinates(index)[1]

    def coordinates_many(self, indices: Iterable[int]) -> np.ndarray:
        """Vectorised lookup: ``[[lat, lon], ...]`` for many indices."""
        idx = np.asarray(list(indices), dtype=np.int64)
        if idx.size and (idx.min() < 0 or idx.max() >= self.num_nodes):
            raise IndexError("node index out of range")
        return self._coords[idx]

    def osmid(self, index: int) -> int:
        """OSM node id for a contiguous index."""
        return self.index_to_osmid[self._check_index(index)]

    def index_of(self, osmid: int) -> int:
        """Contiguous index for an OSM node id."""
        try:
            return self.osmid_to_index[int(osmid)]
        except KeyError as exc:
            raise KeyError(f"OSM node {osmid} is not in the processed graph") from exc

    def coordinates_of_osmid(self, osmid: int) -> tuple[float, float]:
        return self.coordinates(self.index_of(osmid))

    def nearest_node(self, latitude: float, longitude: float) -> int:
        """Index of the node closest to a lat/lon, by haversine distance.

        Brute force over all nodes. Irvine is ~10^4 nodes, so this is well
        under a millisecond; swap in a BallTree if the graph grows.
        """
        lat1, lon1 = np.radians(latitude), np.radians(longitude)
        lat2 = np.radians(self._coords[:, 0])
        lon2 = np.radians(self._coords[:, 1])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return int(np.argmin(2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))))

    # -- edge lookups -------------------------------------------------------
    def edge_endpoints(self, edge_id: int) -> tuple[int, int]:
        """``(source_index, target_index)`` for an edge id."""
        edge_id = int(edge_id)
        if not 0 <= edge_id < self.num_edges:
            raise IndexError(
                f"edge id {edge_id} out of range for {self.num_edges} edges"
            )
        return int(self.edge_index[0, edge_id]), int(self.edge_index[1, edge_id])

    def edge_features(self, edge_id: int) -> dict[str, float]:
        """Named feature dict for an edge id."""
        self.edge_endpoints(edge_id)  # bounds check
        row = self.edge_attr[int(edge_id)]
        return {name: float(row[i]) for i, name in enumerate(self.edge_feature_names)}

    # -- turn lookups -------------------------------------------------------
    def turns_at(self, node_index: int) -> dict[str, np.ndarray]:
        """Every movement through one intersection, as parallel arrays.

        Empty arrays when the bundle predates the turn table, so a caller can
        treat "no turns recorded" and "no turns here" the same way.
        """
        node_index = self._check_index(node_index)
        turns = self.turns
        if turns is None:
            empty = np.zeros(0)
            return {
                "in_edge_id": empty.astype(np.int32),
                "out_edge_id": empty.astype(np.int32),
                "delta_deg": empty.astype(np.float32),
                "is_unprotected_left": empty.astype(bool),
                "is_aggressive_merge": empty.astype(bool),
            }

        rows = turns.node_id == node_index
        return {
            "in_edge_id": turns.in_edge_id[rows],
            "out_edge_id": turns.out_edge_id[rows],
            "delta_deg": turns.delta_deg[rows],
            "is_unprotected_left": turns.is_unprotected_left[rows],
            "is_aggressive_merge": turns.is_aggressive_merge[rows],
        }

    def to_pyg_data(self):
        """Return a ``torch_geometric.data.Data`` object.

        Imported lazily so the rest of this module stays usable without PyG.
        """
        from torch_geometric.data import Data

        return Data(
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            pos=self.pos,
            num_nodes=self.num_nodes,
        )

    def __repr__(self) -> str:
        turns = f", turns={len(self.turns)}" if self.turns is not None else ""
        return (
            f"IrvineGraphData(nodes={self.num_nodes}, edges={self.num_edges}, "
            f"features={list(self.edge_feature_names)}{turns})"
        )


def _turn_table_from_bundle(bundle: dict[str, Any]) -> TurnTable | None:
    """Rebuild the turn table from a tensor bundle, or None if it has none."""
    if "turn_node_id" not in bundle:
        return None
    return TurnTable(
        node_id=bundle["turn_node_id"].numpy(),
        in_edge_id=bundle["turn_in_edge_id"].numpy(),
        out_edge_id=bundle["turn_out_edge_id"].numpy(),
        bearing_in_deg=bundle["turn_bearing_in_deg"].numpy(),
        bearing_out_deg=bundle["turn_bearing_out_deg"].numpy(),
        delta_deg=bundle["turn_delta_deg"].numpy(),
        is_unprotected_left=bundle["turn_is_unprotected_left"].numpy(),
        is_aggressive_merge=bundle["turn_is_aggressive_merge"].numpy(),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--place", default=DEFAULT_PLACE)
    parser.add_argument("--network-type", default="drive")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--consolidate-tolerance-m",
        type=float,
        default=None,
        help="Merge intersection clusters within N metres (drops OSM node ids)",
    )
    parser.add_argument(
        "--weakly-connected",
        action="store_true",
        help="Keep the largest weakly connected component instead of strongly",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Keep posted speed limits and skip intersection delays "
        "(reproduces a pre-calibration artifact)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    )
    config = GraphBuildConfig(
        place=args.place,
        network_type=args.network_type,
        output_dir=args.output_dir,
        strongly_connected=not args.weakly_connected,
        consolidate_tolerance_m=args.consolidate_tolerance_m,
        use_cache=not args.no_cache,
        calibrate=not args.no_calibrate,
    )
    paths = build_graph_artifacts(config)
    graph = IrvineGraphData.load(config.output_dir)
    LOGGER.info("Built %r", graph)
    for name, path in paths.items():
        LOGGER.info("  %-8s %s", name, path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
