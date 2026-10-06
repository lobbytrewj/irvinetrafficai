"""Synthetic traffic speed/travel-time profiles for Irvine's primary corridors.

Real loop-detector feeds (PeMS, Iteris) need an agreement before they can be
redistributed, so this module stands in a simulator that reproduces the
structure a spatio-temporal model has to learn:

* a **diurnal shape** -- an AM peak around 07:45 and a broader, later PM peak,
  separated by a midday plateau and an overnight free-flow floor;
* **direction asymmetry** -- the AM peak loads the commute-in direction (in
  Irvine, broadly north-west toward the Business Complex and the 405), the PM
  peak the reverse, so opposite sides of the same arterial diverge by hours;
* **day-of-week structure** -- weekends lose both peaks and gain a midday hump;
* **spatial correlation** -- congestion is diffused along the segment graph,
  so neighbouring segments of a corridor move together;
* **incidents** -- rare Poisson shocks that decay over 15-120 minutes and
  propagate *upstream* against the direction of travel, which is the part a
  purely temporal model cannot predict and a graph model can;
* **noise** -- an AR(1) process, not white noise, so consecutive steps stay
  plausibly correlated.

Nodes are **road segments** (directed OSM edges), not intersections: travel
time is a property of a segment, so the prediction target and the
message-passing unit stay the same object. Segments are connected by the
*line graph* -- segment ``i`` neighbours ``j`` when the head of ``i`` is the
tail of ``j``, plus the reverse-direction twin of each segment, since a queue
on one side of an arterial is visible from the other.

Corridors covered: I-405, I-5, Jamboree Rd, Culver Dr and Alton Pkwy.

Outputs ``data/processed/traffic_dataset.pt`` containing the full
``[T, N]`` series plus the segment graph, the corridor labels and the
train-split scalers. Windowing into training samples happens lazily in
:class:`TrafficDataset`: materialising every window would cost terabytes,
since consecutive windows overlap in 11 of 12 steps.

Usage
-----
    python -m src.data.generate_traffic_data --days 90 --seed 7

    from src.data.generate_traffic_data import TrafficDataset, load_bundle
    bundle = load_bundle("data/processed/traffic_dataset.pt")
    train = TrafficDataset(bundle, split="train")
    x, y = train[0]          # x: [N, F, L]   y: [N, H]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)

DEFAULT_PROCESSED_DIR = Path("data/processed")
DATASET_FILENAME = "traffic_dataset.pt"
GRAPH_FILENAME = "irvine_graph.gpkg"
TENSOR_FILENAME = "graph_tensors.pt"

# Channel order of the feature tensor. Downstream code must index through this
# tuple rather than hard-coding positions.
FEATURE_NAMES: tuple[str, ...] = (
    "speed_kph",
    "travel_time_s",
    "tod_sin",
    "tod_cos",
    "is_weekend",
)
# Channels carrying physical units, and therefore the ones that get z-scored.
# The cyclical/binary calendar channels are already O(1) and are passed
# through unchanged so their geometry survives.
SCALED_CHANNELS: tuple[int, ...] = (0, 1)

TARGET_NAME = "travel_time_s"

# Speeds below this are treated as gridlock; keeps travel time finite.
MIN_SPEED_KPH = 6.0
# Congestion index is a fraction of free-flow speed lost. 0.95 => 5% of
# free flow, i.e. stop-and-go but still moving.
MAX_CONGESTION = 0.95

# Bearing (degrees) of the morning commute-in direction. Irvine's job cores
# (Irvine Business Complex, John Wayne, the 405 corridor) sit north-west of
# the residential villages, so NW-bound segments carry the AM load and the
# opposite bearing carries the PM load.
AM_INBOUND_BEARING = 315.0


# ---------------------------------------------------------------------------
# Corridor definitions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CorridorSpec:
    """One named corridor and how congested it gets.

    Matching is by OSM tag: freeways carry a ``ref`` (``"I 405"``), arterials
    carry a ``name`` (``"Culver Drive"``). Both are checked, and both may be
    multi-valued in OSMnx output, so any token matching counts.

    Attributes
    ----------
    severity:
        Peak fraction of free-flow speed lost at full demand. Freeways queue
        harder than arterials, which are metered by signals long before they
        reach a standstill.
    incidents_per_day:
        Expected number of incidents anywhere on the corridor per day.
    base_load:
        Off-peak congestion floor: even at 03:00 a freeway is not perfectly
        free-flowing.
    """

    key: str
    refs: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    severity: float = 0.55
    incidents_per_day: float = 0.4
    base_load: float = 0.06

    def matches(self, ref_cell: Any, name_cell: Any) -> bool:
        return bool(
            (self.refs and _tokens(ref_cell) & set(self.refs))
            or (self.names and _tokens(name_cell) & set(self.names))
        )


# Ordered: corridor id is the index into this tuple.
IRVINE_CORRIDORS: tuple[CorridorSpec, ...] = (
    CorridorSpec(
        key="I-405",
        # HOV/Truck carriageways are tagged separately but are the same road.
        refs=("I 405", "I 405 HOV", "I 405 Truck"),
        # ~105 kph free flow -> ~38 kph at the peak of the commute direction,
        # which is what the 405 through Irvine actually does on a weekday.
        severity=0.62,
        incidents_per_day=1.1,
        base_load=0.08,
    ),
    CorridorSpec(
        key="I-5",
        refs=("I 5", "I 5 Truck", "I 5 HOV"),
        severity=0.58,
        incidents_per_day=0.9,
        base_load=0.07,
    ),
    CorridorSpec(
        key="Jamboree Rd",
        names=("Jamboree Road",),
        # Arterials are signal-metered: they degrade earlier but bottom out
        # higher than a freeway, since the queue discharges every cycle.
        severity=0.44,
        incidents_per_day=0.35,
        base_load=0.06,
    ),
    CorridorSpec(
        key="Culver Dr",
        names=("Culver Drive",),
        severity=0.42,
        incidents_per_day=0.30,
        base_load=0.05,
    ),
    CorridorSpec(
        key="Alton Pkwy",
        names=("Alton Parkway",),
        severity=0.38,
        incidents_per_day=0.25,
        base_load=0.04,
    ),
)


@dataclass(frozen=True)
class TrafficGenConfig:
    """Inputs to :func:`build_dataset`."""

    processed_dir: Path = DEFAULT_PROCESSED_DIR
    days: int = 90
    step_minutes: int = 5
    # A Monday, so `days` whole weeks start cleanly on a weekday.
    start: datetime = datetime(2025, 1, 6, tzinfo=timezone.utc)
    seq_len: int = 12
    # Steps ahead to predict: 15, 30 and 60 minutes at a 5-minute cadence.
    horizons: tuple[int, ...] = (3, 6, 12)
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    corridors: tuple[CorridorSpec, ...] = IRVINE_CORRIDORS
    # AR(1) coefficient and innovation scale of the congestion noise.
    noise_rho: float = 0.85
    noise_sigma: float = 0.035
    # Fraction of a segment's congestion replaced by its neighbours' average,
    # applied `diffusion_steps` times. Higher => smoother along a corridor.
    diffusion_alpha: float = 0.25
    diffusion_steps: int = 2
    seed: int = 1234
    # Fall back to a synthetic corridor topology when the OSM artifacts from
    # src.data.graph_builder are absent.
    allow_synthetic_topology: bool = False

    @property
    def steps_per_day(self) -> int:
        if (24 * 60) % self.step_minutes:
            raise ValueError(
                f"step_minutes must divide 1440, got {self.step_minutes}"
            )
        return (24 * 60) // self.step_minutes

    @property
    def num_steps(self) -> int:
        return self.days * self.steps_per_day

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)


# ---------------------------------------------------------------------------
# Segment selection
# ---------------------------------------------------------------------------
@dataclass
class SegmentTable:
    """The corridor segments that become graph nodes.

    All arrays are parallel and indexed by the *segment index* ``0..N-1``,
    which is the node id everything downstream uses.
    """

    edge_id: np.ndarray  # [N] column in the full graph's edge_index, -1 if synthetic
    u: np.ndarray  # [N] tail node index in the road graph
    v: np.ndarray  # [N] head node index in the road graph
    length_m: np.ndarray  # [N]
    free_flow_kph: np.ndarray  # [N]
    bearing_deg: np.ndarray  # [N] direction of travel, 0=N, 90=E
    corridor_id: np.ndarray  # [N] index into `corridors`
    corridors: tuple[CorridorSpec, ...]

    def __len__(self) -> int:
        return int(self.edge_id.shape[0])

    def counts(self) -> dict[str, int]:
        return {
            spec.key: int((self.corridor_id == i).sum())
            for i, spec in enumerate(self.corridors)
        }


def _tokens(cell: Any) -> set[str]:
    """Normalise an OSM tag cell to a set of string tokens.

    OSMnx leaves multi-valued tags as Python lists, and the GeoPackage
    round-trip turns those into their ``str()`` form (``"['I 5', 'I 405']"``),
    so both shapes have to be handled.
    """
    if cell is None:
        return set()
    if isinstance(cell, (list, tuple, set)):
        return {str(item).strip() for item in cell if str(item).strip()}
    text = str(cell).strip()
    if not text or text.lower() in {"nan", "none"}:
        return set()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text.replace("'", '"'))
        except json.JSONDecodeError:
            parsed = [part.strip(" '\"") for part in text[1:-1].split(",")]
        return {str(item).strip() for item in parsed if str(item).strip()}
    return {text}


def _bearing_deg(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Initial great-circle bearing from (lat1, lon1) to (lat2, lon2)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(y, x)) % 360.0


def select_corridor_segments(
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    corridors: Sequence[CorridorSpec] = IRVINE_CORRIDORS,
) -> SegmentTable:
    """Pull the corridor segments out of the graph_builder artifacts.

    Edge ids must line up with the columns of ``graph_tensors.pt``'s
    ``edge_index``. ``graph_builder`` assigns them by sorting ``(u, v, key)``,
    while the GeoPackage preserves graph iteration order, so the frame is
    re-sorted here and the result cross-checked against the tensor bundle. A
    mismatch would silently attach every segment's history to the wrong road.
    """
    processed_dir = Path(processed_dir)
    graph_path = processed_dir / GRAPH_FILENAME
    tensor_path = processed_dir / TENSOR_FILENAME
    for path in (graph_path, tensor_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m src.data.graph_builder` first."
            )

    import geopandas as gpd  # imported lazily: heavy, and only needed here

    edges = gpd.read_file(graph_path, layer="edges")
    edges = edges.sort_values(["u", "v", "key"], kind="stable").reset_index(drop=True)
    edges["edge_id"] = np.arange(len(edges), dtype=np.int64)

    bundle = torch.load(tensor_path, weights_only=True)
    edge_index = bundle["edge_index"].numpy()
    if len(edges) != edge_index.shape[1]:
        raise ValueError(
            f"GeoPackage has {len(edges)} edges but graph_tensors.pt has "
            f"{edge_index.shape[1]}; regenerate both from graph_builder."
        )
    if not (
        np.array_equal(edges["u"].to_numpy(), edge_index[0])
        and np.array_equal(edges["v"].to_numpy(), edge_index[1])
    ):
        raise ValueError(
            "edge ordering in the GeoPackage does not match edge_index; "
            "regenerate both artifacts from the same graph_builder run."
        )

    ref_col = edges["ref"] if "ref" in edges.columns else [None] * len(edges)
    name_col = edges["name"] if "name" in edges.columns else [None] * len(edges)

    corridor_id = np.full(len(edges), -1, dtype=np.int64)
    for spec_idx, spec in enumerate(corridors):
        for row, (ref_cell, name_cell) in enumerate(zip(ref_col, name_col)):
            # First corridor to claim a segment wins; overlaps are rare
            # (a shared ramp) and arbitrary attribution is fine there.
            if corridor_id[row] == -1 and spec.matches(ref_cell, name_cell):
                corridor_id[row] = spec_idx

    keep = np.flatnonzero(corridor_id >= 0)
    if keep.size == 0:
        raise ValueError(
            "no segments matched any corridor; check the ref/name patterns "
            "against the tags in irvine_graph.gpkg"
        )
    selected = edges.iloc[keep]

    pos = bundle["pos"].numpy()
    u = selected["u"].to_numpy(dtype=np.int64).copy()
    v = selected["v"].to_numpy(dtype=np.int64).copy()
    bearing = _bearing_deg(pos[u, 0], pos[u, 1], pos[v, 0], pos[v, 1])

    table = SegmentTable(
        # .copy() throughout: pandas hands back read-only views, which
        # torch.from_numpy accepts but then warns about on every call.
        edge_id=selected["edge_id"].to_numpy(dtype=np.int64).copy(),
        u=u,
        v=v,
        length_m=selected["length"].to_numpy(dtype=np.float64).copy(),
        free_flow_kph=selected["speed_kph"].to_numpy(dtype=np.float64).copy(),
        bearing_deg=bearing,
        corridor_id=corridor_id[keep],
        corridors=tuple(corridors),
    )
    LOGGER.info("Selected %d corridor segments: %s", len(table), table.counts())
    return table


def synthetic_segment_table(
    corridors: Sequence[CorridorSpec] = IRVINE_CORRIDORS,
    segments_per_direction: int = 40,
    rng: np.random.Generator | None = None,
) -> SegmentTable:
    """Build a stand-in topology when the OSM artifacts are unavailable.

    Each corridor becomes a straight chain of ``segments_per_direction``
    segments plus its opposite carriageway. Useful for CI and for exercising
    the pipeline before ``graph_builder`` has been run; not a substitute for
    the real network, since the true corridor's branching is what makes the
    spatial convolution earn its keep.
    """
    rng = rng or np.random.default_rng(0)
    edge_id, u, v, length, free_flow, bearing, corridor_id = [], [], [], [], [], [], []
    next_node = 0

    for spec_idx, spec in enumerate(corridors):
        is_freeway = bool(spec.refs)
        base_speed = 105.0 if is_freeway else 64.0
        seg_len = 900.0 if is_freeway else 420.0
        # Corridors fan out around the AM inbound bearing so both commute
        # directions are represented.
        heading = (AM_INBOUND_BEARING + 30.0 * spec_idx) % 360.0

        chain = list(range(next_node, next_node + segments_per_direction + 1))
        next_node = chain[-1] + 1
        for a, b in zip(chain, chain[1:]):
            for tail, head, direction in ((a, b, heading), (b, a, (heading + 180) % 360)):
                edge_id.append(-1)
                u.append(tail)
                v.append(head)
                length.append(seg_len * float(rng.uniform(0.8, 1.2)))
                free_flow.append(base_speed * float(rng.uniform(0.95, 1.05)))
                bearing.append(direction)
                corridor_id.append(spec_idx)

    return SegmentTable(
        edge_id=np.asarray(edge_id, dtype=np.int64),
        u=np.asarray(u, dtype=np.int64),
        v=np.asarray(v, dtype=np.int64),
        length_m=np.asarray(length, dtype=np.float64),
        free_flow_kph=np.asarray(free_flow, dtype=np.float64),
        bearing_deg=np.asarray(bearing, dtype=np.float64),
        corridor_id=np.asarray(corridor_id, dtype=np.int64),
        corridors=tuple(corridors),
    )


# ---------------------------------------------------------------------------
# Segment (line) graph
# ---------------------------------------------------------------------------
def build_segment_graph(table: SegmentTable) -> tuple[np.ndarray, np.ndarray]:
    """Line graph over the selected segments.

    Returns
    -------
    (downstream, undirected):
        ``downstream`` is ``[2, E]`` with an edge ``i -> j`` when traffic
        flows out of ``i`` into ``j`` (head of ``i`` == tail of ``j``). It is
        what incident shockwaves travel *backwards* along.

        ``undirected`` is the symmetric closure plus opposite-carriageway
        links, which is what the ChebConv operator consumes: a symmetric
        normalised Laplacian needs a symmetric adjacency.
    """
    by_tail: dict[int, list[int]] = {}
    for seg, tail in enumerate(table.u):
        by_tail.setdefault(int(tail), []).append(seg)

    downstream: list[tuple[int, int]] = []
    for seg, head in enumerate(table.v):
        for nxt in by_tail.get(int(head), ()):
            if nxt != seg:
                downstream.append((seg, nxt))

    pairs = set(downstream)
    pairs |= {(j, i) for i, j in downstream}

    # Opposite carriageway: same endpoints, reversed. Congestion on one side
    # is observable from the other (shared signals, rubbernecking, diversions).
    by_endpoints: dict[tuple[int, int], list[int]] = {}
    for seg, (tail, head) in enumerate(zip(table.u, table.v)):
        by_endpoints.setdefault((int(tail), int(head)), []).append(seg)
    for seg, (tail, head) in enumerate(zip(table.u, table.v)):
        for twin in by_endpoints.get((int(head), int(tail)), ()):
            if twin != seg:
                pairs.add((seg, twin))
                pairs.add((twin, seg))

    if not pairs:
        raise ValueError(
            "the selected segments share no endpoints; the corridor set is "
            "entirely disconnected"
        )

    downstream_arr = np.asarray(sorted(set(downstream)), dtype=np.int64).T
    undirected_arr = np.asarray(sorted(pairs), dtype=np.int64).T
    LOGGER.info(
        "Segment graph: %d nodes, %d directed / %d undirected links",
        len(table),
        downstream_arr.shape[1],
        undirected_arr.shape[1],
    )
    return downstream_arr, undirected_arr


def _row_normalised_neighbour_matrix(
    edge_index: np.ndarray, num_nodes: int
) -> np.ndarray:
    """Dense ``[N, N]`` row-stochastic neighbour-average operator.

    Dense is fine at corridor scale (N in the hundreds) and keeps the
    diffusion a single matmul per step.
    """
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adjacency[edge_index[0], edge_index[1]] = 1.0
    degree = adjacency.sum(axis=1, keepdims=True)
    # Isolated segments keep their own value rather than becoming NaN.
    np.divide(adjacency, degree, out=adjacency, where=degree > 0)
    isolated = (degree.ravel() == 0)
    adjacency[isolated, isolated] = 1.0
    return adjacency


# ---------------------------------------------------------------------------
# Demand simulation
# ---------------------------------------------------------------------------
def calendar_features(config: TrafficGenConfig) -> tuple[np.ndarray, np.ndarray]:
    """``(hour_of_day, is_weekend)`` for every step in the horizon.

    The simulator and the feature tensor both need this, and they must agree:
    if the peaks were driven by one weekday calendar while the model saw
    another, the ``is_weekend`` channel would be actively misleading.
    """
    step_hours = config.step_minutes / 60.0
    absolute_hours = np.arange(config.num_steps, dtype=np.float64) * step_hours
    hours = absolute_hours % 24.0
    weekday = (
        config.start.weekday() + (absolute_hours // 24).astype(np.int64)
    ) % 7
    return hours, (weekday >= 5).astype(np.float64)


def _gaussian_bump(
    hours: np.ndarray, centre: np.ndarray | float, width_h: float
) -> np.ndarray:
    """Periodic Gaussian over a 24-hour clock (wraps across midnight)."""
    delta = np.abs(hours - centre)
    delta = np.minimum(delta, 24.0 - delta)
    return np.exp(-0.5 * (delta / width_h) ** 2)


def temporal_demand(
    hours: np.ndarray,
    is_weekend: np.ndarray,
    am_offset_h: np.ndarray | float = 0.0,
    pm_offset_h: np.ndarray | float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split demand into AM-peak, PM-peak and undirected background parts.

    Returns three curves in ``[0, 1]``. Keeping the peaks separate is what
    lets a segment's direction weight them differently -- the whole point of
    modelling a commute rather than a generic rush.

    ``*_offset_h`` shifts each segment's peak in time. Passing a ``[N]`` array
    (with ``hours`` shaped ``[T, 1]``) returns ``[T, N]`` peaks: real queues
    form at bottlenecks first and take minutes to reach the segments behind
    them, and without that jitter every segment peaks on the same step and the
    whole network collapses to one shared curve.
    """
    weekday = 1.0 - is_weekend
    # Widths: the AM peak is sharp (school + work start on the same clock),
    # the PM peak is smeared over staggered departures and errands.
    am = weekday * _gaussian_bump(hours, centre=7.75 + am_offset_h, width_h=0.85)
    pm = weekday * _gaussian_bump(hours, centre=17.25 + pm_offset_h, width_h=1.35)
    # Weekday midday plateau plus a weekend shopping hump; both taper to an
    # overnight floor.
    background = (
        weekday * 0.34 * _gaussian_bump(hours, centre=12.5, width_h=3.0)
        + is_weekend * 0.55 * _gaussian_bump(hours, centre=14.0, width_h=3.4)
        + 0.08 * _gaussian_bump(hours, centre=12.0, width_h=6.0)
    )
    return am, pm, background


def direction_weights(bearing_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-segment AM/PM loading from the direction of travel.

    A raised cosine centred on :data:`AM_INBOUND_BEARING`: a segment pointing
    exactly inbound gets the full AM peak and a floor of the PM peak, and
    vice versa. The 0.25 floor keeps the reverse-commute direction from going
    perfectly free-flow, which is not what the counter-peak actually looks
    like.
    """
    theta = np.radians(bearing_deg - AM_INBOUND_BEARING)
    inbound = 0.5 * (1.0 + np.cos(theta))  # 1 inbound, 0 outbound
    am = 0.25 + 0.75 * inbound
    pm = 0.25 + 0.75 * (1.0 - inbound)
    return am, pm


def _incident_shock(
    config: TrafficGenConfig,
    table: SegmentTable,
    downstream: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample incidents and spread them upstream. Returns ``[T, N]``.

    An incident is a capacity drop at one segment. The queue it creates grows
    backwards along the direction of travel, so the shock is propagated over
    the *reverse* of the downstream graph with a per-hop decay. Onset is
    faster than recovery, matching how a lane closure clears.
    """
    num_steps, num_nodes = config.num_steps, len(table)
    shock = np.zeros((num_steps, num_nodes), dtype=np.float32)

    upstream_of: dict[int, list[int]] = {}
    for src, dst in zip(downstream[0], downstream[1]):
        upstream_of.setdefault(int(dst), []).append(int(src))

    severity = np.array(
        [table.corridors[c].severity for c in table.corridor_id], dtype=np.float64
    )
    per_day = np.array(
        [table.corridors[c].incidents_per_day for c in table.corridor_id],
        dtype=np.float64,
    )
    # Corridor rate is for the whole corridor, so split it across its segments.
    counts = np.bincount(table.corridor_id, minlength=len(table.corridors))
    rate_per_segment = per_day / np.maximum(counts[table.corridor_id], 1)
    expected = rate_per_segment.sum() * config.days
    num_incidents = int(rng.poisson(expected))

    steps_per_hour = 60 // config.step_minutes
    probabilities = rate_per_segment / rate_per_segment.sum()

    for _ in range(num_incidents):
        origin = int(rng.choice(num_nodes, p=probabilities))
        start = int(rng.integers(0, num_steps))
        # 15 minutes to 2 hours, log-uniform: most incidents are minor.
        duration = int(
            np.clip(
                rng.lognormal(mean=math.log(4 * steps_per_hour / 3), sigma=0.7),
                3,
                2 * steps_per_hour,
            )
        )
        magnitude = float(rng.uniform(0.3, 0.9)) * severity[origin]

        onset = max(1, duration // 4)
        profile = np.concatenate(
            [
                np.linspace(0.0, 1.0, onset, endpoint=False),
                np.exp(-np.arange(duration) / max(duration / 2.0, 1.0)),
            ]
        )
        window = min(len(profile), num_steps - start)
        if window <= 0:
            continue
        profile = profile[:window] * magnitude

        # Breadth-first upstream spread, 35% of the shock lost per hop.
        frontier, visited, decay = [origin], {origin}, 1.0
        for _hop in range(4):
            for seg in frontier:
                shock[start : start + window, seg] += profile * decay
            decay *= 0.65
            next_frontier = []
            for seg in frontier:
                for up in upstream_of.get(seg, ()):
                    if up not in visited:
                        visited.add(up)
                        next_frontier.append(up)
            frontier = next_frontier
            if not frontier:
                break

    LOGGER.info("Simulated %d incidents over %d days", num_incidents, config.days)
    return shock


def simulate_congestion(
    config: TrafficGenConfig,
    table: SegmentTable,
    downstream: np.ndarray,
    undirected: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the simulator.

    Returns ``(speed_kph, travel_time_s, hour_of_day)`` with the first two
    shaped ``[T, N]`` and the last ``[T]``.
    """
    num_steps, num_nodes = config.num_steps, len(table)
    hours, is_weekend = calendar_features(config)

    am_w, pm_w = (w.astype(np.float32) for w in direction_weights(table.bearing_deg))
    severity = np.array(
        [table.corridors[c].severity for c in table.corridor_id], dtype=np.float32
    )
    base_load = np.array(
        [table.corridors[c].base_load for c in table.corridor_id], dtype=np.float32
    )
    # Per-segment sensitivity: a segment beside a signalised intersection or a
    # freeway merge queues sooner than a mid-block one. Static across time, so
    # the model can learn it from the segment's own history.
    sensitivity = rng.uniform(0.65, 1.35, size=num_nodes)
    # Peak timing jitter, ~+/-20 minutes: bottlenecks break down before the
    # segments feeding them do.
    am_offset = rng.normal(0.0, 0.35, size=num_nodes)
    pm_offset = rng.normal(0.0, 0.45, size=num_nodes)

    # [T, 1] against [1, N] -> per-segment [T, N] curves, in float32: at 90
    # days x 5 minutes these are ~12M elements each.
    am_curve, pm_curve, background = temporal_demand(
        hours[:, None].astype(np.float32),
        is_weekend[:, None].astype(np.float32),
        am_offset[None, :].astype(np.float32),
        pm_offset[None, :].astype(np.float32),
    )
    demand = am_curve * am_w[None, :] + pm_curve * pm_w[None, :] + background
    del am_curve, pm_curve

    congestion = base_load[None, :] + severity[None, :] * sensitivity[None, :] * demand
    congestion = congestion.astype(np.float32, copy=False)
    del demand

    # AR(1) noise: consecutive steps stay correlated, so a model cannot beat
    # the series by memorising an i.i.d. mean.
    innovations = rng.normal(
        0.0, config.noise_sigma, size=(num_steps, num_nodes)
    ).astype(np.float32)
    noise = np.empty_like(innovations)
    noise[0] = innovations[0]
    for t in range(1, num_steps):
        noise[t] = config.noise_rho * noise[t - 1] + innovations[t]
    congestion += noise
    del noise, innovations

    congestion += _incident_shock(config, table, downstream, rng)

    # Spatial smoothing last, so incidents and noise are correlated across
    # neighbours too -- otherwise the graph convolution has nothing to exploit
    # that a per-segment temporal model could not learn on its own.
    neighbour_mean = _row_normalised_neighbour_matrix(undirected, num_nodes)
    mixer = (
        (1.0 - config.diffusion_alpha) * np.eye(num_nodes, dtype=np.float32)
        + config.diffusion_alpha * neighbour_mean
    ).astype(np.float32)
    for _ in range(config.diffusion_steps):
        congestion = congestion @ mixer.T

    np.clip(congestion, 0.0, MAX_CONGESTION, out=congestion)
    speed = np.maximum(
        table.free_flow_kph[None, :].astype(np.float32) * (1.0 - congestion),
        MIN_SPEED_KPH,
    )
    travel_time = table.length_m[None, :].astype(np.float32) / (speed / 3.6)
    return speed, travel_time, hours


# ---------------------------------------------------------------------------
# Assembly and persistence
# ---------------------------------------------------------------------------
def _split_bounds(config: TrafficGenConfig) -> dict[str, tuple[int, int]]:
    """Chronological ``[start, end)`` step ranges per split.

    Splitting by time, not at random, is the only honest option for a
    forecaster: random windows overlap in 11 of their 12 steps, so a shuffled
    split leaks almost the entire validation set into training.
    """
    total = config.num_steps
    train_end = int(total * config.train_fraction)
    val_end = int(total * (config.train_fraction + config.val_fraction))
    if not 0 < train_end < val_end < total:
        raise ValueError(
            f"split fractions {config.train_fraction}/{config.val_fraction} "
            f"leave an empty split for {total} steps"
        )
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, total),
    }


def build_dataset(config: TrafficGenConfig | None = None) -> dict[str, Any]:
    """Generate the full dataset bundle. Does not write to disk."""
    config = config or TrafficGenConfig()
    rng = np.random.default_rng(config.seed)

    try:
        table = select_corridor_segments(config.processed_dir, config.corridors)
    except FileNotFoundError:
        if not config.allow_synthetic_topology:
            raise
        LOGGER.warning(
            "Graph artifacts missing; falling back to a synthetic corridor "
            "topology. Results are not tied to the real Irvine network."
        )
        table = synthetic_segment_table(config.corridors, rng=rng)

    downstream, undirected = build_segment_graph(table)
    speed, travel_time, hours = simulate_congestion(
        config, table, downstream, undirected, rng
    )

    num_steps = config.num_steps
    step_seconds = config.step_minutes * 60
    timestamps = np.array(
        [
            int((config.start + timedelta(seconds=i * step_seconds)).timestamp())
            for i in range(num_steps)
        ],
        dtype=np.int64,
    )
    _, is_weekend = calendar_features(config)
    is_weekend = is_weekend.astype(np.float32)
    tod = hours / 24.0

    bounds = _split_bounds(config)
    train_slice = slice(*bounds["train"])
    speed_t = torch.from_numpy(speed).float()
    travel_t = torch.from_numpy(travel_time).float()

    feature_mean, feature_std = _feature_scaler(
        speed_t[train_slice], travel_t[train_slice]
    )
    target_mean = travel_t[train_slice].mean(dim=0)
    # Guard against a segment that never moves off free flow in the training
    # window: a zero std would produce inf targets.
    target_std = travel_t[train_slice].std(dim=0).clamp_min(1e-3)

    free_flow_tt = torch.from_numpy(
        table.length_m / (table.free_flow_kph / 3.6)
    ).float()

    return {
        # --- series -------------------------------------------------------
        "speed_kph": speed_t,  # [T, N]
        "travel_time_s": travel_t,  # [T, N]
        "tod_sin": torch.from_numpy(np.sin(2 * np.pi * tod)).float(),  # [T]
        "tod_cos": torch.from_numpy(np.cos(2 * np.pi * tod)).float(),  # [T]
        "is_weekend": torch.from_numpy(is_weekend),  # [T]
        "timestamps": torch.from_numpy(timestamps),  # [T]
        # --- graph --------------------------------------------------------
        "edge_index": torch.from_numpy(undirected).long(),  # [2, E] symmetric
        "downstream_edge_index": torch.from_numpy(downstream).long(),
        # --- segment metadata ---------------------------------------------
        "segment_edge_id": torch.from_numpy(table.edge_id),
        "segment_u": torch.from_numpy(table.u),
        "segment_v": torch.from_numpy(table.v),
        "segment_length_m": torch.from_numpy(table.length_m).float(),
        "segment_free_flow_kph": torch.from_numpy(table.free_flow_kph).float(),
        "segment_bearing_deg": torch.from_numpy(table.bearing_deg).float(),
        "segment_free_flow_time_s": free_flow_tt,
        "corridor_id": torch.from_numpy(table.corridor_id),
        "corridor_names": [spec.key for spec in table.corridors],
        # --- scalers (train split only) -----------------------------------
        "feature_mean": feature_mean,  # [N, F]
        "feature_std": feature_std,  # [N, F]
        "target_mean": target_mean,  # [N]
        "target_std": target_std,  # [N]
        # --- protocol -----------------------------------------------------
        "feature_names": list(FEATURE_NAMES),
        "target_name": TARGET_NAME,
        "seq_len": config.seq_len,
        "horizons": list(config.horizons),
        "step_minutes": config.step_minutes,
        "num_nodes": len(table),
        "num_steps": num_steps,
        "splits": {name: [int(lo), int(hi)] for name, (lo, hi) in bounds.items()},
        "config": _config_payload(config),
    }


def _feature_scaler(
    speed: torch.Tensor, travel_time: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-segment z-score statistics, identity for the calendar channels.

    Per-segment rather than global: a 900 m freeway segment and a 200 m
    arterial block differ by an order of magnitude in travel time, and a
    single global scaler would let the long segments dominate the loss.
    """
    num_nodes = speed.size(1)
    mean = torch.zeros(num_nodes, len(FEATURE_NAMES))
    std = torch.ones(num_nodes, len(FEATURE_NAMES))
    for channel, series in zip(SCALED_CHANNELS, (speed, travel_time)):
        mean[:, channel] = series.mean(dim=0)
        std[:, channel] = series.std(dim=0).clamp_min(1e-3)
    return mean, std


def _config_payload(config: TrafficGenConfig) -> dict[str, Any]:
    """JSON/`weights_only`-safe view of the config for provenance."""
    payload = asdict(config)
    payload["processed_dir"] = str(config.processed_dir)
    payload["start"] = config.start.isoformat()
    payload["horizons"] = list(config.horizons)
    payload["corridors"] = [spec.key for spec in config.corridors]
    return payload


def save_dataset(
    bundle: dict[str, Any], output_dir: Path | str = DEFAULT_PROCESSED_DIR
) -> Path:
    """Write the bundle to ``<output_dir>/traffic_dataset.pt``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / DATASET_FILENAME
    torch.save(bundle, path)
    LOGGER.info(
        "Wrote %s (%d steps x %d segments, %.1f MB)",
        path,
        bundle["num_steps"],
        bundle["num_nodes"],
        path.stat().st_size / 1e6,
    )
    return path


def load_bundle(path: Path | str = DEFAULT_PROCESSED_DIR / DATASET_FILENAME):
    """Load a saved bundle. Directories are resolved to the default filename."""
    path = Path(path)
    if path.is_dir():
        path = path / DATASET_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.data.generate_traffic_data` first."
        )
    # weights_only=True: our own plain tensors/primitives, and it keeps
    # torch.load off the arbitrary-code-execution path.
    return torch.load(path, weights_only=True)


# ---------------------------------------------------------------------------
# Windowed view for training
# ---------------------------------------------------------------------------
class TrafficDataset(Dataset):
    """Sliding windows over one chronological split.

    Sample ``k`` starting at step ``t`` yields

        x : ``[N, F, L]``  normalised features over ``t .. t+L-1``
        y : ``[N, H]``     normalised travel time at ``t+L-1+h`` for each h

    Windows are constrained to lie entirely inside their split, so no sample
    ever straddles a boundary and no target is drawn from a later split.

    ``stride`` keeps every ``stride``-th window. Consecutive windows share
    ``L - 1`` of their ``L`` input steps, so striding the *training* split
    trades a little redundancy for a proportionally shorter epoch; leave it at
    1 for validation and test, where every window should be scored.
    """

    def __init__(
        self,
        bundle: dict[str, Any],
        split: str = "train",
        normalise: bool = True,
        stride: int = 1,
    ) -> None:
        if split not in bundle["splits"]:
            raise KeyError(
                f"unknown split {split!r}; have {sorted(bundle['splits'])}"
            )
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.bundle = bundle
        self.split = split
        self.normalise = normalise
        self.stride = stride
        self.seq_len = int(bundle["seq_len"])
        self.horizons = [int(h) for h in bundle["horizons"]]
        self.num_nodes = int(bundle["num_nodes"])

        start, end = (int(v) for v in bundle["splits"][split])
        span = self.seq_len + max(self.horizons)
        self.starts = torch.arange(
            start, max(start, end - span + 1), stride, dtype=torch.long
        )
        if not len(self.starts):
            raise ValueError(
                f"split {split!r} spans {end - start} steps, too few for a "
                f"{span}-step window"
            )

        # Views into the bundle, never copies: a materialised [T, N, F] tensor
        # would cost hundreds of MB and be duplicated across the three splits.
        # Assembling and scaling one [L, N, F] window per sample is cheap.
        self.speed = bundle["speed_kph"]
        self.travel_time = bundle["travel_time_s"]
        self.calendar = torch.stack(
            [bundle["tod_sin"], bundle["tod_cos"], bundle["is_weekend"]], dim=-1
        )  # [T, 3]
        self.feature_mean = bundle["feature_mean"]
        self.feature_std = bundle["feature_std"]
        self.target_mean = bundle["target_mean"]
        self.target_std = bundle["target_std"]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[index])
        stop = start + self.seq_len
        calendar = self.calendar[start:stop]  # [L, 3]
        window = torch.cat(
            [
                self.speed[start:stop].unsqueeze(-1),
                self.travel_time[start:stop].unsqueeze(-1),
                calendar.unsqueeze(1).expand(-1, self.num_nodes, -1),
            ],
            dim=-1,
        )  # [L, N, F]
        target_steps = [stop - 1 + h for h in self.horizons]
        target = self.travel_time[target_steps]  # [H, N]

        if self.normalise:
            window = (window - self.feature_mean) / self.feature_std
            target = (target - self.target_mean) / self.target_std

        # [L, N, F] -> [N, F, L]; [H, N] -> [N, H]
        return window.permute(1, 2, 0).contiguous(), target.t().contiguous()

    def __repr__(self) -> str:
        return (
            f"TrafficDataset(split={self.split!r}, windows={len(self)}, "
            f"nodes={self.num_nodes}, seq_len={self.seq_len}, "
            f"horizons={self.horizons}, stride={self.stride})"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--step-minutes", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[3, 6, 12],
        help="forecast steps ahead (default: 3 6 12 == 15/30/60 min)",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--allow-synthetic-topology",
        action="store_true",
        help="fabricate a corridor topology if the OSM artifacts are missing",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    )
    config = TrafficGenConfig(
        processed_dir=args.processed_dir,
        days=args.days,
        step_minutes=args.step_minutes,
        seq_len=args.seq_len,
        horizons=tuple(args.horizons),
        seed=args.seed,
        allow_synthetic_topology=args.allow_synthetic_topology,
    )
    bundle = build_dataset(config)
    path = save_dataset(bundle, args.output_dir or args.processed_dir)

    for split in ("train", "val", "test"):
        LOGGER.info("  %s", TrafficDataset(bundle, split=split))
    speed = bundle["speed_kph"]
    LOGGER.info(
        "Speed range %.1f-%.1f kph (mean %.1f); wrote %s",
        float(speed.min()),
        float(speed.max()),
        float(speed.mean()),
        path,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
