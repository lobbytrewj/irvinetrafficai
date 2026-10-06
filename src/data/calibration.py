"""Drive-time calibration: what a road costs beyond its length and speed limit.

The problem this exists to fix
------------------------------
Free-flow travel time is ``length / posted_speed``, and a router built on it
describes a city with no signals, no cross traffic and no left turns. Across
Irvine that is not a small error: the UTC to Tustin Legacy trip is 7.3 km of
arterial, which the uncalibrated graph covers in 5.6 minutes at an implied
78 km/h. Nobody has ever driven Jamboree at 78 km/h. The trip takes 13 to 16
minutes, and the missing eight are spent at red lights.

Two corrections close that gap, and they are separate things:

1. **Cruise speed.** The posted limit is an upper bound reached midway between
   signals, not an average. Progression, lane changes, and the car ahead pull
   the achievable cruise well below it, so arterial speeds are capped.
2. **Intersection delay.** Every junction on the path costs time, and how much
   depends on who controls it and which way you are going.

Both are calibration constants, not measurements. They are set to the middle of
the published ranges for a signalised suburban arterial grid (HCM-style
uniform delay at a 120 s cycle with 45-55 % green is roughly 25-35 s per
signal) and checked end to end against known drive times in
``tests/python/test_router_calibration.py``. Retune them there.

Why this module and not graph_builder
--------------------------------------
Both the offline pipeline and the serving path need the same numbers:
``graph_builder`` bakes them into new artifacts, while ``app.network`` applies
them to artifacts already on disk. Keeping them here means the two cannot
drift, and lets the service import them without pulling OSMnx into startup.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "RoadClass",
    "IntersectionDelays",
    "ARTERIAL_CRUISE_CAP_KPH",
    "MINOR_CRUISE_CAP_KPH",
    "SIGNAL_DELAY_SEC",
    "MINOR_SIGNAL_DELAY_SEC",
    "YIELD_DELAY_SEC",
    "TURN_PENALTY_SEC",
    "normalize_highway",
    "road_class",
    "road_classes",
    "cruise_speed_kph",
    "cap_speeds_kph",
    "classify_intersections",
    "polyline_bearings",
]


class RoadClass(IntEnum):
    """Functional class, matching ``irvine::RoadClass`` in the native engine."""

    FREEWAY = 0
    ARTERIAL = 1
    MINOR = 2


# --------------------------------------------------------------------------
# Speed
# --------------------------------------------------------------------------
# Highest sustained speed an arterial actually delivers between signals. Irvine
# posts 45-50 mph (72-80 km/h) on Jamboree, Culver and Alton; GPS traces put
# free-flow midblock cruise nearer 37 mph, because the platoon leaves each
# signal together and never gets the whole block to accelerate through.
ARTERIAL_CRUISE_CAP_KPH = 60.0

# Collectors, residential streets and the link roads/ramps that join arterials.
MINOR_CRUISE_CAP_KPH = 45.0

# Freeways are deliberately uncapped: they are grade separated, they hold their
# posted speed off-peak, and congestion on them is exactly what the ST-GNN is
# trained to predict. Capping here would double-count it.

# --------------------------------------------------------------------------
# Intersection delay
# --------------------------------------------------------------------------
# Average delay for a movement through a signalised arterial crossing, taken
# over the whole cycle -- greens included, which is why it is well below the
# ~60 s a red costs. Charged to every movement: a light stops through traffic
# and turning traffic alike.
SIGNAL_DELAY_SEC = 30.0

# A signal on a minor road runs a shorter cycle and gives the minor approach a
# smaller share of it.
MINOR_SIGNAL_DELAY_SEC = 25.0

# Stop sign or uncontrolled junction. Charged only to movements that enter or
# leave the minor road, which is the whole point of the distinction: an
# arterial does not slow down for the residential street crossing it.
YIELD_DELAY_SEC = 10.0

# The turn itself, on top of any node delay. A left crosses oncoming traffic
# and waits for a gap or a protected phase; a right merges with it.
TURN_PENALTY_SEC = {
    "through": 0.0,
    "right": 5.0,
    "left": 15.0,
    "uturn": 25.0,
}

# OSM highway tags by class. Anything unrecognised is treated as minor, which
# is the conservative direction: it can only add delay, never remove it.
_FREEWAY_TAGS = frozenset(
    {"motorway", "motorway_link", "trunk", "trunk_link"}
)
_ARTERIAL_TAGS = frozenset(
    {"primary", "secondary", "primary_link", "secondary_link"}
)
# Links are ramps and turn lanes: an arterial's slip road is not itself an
# arterial to cruise along.
_LINK_TAGS = frozenset(
    {"motorway_link", "trunk_link", "primary_link", "secondary_link",
     "tertiary_link"}
)

# The classes that make a junction a *crossing* worth signalising. Links are
# excluded: a ramp meeting its parent road is not a crossroads.
_CROSSING_TAGS = frozenset({"primary", "secondary", "trunk", "motorway"})

# Importance order, most important first. Used to pick one tag out of a way
# that carries several: a road tagged ``['tertiary', 'residential']`` behaves
# like the tertiary it also is. Anything unlisted sorts last.
_TAG_RANK = {
    tag: rank
    for rank, tag in enumerate(
        (
            "motorway", "trunk", "primary", "secondary", "tertiary",
            "motorway_link", "trunk_link", "primary_link", "secondary_link",
            "tertiary_link", "unclassified", "residential", "living_street",
            "service", "track",
        )
    )
}

# OSM node tags that pin down intersection control directly, where mappers have
# bothered. Irvine's coverage is partial -- 383 of 6,864 nodes -- so these
# refine the class-based rule below rather than replacing it.
_SIGNAL_NODE_TAGS = frozenset({"traffic_signals", "crossing"})
_STOP_NODE_TAGS = frozenset({"stop", "give_way", "mini_roundabout"})


@dataclass(frozen=True)
class IntersectionDelays:
    """Per-node delay in seconds, in the two forms the engine charges them.

    ``priority_sec`` is paid by every movement through the node; ``yield_sec``
    only by movements that touch a minor road. ``kind`` is the label each node
    was classified as, kept for reporting and for tests to assert against.
    """

    priority_sec: np.ndarray  # float32 [N]
    yield_sec: np.ndarray     # float32 [N]
    kind: list[str]

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for label in self.kind:
            counts[label] = counts.get(label, 0) + 1
        return counts


# --------------------------------------------------------------------------
# Tag handling
# --------------------------------------------------------------------------
def normalize_highway(value: object) -> str:
    """One tag from an OSM ``highway`` value.

    OSMnx keeps a *list* where a way carries several classes, and the
    GeoPackage round-trips it as the string ``"['tertiary', 'residential']"``.
    The most important class wins, since that is the one that governs how the
    junction behaves.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        tags = [str(v) for v in value]
    else:
        text = str(value).strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                tags = [str(v) for v in parsed] if isinstance(parsed, (list, tuple)) else [text]
            except (ValueError, SyntaxError):
                tags = [text.strip("[]'\" ")]
        else:
            tags = [text]

    if not tags:
        return ""
    # Most important tag wins, so "['tertiary', 'residential']" resolves to
    # tertiary rather than to whichever happened to come first.
    return min(tags, key=lambda tag: (_TAG_RANK.get(tag, len(_TAG_RANK)), tag))


def road_class(highway: object) -> RoadClass:
    """Functional class for one OSM highway tag."""
    tag = highway if isinstance(highway, str) else normalize_highway(highway)
    if tag in _FREEWAY_TAGS:
        return RoadClass.FREEWAY
    if tag in _ARTERIAL_TAGS:
        return RoadClass.ARTERIAL
    return RoadClass.MINOR


def road_classes(highways: Iterable[object]) -> np.ndarray:
    """``uint8`` array of :class:`RoadClass` values, for the engine."""
    return np.fromiter(
        (int(road_class(normalize_highway(h))) for h in highways),
        dtype=np.uint8,
    )


# --------------------------------------------------------------------------
# Speed capping
# --------------------------------------------------------------------------
def cruise_speed_kph(highway: object, posted_kph: float) -> float:
    """Achievable cruise speed for one edge, in km/h.

    Never raises a speed: the cap is a ceiling on the posted limit, so a
    residential street posted at 40 stays at 40.
    """
    tag = normalize_highway(highway)
    if tag in _FREEWAY_TAGS:
        return float(posted_kph)
    if tag in _ARTERIAL_TAGS and tag not in _LINK_TAGS:
        return float(min(posted_kph, ARTERIAL_CRUISE_CAP_KPH))
    return float(min(posted_kph, MINOR_CRUISE_CAP_KPH))


def cap_speeds_kph(
    highways: Sequence[object], posted_kph: np.ndarray
) -> np.ndarray:
    """Vectorised :func:`cruise_speed_kph` over a whole edge table."""
    posted = np.asarray(posted_kph, dtype=np.float64)
    if len(highways) != posted.shape[0]:
        raise ValueError(
            f"{len(highways)} highway tags but {posted.shape[0]} speeds"
        )
    return np.fromiter(
        (cruise_speed_kph(h, s) for h, s in zip(highways, posted)),
        dtype=np.float64,
        count=posted.shape[0],
    )


# --------------------------------------------------------------------------
# Intersection classification
# --------------------------------------------------------------------------
def classify_intersections(
    num_nodes: int,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_highway: Sequence[object],
    edge_name: Sequence[object] | None = None,
    node_highway: Sequence[object] | None = None,
) -> IntersectionDelays:
    """Assign every node a delay from the roads that meet there.

    The rules, in order:

    ``freeway``
        Every incident road is a motorway, a trunk or one of their links. Grade
        separated, so no delay at all -- this is what keeps a freeway route
        from being charged for the interchanges it flies over.
    ``through``
        Fewer than three distinct neighbours and no signal tag. A dead end, or
        a node OSMnx kept because an attribute changed there. Not an
        intersection, so no delay.
    ``signal``
        Tagged ``highway=traffic_signals``, **or** two or more distinctly named
        arterial-class streets cross here. The second half of that rule is
        doing most of the work: OSM's signal coverage in Irvine is partial,
        while its arterial grid is signalised at essentially every crossing of
        two named arterials, so classifying the crossing is more reliable than
        trusting the tag to be present.
    ``minor``
        Anything else that is a real junction: an arterial meeting a driveway
        or a residential street, or two residential streets meeting. Charged as
        a yield, so it costs the traffic turning on or off the minor road and
        not the arterial running past it.
    """
    edge_u = np.asarray(edge_u, dtype=np.int64)
    edge_v = np.asarray(edge_v, dtype=np.int64)
    if edge_u.shape != edge_v.shape:
        raise ValueError("edge_u and edge_v must be the same length")
    if len(edge_highway) != edge_u.shape[0]:
        raise ValueError(
            f"{len(edge_highway)} highway tags but {edge_u.shape[0]} edges"
        )

    tags = [normalize_highway(h) for h in edge_highway]
    names = (
        ["" if not isinstance(n, str) else n.strip() for n in edge_name]
        if edge_name is not None
        else [""] * len(tags)
    )
    node_tags = (
        [normalize_highway(t) for t in node_highway]
        if node_highway is not None
        else [""] * num_nodes
    )
    if len(node_tags) != num_nodes:
        raise ValueError(
            f"{len(node_tags)} node tags but {num_nodes} nodes"
        )

    # One pass over the edge list collects, per node: the classes that meet
    # there, the distinct arterial street names, and the distinct neighbours.
    incident_class: list[set[int]] = [set() for _ in range(num_nodes)]
    crossing_names: list[set[str]] = [set() for _ in range(num_nodes)]
    neighbours: list[set[int]] = [set() for _ in range(num_nodes)]

    for i in range(edge_u.shape[0]):
        u = int(edge_u[i])
        v = int(edge_v[i])
        tag = tags[i]
        klass = int(road_class(tag))
        # An unnamed arterial still counts as its own approach; keying by tag
        # keeps two unnamed ones from collapsing into a single street.
        label = names[i] or f"<unnamed {tag}>"
        for node, other in ((u, v), (v, u)):
            incident_class[node].add(klass)
            neighbours[node].add(other)
            if tag in _CROSSING_TAGS:
                crossing_names[node].add(label)

    priority = np.zeros(num_nodes, dtype=np.float32)
    yielding = np.zeros(num_nodes, dtype=np.float32)
    kinds: list[str] = []

    freeway_only = {int(RoadClass.FREEWAY)}
    for node in range(num_nodes):
        classes = incident_class[node]
        tag = node_tags[node]
        signalled = tag in _SIGNAL_NODE_TAGS

        if classes and classes <= freeway_only:
            kinds.append("freeway")
            continue
        if len(neighbours[node]) < 3 and not signalled:
            kinds.append("through")
            continue

        arterial_here = bool(classes & {int(RoadClass.FREEWAY), int(RoadClass.ARTERIAL)})
        if signalled or len(crossing_names[node]) >= 2:
            priority[node] = (
                SIGNAL_DELAY_SEC if arterial_here else MINOR_SIGNAL_DELAY_SEC
            )
            kinds.append("signal" if arterial_here else "minor_signal")
            continue

        yielding[node] = YIELD_DELAY_SEC
        kinds.append("minor")

    return IntersectionDelays(priority_sec=priority, yield_sec=yielding, kind=kinds)


# --------------------------------------------------------------------------
# Bearings
# --------------------------------------------------------------------------
# Ignore the first and last few metres of a polyline when reading its heading:
# the vertex right at an intersection often sits in the turn pocket, and a
# two-metre stub there points somewhere the road does not go.
BEARING_BASELINE_M = 20.0

_EARTH_RADIUS_M = 6_371_008.8


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing in degrees, clockwise from north."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlambda = np.radians(lon2 - lon1)
    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    return float(np.degrees(np.arctan2(y, x)) % 360.0)


def _chord_m(a: np.ndarray, b: np.ndarray) -> float:
    """Great-circle metres between two ``[lat, lon]`` points."""
    phi1, phi2 = np.radians(a[0]), np.radians(b[0])
    dphi = phi2 - phi1
    dlambda = np.radians(b[1] - a[1])
    h = (
        np.sin(dphi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    )
    return float(2 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(min(1.0, h))))


def polyline_bearings(
    geometry: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Heading at the start and end of each ``[k, 2]`` lat/lon polyline.

    Returned in degrees, one pair per edge, for ``Graph.set_edge_bearings``.
    Measured over :data:`BEARING_BASELINE_M` where the geometry is long enough,
    which matters on Irvine's curving arterials: the straight-chord heading of
    a 900 m arc through Turtle Rock can be 40 degrees off the direction the
    driver is actually pointing when they reach the junction.
    """
    entry = np.zeros(len(geometry), dtype=np.float32)
    exit_ = np.zeros(len(geometry), dtype=np.float32)

    for i, line in enumerate(geometry):
        points = np.asarray(line, dtype=np.float64)
        if points.shape[0] < 2:
            continue

        # Walk in from the start until the baseline is covered.
        head = points[-1]
        for j in range(1, points.shape[0]):
            head = points[j]
            if _chord_m(points[0], head) >= BEARING_BASELINE_M:
                break
        entry[i] = _bearing(points[0, 0], points[0, 1], head[0], head[1])

        # And back in from the end.
        tail = points[0]
        for j in range(points.shape[0] - 2, -1, -1):
            tail = points[j]
            if _chord_m(tail, points[-1]) >= BEARING_BASELINE_M:
                break
        exit_[i] = _bearing(tail[0], tail[1], points[-1, 0], points[-1, 1])

    return entry, exit_
