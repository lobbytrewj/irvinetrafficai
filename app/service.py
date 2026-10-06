"""The service object: native graphs, forecast application, route queries.

Why one graph per horizon
-------------------------
The forecaster emits three lead times (15, 30 and 60 minutes ahead) and a
request may carry a ``departure_time``. Holding one graph and re-weighting it
per request would put a 444-edge write in the middle of the query path and
serialise every caller behind it. Instead the background loop writes all three
horizons into three pre-built graphs on the same tick, and a request picks the
graph matching its departure time. Queries never mutate anything, so they cost
exactly one A* search.

A fourth graph holds the static free-flow weights and is never updated. That
is the baseline: the route a router with no traffic model would return.

Cost of the extra graphs is negligible -- the topology is ~15,553 edges of
about 20 bytes, well under a megabyte each -- and it buys a query path with no
locking against the writer beyond the engine's own per-graph mutex.

Comparing the two routes
------------------------
The baseline *path* is chosen under free-flow weights, but reporting its
free-flow duration next to the dynamic route's predicted duration would
compare two different worlds and make ``time_saved_sec`` meaningless (free
flow is always optimistic, so the "saving" would come out negative). Both
paths are therefore scored under the *same* predicted weights: what a driver
would actually experience on each. The baseline's free-flow number is still
reported separately, as ``free_flow_duration_sec``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.config import Settings
from app.schemas import RoutePreferences
from app.forecast import (
    FreeFlowFeed,
    Prediction,
    SegmentIndex,
    TrafficForecaster,
    load_segment_index,
)
from app.network import RoadNetwork, SnapResult

__all__ = ["RoutingService", "RouteLeg", "RouteComparison", "ServiceStatus"]

LOGGER = logging.getLogger(__name__)

# Congestion buckets for the heatmap, as a multiple of free-flow travel time.
CONGESTION_LEVELS: tuple[tuple[float, str], ...] = (
    (1.10, "free"),
    (1.35, "light"),
    (1.80, "moderate"),
    (2.50, "heavy"),
)


@dataclass(frozen=True)
class RouteLeg:
    """One computed route."""

    duration_sec: float
    distance_meters: float
    coordinates: list[list[float]]
    free_flow_duration_sec: float
    node_count: int
    nodes_expanded: int


@dataclass(frozen=True)
class RouteComparison:
    dynamic: RouteLeg
    baseline: RouteLeg
    # Baseline duration minus dynamic duration, both real time on the road.
    # Negative when preferences bought comfort at the cost of speed, which is
    # the driver getting what they asked for rather than an error.
    time_saved_sec: float
    # Bias the dynamic route still carries: flagged movements it could not
    # avoid. Never inside any duration.
    preference_penalty_sec: float
    origin: SnapResult
    destination: SnapResult
    horizon_minutes: int
    identical_path: bool
    # Wall time inside route(): snapping, both A* searches, re-scoring and
    # polyline assembly. Excludes the thread-pool hop and all HTTP framing,
    # so it is the number the "queries are sub-millisecond" claim refers to.
    compute_ms: float


@dataclass
class ServiceStatus:
    """Mutable health/observability state, written by the update loop."""

    started_at: datetime
    last_update: datetime | None = None
    last_inference_ms: float | None = None
    last_apply_ms: float | None = None
    update_count: int = 0
    failure_count: int = 0
    last_error: str | None = None
    feed_name: str = "unknown"
    degraded_reason: str | None = None
    # Per-horizon mean congestion ratio, cheap enough to keep for /health.
    mean_congestion: tuple[float, ...] = field(default_factory=tuple)


class RoutingService:
    """Everything the HTTP layer needs, with no HTTP in it."""

    def __init__(
        self,
        network: RoadNetwork,
        forecaster: TrafficForecaster,
        segments: SegmentIndex,
        settings: Settings,
        degraded_reason: str | None = None,
    ) -> None:
        self.network = network
        self.forecaster = forecaster
        self.segments = segments
        self.settings = settings

        segments.validate_against(network.num_edges)

        self.horizon_minutes = forecaster.horizon_minutes
        self.num_horizons = len(self.horizon_minutes)

        LOGGER.info("Building %d native graphs", self.num_horizons + 1)
        started = time.perf_counter()
        self.baseline_graph = network.build_engine_graph()
        self.dynamic_graphs = [
            network.build_engine_graph() for _ in range(self.num_horizons)
        ]
        LOGGER.info(
            "Graphs ready in %.2fs: %s",
            time.perf_counter() - started,
            self.baseline_graph,
        )

        # Python-side mirror of what each dynamic graph currently holds. Used
        # to re-score the baseline path under predicted conditions without a
        # second trip into the engine.
        self.travel_times = np.tile(
            network.edge_free_flow_s.astype(np.float32), (self.num_horizons, 1)
        )

        # Serialises writers against each other; the engine already guards
        # each graph's internals against a concurrent reader.
        self._write_lock = threading.Lock()
        self._segment_edge_ids = [int(e) for e in segments.edge_id]
        # Calibrated free flow on the corridor edges: the fastest any of them
        # can legitimately be predicted to run. See apply().
        self._segment_free_flow_floor = network.edge_free_flow_s[
            np.asarray(self._segment_edge_ids, dtype=np.int64)
        ].astype(np.float32)
        self.latest_prediction: Prediction | None = None
        self.status = ServiceStatus(
            started_at=datetime.now(timezone.utc),
            feed_name=forecaster.feed.name,
            degraded_reason=degraded_reason,
        )

    # -- construction --------------------------------------------------------
    @classmethod
    def create(cls, settings: Settings) -> "RoutingService":
        """Load every artifact and build the service. Blocking; ~seconds."""
        network = RoadNetwork.load(
            mapping_path=settings.node_mapping_path,
            tensors_path=settings.graph_tensors_path,
            geopackage_path=settings.geopackage_path,
        )
        LOGGER.info(
            "Road network: %d nodes, %d edges", network.num_nodes, network.num_edges
        )

        segments, feed, degraded = _load_feed(settings, network)
        forecaster = TrafficForecaster(
            onnx_path=settings.onnx_path, segments=segments, feed=feed
        )
        return cls(
            network=network,
            forecaster=forecaster,
            segments=segments,
            settings=settings,
            degraded_reason=degraded,
        )

    # -- forecast application ------------------------------------------------
    def tick(self, now: datetime | None = None) -> Prediction:
        """One forecast pass, pushed into every dynamic graph. Blocking."""
        prediction = self.forecaster.predict(now)
        apply_ms = self.apply(prediction)

        free_flow = self.segments.free_flow_s
        self.latest_prediction = prediction
        self.status.last_update = prediction.generated_at
        self.status.last_inference_ms = prediction.inference_ms
        self.status.last_apply_ms = apply_ms
        self.status.update_count += 1
        self.status.last_error = None
        self.status.mean_congestion = tuple(
            float(np.mean(prediction.travel_time_s[h] / free_flow))
            for h in range(prediction.travel_time_s.shape[0])
        )
        return prediction

    def apply(self, prediction: Prediction) -> float:
        """Write predicted travel times into the dynamic graphs. Returns ms."""
        started = time.perf_counter()
        edge_ids = self._segment_edge_ids
        segment_edge_index = self.segments.edge_id
        floor = self._segment_free_flow_floor

        with self._write_lock:
            for horizon, graph in enumerate(self.dynamic_graphs):
                # The model was trained against posted-limit free-flow times
                # and clamps its predictions there. Calibrated free flow is
                # slower than that on every arterial, so a light prediction can
                # come out *faster* than the road can physically be driven --
                # which would make the corridor look like a shortcut. Floor it.
                row = np.maximum(prediction.travel_time_s[horizon], floor)
                # O(k) over the 444 corridor segments; the other 15,109 edges
                # keep their free-flow weights and are never touched.
                graph.update_edge_weights_subset(edge_ids, [float(v) for v in row])
                self.travel_times[horizon, segment_edge_index] = row
        return (time.perf_counter() - started) * 1000.0

    # -- routing -------------------------------------------------------------
    def horizon_for(self, departure_time: datetime | None, now: datetime) -> int:
        """Index of the horizon closest to a requested departure.

        No departure time, or one already past, means "leaving now" and takes
        the nearest horizon.
        """
        if departure_time is None:
            return 0
        lead_minutes = (departure_time - now).total_seconds() / 60.0
        if lead_minutes <= 0:
            return 0
        return int(
            np.argmin([abs(lead_minutes - m) for m in self.horizon_minutes])
        )

    def route(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        departure_time: datetime | None = None,
        now: datetime | None = None,
        preferences: RoutePreferences | None = None,
    ) -> RouteComparison:
        """Dynamic and baseline routes between two coordinates.

        Blocking, and the two searches are the only expensive part. Raises
        ``LookupError`` when either endpoint is off-network or no path exists.

        Preferences go to *both* searches. The comparison exists to isolate
        what the traffic model is worth, so the only difference between the
        two sides has to stay the edge weights: penalising a left turn on one
        side and not the other would fold two unrelated changes into one
        number.
        """
        compute_started = time.perf_counter()
        now = now or datetime.now(timezone.utc)
        origin = self.network.snap(origin_lat, origin_lon)
        destination = self.network.snap(dest_lat, dest_lon)

        limit = self.settings.max_snap_distance_m
        for label, snapped in (("origin", origin), ("destination", destination)):
            if snapped.distance_m > limit:
                raise LookupError(
                    f"{label} is {snapped.distance_m:,.0f} m from the nearest "
                    f"Irvine intersection (limit {limit:,.0f} m)"
                )

        horizon = self.horizon_for(departure_time, now)
        native_preferences = _native_preferences(preferences)
        dynamic_graph = self.dynamic_graphs[horizon]
        dynamic_result = dynamic_graph.find_shortest_path(
            origin.node_id, destination.node_id, native_preferences
        )
        baseline_result = self.baseline_graph.find_shortest_path(
            origin.node_id, destination.node_id, native_preferences
        )

        if not dynamic_result.found or not baseline_result.found:
            raise LookupError(
                f"no route between node {origin.node_id} and node "
                f"{destination.node_id}; the graph is not strongly connected here"
            )

        # Durations are scored without preferences throughout: a penalty is
        # what the driver would pay to skip a movement, not time they spend on
        # the road, so it must never reach an ETA.
        dynamic = self._leg(dynamic_result, predicted_duration=None)
        baseline = self._leg(
            baseline_result,
            # A* on the baseline graph optimised free-flow time; what the
            # driver would actually experience is that path under prediction.
            # Scored by the dynamic graph itself rather than by summing edge
            # times in Python, so the baseline path is charged the same signal
            # delays and turn penalties the dynamic path was.
            predicted_duration=dynamic_graph.score_path(
                list(baseline_result.path_edge_ids)
            ),
        )

        return RouteComparison(
            dynamic=dynamic,
            baseline=baseline,
            time_saved_sec=baseline.duration_sec - dynamic.duration_sec,
            preference_penalty_sec=round(
                float(dynamic_result.preference_penalty_sec), 2
            ),
            origin=origin,
            destination=destination,
            horizon_minutes=self.horizon_minutes[horizon],
            identical_path=(
                list(dynamic_result.path_edge_ids) == list(baseline_result.path_edge_ids)
            ),
            compute_ms=(time.perf_counter() - compute_started) * 1000.0,
        )

    def _leg(self, result, predicted_duration: float | None) -> RouteLeg:
        edge_ids = list(result.path_edge_ids)
        # "If the road were empty" still means stopping at the lights: the
        # baseline graph holds free-flow edge weights and the same intersection
        # delays, so scoring there isolates congestion from control.
        free_flow = float(self.baseline_graph.score_path(edge_ids))
        duration = (
            result.total_duration_sec
            if predicted_duration is None
            else predicted_duration
        )
        return RouteLeg(
            duration_sec=round(float(duration), 2),
            distance_meters=round(float(result.total_distance_meters), 2),
            coordinates=self.network.polyline(edge_ids)
            or self.network.node_polyline(list(result.path_node_ids)),
            free_flow_duration_sec=round(free_flow, 2),
            node_count=len(result.path_node_ids),
            nodes_expanded=int(result.nodes_expanded),
        )

    # -- traffic state -------------------------------------------------------
    def traffic_state(
        self, horizon: int = 0, include_geometry: bool = True
    ) -> dict:
        """Predicted congestion on every corridor segment, for the heatmap."""
        if not 0 <= horizon < self.num_horizons:
            raise ValueError(
                f"horizon {horizon} out of range for {self.num_horizons} horizons"
            )

        segments = self.segments
        free_flow = segments.free_flow_s
        prediction = self.latest_prediction

        predicted = (
            prediction.travel_time_s[horizon]
            if prediction is not None
            else free_flow.copy()
        )
        ratio = predicted / np.maximum(free_flow, 1e-6)
        speed_kph = segments.length_m / np.maximum(predicted, 1e-6) * 3.6

        geometry = self.network.edge_geometry
        records = []
        for i, edge_id in enumerate(self._segment_edge_ids):
            record = {
                "segment_id": i,
                "edge_id": edge_id,
                "corridor": segments.corridor_of(i),
                "name": self.network.edge_label[edge_id],
                "length_meters": round(float(segments.length_m[i]), 2),
                "free_flow_sec": round(float(free_flow[i]), 2),
                "predicted_sec": round(float(predicted[i]), 2),
                "congestion_ratio": round(float(ratio[i]), 4),
                "speed_kph": round(float(speed_kph[i]), 2),
                "level": _congestion_level(float(ratio[i])),
            }
            if include_geometry:
                record["coordinates"] = geometry[edge_id].round(6).tolist()
            records.append(record)

        return {
            "generated_at": (
                prediction.generated_at.isoformat() if prediction else None
            ),
            "horizon_minutes": self.horizon_minutes[horizon],
            "available_horizon_minutes": list(self.horizon_minutes),
            "segment_count": len(records),
            "mean_congestion_ratio": round(float(ratio.mean()), 4),
            "max_congestion_ratio": round(float(ratio.max()), 4),
            "stale": prediction is None,
            "segments": records,
        }


def _native_preferences(preferences: RoutePreferences | None):
    """The API model as the engine's own struct.

    Built per request rather than cached: it is four fields, and holding one
    on the service would make a per-query choice look like server state.
    """
    import traffic_engine

    if preferences is None:
        return traffic_engine.RoutePreferences()
    return traffic_engine.RoutePreferences(
        avoid_unprotected_lefts=preferences.avoid_unprotected_lefts,
        low_stress_merges=preferences.low_stress_merges,
        unprotected_left_penalty_sec=preferences.unprotected_left_penalty_sec,
        aggressive_merge_penalty_sec=preferences.aggressive_merge_penalty_sec,
    )


def _congestion_level(ratio: float) -> str:
    for threshold, label in CONGESTION_LEVELS:
        if ratio < threshold:
            return label
    return "severe"


def _load_feed(settings: Settings, network: RoadNetwork):
    """Segment metadata plus an observation feed, degrading if need be."""
    dataset_path: Path = settings.dataset_path
    if dataset_path.exists():
        segments, feed = load_segment_index(dataset_path)
        LOGGER.info(
            "Replay feed over %d steps x %d segments from %s",
            feed.num_steps,
            feed.num_segments,
            dataset_path.name,
        )
        return segments, feed, None

    if settings.strict_artifacts:
        raise FileNotFoundError(
            f"{dataset_path} not found. It carries the 444-segment metadata and "
            "the observation history the forecaster needs. Run "
            "`python -m src.data.generate_traffic_data`, or set "
            "IRVINE_STRICT_ARTIFACTS=0 to serve free-flow only."
        )

    # No history: re-derive the segment table from the GeoPackage and feed the
    # model an empty road. Every endpoint answers; predictions are inert.
    from src.data.generate_traffic_data import select_corridor_segments

    table = select_corridor_segments(settings.processed_dir)
    free_flow_s = (table.length_m / (table.free_flow_kph / 3.6)).astype(np.float32)
    segments = SegmentIndex(
        edge_id=table.edge_id.astype(np.int32),
        length_m=table.length_m.astype(np.float64),
        free_flow_s=free_flow_s,
        corridor_id=table.corridor_id,
        corridor_names=tuple(spec.key for spec in table.corridors),
    )
    reason = f"{dataset_path.name} missing; serving free-flow observations"
    LOGGER.warning("%s", reason)
    return segments, FreeFlowFeed(free_flow_s, segments.length_m), reason
