"""Request and response models for the public API.

Pydantic is doing real work here rather than decoration: a routing request is
four floats from a browser, and rejecting a longitude that landed in the
latitude field at the edge of the service is much cheaper than snapping it to
whichever Irvine intersection happens to be closest to a nonsense coordinate.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "GeocodeResult",
    "RoutePreferences",
    "MapConfig",
    "RouteRequest",
    "RouteLegModel",
    "RouteResponse",
    "SegmentState",
    "TrafficStateResponse",
    "HealthResponse",
]


class RoutePreferences(BaseModel):
    """What the driver will trade time for.

    Mirrors ``irvine::RoutePreferences`` in the native engine field for field.
    Both flags default off, so a request that says nothing about preferences
    gets the fastest route -- the behaviour every existing client already
    depends on.

    The penalties are exposed because they are the honest way to express
    "how much is this worth to you": at 45 s a driver takes a detour that
    costs less than that and makes the turn otherwise. They are bounded rather
    than free-form, since a penalty of an hour is not a preference, it is a
    prohibition, and a client wanting that should say so a different way.
    """

    model_config = ConfigDict(extra="forbid")

    avoid_unprotected_lefts: bool = Field(
        False,
        description=(
            "Penalise left turns across oncoming traffic at junctions with no "
            "signal holding it."
        ),
    )
    low_stress_merges: bool = Field(
        False,
        description="Penalise freeway ramps joining or leaving fast traffic.",
    )
    unprotected_left_penalty_sec: float = Field(
        45.0,
        ge=0.0,
        le=600.0,
        description="Seconds of detour an unprotected left is worth avoiding.",
    )
    aggressive_merge_penalty_sec: float = Field(
        60.0,
        ge=0.0,
        le=600.0,
        description="Seconds of detour an aggressive merge is worth avoiding.",
    )

    def any(self) -> bool:
        """True when at least one avoidance is switched on."""
        return self.avoid_unprotected_lefts or self.low_stress_merges


class RouteRequest(BaseModel):
    """A door-to-door query."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "origin_lat": 33.6846,
                    "origin_lon": -117.8265,
                    "dest_lat": 33.6595,
                    "dest_lon": -117.7467,
                    "departure_time": "2026-08-11T17:30:00Z",
                    "preferences": {"avoid_unprotected_lefts": True},
                }
            ]
        },
    )

    origin_lat: float = Field(..., ge=-90.0, le=90.0)
    origin_lon: float = Field(..., ge=-180.0, le=180.0)
    dest_lat: float = Field(..., ge=-90.0, le=90.0)
    dest_lon: float = Field(..., ge=-180.0, le=180.0)
    # ISO 8601. Naive values are read as UTC. Selects the forecast horizon;
    # absent or already past means "leaving now".
    departure_time: datetime | None = None
    # Absent means "all off", which is the fastest route.
    preferences: RoutePreferences = Field(default_factory=RoutePreferences)


class RouteLegModel(BaseModel):
    """One route, as returned to the client."""

    duration_sec: float = Field(
        ..., description="Travel time under the predicted conditions, in seconds."
    )
    distance_meters: float
    coordinates: list[list[float]] = Field(
        ..., description="Road centreline as [[lat, lon], ...]."
    )
    free_flow_duration_sec: float = Field(
        ...,
        description=(
            "The same path at posted speed limits. Reported so the client can "
            "see how much of the duration is congestion."
        ),
    )
    node_count: int
    nodes_expanded: int = Field(
        ..., description="Nodes popped from the A* open set; search effort."
    )


class SnappedPoint(BaseModel):
    node_id: int
    latitude: float
    longitude: float
    distance_m: float


class RouteResponse(BaseModel):
    dynamic_route: RouteLegModel
    baseline_route: RouteLegModel
    time_saved_sec: float = Field(
        ...,
        description=(
            "Baseline path minus dynamic path, both scored under the same "
            "predicted travel times. Zero when the two paths agree, and "
            "negative when routing preferences made the chosen route slower "
            "on purpose -- an avoided left turn is a longer drive."
        ),
    )
    preference_penalty_sec: float = Field(
        0.0,
        description=(
            "Search bias the returned route still carries: the cost of the "
            "flagged movements it could not avoid. Zero when nothing was "
            "requested, or when everything requested was successfully "
            "avoided. Never part of any duration."
        ),
    )
    query_latency_ms: float = Field(
        ...,
        description=(
            "Server-side handler time: the compute below plus the thread-pool "
            "hop. Excludes HTTP framing and the network."
        ),
    )
    compute_latency_ms: float = Field(
        ...,
        description=(
            "Time inside the routing service alone -- snapping, both A* "
            "searches, re-scoring, polyline assembly. Sub-millisecond, because "
            "the graph was already updated by the background forecaster."
        ),
    )

    # -- context -------------------------------------------------------------
    origin: SnappedPoint
    destination: SnappedPoint
    horizon_minutes: int = Field(
        ..., description="Forecast lead time the dynamic route was scored at."
    )
    identical_path: bool
    prediction_generated_at: datetime | None


class SegmentState(BaseModel):
    segment_id: int
    edge_id: int
    corridor: str
    name: str
    length_meters: float
    free_flow_sec: float
    predicted_sec: float
    congestion_ratio: float
    speed_kph: float
    level: str
    coordinates: list[list[float]] | None = None


class TrafficStateResponse(BaseModel):
    generated_at: datetime | None
    horizon_minutes: int
    available_horizon_minutes: list[int]
    segment_count: int
    mean_congestion_ratio: float
    max_congestion_ratio: float
    stale: bool = Field(
        ..., description="True before the first forecast tick has completed."
    )
    segments: list[SegmentState]


class GeocodeResult(BaseModel):
    """One place-name match, ready to drop on the map.

    Nominatim hands back latitude and longitude as strings; they are parsed
    here so the client never has to, and so a malformed row is dropped at the
    boundary rather than becoming ``NaN`` inside a route request.
    """

    display_name: str = Field(
        ..., description="Full human-readable label, e.g. 'University of California, "
        "Irvine, Irvine, Orange County, California'."
    )
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)


class MapConfig(BaseModel):
    """Basemap settings handed to the dashboard at boot."""

    mapbox_token: str = Field(
        ...,
        description=(
            "Publishable Mapbox token, or empty. Empty means the client falls "
            "back to a keyless basemap: Mapbox GL JS will not build a map "
            "without a token, whatever the style."
        ),
    )
    map_style: str
    center: list[float] = Field(..., description="[longitude, latitude], GL order.")
    zoom: float
    update_interval_sec: float = Field(
        ..., description="Forecast cadence, so the client can pace its polling."
    )


class HealthResponse(BaseModel):
    status: str
    engine_version: str
    num_nodes: int
    num_edges: int
    num_segments: int
    horizon_minutes: list[int]
    feed: str
    uptime_sec: float
    update_count: int
    failure_count: int
    last_update: datetime | None
    seconds_since_update: float | None
    last_inference_ms: float | None
    last_apply_ms: float | None
    mean_congestion: list[float]
    last_error: str | None
    degraded_reason: str | None
