"""Integration tests for the FastAPI routing service.

These run the real thing: the real GeoPackage, the real ONNX graph, the real
native engine. Startup is expensive (~20 s, most of it the 92 MB dataset
bundle) so the client is session-scoped and every test shares one warm
service.

What is worth pinning here is the machinery that fails *quietly*:

* **Edge-id alignment.** ``graph_tensors.pt`` sorts edges by ``(u, v, key)``
  and the GeoPackage does not. If the service ever adopted the wrong
  convention, every prediction would land on a real road -- just the wrong
  one -- and every response would stay perfectly well-formed. Asserted by
  reading travel times back out of the native graph.

* **Route comparability.** ``time_saved_sec`` is only meaningful if both
  paths are scored under the same weights. Since A* on the dynamic graph
  minimises predicted travel time, the dynamic path can never be slower than
  the baseline path under those weights, so a negative saving is proof the
  two numbers came from different worlds.

* **Loop survival.** A background task that dies on one bad tick leaves a
  service that answers every request with stale weights and a 200.

* **Geocoding's bounding box.** ``/api/v1/geocode`` is the one endpoint that
  leaves the process, and the one whose correctness is invisible in its own
  response: an unbounded search returns a perfectly well-formed hit in
  Sacramento that the router then rejects. The outbound parameters are
  asserted directly, against a stubbed transport.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np
import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("httpx", reason="httpx not installed; needed by TestClient")
pytest.importorskip(
    "traffic_engine", reason="native extension not built; run `pip install -e .`"
)
pytest.importorskip("sklearn", reason="scikit-learn not installed")

from fastapi.testclient import TestClient  # noqa: E402

from app import main as app_main  # noqa: E402
from app.config import Settings  # noqa: E402
from app.main import create_app, update_traffic_predictions_loop  # noqa: E402

SETTINGS = Settings()

# Real Irvine coordinates. Origin is near the Irvine Business Complex, the
# destination is out by the Great Park -- far enough apart that the corridor
# network is genuinely in play.
ORIGIN = {"origin_lat": 33.6846, "origin_lon": -117.8265}
DESTINATION = {"dest_lat": 33.6595, "dest_lon": -117.7467}
TRIP = {**ORIGIN, **DESTINATION}

EXPECTED_NODES = 6864
EXPECTED_EDGES = 15553
EXPECTED_SEGMENTS = 444


def _artifacts_present() -> bool:
    return all(
        path.exists()
        for path in (
            SETTINGS.onnx_path,
            SETTINGS.geopackage_path,
            SETTINGS.node_mapping_path,
            SETTINGS.graph_tensors_path,
            SETTINGS.dataset_path,
        )
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def client():
    """A started app, with the first forecast tick already applied.

    The artifact check lives here rather than in a module-level ``pytestmark``
    so that the endpoints which need nothing from the router -- geocoding, and
    the 503 guard -- still run on a checkout without the processed data.
    """
    if not _artifacts_present():
        pytest.skip("processed artifacts missing; run the data and training pipelines")

    with TestClient(create_app(SETTINGS)) as test_client:
        # The loop ticks immediately on startup, but the task is scheduled
        # after the lifespan yields, so a fast first request can beat it.
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if test_client.get("/health").json()["update_count"] >= 1:
                break
            time.sleep(0.05)
        else:
            pytest.fail("no forecast tick completed within 60 s of startup")
        yield test_client


@pytest.fixture(scope="session")
def service(client):
    return client.app.state.service


@pytest.fixture(scope="session")
def route_payload(client) -> dict:
    response = client.post("/api/v1/route", json=TRIP)
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
def test_health_reports_the_full_network(client):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["num_nodes"] == EXPECTED_NODES
    assert body["num_edges"] == EXPECTED_EDGES
    assert body["num_segments"] == EXPECTED_SEGMENTS
    assert body["horizon_minutes"] == [15, 30, 60]
    assert body["failure_count"] == 0
    assert body["last_error"] is None
    assert body["degraded_reason"] is None
    assert body["engine_version"]


def test_health_shows_a_fresh_forecast(client):
    body = client.get("/health").json()

    assert body["update_count"] >= 1
    assert body["last_update"] is not None
    assert body["seconds_since_update"] < SETTINGS.update_interval_sec
    assert body["last_inference_ms"] > 0
    # Every horizon should be at or above free flow: the export clamps there.
    assert len(body["mean_congestion"]) == 3
    assert all(ratio >= 1.0 for ratio in body["mean_congestion"])


# ---------------------------------------------------------------------------
# POST /api/v1/route
# ---------------------------------------------------------------------------
def test_route_returns_the_documented_payload(route_payload):
    for key in (
        "dynamic_route",
        "baseline_route",
        "time_saved_sec",
        "query_latency_ms",
    ):
        assert key in route_payload

    for leg in (route_payload["dynamic_route"], route_payload["baseline_route"]):
        assert leg["duration_sec"] > 0
        assert leg["distance_meters"] > 0
        assert len(leg["coordinates"]) >= 2
        assert all(len(point) == 2 for point in leg["coordinates"])


def test_route_coordinates_trace_the_requested_trip(route_payload):
    """The polyline must start at the snapped origin and end at the destination."""
    origin = route_payload["origin"]
    destination = route_payload["destination"]

    for leg in (route_payload["dynamic_route"], route_payload["baseline_route"]):
        first_lat, first_lon = leg["coordinates"][0]
        last_lat, last_lon = leg["coordinates"][-1]
        assert first_lat == pytest.approx(origin["latitude"], abs=1e-4)
        assert first_lon == pytest.approx(origin["longitude"], abs=1e-4)
        assert last_lat == pytest.approx(destination["latitude"], abs=1e-4)
        assert last_lon == pytest.approx(destination["longitude"], abs=1e-4)

    # And the whole path should stay inside a box around Irvine.
    for lat, lon in route_payload["dynamic_route"]["coordinates"]:
        assert 33.5 < lat < 33.85
        assert -118.0 < lon < -117.6


def test_snapping_reports_a_plausible_distance(route_payload):
    for point in (route_payload["origin"], route_payload["destination"]):
        assert 0 <= point["distance_m"] <= SETTINGS.max_snap_distance_m
        assert 0 <= point["node_id"] < EXPECTED_NODES


def test_time_saved_is_a_like_for_like_comparison(route_payload):
    """Both paths are scored under the predicted weights, so saving >= 0.

    A* on the dynamic graph returns the minimum-predicted-time path, so no
    other path -- the baseline's included -- can beat it under those weights.
    A negative saving means the two durations came from different weight sets.
    """
    dynamic = route_payload["dynamic_route"]
    baseline = route_payload["baseline_route"]

    assert route_payload["time_saved_sec"] == pytest.approx(
        baseline["duration_sec"] - dynamic["duration_sec"], abs=0.02
    )
    assert route_payload["time_saved_sec"] >= -0.01

    if route_payload["identical_path"]:
        assert route_payload["time_saved_sec"] == pytest.approx(0.0, abs=0.01)


def test_baseline_free_flow_is_faster_than_its_predicted_duration(route_payload):
    """Congestion can only slow a path down: the model clamps at free flow."""
    baseline = route_payload["baseline_route"]
    assert baseline["free_flow_duration_sec"] <= baseline["duration_sec"] + 0.01

    # And the baseline path is the fastest one at free flow, so the dynamic
    # detour must cost more when both are scored that way.
    dynamic = route_payload["dynamic_route"]
    assert baseline["free_flow_duration_sec"] <= dynamic["free_flow_duration_sec"] + 0.01


def test_query_path_stays_an_order_below_inference(service):
    """The architectural claim: a request hits an already-current graph.

    Measured at the service boundary -- snapping, both A* searches, path
    re-scoring and polyline assembly, but no HTTP and no thread-pool hop.
    Inference is ~7 ms, so anything in this range proves the forecast is not
    running on the request path.

    The bound was 1 ms before the intersection model landed. Searching over
    directed edges rather than nodes is what turn penalties require -- a turn
    is a property of an edge pair -- and it costs roughly 3x: the state space
    is E rather than N, and each relaxation now evaluates a transition. That
    is a millisecond spent to stop reporting a 14-minute drive as a 6-minute
    one, and it leaves the gap to inference intact.
    """
    timings = []
    for _ in range(50):
        started = time.perf_counter()
        service.route(**TRIP)
        timings.append((time.perf_counter() - started) * 1000.0)

    assert float(np.median(timings)) < 3.0


def test_endpoint_latency_stays_in_the_low_milliseconds(client):
    """End to end, the thread-pool hop roughly triples the native cost."""
    latencies = []
    for _ in range(20):
        response = client.post("/api/v1/route", json=TRIP)
        assert response.status_code == 200
        latencies.append(response.json()["query_latency_ms"])

    assert float(np.median(latencies)) < 5.0


def test_departure_time_selects_a_forecast_horizon(client):
    now = datetime.now(timezone.utc)
    cases = {2: 15, 28: 30, 55: 60}

    for lead_minutes, expected_horizon in cases.items():
        payload = {
            **TRIP,
            "departure_time": (now + timedelta(minutes=lead_minutes)).isoformat(),
        }
        response = client.post("/api/v1/route", json=payload)
        assert response.status_code == 200, response.text
        assert response.json()["horizon_minutes"] == expected_horizon


def test_departure_time_in_the_past_means_now(client):
    payload = {
        **TRIP,
        "departure_time": (
            datetime.now(timezone.utc) - timedelta(hours=3)
        ).isoformat(),
    }
    response = client.post("/api/v1/route", json=payload)
    assert response.status_code == 200
    assert response.json()["horizon_minutes"] == 15


def test_route_to_self_is_empty_not_an_error(client):
    payload = {
        "origin_lat": 33.6846,
        "origin_lon": -117.8265,
        "dest_lat": 33.6846,
        "dest_lon": -117.8265,
    }
    response = client.post("/api/v1/route", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["dynamic_route"]["duration_sec"] == pytest.approx(0.0, abs=1e-6)
    assert body["dynamic_route"]["distance_meters"] == pytest.approx(0.0, abs=1e-6)
    assert body["time_saved_sec"] == pytest.approx(0.0, abs=1e-6)


def test_off_network_coordinates_are_rejected(client):
    """Times Square is not in Irvine; snapping it silently would be worse."""
    payload = {
        "origin_lat": 40.7580,
        "origin_lon": -73.9855,
        "dest_lat": 33.6595,
        "dest_lon": -117.7467,
    }
    response = client.post("/api/v1/route", json=payload)

    assert response.status_code == 404
    assert "origin" in response.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({**TRIP, "origin_lat": 200.0}, id="latitude-out-of-range"),
        pytest.param({**TRIP, "dest_lon": -400.0}, id="longitude-out-of-range"),
        pytest.param({"origin_lat": 33.68, "origin_lon": -117.82}, id="missing-fields"),
        pytest.param(
            {**TRIP, "departure_time": "next tuesday"}, id="unparseable-departure"
        ),
        pytest.param({**TRIP, "origin_latitude": 33.68}, id="unknown-field"),
    ],
)
def test_malformed_requests_are_rejected(client, payload):
    assert client.post("/api/v1/route", json=payload).status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/traffic-state
# ---------------------------------------------------------------------------
def test_traffic_state_covers_every_corridor_segment(client):
    body = client.get("/api/v1/traffic-state").json()

    assert body["segment_count"] == EXPECTED_SEGMENTS
    assert len(body["segments"]) == EXPECTED_SEGMENTS
    assert body["stale"] is False
    assert body["generated_at"] is not None
    assert body["horizon_minutes"] == 15
    assert body["available_horizon_minutes"] == [15, 30, 60]

    ids = [segment["segment_id"] for segment in body["segments"]]
    assert ids == list(range(EXPECTED_SEGMENTS))


def test_traffic_state_values_are_physically_sane(client):
    body = client.get("/api/v1/traffic-state").json()

    corridors = {segment["corridor"] for segment in body["segments"]}
    assert corridors <= {"I-405", "I-5", "Jamboree Rd", "Culver Dr", "Alton Pkwy"}

    for segment in body["segments"]:
        assert segment["predicted_sec"] > 0
        assert segment["free_flow_sec"] > 0
        # The ONNX export clamps predictions at free flow.
        assert segment["congestion_ratio"] >= 1.0 - 1e-6
        assert 0 < segment["speed_kph"] <= 130
        assert segment["level"] in {"free", "light", "moderate", "heavy", "severe"}
        assert 0 <= segment["edge_id"] < EXPECTED_EDGES
        assert len(segment["coordinates"]) >= 2


def test_traffic_state_congestion_grows_with_the_horizon(client):
    """Uncertainty and peak build both push the longer horizons higher."""
    means = [
        client.get(f"/api/v1/traffic-state?horizon={h}&include_geometry=false").json()[
            "mean_congestion_ratio"
        ]
        for h in range(3)
    ]
    assert all(mean >= 1.0 for mean in means)
    assert len(set(means)) > 1, "every horizon returned an identical forecast"


def test_traffic_state_geometry_is_optional(client):
    body = client.get("/api/v1/traffic-state?include_geometry=false").json()
    assert all(segment["coordinates"] is None for segment in body["segments"])


def test_traffic_state_rejects_an_unknown_horizon(client):
    assert client.get("/api/v1/traffic-state?horizon=9").status_code == 422
    assert client.get("/api/v1/traffic-state?horizon=-1").status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/geocode
# ---------------------------------------------------------------------------
# Two real rows, in Nominatim's own shape: a dozen fields, coordinates as
# strings, display_name as a comma-separated hierarchy.
UCI_ROW = {
    "place_id": 297_016_154,
    "osm_type": "relation",
    "osm_id": 111_968,
    "lat": "33.6459798",
    "lon": "-117.8427265",
    "category": "amenity",
    "type": "university",
    "importance": 0.5423,
    "display_name": (
        "University of California, Irvine, Irvine, Orange County, "
        "California, 92697, United States"
    ),
    "boundingbox": ["33.6335", "33.6607", "-117.8617", "-117.8168"],
}
SPECTRUM_ROW = {
    "place_id": 123_456_789,
    "lat": "33.6503",
    "lon": "-117.7460",
    "display_name": (
        "Irvine Spectrum Center, 670, Spectrum Center Drive, Irvine, "
        "Orange County, California, 92618, United States"
    ),
}


class FakeNominatim:
    """A stand-in for ``httpx.AsyncClient`` that records the outbound call.

    Real ``httpx.Response`` objects come back out of it, so ``raise_for_status``
    and ``.json()`` behave exactly as they do in production -- including the
    ``JSONDecodeError`` a non-JSON body raises, which is the case the handler
    has to turn into a 502 rather than a 500.
    """

    def __init__(self, *, response=None, error=None):
        self.response = response
        self.error = error
        self.init_kwargs = None
        self.calls = []

    def install(self, monkeypatch):
        stub = self

        class _Client:
            def __init__(self, **kwargs):
                stub.init_kwargs = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc_info):
                return False

            async def get(self, url, params=None):
                stub.calls.append({"url": url, "params": dict(params or {})})
                if stub.error is not None:
                    raise stub.error
                return stub.response

        monkeypatch.setattr(app_main.httpx, "AsyncClient", _Client)
        return self

    @property
    def params(self) -> dict:
        assert self.calls, "no request was made to Nominatim"
        return self.calls[-1]["params"]


def _nominatim_response(payload, status_code: int = 200) -> httpx.Response:
    request = httpx.Request("GET", app_main.NOMINATIM_URL)
    if isinstance(payload, str):
        return httpx.Response(status_code, text=payload, request=request)
    return httpx.Response(status_code, json=payload, request=request)


@pytest.fixture(scope="module")
def geocode_client():
    """A client that never enters the lifespan.

    ``/api/v1/geocode`` does not depend on ``get_service``, so these tests skip
    the ~20 s artifact load entirely and run anywhere.
    """
    return TestClient(create_app(SETTINGS))


def test_geocode_returns_display_name_lat_lon(geocode_client, monkeypatch):
    """The three fields the dropdown binds to, and coordinates as numbers."""
    FakeNominatim(
        response=_nominatim_response([UCI_ROW, SPECTRUM_ROW])
    ).install(monkeypatch)

    response = geocode_client.get("/api/v1/geocode", params={"q": "UC Irvine"})
    assert response.status_code == 200, response.text
    body = response.json()

    assert isinstance(body, list)
    assert len(body) == 2
    for match in body:
        assert set(match) == {"display_name", "lat", "lon"}
        assert isinstance(match["display_name"], str) and match["display_name"]
        # Parsed out of Nominatim's strings, not passed through as strings.
        assert isinstance(match["lat"], float)
        assert isinstance(match["lon"], float)

    assert body[0]["display_name"].startswith("University of California, Irvine")
    assert body[0]["lat"] == pytest.approx(33.6459798)
    assert body[0]["lon"] == pytest.approx(-117.8427265)


def test_geocode_bounds_the_search_to_greater_irvine(geocode_client, monkeypatch):
    """``viewbox`` without ``bounded=1`` only re-ranks -- it does not filter.

    This is the assertion that keeps "Main Street" from resolving to Los
    Angeles, and it is invisible in the response body: an out-of-area hit comes
    back just as well-formed as a local one.
    """
    stub = FakeNominatim(response=_nominatim_response([UCI_ROW])).install(monkeypatch)

    assert geocode_client.get(
        "/api/v1/geocode", params={"q": "Main Street"}
    ).status_code == 200

    assert stub.calls[-1]["url"] == "https://nominatim.openstreetmap.org/search"
    assert stub.params["viewbox"] == "-117.92,33.58,-117.68,33.78"
    assert stub.params["bounded"] == "1"
    assert stub.params["q"] == "Main Street"
    assert stub.params["format"] == "jsonv2"
    assert int(stub.params["limit"]) > 0


def test_geocode_viewbox_contains_the_routable_network():
    """The box has to cover what the router can actually serve.

    Read straight off the constant so a future edit that shrinks it past the
    test trip -- or flips the corner order -- fails here rather than as an
    unexplained 404 from ``/api/v1/route``.
    """
    # Nominatim's viewbox is two opposite corners: x1,y1,x2,y2.
    left, bottom, right, top = (float(v) for v in app_main.IRVINE_VIEWBOX.split(","))
    assert left < right
    assert bottom < top

    for latitude, longitude in (
        (ORIGIN["origin_lat"], ORIGIN["origin_lon"]),
        (DESTINATION["dest_lat"], DESTINATION["dest_lon"]),
        (33.6459798, -117.8427265),   # UC Irvine
    ):
        assert left <= longitude <= right
        assert bottom <= latitude <= top


def test_geocode_identifies_itself_and_bounds_its_wait(geocode_client, monkeypatch):
    """Nominatim's policy demands a real User-Agent; the timeout is ours."""
    stub = FakeNominatim(response=_nominatim_response([])).install(monkeypatch)

    geocode_client.get("/api/v1/geocode", params={"q": "Jamboree"})

    assert stub.init_kwargs["headers"]["User-Agent"] == "IrvineTrafficAI/1.0"
    assert stub.init_kwargs["timeout"] == 3.0


def test_geocode_returns_an_empty_list_when_nothing_matches(
    geocode_client, monkeypatch
):
    """No match is not an error -- the dropdown says so and the page carries on."""
    FakeNominatim(response=_nominatim_response([])).install(monkeypatch)

    response = geocode_client.get("/api/v1/geocode", params={"q": "zzzzqqq"})
    assert response.status_code == 200
    assert response.json() == []


def test_geocode_drops_unusable_rows_instead_of_failing(geocode_client, monkeypatch):
    """One malformed row must not cost the user the other suggestions."""
    rows = [
        UCI_ROW,
        {"display_name": "No coordinates here"},
        {"lat": "33.65", "lon": "-117.75"},               # no label
        {"display_name": "Bad numbers", "lat": "n/a", "lon": "-117.7"},
        {"display_name": "Off the planet", "lat": "999", "lon": "-117.7"},
        "not even a row",
        SPECTRUM_ROW,
    ]
    FakeNominatim(response=_nominatim_response(rows)).install(monkeypatch)

    body = geocode_client.get("/api/v1/geocode", params={"q": "irvine"}).json()

    assert [match["display_name"].split(",")[0] for match in body] == [
        "University of California",
        "Irvine Spectrum Center",
    ]


def test_geocode_handles_an_unexpected_payload_shape(geocode_client, monkeypatch):
    """Nominatim answers errors with an object, not a list."""
    FakeNominatim(
        response=_nominatim_response({"error": "Unable to geocode"})
    ).install(monkeypatch)

    response = geocode_client.get("/api/v1/geocode", params={"q": "irvine"})
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({}, id="missing-query"),
        pytest.param({"q": "a"}, id="one-character"),
        pytest.param({"q": "x" * 201}, id="absurdly-long"),
    ],
)
def test_geocode_rejects_queries_it_should_not_forward(
    geocode_client, monkeypatch, params
):
    stub = FakeNominatim(response=_nominatim_response([UCI_ROW])).install(monkeypatch)

    assert geocode_client.get("/api/v1/geocode", params=params).status_code == 422
    assert stub.calls == [], "a rejected query still reached Nominatim"


def test_geocode_whitespace_only_query_makes_no_upstream_call(
    geocode_client, monkeypatch
):
    """Long enough to pass validation, empty once trimmed."""
    stub = FakeNominatim(response=_nominatim_response([UCI_ROW])).install(monkeypatch)

    response = geocode_client.get("/api/v1/geocode", params={"q": "   "})
    assert response.status_code == 200
    assert response.json() == []
    assert stub.calls == []


def test_geocode_reports_an_upstream_timeout_as_504(geocode_client, monkeypatch):
    """A slow third party is not this service failing -- and must not read as one."""
    FakeNominatim(
        error=httpx.ReadTimeout("timed out", request=None)
    ).install(monkeypatch)

    response = geocode_client.get("/api/v1/geocode", params={"q": "UC Irvine"})
    assert response.status_code == 504
    assert "timed out" in response.json()["detail"]


@pytest.mark.parametrize(
    "stub_kwargs",
    [
        pytest.param(
            {"error": httpx.ConnectError("name resolution failed", request=None)},
            id="unreachable",
        ),
        pytest.param(
            {"response": _nominatim_response({"error": "blocked"}, status_code=403)},
            id="rejected",
        ),
        pytest.param(
            {"response": _nominatim_response({}, status_code=500)},
            id="upstream-error",
        ),
        pytest.param(
            {"response": _nominatim_response("<html>maintenance</html>")},
            id="not-json",
        ),
    ],
)
def test_geocode_turns_an_upstream_failure_into_502(
    geocode_client, monkeypatch, stub_kwargs
):
    """Never a 500: the fault is upstream and the detail says so."""
    FakeNominatim(**stub_kwargs).install(monkeypatch)

    response = geocode_client.get("/api/v1/geocode", params={"q": "UC Irvine"})
    assert response.status_code == 502
    assert "geocoding service" in response.json()["detail"]


def test_geocode_works_before_the_router_has_loaded(geocode_client, monkeypatch):
    """Search is useful the moment the process is listening.

    ``geocode_client`` never enters the lifespan, so there is no service on
    ``app.state``: routing correctly answers 503 here while geocoding answers
    200. That is the whole point of leaving ``get_service`` off this endpoint.
    """
    FakeNominatim(response=_nominatim_response([UCI_ROW])).install(monkeypatch)

    assert geocode_client.post("/api/v1/route", json=TRIP).status_code == 503
    assert geocode_client.get(
        "/api/v1/geocode", params={"q": "UC Irvine"}
    ).status_code == 200


def test_geocode_is_published_in_the_schema(geocode_client):
    schema = geocode_client.get("/openapi.json").json()
    operation = schema["paths"]["/api/v1/geocode"]["get"]

    names = {parameter["name"] for parameter in operation["parameters"]}
    assert "q" in names


# ---------------------------------------------------------------------------
# Predictions really reach the native graph
# ---------------------------------------------------------------------------
def test_predictions_land_on_the_right_edges(service):
    """The alignment test: read the weights back out of the engine.

    ``segment_edge_id`` follows the ``(u, v, key)``-sorted convention. If the
    service used GeoPackage row order instead, this would still pass a shape
    check and still route -- along the wrong roads.

    What lands on the edge is the prediction floored at that edge's calibrated
    free-flow time: the model predicts against posted limits, which are faster
    than the capped cruise speeds the graph is built on, so an uncongested
    forecast would otherwise make a corridor edge faster than it can be
    driven. See ``RoutingService.apply``.
    """
    prediction = service.latest_prediction
    assert prediction is not None

    for horizon, graph in enumerate(service.dynamic_graphs):
        predicted = prediction.travel_time_s[horizon]
        for i in range(0, EXPECTED_SEGMENTS, 37):
            edge_id = int(service.segments.edge_id[i])
            edge = graph.edge(edge_id)
            expected = max(
                float(predicted[i]),
                float(service.network.edge_free_flow_s[edge_id]),
            )
            assert edge.travel_time_sec == pytest.approx(expected, rel=1e-5)
            # The prediction must sit on the segment the dataset says it does.
            assert edge.src == int(service.network.edge_src[edge_id])
            assert edge.dst == int(service.network.edge_dst[edge_id])


def test_non_corridor_edges_keep_free_flow_weights(service):
    """``update_edge_weights_subset`` must touch 444 edges, not 15,553."""
    corridor = set(int(e) for e in service.segments.edge_id)
    others = [e for e in range(0, EXPECTED_EDGES, 211) if e not in corridor]
    assert others

    graph = service.dynamic_graphs[0]
    for edge_id in others:
        assert graph.edge(edge_id).travel_time_sec == pytest.approx(
            float(service.network.edge_free_flow_s[edge_id]), rel=1e-5
        )


def test_baseline_graph_is_never_updated(service):
    """The baseline must stay at posted speed limits for the comparison to mean anything."""
    for i in range(0, EXPECTED_SEGMENTS, 53):
        edge_id = int(service.segments.edge_id[i])
        assert service.baseline_graph.edge(edge_id).travel_time_sec == pytest.approx(
            float(service.network.edge_free_flow_s[edge_id]), rel=1e-5
        )


def test_python_weight_mirror_matches_the_engine(service):
    """``travel_times`` is used to re-score the baseline path; it must not drift."""
    for horizon, graph in enumerate(service.dynamic_graphs):
        for edge_id in range(0, EXPECTED_EDGES, 397):
            assert graph.edge(edge_id).travel_time_sec == pytest.approx(
                float(service.travel_times[horizon, edge_id]), rel=1e-5
            )


# ---------------------------------------------------------------------------
# Background loop
# ---------------------------------------------------------------------------
def test_update_loop_ticks_repeatedly(service):
    """Driven directly at a fast cadence; the app's own loop sleeps for 300 s."""

    async def drive() -> int:
        before = service.status.update_count
        task = asyncio.create_task(update_traffic_predictions_loop(service, 0.01))
        try:
            deadline = asyncio.get_running_loop().time() + 20.0
            while service.status.update_count < before + 3:
                if asyncio.get_running_loop().time() > deadline:
                    break
                await asyncio.sleep(0.05)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        return service.status.update_count - before

    assert asyncio.run(drive()) >= 3


def test_update_loop_survives_a_failing_tick(service, monkeypatch):
    """One bad tick must not silently take the forecaster offline."""

    calls = {"n": 0}
    real_tick = service.tick

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("simulated feed outage")
        return real_tick(*args, **kwargs)

    monkeypatch.setattr(service, "tick", flaky)

    async def drive() -> None:
        failures = service.status.failure_count
        updates = service.status.update_count
        task = asyncio.create_task(update_traffic_predictions_loop(service, 0.01))
        try:
            deadline = asyncio.get_running_loop().time() + 20.0
            while service.status.update_count <= updates:
                if asyncio.get_running_loop().time() > deadline:
                    pytest.fail("loop never recovered from the simulated failure")
                await asyncio.sleep(0.05)
            assert service.status.failure_count == failures + 2
            # A successful tick clears the error banner.
            assert service.status.last_error is None
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(drive())


# ---------------------------------------------------------------------------
# Middleware and plumbing
# ---------------------------------------------------------------------------
def test_cors_preflight_is_allowed_for_the_frontend(client):
    response = client.options(
        "/api/v1/route",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_openapi_schema_is_served(client):
    schema = client.get("/openapi.json").json()
    assert "/api/v1/route" in schema["paths"]
    assert "/api/v1/traffic-state" in schema["paths"]
    assert "/health" in schema["paths"]


def test_service_is_unavailable_before_startup():
    """Requests that beat the lifespan get a 503, not an AttributeError."""
    from app.main import get_service
    from fastapi import HTTPException

    class _Request:
        app = type("_App", (), {"state": type("_State", (), {})()})()

    with pytest.raises(HTTPException) as excinfo:
        get_service(_Request())
    assert excinfo.value.status_code == 503
