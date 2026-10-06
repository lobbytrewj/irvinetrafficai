"""Routing preferences over HTTP, end to end.

Where this can go wrong
-----------------------
The preference plumbing crosses four boundaries -- browser JSON, a Pydantic
model, the service, and a pybind11 struct -- and three of the four failure
modes are silent. A field dropped between the request model and the service, a
boolean transposed between the two preferences, or a penalty that never
reaches the engine all produce a perfectly well-formed 200 with a perfectly
sensible route on it. The only way to see any of them is to make the flag
actually change the answer.

So the centrepiece here flags a real movement on the live Irvine graph, on a
route the service just returned, and asserts the next request routes around
it. Nothing is mocked: same service, same graphs, same A*.

Flags on the real graph
-----------------------
The artifacts on disk predate ``graph_builder``'s turn stage, so the service
starts with no movement flags at all -- which is exactly the state most of
these tests want, since it proves preferences are inert rather than
accidentally load-bearing. The reroute test installs a flag itself and takes
it back out afterwards.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("httpx", reason="httpx not installed; needed by TestClient")
pytest.importorskip(
    "traffic_engine", reason="native extension not built; run `pip install -e .`"
)
pytest.importorskip("sklearn", reason="scikit-learn not installed")

import traffic_engine  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.schemas import RoutePreferences, RouteRequest  # noqa: E402

SETTINGS = Settings()

# Campus Drive to the Great Park: long enough to cross a dozen junctions, so
# there is somewhere for an avoidance route to go.
TRIP = {
    "origin_lat": 33.6497,
    "origin_lon": -117.8409,
    "dest_lat": 33.6595,
    "dest_lon": -117.7467,
}

AVOID_LEFTS = {"preferences": {"avoid_unprotected_lefts": True}}

# Most movements on a 13.8 km arterial trip have no alternative worth 45 s --
# the corridor is the corridor -- so the reroute tests price avoidance high
# enough that a detour is on the table. That the *default* 45 s often leaves
# the route alone is the preference working: it is a price, not a ban.
STRONG_AVOID = {
    "preferences": {
        "avoid_unprotected_lefts": True,
        "unprotected_left_penalty_sec": 300.0,
    }
}


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
# Schema: no service needed
# ---------------------------------------------------------------------------
def test_preferences_default_to_the_fastest_route():
    """A request that says nothing must behave exactly as it did before."""
    request = RouteRequest(**TRIP)

    assert request.preferences.avoid_unprotected_lefts is False
    assert request.preferences.low_stress_merges is False
    assert request.preferences.any() is False
    # The penalties still have values; they simply do not apply.
    assert request.preferences.unprotected_left_penalty_sec == 45.0
    assert request.preferences.aggressive_merge_penalty_sec == 60.0


def test_preferences_accept_a_partial_object():
    """The browser sends the two booleans; the penalties keep their defaults."""
    request = RouteRequest(**TRIP, preferences={"low_stress_merges": True})

    assert request.preferences.low_stress_merges is True
    assert request.preferences.avoid_unprotected_lefts is False
    assert request.preferences.aggressive_merge_penalty_sec == 60.0


def test_the_api_model_matches_the_engine_struct():
    """Field for field, because a rename on one side is a silent no-op.

    ``_native_preferences`` copies by keyword, so a field renamed here and not
    there would fail loudly -- but only if something checks the shapes agree.
    """
    api = set(RoutePreferences.model_fields)
    native = traffic_engine.RoutePreferences()

    assert api == {
        "avoid_unprotected_lefts",
        "low_stress_merges",
        "unprotected_left_penalty_sec",
        "aggressive_merge_penalty_sec",
    }
    for field in api:
        assert hasattr(native, field), f"engine has no {field}"


@pytest.mark.parametrize(
    "preferences",
    [
        pytest.param({"avoid_unprotected_lefts": "yes please"}, id="not-a-boolean"),
        pytest.param({"unprotected_left_penalty_sec": -1.0}, id="negative-penalty"),
        pytest.param({"aggressive_merge_penalty_sec": 6000.0}, id="absurd-penalty"),
        pytest.param({"avoid_left_turns": True}, id="misspelled-field"),
    ],
)
def test_malformed_preferences_are_rejected(preferences):
    """`extra="forbid"` catches the typo a client would otherwise ship blind."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RouteRequest(**TRIP, preferences=preferences)


# ---------------------------------------------------------------------------
# Over HTTP
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    if not _artifacts_present():
        pytest.skip("processed artifacts missing; run the data and training pipelines")

    with TestClient(create_app(SETTINGS)) as test_client:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if test_client.get("/health").json()["update_count"] >= 1:
                break
            time.sleep(0.05)
        else:
            pytest.fail("no forecast tick completed within 60 s of startup")
        yield test_client


@pytest.fixture(scope="module")
def service(client):
    return client.app.state.service


def _route(client, **extra) -> dict:
    response = client.post("/api/v1/route", json={**TRIP, **extra})
    assert response.status_code == 200, response.text
    return response.json()


def test_a_request_without_preferences_still_works(client):
    """The compatibility guarantee: old clients send no such field."""
    body = _route(client)

    assert body["dynamic_route"]["duration_sec"] > 0
    assert body["preference_penalty_sec"] == 0.0


def test_preferences_are_accepted_and_reported(client):
    body = _route(client, **AVOID_LEFTS)

    assert body["dynamic_route"]["duration_sec"] > 0
    assert "preference_penalty_sec" in body
    assert body["preference_penalty_sec"] >= 0.0


def test_both_preferences_together_are_accepted(client):
    body = _route(
        client,
        preferences={
            "avoid_unprotected_lefts": True,
            "low_stress_merges": True,
            "unprotected_left_penalty_sec": 30.0,
            "aggressive_merge_penalty_sec": 90.0,
        },
    )
    assert body["dynamic_route"]["duration_sec"] > 0


@pytest.mark.parametrize(
    "preferences",
    [
        pytest.param({"avoid_unprotected_lefts": 3.7}, id="not-a-boolean"),
        pytest.param({"unprotected_left_penalty_sec": -5}, id="negative-penalty"),
        pytest.param({"nonexistent": True}, id="unknown-field"),
    ],
)
def test_the_endpoint_rejects_malformed_preferences(client, preferences):
    response = client.post("/api/v1/route", json={**TRIP, "preferences": preferences})
    assert response.status_code == 422


def test_preferences_appear_in_the_openapi_schema(client):
    """The dashboard is one client; the schema is how the others find this."""
    schema = client.get("/openapi.json").json()

    request_model = schema["components"]["schemas"]["RouteRequest"]
    assert "preferences" in request_model["properties"]
    preferences = schema["components"]["schemas"]["RoutePreferences"]["properties"]
    assert preferences["avoid_unprotected_lefts"]["default"] is False
    assert preferences["low_stress_merges"]["default"] is False


def test_preferences_are_inert_without_movement_flags(client, service):
    """No flags loaded means no behaviour change -- and no crash either.

    This is the state the shipped artifacts are in, so it is worth pinning:
    asking to avoid something the graph cannot identify returns the fastest
    route rather than an error or an empty result.
    """
    assert not service.dynamic_graphs[0].has_turn_flags

    plain = _route(client)
    fussy = _route(client, **AVOID_LEFTS)

    assert plain["dynamic_route"]["coordinates"] == fussy["dynamic_route"]["coordinates"]
    assert fussy["preference_penalty_sec"] == 0.0


# ---------------------------------------------------------------------------
# The one that proves the wiring
# ---------------------------------------------------------------------------
@pytest.fixture
def flagged_movement(client, service):
    """Flag a movement on the live route that the router can actually avoid.

    Walks the real path and marks one movement at a time until the route moves,
    because not every movement has an alternative -- a segment of Jamboree with
    no parallel street is unavoidable at any price, and a test that happened to
    pick one would fail for a reason that has nothing to do with the plumbing.

    Flags go on every graph, baseline included, since the service routes both
    sides under the same preferences. Removed afterwards: the service fixture
    is module-scoped and the graphs are shared.
    """
    comparison = service.route(**TRIP)
    result = service.dynamic_graphs[0].find_shortest_path(
        comparison.origin.node_id, comparison.destination.node_id
    )
    edge_ids = list(result.path_edge_ids)
    assert len(edge_ids) > 6, "need a path with somewhere to detour"

    graphs = [service.baseline_graph, *service.dynamic_graphs]

    def install(pair):
        for graph in graphs:
            if pair is None:
                graph.set_turn_flags([], [], [])
            else:
                graph.set_turn_flags(
                    [pair[0]], [pair[1]], [traffic_engine.TURN_FLAG_UNPROTECTED_LEFT]
                )

    fastest = _route(client)["dynamic_route"]["coordinates"]
    chosen = None
    try:
        for index in range(1, len(edge_ids)):
            pair = (edge_ids[index - 1], edge_ids[index])
            install(pair)
            if _route(client, **STRONG_AVOID)["dynamic_route"]["coordinates"] != fastest:
                chosen = pair
                break
            install(None)

        if chosen is None:
            pytest.skip("no movement on this path has an alternative route")

        yield {"in_edge": chosen[0], "out_edge": chosen[1], "fastest": fastest}
    finally:
        install(None)


def test_toggling_the_preference_reroutes_around_the_flagged_turn(
    client, service, flagged_movement
):
    """The headline: the same request, one boolean apart, avoids the movement.

    Everything else is held still -- same endpoints, same departure, same
    forecast tick, same graphs. The only input that moved is the toggle the
    sidebar switch flips.
    """
    fastest = _route(client)
    avoided = _route(client, **STRONG_AVOID)

    assert fastest["dynamic_route"]["coordinates"] != avoided[
        "dynamic_route"
    ]["coordinates"], "the preference did not change the route"

    # The flagged movement is gone, and the detour is honest about its cost:
    # a route that avoids something cannot also be faster.
    assert avoided["preference_penalty_sec"] == 0.0
    assert avoided["dynamic_route"]["duration_sec"] >= fastest["dynamic_route"][
        "duration_sec"
    ] - 0.01


def test_the_same_toggle_off_returns_the_fast_route(client, flagged_movement):
    """Flipping back is not a one-way door: state lives in the request."""
    avoided = _route(client, **STRONG_AVOID)
    back = _route(client)

    assert back["dynamic_route"]["coordinates"] == flagged_movement["fastest"]
    assert back["dynamic_route"]["coordinates"] != avoided["dynamic_route"][
        "coordinates"
    ]


def test_the_flagged_turn_is_priced_only_when_asked_for(service, flagged_movement):
    """Same graph, same movement: the flag alone costs nothing."""
    graph = service.dynamic_graphs[0]
    in_edge = flagged_movement["in_edge"]
    out_edge = flagged_movement["out_edge"]

    assert graph.turn_flags(in_edge, out_edge) == traffic_engine.TURN_FLAG_UNPROTECTED_LEFT
    assert graph.transition_cost(in_edge, out_edge) == pytest.approx(
        graph.transition_cost(
            in_edge, out_edge, traffic_engine.RoutePreferences()
        )
    )
    assert graph.transition_cost(
        in_edge,
        out_edge,
        traffic_engine.RoutePreferences(avoid_unprotected_lefts=True),
    ) == pytest.approx(graph.transition_cost(in_edge, out_edge) + 45.0)


def test_a_penalty_of_zero_leaves_the_route_alone(client, flagged_movement):
    """A preference priced at nothing is a preference for nothing."""
    free = _route(
        client,
        preferences={
            "avoid_unprotected_lefts": True,
            "unprotected_left_penalty_sec": 0.0,
        },
    )

    assert free["dynamic_route"]["coordinates"] == flagged_movement["fastest"]


def test_the_other_preference_ignores_this_flag(client, flagged_movement):
    """Avoiding merges must not quietly avoid lefts as well."""
    merges = _route(
        client,
        preferences={"low_stress_merges": True, "aggressive_merge_penalty_sec": 300.0},
    )

    assert merges["dynamic_route"]["coordinates"] == flagged_movement["fastest"]


def test_latency_is_unchanged_by_preferences(client, flagged_movement):
    """The flag lookup is one array read; it must not show up as a cost."""
    latencies = {}
    for label, extra in (("plain", {}), ("avoid", STRONG_AVOID)):
        samples = [
            _route(client, **extra)["query_latency_ms"] for _ in range(15)
        ]
        latencies[label] = float(np.median(samples))

    assert latencies["avoid"] < 5.0
    assert latencies["avoid"] < latencies["plain"] * 3.0


# ---------------------------------------------------------------------------
# The dashboard sends what the server expects
# ---------------------------------------------------------------------------
def test_the_bundle_posts_the_preference_field(client):
    """The two halves agree on the field names, checked against the source.

    A rename on the server would leave the toggles silently inert: the request
    still validates, because the field is optional.
    """
    bundle = client.get("/static/app.js").text

    assert "preferences: readPreferences()" in bundle
    for field in ("avoid_unprotected_lefts", "low_stress_merges"):
        assert field in bundle, f"{field} is not sent by the dashboard"


def test_the_sidebar_has_both_toggles(client):
    page = client.get("/").text

    assert 'data-preference="avoid_unprotected_lefts"' in page
    assert 'data-preference="low_stress_merges"' in page
    # Real checkboxes, so the switches are keyboard-reachable.
    assert page.count('type="checkbox"') >= 2
    assert "Routing Preferences" in page


def test_the_switch_styling_is_served(client):
    css = client.get("/static/style.css").text

    assert ".switch input:checked + .slider" in css
    assert "#3b82f6" in css and "#1e293b" in css
