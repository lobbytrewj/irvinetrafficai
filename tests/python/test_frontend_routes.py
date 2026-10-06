"""Verification tests for the dashboard's HTTP surface.

Scope is deliberately the *wiring*, not the rendering: that the root route
serves the shell, that the static mount serves the bundle, and that the
contract the bundle reads at boot -- ``/api/v1/config`` plus the response
fields the metric cards bind to -- is actually there.

The failure this is really guarding against is a silent one. ``app.js`` is
fetched by a `<script src>`; if the mount broke, the page would still return
200 with a perfectly valid HTML shell and simply render nothing. So the
assertions go past status codes into content type and payload.

Unlike ``test_api.py`` these do not need the ONNX model or the dataset -- the
routes under test are served before the lifespan touches any artifact -- but
they share the same app factory, so the fixture starts the real service when
the artifacts happen to be present.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("httpx", reason="httpx not installed; needed by TestClient")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402

SETTINGS = Settings()

STATIC_ASSETS = ("/static/index.html", "/static/app.js")


@pytest.fixture(scope="module")
def client():
    """A started app. Requires the artifacts, since the lifespan loads them."""
    for path in (
        SETTINGS.onnx_path,
        SETTINGS.geopackage_path,
        SETTINGS.node_mapping_path,
        SETTINGS.graph_tensors_path,
        SETTINGS.dataset_path,
    ):
        if not path.exists():
            pytest.skip(f"{path.name} missing; run the data and training pipelines")

    with TestClient(create_app(SETTINGS)) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Root route
# ---------------------------------------------------------------------------
def test_root_serves_the_dashboard(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>Irvine Traffic AI</title>" in response.text


def test_root_shell_loads_the_bundle_and_libraries(client):
    """The shell is inert without these three tags; a typo in any is silent."""
    body = client.get("/").text

    assert '<script src="/static/app.js"></script>' in body
    assert "mapbox-gl.js" in body
    assert "deck.gl" in body


def test_root_is_not_cached(client):
    """A cached shell pointing at a replaced bundle is the one bad outcome."""
    assert client.get("/").headers.get("cache-control") == "no-store"


# ---------------------------------------------------------------------------
# Static mount
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", STATIC_ASSETS)
def test_static_assets_are_served(client, path):
    response = client.get(path)

    assert response.status_code == 200
    assert len(response.content) > 0


def test_app_js_is_served_as_javascript(client):
    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    # Cheap proof it is the real bundle and not an error page rendered as 200.
    assert "MapboxOverlay" in response.text
    assert "/api/v1/route" in response.text


def test_static_mount_rejects_unknown_files(client):
    assert client.get("/static/does-not-exist.js").status_code == 404


def test_static_mount_does_not_shadow_the_api(client):
    """The mount is added after the routers, so /api and /health still win."""
    for path in ("/health", "/api/v1/config"):
        assert client.get(path).status_code == 200


# ---------------------------------------------------------------------------
# Boot contract
# ---------------------------------------------------------------------------
def test_config_endpoint_describes_the_basemap(client):
    body = client.get("/api/v1/config").json()

    assert set(body) == {
        "mapbox_token",
        "map_style",
        "center",
        "zoom",
        "update_interval_sec",
    }
    assert body["map_style"] == "mapbox://styles/mapbox/dark-v11"

    # GL order: [longitude, latitude], centred on Irvine.
    longitude, latitude = body["center"]
    assert longitude == pytest.approx(-117.8265, abs=1e-3)
    assert latitude == pytest.approx(33.6846, abs=1e-3)
    assert isinstance(body["mapbox_token"], str)


def test_config_carries_no_token_by_default(client):
    """Absent an env var the client must fall back, not render a broken map."""
    assert client.get("/api/v1/config").json()["mapbox_token"] == ""


def test_route_response_carries_both_latency_numbers(client):
    """The dashboard shows service-boundary and HTTP latency side by side."""
    response = client.post(
        "/api/v1/route",
        json={
            "origin_lat": 33.6846,
            "origin_lon": -117.8265,
            "dest_lat": 33.6595,
            "dest_lon": -117.7467,
        },
    )
    assert response.status_code == 200
    body = response.json()

    assert body["compute_latency_ms"] > 0
    # The handler measurement wraps the compute, so it can only be larger.
    assert body["compute_latency_ms"] <= body["query_latency_ms"]


def test_traffic_state_supplies_what_the_heatmap_bands_need(client):
    """Bands are free_flow_sec / predicted_sec; both must be present and > 0."""
    body = client.get("/api/v1/traffic-state?include_geometry=false").json()

    assert body["segment_count"] == 444
    for segment in body["segments"][:50]:
        assert segment["free_flow_sec"] > 0
        assert segment["predicted_sec"] > 0
        ratio = segment["free_flow_sec"] / segment["predicted_sec"]
        assert 0 < ratio <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Bundle sanity
# ---------------------------------------------------------------------------
def test_bundle_pins_every_executable_cdn_dependency(client):
    """Floating CDN tags break a dashboard on someone else's release schedule.

    Scoped to code -- the ``.js`` and ``.css`` the page executes. The CARTO
    fallback style is a ``.json`` document fetched as *data*: it has no
    versioned URL on offer, and the worst a change there can do is restyle the
    basemap. The lookahead keeps ``style.json`` from matching as ``style.js``.
    """
    sources = client.get("/").text + client.get("/static/app.js").text

    urls = re.findall(r"https://[^\s\"'()]+\.(?:js|css)(?=[\"'\s)]|$)", sources)
    assert len(urls) >= 3, f"expected the GL and Deck.gl bundles, found {urls}"

    unpinned = [url for url in urls if not re.search(r"[@/]v?\d+\.\d+\.\d+/", url)]
    assert not unpinned, f"unpinned CDN URLs: {unpinned}"


def test_bundle_flips_coordinates_exactly_once(client):
    """The API speaks [lat, lon] and GL speaks [lng, lat]; one helper owns it."""
    bundle = client.get("/static/app.js").text
    assert bundle.count("function toLngLat") == 1
