"""Asynchronous routing API over the ONNX ST-GNN and the native A* engine.

Shape of the thing
------------------
Two costs sit behind this API and they are orders of magnitude apart: an
ST-GNN forward pass over 444 corridor segments takes tens of milliseconds,
while an A* search over the 15,553-edge graph -- signal delays and turn
penalties included -- takes a few hundred *micro*seconds. Putting inference on
the request path would therefore make every query ~20x slower for a prediction
that only changes every five minutes.

So it does not go there. A background task runs inference on a five-minute
cadence -- the sampling interval the model was trained on -- and writes the
result into the native graphs with ``update_edge_weights_subset``. A request
arrives at a graph that is already current and pays for one A* search and
nothing else.

    startup ──► load artifacts ──► build 4 native graphs ──► ONNX session
                                                                 │
                     ┌───────────────────────────────────────────┘
                     ▼
       update_traffic_predictions_loop()      every 300 s, off the event loop
         features [1,444,5,12] ──► ONNX ──► [3,444] seconds
                                              │
                                              ▼  O(444) in-place write
       ┌──────────────┬──────────────┬──────────────┐   ┌──────────────┐
       │ dynamic +15m │ dynamic +30m │ dynamic +60m │   │ baseline     │
       └──────────────┴──────────────┴──────────────┘   └──────────────┘
                     ▲                                          ▲
       POST /api/v1/route ── one A* on each ──► compare ────────┘

Async model
-----------
Every native call releases the GIL, so the blocking work runs in the thread
pool via ``asyncio.to_thread`` and the event loop keeps accepting connections
throughout -- including during the ~7 ms forecast pass, which is the one piece
of work long enough to be felt if it ran inline.

Run it with:

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings
from app.schemas import (
    GeocodeResult,
    HealthResponse,
    MapConfig,
    RouteLegModel,
    RouteRequest,
    RouteResponse,
    SnappedPoint,
    TrafficStateResponse,
)
from app.service import RoutingService

__all__ = ["app", "create_app", "update_traffic_predictions_loop"]

# The dashboard opens here; the graph covers the City of Irvine.
IRVINE_CENTER_LAT = 33.6846
IRVINE_CENTER_LON = -117.8265

# ---------------------------------------------------------------------------
# Geocoding (OpenStreetMap Nominatim)
# ---------------------------------------------------------------------------
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# Nominatim's usage policy requires an identifying User-Agent and blocks the
# default one sent by HTTP libraries.
NOMINATIM_USER_AGENT = "IrvineTrafficAI/1.0"

# Short by design. This sits on an interactive keystroke path, and a slow
# third-party lookup is worth abandoning rather than queueing behind.
GEOCODE_TIMEOUT_SEC = 3.0

# Greater Irvine / Orange County, as Nominatim's two opposite corners
# "x1,y1,x2,y2" -- here south-west then north-east. With bounded=1
# Nominatim discards anything outside it, so "Main Street" resolves to the one
# in Irvine instead of the one in Los Angeles -- and every hit is a coordinate
# the router can actually snap to the graph.
IRVINE_VIEWBOX = "-117.92,33.58,-117.68,33.78"

GEOCODE_MAX_RESULTS = 8

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
)

API_PREFIX = "/api/v1"


# ---------------------------------------------------------------------------
# Background forecasting
# ---------------------------------------------------------------------------
async def update_traffic_predictions_loop(
    service: RoutingService, interval_sec: float
) -> None:
    """Re-score the network every ``interval_sec`` until cancelled.

    Runs one tick immediately so the service is never serving free-flow
    weights under a "dynamic" label, then settles into the cadence.

    A failed tick is logged and the loop continues: the graphs keep the last
    good prediction, which is a far better answer than an outage. Persistent
    failures surface through ``/health`` as a rising ``failure_count`` and a
    stale ``last_update``.
    """
    while True:
        started = time.perf_counter()
        try:
            # to_thread: ONNX inference and the native weight writes both
            # release the GIL, so this genuinely yields the event loop.
            prediction = await asyncio.to_thread(service.tick)
            LOGGER.info(
                "Forecast tick %d: %.1f ms inference, %.2f ms apply, "
                "mean congestion %s",
                service.status.update_count,
                prediction.inference_ms,
                service.status.last_apply_ms or 0.0,
                ", ".join(f"{c:.2f}x" for c in service.status.mean_congestion),
            )
        except asyncio.CancelledError:
            LOGGER.info("Forecast loop cancelled after %d ticks",
                        service.status.update_count)
            raise
        except Exception as exc:  # noqa: BLE001 - the loop must outlive any tick
            service.status.failure_count += 1
            service.status.last_error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Forecast tick failed; keeping previous weights")

        # Subtract the work already done so the cadence does not drift.
        elapsed = time.perf_counter() - started
        await asyncio.sleep(max(0.0, interval_sec - elapsed))


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI app. Artifacts are loaded in the lifespan, not here."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        LOGGER.info("Loading artifacts from %s", settings.processed_dir)
        started = time.perf_counter()
        # Blocking and heavy (a GeoPackage, a 92 MB tensor bundle, an ONNX
        # graph). Off-thread so the ASGI server stays responsive to signals.
        service = await asyncio.to_thread(RoutingService.create, settings)
        LOGGER.info("Startup complete in %.2fs", time.perf_counter() - started)

        app.state.service = service
        app.state.settings = settings
        app.state.update_task = asyncio.create_task(
            update_traffic_predictions_loop(service, settings.update_interval_sec),
            name="update_traffic_predictions_loop",
        )
        try:
            yield
        finally:
            app.state.update_task.cancel()
            try:
                await app.state.update_task
            except asyncio.CancelledError:
                pass
            LOGGER.info("Shutdown complete")

    app = FastAPI(
        title="Irvine Traffic AI",
        version="1.0.0",
        summary="ST-GNN travel-time forecasts served through a native A* router.",
        description=__doc__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    _register_routes(app, settings)
    _mount_dashboard(app, settings)
    return app


def _mount_dashboard(app: FastAPI, settings: Settings) -> None:
    """Serve the Deck.gl dashboard out of ``app/static``.

    Mounted after the API routes so a future static file can never shadow an
    endpoint. The root route returns ``index.html`` directly rather than
    redirecting, so the dashboard has a clean URL.
    """
    static_dir = settings.static_dir
    if not static_dir.is_dir():
        LOGGER.warning("%s missing; dashboard not mounted", static_dir)
        return

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    index = static_dir / "index.html"

    @app.get("/", include_in_schema=False)
    async def dashboard() -> FileResponse:
        if not index.exists():
            raise HTTPException(status_code=404, detail="dashboard is not built")
        # no-store: the HTML is a thin shell around app.js, and a cached copy
        # pointing at a stale bundle is the one failure worth designing out.
        return FileResponse(index, headers={"Cache-Control": "no-store"})

    LOGGER.info("Dashboard mounted from %s", static_dir)


def get_service(request: Request) -> RoutingService:
    """The service, or 503 if the lifespan has not finished loading it."""
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="service is still starting up",
        )
    return service


def _register_routes(app: FastAPI, settings: Settings) -> None:
    @app.post(
        f"{API_PREFIX}/route",
        response_model=RouteResponse,
        summary="Compare a traffic-aware route against a free-flow baseline",
        tags=["routing"],
    )
    async def route(
        payload: RouteRequest, service: RoutingService = Depends(get_service)
    ) -> RouteResponse:
        """Snap both endpoints to the graph and run two A* searches.

        The dynamic search uses the ST-GNN weights for the horizon nearest the
        requested departure; the baseline uses posted speed limits. Both paths
        are then scored under the predicted weights, so ``time_saved_sec`` is
        the delay the traffic model expects to avoid.
        """
        started = time.perf_counter()
        now = datetime.now(timezone.utc)
        departure = _as_utc(payload.departure_time)

        try:
            # One hop into the thread pool for the whole comparison. The two
            # searches run back to back inside it: each is ~0.2 ms of native
            # work, so a second to_thread to overlap them would cost more in
            # dispatch than it could ever save.
            comparison = await asyncio.to_thread(
                service.route,
                payload.origin_lat,
                payload.origin_lon,
                payload.dest_lat,
                payload.dest_lon,
                departure,
                now,
                payload.preferences,
            )
        except LookupError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc

        latency_ms = (time.perf_counter() - started) * 1000.0
        prediction = service.latest_prediction

        return RouteResponse(
            dynamic_route=_leg(comparison.dynamic),
            baseline_route=_leg(comparison.baseline),
            time_saved_sec=round(comparison.time_saved_sec, 2),
            preference_penalty_sec=comparison.preference_penalty_sec,
            query_latency_ms=round(latency_ms, 3),
            compute_latency_ms=round(comparison.compute_ms, 3),
            origin=_point(comparison.origin),
            destination=_point(comparison.destination),
            horizon_minutes=comparison.horizon_minutes,
            identical_path=comparison.identical_path,
            prediction_generated_at=prediction.generated_at if prediction else None,
        )

    @app.get(
        f"{API_PREFIX}/traffic-state",
        response_model=TrafficStateResponse,
        summary="Predicted congestion on every corridor segment",
        tags=["traffic"],
    )
    async def traffic_state(
        service: RoutingService = Depends(get_service),
        horizon: int = Query(
            0, ge=0, description="Index into available_horizon_minutes."
        ),
        include_geometry: bool = Query(
            True, description="Include per-segment polylines for map rendering."
        ),
    ) -> TrafficStateResponse:
        """Current forecast for the 444 corridor segments, for the heatmap.

        Read straight out of the last tick's prediction -- no inference on the
        request path. ``stale`` is true only in the window before the first
        tick completes.
        """
        try:
            payload = service.traffic_state(
                horizon=horizon, include_geometry=include_geometry
            )
        except ValueError as exc:
            # 422 as a literal: Starlette renamed the constant, and matching
            # FastAPI's own validation status is the point either way.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return TrafficStateResponse(**payload)

    @app.get(
        f"{API_PREFIX}/geocode",
        response_model=list[GeocodeResult],
        summary="Search Irvine addresses and places by name",
        tags=["dashboard"],
    )
    async def geocode(
        q: str = Query(
            ...,
            min_length=2,
            max_length=200,
            description="Free-text address or place name, e.g. 'UC Irvine'.",
        ),
    ) -> list[GeocodeResult]:
        """Proxy a place-name search to Nominatim, bounded to greater Irvine.

        Proxied rather than called from the browser for three reasons: the
        User-Agent that Nominatim's usage policy demands cannot be set on a
        cross-origin ``fetch``, the bounding box stays server-side where it
        cannot drift from the routable network, and the upstream shape (a
        dozen fields, coordinates as strings) is narrowed to the three the
        dashboard uses.

        This endpoint deliberately does not depend on ``get_service``: an
        address search is useful the moment the process is listening, and
        there is nothing here for the router to be warm for.
        """
        query = q.strip()
        if not query:
            return []

        try:
            async with httpx.AsyncClient(
                timeout=GEOCODE_TIMEOUT_SEC,
                headers={"User-Agent": NOMINATIM_USER_AGENT},
            ) as client:
                response = await client.get(
                    NOMINATIM_URL, params=_nominatim_params(query)
                )
                response.raise_for_status()
                rows = response.json()
        except httpx.TimeoutException as exc:
            LOGGER.warning("Geocode timed out after %.1fs: %s", GEOCODE_TIMEOUT_SEC, exc)
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="geocoding service timed out",
            ) from exc
        except httpx.HTTPError as exc:
            LOGGER.warning("Geocode request failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="geocoding service is unavailable",
            ) from exc
        except ValueError as exc:  # response.json() on a non-JSON body
            LOGGER.warning("Geocode returned an unreadable body: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="geocoding service returned an unreadable response",
            ) from exc

        return _geocode_results(rows)

    @app.get(f"{API_PREFIX}/config", response_model=MapConfig, tags=["dashboard"])
    async def map_config() -> MapConfig:
        """Basemap settings for the dashboard, including the Mapbox token.

        The token is a publishable ``pk.*`` key -- it is designed to reach the
        browser -- but it still comes from the environment rather than being
        baked into a static asset, so it can be rotated or scoped per
        deployment. When it is absent the client falls back to a keyless
        basemap; Deck.gl and every metric work either way.
        """
        return MapConfig(
            mapbox_token=settings.mapbox_token,
            map_style=settings.map_style,
            center=[IRVINE_CENTER_LON, IRVINE_CENTER_LAT],
            zoom=11.5,
            update_interval_sec=settings.update_interval_sec,
        )

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health(service: RoutingService = Depends(get_service)) -> HealthResponse:
        """Liveness plus enough forecast state to spot a wedged update loop."""
        import traffic_engine

        now = datetime.now(timezone.utc)
        state = service.status
        since_update = (
            (now - state.last_update).total_seconds() if state.last_update else None
        )
        # Stale once we have missed two cadences; that is a real fault, while
        # a single skipped tick is not.
        interval = service.settings.update_interval_sec
        healthy = state.last_update is not None and (since_update or 0) < interval * 2

        return HealthResponse(
            status="ok" if healthy else "degraded",
            engine_version=traffic_engine.__version__,
            num_nodes=service.network.num_nodes,
            num_edges=service.network.num_edges,
            num_segments=len(service.segments),
            horizon_minutes=list(service.horizon_minutes),
            feed=state.feed_name,
            uptime_sec=round((now - state.started_at).total_seconds(), 1),
            update_count=state.update_count,
            failure_count=state.failure_count,
            last_update=state.last_update,
            seconds_since_update=(
                round(since_update, 1) if since_update is not None else None
            ),
            last_inference_ms=(
                round(state.last_inference_ms, 2) if state.last_inference_ms else None
            ),
            last_apply_ms=(
                round(state.last_apply_ms, 3) if state.last_apply_ms else None
            ),
            mean_congestion=[round(c, 4) for c in state.mean_congestion],
            last_error=state.last_error,
            degraded_reason=state.degraded_reason,
        )


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------
def _as_utc(moment: datetime | None) -> datetime | None:
    """Read a naive timestamp as UTC rather than as local server time."""
    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _nominatim_params(query: str) -> dict[str, str]:
    """Query string for one bounded Nominatim search.

    ``bounded=1`` is what makes ``viewbox`` a filter rather than a hint: on its
    own the viewbox only re-ranks, so a query with no local match would still
    return somewhere unroutable.
    """
    return {
        "q": query,
        "format": "jsonv2",
        "limit": str(GEOCODE_MAX_RESULTS),
        "viewbox": IRVINE_VIEWBOX,
        "bounded": "1",
        "addressdetails": "0",
    }


def _geocode_results(rows: object) -> list[GeocodeResult]:
    """Narrow Nominatim's rows to ``display_name``/``lat``/``lon``.

    Coordinates arrive as strings and the occasional row arrives without them.
    A bad row is skipped rather than failing the whole search: the user is
    typing, and eight good suggestions minus one is still a usable dropdown.
    """
    if not isinstance(rows, list):
        LOGGER.warning("Geocode expected a list, got %s", type(rows).__name__)
        return []

    results: list[GeocodeResult] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("display_name")
        if not name:
            continue
        try:
            # ValueError covers both the float() parse and pydantic's own
            # range check, which raises a ValidationError subclass of it.
            results.append(
                GeocodeResult(
                    display_name=str(name),
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return results


def _leg(leg) -> RouteLegModel:
    return RouteLegModel(
        duration_sec=leg.duration_sec,
        distance_meters=leg.distance_meters,
        coordinates=leg.coordinates,
        free_flow_duration_sec=leg.free_flow_duration_sec,
        node_count=leg.node_count,
        nodes_expanded=leg.nodes_expanded,
    )


def _point(snap) -> SnappedPoint:
    return SnappedPoint(
        node_id=snap.node_id,
        latitude=snap.latitude,
        longitude=snap.longitude,
        distance_m=round(snap.distance_m, 2),
    )


app = create_app()
