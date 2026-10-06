/**
 * Irvine Traffic AI — dashboard.
 *
 * Boot order
 * ----------
 *   GET /api/v1/config   → basemap settings, decides Mapbox vs. fallback
 *   load the GL library  → Mapbox GL JS, or MapLibre when no token exists
 *   attach Deck.gl       → MapboxOverlay, which drives both libraries
 *   GET /health          → feed status, forecast freshness
 *   GET /traffic-state   → 444 corridor segments
 *   click, click         → POST /route
 *   (or type, type       → GET /geocode → the same two endpoints)
 *   toggle a preference  → POST /route again, same endpoints
 *
 * Two things here are deliberate rather than incidental:
 *
 * **Geometry is fetched once.** Segment polylines are a property of the road
 * network, not of the forecast, so they never change between ticks. The first
 * traffic-state call takes the full 175 KB payload and caches the geometry by
 * segment id; every refresh after that asks for `include_geometry=false` and
 * merges into the cache. Switching horizons therefore costs no geometry at all.
 *
 * **Refreshes are gated on /health, not on a timer.** The forecaster updates
 * every five minutes. Polling traffic-state on a short timer would re-download
 * an unchanged forecast; instead a small /health poll watches `last_update`
 * and only then pulls new congestion values.
 *
 * Coordinate order is the one persistent trap: the API speaks [lat, lon]
 * throughout and every GL library speaks [lon, lat]. `toLngLat` is the single
 * place that flip happens.
 */

'use strict';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const MAPLIBRE_JS = 'https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js';
const MAPLIBRE_CSS = 'https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css';
const FALLBACK_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

const HEALTH_POLL_MS = 20000;
const BASEMAP_TIMEOUT_MS = 12000;

// Deck.gl wants RGBA byte tuples, so the palette is duplicated from the CSS
// custom properties rather than read back out of it.
const COLOR = {
  free: [34, 197, 94],
  moderate: [234, 179, 8],
  bottleneck: [239, 68, 68],
  dynamic: [0, 229, 255],
  baseline: [255, 58, 240],
};

const ORIGIN_COLOR = '#10b981';   // emerald
const DEST_COLOR = '#dc143c';     // crimson

// Speed as a share of the posted limit. Above FREE the corridor is running
// essentially at the limit; below BOTTLENECK it has lost more than half its
// speed. Matches the legend in index.html.
const BAND_FREE = 0.85;
const BAND_BOTTLENECK = 0.5;

// Each button's lead time in minutes, and the forecast horizon it lands on.
// The model produces +15 / +30 / +60, so "Now" and "+15m" share horizon 0 —
// the note under the selector says so rather than hiding it.
const HORIZON_FOR_LEAD = { 0: 0, 15: 0, 30: 1, 60: 2 };

// Address search. 300 ms is comfortably longer than a fast typist's inter-key
// gap, so a word turns into one Nominatim call rather than one per letter —
// which their usage policy asks for and which keeps the dropdown from
// flickering through partial-word matches.
const SEARCH_DEBOUNCE_MS = 300;
const SEARCH_MIN_CHARS = 2;

// Each field's DOM ids and the endpoint it drives. The keys are the `state`
// keys, so a field can set its endpoint without a lookup table.
const SEARCH_FIELDS = {
  origin: {
    input: 'origin-search',
    box: 'origin-box',
    results: 'origin-results',
    color: ORIGIN_COLOR,
  },
  destination: {
    input: 'dest-search',
    box: 'dest-box',
    results: 'dest-results',
    color: DEST_COLOR,
  },
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  config: null,
  // Mirrors the RoutePreferences model on the server. Read from the DOM on
  // every request rather than kept in sync with it, so the checkboxes are the
  // single source of truth for what the user asked for.
  preferences: { avoid_unprotected_lefts: false, low_stress_merges: false },
  map: null,
  GL: null,          // the active GL library namespace (mapbox or maplibre)
  overlay: null,
  origin: null,      // {lat, lon}
  destination: null,
  markers: { origin: null, destination: null },
  lead: 0,
  segments: [],      // traffic-state records, geometry merged in
  geometry: new Map(),   // segment_id -> [[lng, lat], ...]
  route: null,
  health: null,
  lastForecastAt: null,
  busy: false,
  search: {},        // field key -> the record built by setupSearchField
};

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------
async function getJSON(url) {
  const response = await fetch(url, { headers: { Accept: 'application/json' } });
  if (!response.ok) {
    throw new Error(`GET ${url} → ${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function postJSON(url, body) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    // FastAPI puts the useful message in `detail`; a 422 nests it further.
    const detail = payload && payload.detail;
    const message = typeof detail === 'string'
      ? detail
      : Array.isArray(detail) && detail.length
        ? detail.map((d) => d.msg).join('; ')
        : `${response.status} ${response.statusText}`;
    throw new Error(message);
  }
  return payload;
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const el = document.createElement('script');
    el.src = src;
    el.onload = resolve;
    el.onerror = () => reject(new Error(`failed to load ${src}`));
    document.head.appendChild(el);
  });
}

function loadStylesheet(href) {
  const el = document.createElement('link');
  el.rel = 'stylesheet';
  el.href = href;
  document.head.appendChild(el);
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------
/** Seconds to "1 hr 4 min", "12 min 45 sec" or "45 sec". */
function formatDuration(seconds) {
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;

  if (hours > 0) return `${hours} hr ${minutes} min`;
  if (minutes > 0) return `${minutes} min ${secs} sec`;
  return `${secs} sec`;
}

function formatLatency(ms) {
  if (ms === null || ms === undefined) return '—';
  return ms < 10 ? `${ms.toFixed(2)} ms` : `${ms.toFixed(1)} ms`;
}

function formatCoord(point) {
  return `${point.lat.toFixed(5)}, ${point.lon.toFixed(5)}`;
}

function escapeHTML(value) {
  return String(value).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

/** API [lat, lon] pairs → GL [lng, lat] pairs. The one place this flips. */
function toLngLat(coordinates) {
  return coordinates.map(([lat, lon]) => [lon, lat]);
}

// ---------------------------------------------------------------------------
// Congestion banding
// ---------------------------------------------------------------------------
/**
 * Current speed as a share of the free-flow speed.
 *
 * The API reports travel times, and the ratio of those is the inverse of the
 * ratio of speeds over a fixed length: free_flow_sec / predicted_sec. The
 * model clamps predictions at free flow, so this cannot exceed 1.
 */
function speedRatio(segment) {
  if (!segment.predicted_sec) return 1;
  return segment.free_flow_sec / segment.predicted_sec;
}

function bandOf(ratio) {
  if (ratio > BAND_FREE) return 'free';
  if (ratio >= BAND_BOTTLENECK) return 'moderate';
  return 'bottleneck';
}

// ---------------------------------------------------------------------------
// Basemap
// ---------------------------------------------------------------------------
async function initMap(config) {
  const usingMapbox = Boolean(config.mapbox_token);
  let style = config.map_style;

  if (usingMapbox) {
    state.GL = window.mapboxgl;
    state.GL.accessToken = config.mapbox_token;
  } else {
    // Mapbox GL JS throws without a token regardless of the style, so the
    // fallback swaps in MapLibre — same API surface, keyless basemap.
    loadStylesheet(MAPLIBRE_CSS);
    await loadScript(MAPLIBRE_JS);
    state.GL = window.maplibregl;
    style = FALLBACK_STYLE;
    showBasemapNote();
  }

  state.map = new state.GL.Map({
    container: 'map',
    style,
    center: config.center,
    zoom: config.zoom,
    attributionControl: true,
  });
  state.map.addControl(new state.GL.NavigationControl({ visualizePitch: true }), 'bottom-right');

  // Deliberately *not* awaiting the map's 'load' event. The overlay creates
  // its own canvas and the click handler needs nothing from the style, so
  // gating on 'load' would only mean that a slow or unreachable tile CDN took
  // the metrics, the feed panel and the congestion data down with it. The
  // basemap is the one part of this page that is allowed to be late.
  state.overlay = new deck.MapboxOverlay({ layers: [] });
  state.map.addControl(state.overlay);
  state.map.on('click', onMapClick);

  watchBasemap();
}

/** Warn if the basemap never finishes; everything else carries on regardless. */
function watchBasemap() {
  let settled = false;
  const done = () => { settled = true; };
  state.map.on('load', done);
  state.map.on('error', done);

  setTimeout(() => {
    if (settled || !state.map) return;
    const note = $('basemap-note');
    if (note.style.display === 'block') return;  // already showing the keyless notice
    note.innerHTML =
      '<b>Basemap is slow to load.</b> Tiles come from a third-party CDN. ' +
      'Routing, congestion and every metric are served locally and are ' +
      'unaffected.';
    note.style.display = 'block';
  }, BASEMAP_TIMEOUT_MS);
}

function showBasemapNote() {
  const note = $('basemap-note');
  note.innerHTML =
    '<b>Keyless basemap.</b> No Mapbox token configured, so the dark tiles ' +
    'come from CARTO via MapLibre. Set <code>MAPBOX_TOKEN</code> and restart ' +
    'for <code>mapbox://styles/mapbox/dark-v11</code>. Every layer and metric ' +
    'below is unaffected.';
  note.style.display = 'block';
}

// ---------------------------------------------------------------------------
// Deck.gl layers
// ---------------------------------------------------------------------------
function renderLayers() {
  // The map may still be booting; whoever finishes last calls this again.
  if (!state.overlay) return;
  const layers = [];

  if (state.segments.length) {
    layers.push(new deck.PathLayer({
      id: 'congestion',
      data: state.segments,
      getPath: (d) => d.path,
      getColor: (d) => COLOR[d.band],
      // Bottlenecks draw thicker: they are the few segments worth spotting
      // from a zoomed-out view, and there are only a handful of them.
      getWidth: (d) => (d.band === 'bottleneck' ? 5 : d.band === 'moderate' ? 3.5 : 2.5),
      widthUnits: 'pixels',
      widthMinPixels: 2,
      capRounded: true,
      jointRounded: true,
      opacity: 0.85,
      pickable: true,
      onHover: onSegmentHover,
      updateTriggers: {
        // Only the colour and width depend on the forecast; the paths do not,
        // so a horizon switch re-uploads no geometry.
        getColor: state.lastForecastAt,
        getWidth: state.lastForecastAt,
      },
    }));
  }

  if (state.route) {
    // Baseline first so the AI path draws over it where they overlap.
    layers.push(new deck.PathLayer({
      id: 'baseline-route',
      data: [{ path: toLngLat(state.route.baseline_route.coordinates) }],
      getPath: (d) => d.path,
      getColor: COLOR.baseline,
      getWidth: 4,
      widthUnits: 'pixels',
      capRounded: true,
      jointRounded: true,
      getDashArray: [7, 4],
      dashJustified: true,
      extensions: [new deck.PathStyleExtension({ dash: true })],
    }));

    layers.push(new deck.PathLayer({
      id: 'dynamic-route',
      data: [{ path: toLngLat(state.route.dynamic_route.coordinates) }],
      getPath: (d) => d.path,
      getColor: COLOR.dynamic,
      getWidth: 5,
      widthUnits: 'pixels',
      capRounded: true,
      jointRounded: true,
    }));
  }

  state.overlay.setProps({ layers });
}

function onSegmentHover(info) {
  const tooltip = $('tooltip');
  if (!info.object || info.x === undefined) {
    tooltip.style.display = 'none';
    return;
  }

  const d = info.object;
  const pct = (d.ratio * 100).toFixed(0);
  const delay = d.predicted_sec - d.free_flow_sec;
  tooltip.innerHTML = `
    <div class="t-name">${escapeHTML(d.name)}</div>
    <div class="t-row">${escapeHTML(d.corridor)} · segment ${d.segment_id}</div>
    <div class="t-row">speed <b>${d.speed_kph.toFixed(0)} km/h</b> (${pct}% of limit)</div>
    <div class="t-row">travel <b>${d.predicted_sec.toFixed(0)}s</b> vs ${d.free_flow_sec.toFixed(0)}s free flow</div>
    <div class="t-row">delay <b>+${delay.toFixed(0)}s</b> over ${Math.round(d.length_meters)} m</div>`;
  tooltip.style.display = 'block';
  tooltip.style.left = `${info.x + 14}px`;
  tooltip.style.top = `${info.y + 14}px`;
}

// ---------------------------------------------------------------------------
// Traffic state
// ---------------------------------------------------------------------------
async function loadTrafficState() {
  const horizon = HORIZON_FOR_LEAD[state.lead];
  // Geometry is a property of the road network, not the forecast: pull it on
  // the first call only, then reuse the cache forever.
  const needGeometry = state.geometry.size === 0;
  const payload = await getJSON(
    `/api/v1/traffic-state?horizon=${horizon}&include_geometry=${needGeometry}`
  );

  state.segments = payload.segments.map((segment) => {
    if (needGeometry && segment.coordinates) {
      state.geometry.set(segment.segment_id, toLngLat(segment.coordinates));
    }
    const ratio = speedRatio(segment);
    return {
      ...segment,
      ratio,
      band: bandOf(ratio),
      path: state.geometry.get(segment.segment_id) || [],
    };
  }).filter((segment) => segment.path.length >= 2);

  state.lastForecastAt = payload.generated_at;
  updateLegendCounts();
  renderLayers();
}

function updateLegendCounts() {
  const counts = { free: 0, moderate: 0, bottleneck: 0 };
  state.segments.forEach((segment) => { counts[segment.band] += 1; });
  $('count-free').textContent = `${counts.free}`;
  $('count-moderate').textContent = `${counts.moderate}`;
  $('count-bottleneck').textContent = `${counts.bottleneck}`;
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------
async function pollHealth() {
  try {
    const health = await getJSON('/health');
    const previous = state.health && state.health.last_update;
    state.health = health;
    renderStatus();
    renderFeed();

    // Only pull the forecast when it has actually moved.
    if (health.last_update && health.last_update !== previous) {
      await loadTrafficState();
    }
  } catch (error) {
    setStatus('bad', 'backend unreachable');
  }
}

function setStatus(kind, text) {
  $('status-dot').className = `dot ${kind}`;
  $('status-text').textContent = text;
}

function renderStatus() {
  const health = state.health;
  if (!health) return;
  const nodes = health.num_nodes.toLocaleString();
  const edges = health.num_edges.toLocaleString();
  if (health.status === 'ok') {
    setStatus('ok', `${nodes} nodes · ${edges} edges · ${health.num_segments} segments`);
  } else {
    setStatus('warn', health.degraded_reason || 'forecast is stale');
  }
}

/**
 * The replay feed maps wall clock onto the same weekday and time of day in
 * UTC, so the alignment it is currently serving *is* the UTC clock at the
 * last tick — no extra endpoint needed to report it.
 */
function feedAlignment(timestamp) {
  if (!timestamp) return null;
  const when = new Date(timestamp);
  const day = when.toLocaleDateString('en-US', { weekday: 'long', timeZone: 'UTC' });
  const time = when.toLocaleTimeString('en-US', {
    hour: '2-digit', minute: '2-digit', hour12: true, timeZone: 'UTC',
  });
  return `${day} ${time} UTC`;
}

function regimeOf(meanCongestion) {
  if (!meanCongestion || !meanCongestion.length) return null;
  const ratio = meanCongestion[0];
  if (ratio >= 1.5) return 'Peak';
  if (ratio >= 1.2) return 'Building';
  return 'Off-peak';
}

const FEED_LABELS = {
  'dataset-replay': 'DatasetReplayFeed',
  'free-flow': 'FreeFlowFeed',
};

function renderFeed() {
  const health = state.health;
  if (!health) return;

  const label = FEED_LABELS[health.feed] || health.feed;
  const alignment = feedAlignment(health.last_update);
  const regime = regimeOf(health.mean_congestion);
  const suffix = alignment ? ` [${alignment}${regime ? ` · ${regime}` : ''}]` : '';
  $('feed-name').textContent = `${label}${suffix}`;

  const rows = [
    ['Forecast ticks', health.update_count],
    ['Last inference', formatLatency(health.last_inference_ms)],
    ['Weight apply', formatLatency(health.last_apply_ms)],
    ['Mean delay (+15m)', health.mean_congestion.length
      ? `${health.mean_congestion[0].toFixed(2)}× free flow`
      : '—'],
  ];
  if (health.failure_count) rows.push(['Failed ticks', health.failure_count]);

  $('feed-detail').innerHTML = rows
    .map(([k, v]) => `<div class="fr"><span>${escapeHTML(k)}</span><span>${escapeHTML(v)}</span></div>`)
    .join('');
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------
function onMapClick(event) {
  const point = { lat: event.lngLat.lat, lon: event.lngLat.lng };
  // Third click starts a fresh pair rather than nudging the old one.
  const key = !state.origin || state.destination ? 'origin' : 'destination';

  if (key === 'origin') clearRoute();
  // A clicked point has no address, so the box drops back to its placeholder
  // and the coordinate line underneath carries the truth.
  resetSearchInput(key);
  setEndpoint(key, point);
}

/**
 * Write one endpoint and let everything downstream follow.
 *
 * The single entry point for both input paths — a picked suggestion and a map
 * click — so the marker, the panel and the route request can never be updated
 * by one of them and missed by the other. `recenter` is on for search, where
 * the point is off-screen by definition, and off for clicks, where moving the
 * map out from under the cursor would be obnoxious.
 */
function setEndpoint(key, point, { recenter = false } = {}) {
  state[key] = point;
  placeMarker(key, point, key === 'origin' ? ORIGIN_COLOR : DEST_COLOR);
  renderEndpoints();

  if (state.origin && state.destination) {
    requestRoute();   // fits the map to the finished route itself
  } else if (recenter) {
    flyTo(point);
  }
}

function placeMarker(key, point, color) {
  // Search can resolve before the basemap does; the endpoint is still valid,
  // it just has nowhere to draw yet.
  if (!state.map || !state.GL) return;
  if (state.markers[key]) state.markers[key].remove();
  state.markers[key] = new state.GL.Marker({ color })
    .setLngLat([point.lon, point.lat])
    .addTo(state.map);
}

function flyTo(point) {
  if (!state.map) return;
  state.map.easeTo({ center: [point.lon, point.lat], zoom: 14, duration: 700 });
}

function clearRoute() {
  state.origin = null;
  state.destination = null;
  state.route = null;
  ['origin', 'destination'].forEach((key) => {
    if (state.markers[key]) state.markers[key].remove();
    state.markers[key] = null;
    resetSearchInput(key);
  });
  hideError();
  renderEndpoints();
  renderMetrics();
  renderPreferenceNote();
  renderLayers();
}

function renderEndpoints() {
  const origin = $('origin-value');
  const destination = $('dest-value');

  const placeholder = 'search above or click the map';
  origin.textContent = state.origin ? formatCoord(state.origin) : placeholder;
  origin.className = state.origin ? 'endpoint-coord' : 'endpoint-coord empty';
  destination.textContent = state.destination ? formatCoord(state.destination) : placeholder;
  destination.className = state.destination ? 'endpoint-coord' : 'endpoint-coord empty';

  $('clear').disabled = !state.origin && !state.destination;

  const hint = $('map-hint');
  if (!state.origin) {
    hint.innerHTML = '<b>Search</b> an address or <b>click</b> to set the origin';
    hint.classList.remove('hidden');
  } else if (!state.destination) {
    hint.innerHTML = '<b>Click again</b> to set the destination';
    hint.classList.remove('hidden');
  } else {
    hint.classList.add('hidden');
  }
}

async function requestRoute() {
  if (!state.origin || !state.destination) return;

  const body = {
    origin_lat: state.origin.lat,
    origin_lon: state.origin.lon,
    dest_lat: state.destination.lat,
    dest_lon: state.destination.lon,
    preferences: readPreferences(),
  };
  if (state.lead > 0) {
    body.departure_time = new Date(Date.now() + state.lead * 60000).toISOString();
  }

  setBusy(true);
  hideError();
  try {
    // Wall-clock round trip, which is the only latency the user actually
    // waits on; the server reports its own two numbers inside the payload.
    const started = performance.now();
    const route = await postJSON('/api/v1/route', body);
    route.round_trip_ms = performance.now() - started;

    state.route = route;
    renderMetrics();
    renderPreferenceNote();
    renderLayers();
    fitToRoute(route);
  } catch (error) {
    state.route = null;
    showError(error.message);
    renderMetrics();
    renderPreferenceNote();
    renderLayers();
  } finally {
    setBusy(false);
  }
}

function fitToRoute(route) {
  const points = route.dynamic_route.coordinates.concat(route.baseline_route.coordinates);
  if (points.length < 2) return;

  const bounds = new state.GL.LngLatBounds();
  points.forEach(([lat, lon]) => bounds.extend([lon, lat]));
  state.map.fitBounds(bounds, { padding: 90, duration: 700, maxZoom: 15 });
}

function setBusy(busy) {
  state.busy = busy;
  $('metrics').classList.toggle('busy', busy);
}

// ---------------------------------------------------------------------------
// Metrics panel
// ---------------------------------------------------------------------------
function renderMetrics() {
  const container = $('metrics');
  const route = state.route;

  if (!route) {
    container.innerHTML =
      '<div class="placeholder">Search for an address or drop an origin and a ' +
      'destination on the map to compare the AI route against a free-flow ' +
      'baseline.</div>';
    return;
  }

  const dynamic = route.dynamic_route;
  const baseline = route.baseline_route;
  const saved = route.time_saved_sec;
  const percent = baseline.duration_sec > 0 ? (saved / baseline.duration_sec) * 100 : 0;
  const meaningful = saved >= 1;

  const savedValue = meaningful ? `+${formatDuration(saved)}` : 'No gain';
  const savedNote = meaningful
    ? `${percent.toFixed(1)}% faster than the baseline path`
    : route.identical_path
      ? 'Both routers chose the same path'
      : 'The baseline path is already optimal under traffic';

  container.innerHTML = `
    <div class="card hero">
      <div class="k">AI travel time</div>
      <div class="v">${formatDuration(dynamic.duration_sec)}</div>
      <div class="note">
        ${(dynamic.distance_meters / 1000).toFixed(2)} km ·
        forecast +${route.horizon_minutes} min
      </div>
    </div>

    <div class="card saved${meaningful ? '' : ' none'}">
      <div class="k">Time saved</div>
      <div class="v">${savedValue}</div>
      <div class="note">${savedNote}</div>
    </div>

    <div class="card">
      <div class="k">Baseline (speed limits)</div>
      <div class="v" style="font-size: 17px">${formatDuration(baseline.duration_sec)}</div>
      <div class="note">
        ${(baseline.distance_meters / 1000).toFixed(2)} km ·
        ${formatDuration(baseline.free_flow_duration_sec)} if the road were empty
      </div>
    </div>

    <div class="latency" style="margin-top: 8px">
      <div class="card">
        <div class="k">Service boundary</div>
        <div class="v">${formatLatency(route.compute_latency_ms)}</div>
        <div class="note">A* on a pre-updated graph</div>
      </div>
      <div class="card">
        <div class="k">HTTP end-to-end</div>
        <div class="v">${formatLatency(route.query_latency_ms)}</div>
        <div class="note">${formatLatency(route.round_trip_ms)} round trip</div>
      </div>
    </div>

    <div class="card">
      <div class="k">Search</div>
      <div class="note" style="margin-top: 6px">
        ${dynamic.nodes_expanded.toLocaleString()} nodes expanded ·
        snapped ${route.origin.distance_m.toFixed(0)} m and
        ${route.destination.distance_m.toFixed(0)} m to the graph
      </div>
    </div>`;
}

function showError(message) {
  const box = $('error');
  box.textContent = message;
  box.style.display = 'block';
}

function hideError() {
  $('error').style.display = 'none';
}

// ---------------------------------------------------------------------------
// Address search
// ---------------------------------------------------------------------------
/**
 * Two comboboxes over GET /api/v1/geocode.
 *
 * Three things make this feel right rather than merely work:
 *
 * **Debounce, then a ticket.** The 300 ms timer collapses a burst of
 * keystrokes into one request, but it cannot stop two in-flight lookups from
 * landing out of order — the network decides that, not the keyboard. Every
 * request takes a monotonic ticket and a response that is not the newest one
 * is dropped, so the dropdown can never show results for a prefix the user has
 * already typed past.
 *
 * **Selection and map clicks share one path.** `setEndpoint` is the only place
 * an endpoint is written, whichever way the user chose it, so the marker, the
 * panel and the automatic route request cannot disagree.
 *
 * **Failures render into the dropdown.** A geocoding outage is local to this
 * widget; putting it in the page-level error banner would suggest the router
 * was down too, when routing by map click still works perfectly.
 */
function initSearch() {
  Object.entries(SEARCH_FIELDS).forEach(([key, ids]) => setupSearchField(key, ids));

  // Click-outside. Capture-phase would fire before the suggestion's own
  // handler, so this listens in the bubble phase and ignores anything that
  // came from inside a search box.
  document.addEventListener('click', (event) => {
    if (!event.target.closest || !event.target.closest('.search-box')) {
      closeAllSuggestions();
    }
  });
}

function setupSearchField(key, ids) {
  const field = {
    key,
    input: $(ids.input),
    box: $(ids.box),
    list: $(ids.results),
    color: ids.color,
    timer: null,
    ticket: 0,      // newest request; stale responses compare unequal
    items: [],
    active: -1,     // keyboard cursor into `items`
  };
  state.search[key] = field;

  field.input.addEventListener('input', () => {
    clearTimeout(field.timer);
    const query = field.input.value.trim();

    if (query.length < SEARCH_MIN_CHARS) {
      // Also void any in-flight lookup: its results are no longer wanted.
      field.ticket += 1;
      setSearchLoading(field, false);
      closeSuggestions(field);
      return;
    }
    field.timer = setTimeout(() => runSearch(field, query), SEARCH_DEBOUNCE_MS);
  });

  field.input.addEventListener('keydown', (event) => onSearchKeydown(field, event));
  field.input.addEventListener('focus', () => {
    if (field.items.length) openSuggestions(field);
  });

  // Delegated so re-rendering the list costs no listener bookkeeping.
  field.list.addEventListener('click', (event) => {
    const item = event.target.closest('.item');
    if (item) selectSuggestion(field, Number(item.dataset.index));
  });
}

async function runSearch(field, query) {
  const ticket = (field.ticket += 1);
  setSearchLoading(field, true);

  try {
    const matches = await getJSON(`/api/v1/geocode?q=${encodeURIComponent(query)}`);
    if (ticket !== field.ticket) return;   // a later keystroke already won
    field.items = Array.isArray(matches) ? matches : [];
    field.active = -1;
    renderSuggestions(field);
  } catch (error) {
    if (ticket !== field.ticket) return;
    field.items = [];
    field.active = -1;
    renderSuggestionNote(
      field,
      'Address search is unavailable — click the map to set this point.',
      true
    );
  } finally {
    if (ticket === field.ticket) setSearchLoading(field, false);
  }
}

function renderSuggestions(field) {
  if (!field.items.length) {
    renderSuggestionNote(field, 'No matches in the Irvine area.', false);
    return;
  }

  // Nominatim's display_name is a comma-separated hierarchy, most specific
  // first: the head is the place, the tail is the address around it.
  field.list.innerHTML = field.items.map((match, index) => {
    const parts = String(match.display_name).split(',');
    const primary = parts[0].trim();
    const secondary = parts.slice(1).join(',').trim();
    return `
      <div class="item" role="option" aria-selected="false" data-index="${index}">
        <div class="primary">${escapeHTML(primary)}</div>
        ${secondary ? `<div class="secondary">${escapeHTML(secondary)}</div>` : ''}
      </div>`;
  }).join('');

  openSuggestions(field);
}

function renderSuggestionNote(field, message, isError) {
  field.list.innerHTML =
    `<div class="note${isError ? ' error' : ''}">${escapeHTML(message)}</div>`;
  openSuggestions(field);
}

function onSearchKeydown(field, event) {
  const open = field.list.classList.contains('open');

  switch (event.key) {
    case 'ArrowDown':
    case 'ArrowUp': {
      if (!open || !field.items.length) return;
      event.preventDefault();
      const step = event.key === 'ArrowDown' ? 1 : -1;
      const count = field.items.length;
      field.active = (field.active + step + count) % count;
      highlightSuggestion(field);
      break;
    }
    case 'Enter':
      if (open && field.active >= 0) {
        event.preventDefault();
        selectSuggestion(field, field.active);
      }
      break;
    case 'Escape':
      closeSuggestions(field);
      break;
    default:
      break;
  }
}

function highlightSuggestion(field) {
  Array.from(field.list.querySelectorAll('.item')).forEach((item, index) => {
    const active = index === field.active;
    item.classList.toggle('active', active);
    item.setAttribute('aria-selected', String(active));
    if (active) item.scrollIntoView({ block: 'nearest' });
  });
}

function selectSuggestion(field, index) {
  const match = field.items[index];
  if (!match) return;

  field.input.value = match.display_name;
  // Drop the list as well as hiding it, so refocusing the field does not
  // reopen suggestions for a search the user has already finished.
  field.items = [];
  closeSuggestions(field);
  setEndpoint(field.key, { lat: match.lat, lon: match.lon }, { recenter: true });
}

function setSearchLoading(field, loading) {
  field.box.classList.toggle('loading', loading);
}

function openSuggestions(field) {
  field.list.classList.add('open');
  field.input.setAttribute('aria-expanded', 'true');
}

function closeSuggestions(field) {
  field.list.classList.remove('open');
  field.input.setAttribute('aria-expanded', 'false');
  field.active = -1;
}

function closeAllSuggestions() {
  Object.values(state.search).forEach(closeSuggestions);
}

/** Clear a field's text without disturbing its endpoint or firing a lookup. */
function resetSearchInput(key) {
  const field = state.search[key];
  if (!field) return;
  clearTimeout(field.timer);
  field.ticket += 1;
  field.items = [];
  field.input.value = '';
  setSearchLoading(field, false);
  closeSuggestions(field);
}

// ---------------------------------------------------------------------------
// Routing preferences
// ---------------------------------------------------------------------------
/**
 * The two toggles, wired straight to the router.
 *
 * Flipping one re-runs the existing query rather than asking the user to drop
 * their pins again: the endpoints have not changed, only what the router is
 * being asked to optimise for. With no route on the map yet there is nothing
 * to recompute, so the state is simply remembered for the next request.
 */
function initPreferences() {
  const toggles = Array.from(document.querySelectorAll('[data-preference]'));

  toggles.forEach((toggle) => {
    toggle.addEventListener('change', async () => {
      readPreferences();
      renderPreferenceNote();
      if (state.origin && state.destination) await requestRoute();
    });
  });

  readPreferences();
  renderPreferenceNote();
}

/** Pull the current toggle states out of the DOM into `state.preferences`. */
function readPreferences() {
  document.querySelectorAll('[data-preference]').forEach((toggle) => {
    state.preferences[toggle.dataset.preference] = toggle.checked;
  });
  return state.preferences;
}

function anyPreference() {
  return Object.values(state.preferences).some(Boolean);
}

/**
 * What the preference did to the route that is actually on screen.
 *
 * Three states worth distinguishing, because "nothing happened" and "it tried
 * and could not" look identical otherwise: nothing asked for, asked and the
 * route came back clean, or asked and the route still has to make one.
 *
 * Deliberately *not* reported here: how much time the preference cost. The
 * server routes once, so the fast route this was measured against does not
 * exist to compare with — and `time_saved_sec` is the traffic model's number,
 * not the preference's. Quoting it as the price of avoidance would be a
 * plausible-looking figure for something nobody computed.
 */
function renderPreferenceNote() {
  const note = $('pref-note');
  const route = state.route;

  if (!anyPreference()) {
    note.textContent = 'Routing for the fastest trip.';
    note.className = '';
    return;
  }

  note.className = 'active';
  if (!route) {
    note.textContent = 'Applied when a route is calculated.';
    return;
  }

  note.innerHTML = (route.preference_penalty_sec || 0) > 0
    ? '<b>Partly avoided.</b> No route between these points skips all of them.'
    : '<b>Avoided.</b> Nothing flagged on this route.';
}

// ---------------------------------------------------------------------------
// Departure selector
// ---------------------------------------------------------------------------
function initDepartureSelector() {
  const buttons = Array.from(document.querySelectorAll('#departure button'));

  buttons.forEach((button) => {
    button.addEventListener('click', async () => {
      const lead = Number(button.dataset.lead);
      if (lead === state.lead) return;

      state.lead = lead;
      buttons.forEach((b) => b.setAttribute('aria-pressed', String(b === button)));
      renderHorizonNote();

      // The congestion layer and the route both key off the horizon.
      await loadTrafficState().catch((error) => showError(error.message));
      if (state.origin && state.destination) await requestRoute();
    });
  });

  renderHorizonNote();
}

function renderHorizonNote() {
  const horizons = [15, 30, 60];
  const horizon = horizons[HORIZON_FOR_LEAD[state.lead]];
  // "Now" is the only selection the model cannot serve literally: its nearest
  // forecast is +15 min. The others land on a horizon exactly, so they need
  // no caveat.
  const note = state.lead === 0
    ? `Scored against the +${horizon} min forecast — the nearest the model produces.`
    : `Scored against the +${horizon} min forecast.`;
  $('horizon-note').textContent = note;
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
async function boot() {
  $('clear').addEventListener('click', clearRoute);
  initSearch();
  initPreferences();
  initDepartureSelector();
  renderEndpoints();

  let config;
  try {
    config = await getJSON('/api/v1/config');
    state.config = config;
  } catch (error) {
    setStatus('bad', 'backend unreachable');
    showError(`Could not reach the API: ${error.message}`);
    return;
  }

  // The two halves are independent: the basemap comes from a CDN, the data
  // comes from this server. Run them concurrently so neither waits on the
  // other, and let the map fail on its own without blanking the panels.
  const dataReady = (async () => {
    try {
      await pollHealth();
      if (!state.segments.length) await loadTrafficState();
    } catch (error) {
      showError(error.message);
    }
  })();

  const mapReady = initMap(config).catch((error) => {
    setStatus('bad', 'map failed to load');
    showError(`Could not initialise the map: ${error.message}`);
  });

  await Promise.all([dataReady, mapReady]);
  renderLayers();   // whichever finished first left this for the other

  setInterval(pollHealth, HEALTH_POLL_MS);
}

document.addEventListener('DOMContentLoaded', boot);
