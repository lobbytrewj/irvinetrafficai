"""Runtime configuration for the routing service.

Everything the service needs to find on disk, plus the two numbers that
define its behaviour under load: how often the forecaster re-scores the
network, and how far a query is allowed to be from the graph before it is
rejected rather than silently snapped to a road on the other side of town.

Overridable from the environment with an ``IRVINE_`` prefix, so a container
can be pointed at a different artifact directory without a rebuild.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Settings"]

# Repository root: app/config.py -> app/ -> <root>.
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
)


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser() if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if not raw:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Where the artifacts live and how the service behaves."""

    processed_dir: Path = field(default_factory=lambda: REPO_ROOT / "data" / "processed")

    # Update cadence of the forecaster. The dataset is sampled every five
    # minutes and the model is trained on that step, so a faster loop would
    # re-run the same inference on the same inputs.
    update_interval_sec: float = 300.0

    # A query further than this from any intersection is off-network. Irvine's
    # graph has a median inter-node spacing well under 200 m, so a kilometre is
    # generous for "the user tapped a parking lot" and tight enough to reject
    # "the user is in Los Angeles".
    max_snap_distance_m: float = 1000.0

    # Fail startup if an artifact is missing, rather than serving a degraded
    # graph. Set false for smoke-testing without the 92 MB dataset bundle.
    strict_artifacts: bool = True

    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS

    # Mapbox GL JS refuses to construct a map without an access token, so the
    # dashboard falls back to a keyless basemap when this is empty. Kept out
    # of the HTML and handed to the client by GET /api/v1/config, so rotating
    # it is an environment change rather than an edit to a static asset.
    mapbox_token: str = ""
    map_style: str = "mapbox://styles/mapbox/dark-v11"

    @property
    def static_dir(self) -> Path:
        return Path(__file__).resolve().parent / "static"

    @property
    def onnx_path(self) -> Path:
        return self.processed_dir / "traffic_stgnn.onnx"

    @property
    def geopackage_path(self) -> Path:
        return self.processed_dir / "irvine_graph.gpkg"

    @property
    def node_mapping_path(self) -> Path:
        return self.processed_dir / "node_mapping.json"

    @property
    def graph_tensors_path(self) -> Path:
        return self.processed_dir / "graph_tensors.pt"

    @property
    def dataset_path(self) -> Path:
        return self.processed_dir / "traffic_dataset.pt"

    @classmethod
    def from_env(cls) -> "Settings":
        default = cls()
        origins = os.getenv("IRVINE_CORS_ORIGINS")
        return cls(
            processed_dir=_env_path("IRVINE_PROCESSED_DIR", default.processed_dir),
            update_interval_sec=_env_float(
                "IRVINE_UPDATE_INTERVAL_SEC", default.update_interval_sec
            ),
            max_snap_distance_m=_env_float(
                "IRVINE_MAX_SNAP_DISTANCE_M", default.max_snap_distance_m
            ),
            strict_artifacts=_env_bool(
                "IRVINE_STRICT_ARTIFACTS", default.strict_artifacts
            ),
            cors_origins=(
                tuple(o.strip() for o in origins.split(",") if o.strip())
                if origins
                else default.cors_origins
            ),
            # MAPBOX_TOKEN is the name the Mapbox tooling already uses, so it
            # is accepted alongside the prefixed form.
            mapbox_token=(
                os.getenv("IRVINE_MAPBOX_TOKEN") or os.getenv("MAPBOX_TOKEN") or ""
            ).strip(),
            map_style=os.getenv("IRVINE_MAP_STYLE", default.map_style),
        )
