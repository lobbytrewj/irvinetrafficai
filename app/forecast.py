"""ST-GNN inference: recent observations in, per-segment travel time out.

Three pieces:

``SegmentIndex``
    The 444 corridor segments the model predicts on, and -- critically -- the
    ``edge_id`` each one occupies in the 15,553-edge router graph. That column
    is what lets a ``[444]`` prediction vector become a subset weight update.

``TrafficFeed``
    Where the last twelve five-minute observations come from. Production would
    put a loop-detector or probe feed behind this interface. This project has
    no live sensor, so :class:`DatasetReplayFeed` replays the synthetic series
    the model was trained on, aligned to wall clock by weekday and time of
    day: at 08:30 on a Tuesday the service sees a Tuesday-morning peak.

``TrafficForecaster``
    Assembles the ``[1, 444, 5, 12]`` feature window and runs the ONNX graph.
    The export wraps the model in ``STGNNRuntime``, so the scalers and the
    free-flow clamp are already baked into the graph: features go in raw and
    seconds come out, with no normalisation constants to keep in sync here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

__all__ = [
    "SegmentIndex",
    "TrafficFeed",
    "DatasetReplayFeed",
    "FreeFlowFeed",
    "TrafficForecaster",
    "Prediction",
]

LOGGER = logging.getLogger(__name__)

# Channel order of the ONNX input. Must match FEATURE_NAMES in
# src/data/generate_traffic_data.py -- the model has no way to notice a swap.
FEATURE_NAMES: tuple[str, ...] = (
    "speed_kph",
    "travel_time_s",
    "tod_sin",
    "tod_cos",
    "is_weekend",
)

STEP_MINUTES = 5
STEPS_PER_DAY = (24 * 60) // STEP_MINUTES
STEPS_PER_WEEK = STEPS_PER_DAY * 7


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SegmentIndex:
    """The corridor segments the forecaster covers.

    ``edge_id[i]`` is the router-graph edge that model node ``i`` predicts,
    in the ``(u, v, key)``-sorted convention shared by ``graph_tensors.pt``.
    """

    edge_id: np.ndarray  # [N] int32, indices into the 15,553-edge graph
    length_m: np.ndarray  # [N] float64
    free_flow_s: np.ndarray  # [N] float32
    corridor_id: np.ndarray  # [N] int64
    corridor_names: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.edge_id.shape[0])

    def corridor_of(self, i: int) -> str:
        cid = int(self.corridor_id[i])
        return self.corridor_names[cid] if 0 <= cid < len(self.corridor_names) else "other"

    def validate_against(self, num_edges: int) -> None:
        if self.edge_id.size and (
            self.edge_id.min() < 0 or self.edge_id.max() >= num_edges
        ):
            raise ValueError(
                f"segment edge ids fall outside [0, {num_edges}); the dataset "
                "and the graph came from different graph_builder runs"
            )


# ---------------------------------------------------------------------------
# Observation feeds
# ---------------------------------------------------------------------------
class TrafficFeed(Protocol):
    """Source of recent per-segment observations."""

    name: str

    def window(self, end: datetime, steps: int) -> np.ndarray:
        """``[steps, N, 2]`` of (speed_kph, travel_time_s), oldest step first.

        The last row is the observation for ``end``.
        """
        ...


class DatasetReplayFeed:
    """Replays ``traffic_dataset.pt``, aligned to wall clock.

    The bundle is 90 days of five-minute steps starting Monday 2025-01-06
    00:00 UTC. A wall-clock instant maps to the step with the same weekday and
    time of day, cycling through the twelve whole weeks the bundle contains so
    consecutive real weeks do not replay identical traffic.

    Alignment matters more than recency: the model's strongest signal is the
    daily demand profile, so serving a Monday-morning peak at Monday morning
    is what makes the predictions look like traffic rather than noise.
    """

    name = "dataset-replay"

    def __init__(self, speed_kph: np.ndarray, travel_time_s: np.ndarray) -> None:
        if speed_kph.shape != travel_time_s.shape:
            raise ValueError(
                f"speed {speed_kph.shape} and travel time {travel_time_s.shape} "
                "must have the same shape"
            )
        self.speed_kph = speed_kph
        self.travel_time_s = travel_time_s
        self.num_steps = int(speed_kph.shape[0])
        self.num_segments = int(speed_kph.shape[1])
        self.whole_weeks = max(1, self.num_steps // STEPS_PER_WEEK)

    def step_for(self, moment: datetime) -> int:
        """Index of the replayed step matching ``moment``'s weekday and time."""
        moment = moment.astimezone(timezone.utc)
        step_of_day = (moment.hour * 60 + moment.minute) // STEP_MINUTES
        # Day 0 of the bundle is a Monday, so weekday() is the day offset.
        week = (moment.toordinal() // 7) % self.whole_weeks
        index = week * STEPS_PER_WEEK + moment.weekday() * STEPS_PER_DAY + step_of_day
        return min(index, self.num_steps - 1)

    def window(self, end: datetime, steps: int) -> np.ndarray:
        if steps > self.num_steps:
            raise ValueError(
                f"feed holds {self.num_steps} steps, too few for a "
                f"{steps}-step window"
            )

        stop = self.step_for(end)
        # Shift forward whole weeks if the window would run off the front;
        # a week offset preserves the weekday/time-of-day alignment.
        while stop < steps - 1:
            stop += STEPS_PER_WEEK
        stop = min(stop, self.num_steps - 1)
        start = stop - steps + 1

        return np.stack(
            [self.speed_kph[start : stop + 1], self.travel_time_s[start : stop + 1]],
            axis=-1,
        )


class FreeFlowFeed:
    """Fallback feed: every segment at its free-flow speed, forever.

    Used when ``traffic_dataset.pt`` is absent. The service stays up and every
    endpoint answers, but the forecaster is being shown an empty road, so the
    dynamic route will track the baseline. Reported through ``/health`` so the
    degradation is never silent.
    """

    name = "free-flow"

    def __init__(self, free_flow_s: np.ndarray, length_m: np.ndarray) -> None:
        speed = (length_m / np.maximum(free_flow_s, 1e-6)) * 3.6
        self._row = np.stack(
            [speed.astype(np.float32), free_flow_s.astype(np.float32)], axis=-1
        )

    def window(self, end: datetime, steps: int) -> np.ndarray:
        return np.repeat(self._row[None, :, :], steps, axis=0)


def load_segment_index(dataset_path: Path) -> tuple[SegmentIndex, DatasetReplayFeed]:
    """Read segment metadata and the replay series from ``traffic_dataset.pt``."""
    bundle = torch.load(dataset_path, weights_only=True)

    segments = SegmentIndex(
        edge_id=bundle["segment_edge_id"].numpy().astype(np.int32),
        length_m=bundle["segment_length_m"].numpy().astype(np.float64),
        free_flow_s=bundle["segment_free_flow_time_s"].numpy().astype(np.float32),
        corridor_id=bundle["corridor_id"].numpy().astype(np.int64),
        corridor_names=tuple(str(n) for n in bundle["corridor_names"]),
    )
    feed = DatasetReplayFeed(
        speed_kph=bundle["speed_kph"].numpy(),
        travel_time_s=bundle["travel_time_s"].numpy(),
    )
    return segments, feed


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Prediction:
    """One forecast pass."""

    # [H, N] predicted travel time in seconds, one row per horizon.
    travel_time_s: np.ndarray
    # Lead time of each horizon, in minutes.
    horizon_minutes: tuple[int, ...]
    generated_at: datetime
    inference_ms: float


class TrafficForecaster:
    """ONNX ST-GNN wrapped in the feature assembly it expects."""

    def __init__(
        self,
        onnx_path: Path,
        segments: SegmentIndex,
        feed: TrafficFeed,
        intra_op_threads: int = 0,
    ) -> None:
        import onnxruntime as ort

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"{onnx_path} not found. Run `python -m src.pipelines.train` first."
            )

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads:
            options.intra_op_num_threads = intra_op_threads

        self.session = ort.InferenceSession(
            str(onnx_path), options, providers=["CPUExecutionProvider"]
        )
        self.segments = segments
        self.feed = feed

        spec = self.session.get_inputs()[0]
        self.input_name = spec.name
        # Shape is [batch, N, F, L] with a dynamic batch axis.
        _, num_nodes, num_features, seq_len = spec.shape
        self.num_nodes = int(num_nodes)
        self.num_features = int(num_features)
        self.seq_len = int(seq_len)
        self.output_name = self.session.get_outputs()[0].name
        self.num_horizons = int(self.session.get_outputs()[0].shape[-1])

        if self.num_features != len(FEATURE_NAMES):
            raise ValueError(
                f"model expects {self.num_features} features but this service "
                f"builds {len(FEATURE_NAMES)}: {FEATURE_NAMES}"
            )
        if self.num_nodes != len(segments):
            raise ValueError(
                f"model predicts {self.num_nodes} segments but the dataset "
                f"describes {len(segments)}"
            )

        # Horizons are [3, 6, 12] steps in the training config; recover the
        # lead times from the export rather than restating them.
        self.horizon_minutes = _horizon_minutes(self.num_horizons)

        LOGGER.info(
            "ONNX session ready: %s -> %s, %d segments, %d-step window, "
            "horizons %s min",
            self.input_name,
            self.output_name,
            self.num_nodes,
            self.seq_len,
            self.horizon_minutes,
        )

    def build_features(self, now: datetime) -> np.ndarray:
        """``[1, N, 5, L]`` raw features for a window ending at ``now``."""
        observations = self.feed.window(now, self.seq_len)  # [L, N, 2]
        if observations.shape != (self.seq_len, self.num_nodes, 2):
            raise ValueError(
                f"feed returned {observations.shape}, expected "
                f"{(self.seq_len, self.num_nodes, 2)}"
            )

        moments = [
            now - timedelta(minutes=STEP_MINUTES * (self.seq_len - 1 - i))
            for i in range(self.seq_len)
        ]
        calendar = _calendar_features(moments)  # [L, 3]

        features = np.empty(
            (1, self.num_nodes, self.num_features, self.seq_len), dtype=np.float32
        )
        # [L, N] -> [N, L] for the two observed channels.
        features[0, :, 0, :] = observations[:, :, 0].T
        features[0, :, 1, :] = observations[:, :, 1].T
        # Calendar channels are global: one value per step, shared by segments.
        for channel in range(3):
            features[0, :, 2 + channel, :] = calendar[:, channel]
        return features

    def predict(self, now: datetime | None = None) -> Prediction:
        """Run one forward pass. Blocking; call it off the event loop."""
        now = now or datetime.now(timezone.utc)
        features = self.build_features(now)

        started = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: features})
        inference_ms = (time.perf_counter() - started) * 1000.0

        # [1, N, H] -> [H, N]
        travel_time = np.ascontiguousarray(outputs[0][0].T, dtype=np.float32)
        # The export clamps at free flow, but a corrupt input should not be
        # able to hand the router a zero or negative edge weight.
        np.maximum(travel_time, 0.1, out=travel_time)

        return Prediction(
            travel_time_s=travel_time,
            horizon_minutes=self.horizon_minutes,
            generated_at=now,
            inference_ms=inference_ms,
        )


def _horizon_minutes(num_horizons: int) -> tuple[int, ...]:
    """Lead times for the exported horizons, in minutes.

    The training config forecasts 3, 6 and 12 steps ahead. Anything else is a
    re-trained model, for which evenly spaced steps are the honest guess.
    """
    default_steps = (3, 6, 12)
    steps = (
        default_steps
        if num_horizons == len(default_steps)
        else tuple(range(1, num_horizons + 1))
    )
    return tuple(s * STEP_MINUTES for s in steps)


def _calendar_features(moments: list[datetime]) -> np.ndarray:
    """``[L, 3]`` of (tod_sin, tod_cos, is_weekend), matching the training set.

    Time of day is a UTC fraction, as in ``calendar_features``; the simulator
    and the model agree on that convention, so the service has to as well.
    """
    out = np.empty((len(moments), 3), dtype=np.float32)
    for i, moment in enumerate(moments):
        moment = moment.astimezone(timezone.utc)
        fraction = (moment.hour + moment.minute / 60.0 + moment.second / 3600.0) / 24.0
        out[i, 0] = np.sin(2 * np.pi * fraction)
        out[i, 1] = np.cos(2 * np.pi * fraction)
        out[i, 2] = 1.0 if moment.weekday() >= 5 else 0.0
    return out
