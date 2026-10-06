"""Unit tests for src.data.generate_traffic_data.

These run entirely on the synthetic corridor topology, so nothing here needs
``data/processed`` to exist or touches OSM.

Two things carry real risk in this module and get most of the attention:

* the **split logic** -- a windowing bug that lets a training sample read a
  step from the validation range produces a model that scores beautifully and
  forecasts nothing;
* the **simulator's structure** -- if the generated series has no diurnal
  shape, no direction asymmetry and no spatial correlation, then a model can
  score well on it while learning none of the things it exists to learn, and
  the whole dataset is a decoration.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.generate_traffic_data import (
    FEATURE_NAMES,
    IRVINE_CORRIDORS,
    MIN_SPEED_KPH,
    CorridorSpec,
    TrafficDataset,
    TrafficGenConfig,
    build_dataset,
    build_segment_graph,
    direction_weights,
    load_bundle,
    save_dataset,
    synthetic_segment_table,
    _tokens,
)

DAYS = 6


@pytest.fixture(scope="module")
def config() -> TrafficGenConfig:
    return TrafficGenConfig(days=DAYS, seed=3, allow_synthetic_topology=True)


@pytest.fixture(scope="module")
def bundle(config: TrafficGenConfig) -> dict:
    # The synthetic topology keeps this fast; the OSM path is exercised by
    # running the CLI, not by the unit suite.
    return build_dataset(config)


# ---------------------------------------------------------------------------
# Tag matching and corridor selection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cell,expected",
    [
        ("I 405", {"I 405"}),
        ("", set()),
        (None, set()),
        ("nan", set()),
        (["I 5", "I 405"], {"I 5", "I 405"}),
        # OSMnx multi-valued tags survive the GeoPackage round-trip as the
        # str() of a Python list.
        ("['I 5', 'I 405']", {"I 5", "I 405"}),
        ('["Culver Drive"]', {"Culver Drive"}),
    ],
)
def test_tokens_normalises_osm_tag_cells(cell, expected) -> None:
    assert _tokens(cell) == expected


def test_corridor_matches_ref_or_name() -> None:
    spec = CorridorSpec(key="X", refs=("I 405",), names=("Culver Drive",))
    assert spec.matches("I 405", "")
    assert spec.matches("", "Culver Drive")
    assert spec.matches("['I 5', 'I 405']", "")
    assert not spec.matches("I 5", "Jamboree Road")


def test_all_five_corridors_are_represented() -> None:
    keys = [spec.key for spec in IRVINE_CORRIDORS]
    assert keys == ["I-405", "I-5", "Jamboree Rd", "Culver Dr", "Alton Pkwy"]


# ---------------------------------------------------------------------------
# Segment graph
# ---------------------------------------------------------------------------
def test_segment_graph_is_symmetric_and_in_range() -> None:
    table = synthetic_segment_table(segments_per_direction=5)
    downstream, undirected = build_segment_graph(table)

    assert undirected.shape[0] == 2
    assert undirected.min() >= 0 and undirected.max() < len(table)

    pairs = {(int(u), int(v)) for u, v in undirected.T}
    assert all((v, u) in pairs for u, v in pairs)
    # Every directed link must appear in the symmetric closure.
    assert all((int(u), int(v)) in pairs for u, v in downstream.T)


def test_downstream_links_follow_the_direction_of_travel() -> None:
    table = synthetic_segment_table(segments_per_direction=4)
    downstream, _ = build_segment_graph(table)
    for source, target in downstream.T:
        assert table.v[source] == table.u[target]


def test_direction_weights_are_opposed() -> None:
    """Inbound and outbound segments must load different peaks."""
    inbound, outbound = 315.0, 135.0
    am, pm = direction_weights(np.array([inbound, outbound]))
    assert am[0] > am[1]  # AM loads the inbound direction
    assert pm[1] > pm[0]  # PM loads the outbound one
    assert float(am.min()) >= 0.25  # counter-peak never goes free-flow


# ---------------------------------------------------------------------------
# Simulated series
# ---------------------------------------------------------------------------
def test_bundle_shapes_and_protocol(bundle: dict, config: TrafficGenConfig) -> None:
    num_nodes, num_steps = bundle["num_nodes"], bundle["num_steps"]
    assert num_steps == DAYS * 288
    assert bundle["speed_kph"].shape == (num_steps, num_nodes)
    assert bundle["travel_time_s"].shape == (num_steps, num_nodes)
    assert bundle["feature_mean"].shape == (num_nodes, len(FEATURE_NAMES))
    assert bundle["target_mean"].shape == (num_nodes,)
    assert bundle["feature_names"] == list(FEATURE_NAMES)
    assert bundle["horizons"] == list(config.horizons)


def test_speeds_are_physical(bundle: dict) -> None:
    speed = bundle["speed_kph"]
    free_flow = bundle["segment_free_flow_kph"]
    assert torch.isfinite(speed).all()
    assert float(speed.min()) >= MIN_SPEED_KPH - 1e-3
    # Congestion only ever removes speed; nothing exceeds free flow.
    assert bool((speed <= free_flow.unsqueeze(0) + 1e-3).all())
    assert torch.isfinite(bundle["travel_time_s"]).all()
    assert float(bundle["travel_time_s"].min()) > 0.0


def test_series_has_a_rush_hour(bundle: dict) -> None:
    """Peak-hour speeds must be materially below the overnight baseline."""
    speed = bundle["speed_kph"].numpy().reshape(DAYS, 288, -1)
    weekday = speed[:5]  # the config starts on a Monday
    by_hour = weekday.mean(axis=(0, 2)).reshape(24, 12).mean(axis=1)

    overnight = by_hour[2:5].mean()
    am_peak, pm_peak = by_hour[7:9].mean(), by_hour[16:18].mean()
    assert am_peak < 0.85 * overnight
    assert pm_peak < 0.85 * overnight


def test_weekends_are_faster_than_weekdays(bundle: dict) -> None:
    speed = bundle["speed_kph"].numpy()
    is_weekend = bundle["is_weekend"].numpy().astype(bool)
    assert speed[is_weekend].mean() > speed[~is_weekend].mean()


def test_commute_directions_peak_at_different_times(bundle: dict) -> None:
    """AM and PM peaks must fall on opposite carriageways."""
    speed = bundle["speed_kph"].numpy().reshape(DAYS, 288, -1)[:5]
    bearing = bundle["segment_bearing_deg"].numpy()
    inbound = np.cos(np.radians(bearing - 315.0)) > 0.5
    outbound = np.cos(np.radians(bearing - 315.0)) < -0.5
    assert inbound.any() and outbound.any()

    def hourly(mask):
        return speed[:, :, mask].mean(axis=(0, 2)).reshape(24, 12).mean(axis=1)

    inbound_hours, outbound_hours = hourly(inbound), hourly(outbound)
    # Inbound is slowest in the morning, outbound in the evening.
    assert int(inbound_hours.argmin()) < 12
    assert int(outbound_hours.argmin()) >= 12


def test_neighbours_are_more_correlated_than_random_pairs(bundle: dict) -> None:
    """Spatial structure is the only reason to reach for a graph model."""
    speed = bundle["speed_kph"].numpy()
    z = (speed - speed.mean(axis=0)) / (speed.std(axis=0) + 1e-9)
    edge_index = bundle["edge_index"].numpy()

    neighbour = (z[:, edge_index[0]] * z[:, edge_index[1]]).mean()
    rng = np.random.default_rng(0)
    left = rng.integers(0, speed.shape[1], 4000)
    right = rng.integers(0, speed.shape[1], 4000)
    random_pair = (z[:, left] * z[:, right]).mean()
    assert neighbour > random_pair + 0.05


def test_generation_is_deterministic(config: TrafficGenConfig) -> None:
    first = build_dataset(config)["speed_kph"]
    second = build_dataset(config)["speed_kph"]
    assert torch.equal(first, second)


# ---------------------------------------------------------------------------
# Splits and windowing
# ---------------------------------------------------------------------------
def test_splits_are_chronological_and_disjoint(bundle: dict) -> None:
    train, val, test = (bundle["splits"][s] for s in ("train", "val", "test"))
    assert train[0] == 0
    assert train[1] == val[0]
    assert val[1] == test[0]
    assert test[1] == bundle["num_steps"]


def test_windows_never_cross_a_split_boundary(bundle: dict) -> None:
    """The one bug that would invalidate every reported number."""
    span = int(bundle["seq_len"]) + max(int(h) for h in bundle["horizons"])
    for split, (start, end) in bundle["splits"].items():
        dataset = TrafficDataset(bundle, split=split)
        assert int(dataset.starts.min()) >= start
        assert int(dataset.starts.max()) + span <= end


def test_scalers_are_fitted_on_the_training_split_only(bundle: dict) -> None:
    train_end = bundle["splits"]["train"][1]
    expected = bundle["travel_time_s"][:train_end].mean(dim=0)
    assert torch.allclose(bundle["target_mean"], expected, atol=1e-3)
    # Statistics over the whole series would differ; confirm they do, or the
    # assertion above proves nothing.
    assert not torch.allclose(
        bundle["target_mean"], bundle["travel_time_s"].mean(dim=0), atol=1e-3
    )


def test_sample_shapes_and_target_alignment(bundle: dict) -> None:
    dataset = TrafficDataset(bundle, split="train", normalise=False)
    x, y = dataset[0]
    num_nodes, seq_len = int(bundle["num_nodes"]), int(bundle["seq_len"])
    horizons = [int(h) for h in bundle["horizons"]]

    assert x.shape == (num_nodes, len(FEATURE_NAMES), seq_len)
    assert y.shape == (num_nodes, len(horizons))

    start = int(dataset.starts[0])
    # Channel 1 is travel time; its last input step is the window's end.
    assert torch.allclose(x[:, 1, -1], bundle["travel_time_s"][start + seq_len - 1])
    for i, horizon in enumerate(horizons):
        expected = bundle["travel_time_s"][start + seq_len - 1 + horizon]
        assert torch.allclose(y[:, i], expected)


def test_normalisation_is_invertible(bundle: dict) -> None:
    raw = TrafficDataset(bundle, split="train", normalise=False)
    scaled = TrafficDataset(bundle, split="train", normalise=True)
    x_raw, y_raw = raw[5]
    x_scaled, y_scaled = scaled[5]

    # feature_mean/std are [N, F]; a sample is [N, F, L].
    recovered = x_scaled * bundle["feature_std"].unsqueeze(-1) + bundle[
        "feature_mean"
    ].unsqueeze(-1)
    assert torch.allclose(recovered, x_raw, atol=1e-2)

    recovered_y = y_scaled * bundle["target_std"].unsqueeze(-1) + bundle[
        "target_mean"
    ].unsqueeze(-1)
    assert torch.allclose(recovered_y, y_raw, atol=1e-2)


def test_calendar_channels_are_left_unscaled(bundle: dict) -> None:
    """The cyclical encoding is already O(1); scaling would distort it."""
    for channel in (2, 3, 4):
        assert torch.allclose(
            bundle["feature_mean"][:, channel], torch.zeros(bundle["num_nodes"])
        )
        assert torch.allclose(
            bundle["feature_std"][:, channel], torch.ones(bundle["num_nodes"])
        )


def test_stride_subsamples_windows(bundle: dict) -> None:
    full = TrafficDataset(bundle, split="train")
    strided = TrafficDataset(bundle, split="train", stride=4)
    assert len(strided) == (len(full) + 3) // 4
    assert torch.equal(strided.starts, full.starts[::4])
    assert torch.allclose(strided[1][0], full[4][0])


def test_unknown_split_is_rejected(bundle: dict) -> None:
    with pytest.raises(KeyError):
        TrafficDataset(bundle, split="holdout")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def test_bundle_round_trips_through_disk(bundle: dict, tmp_path) -> None:
    path = save_dataset(bundle, tmp_path)
    # weights_only=True must be able to read it back: everything in the
    # bundle has to stay a tensor or a plain primitive.
    restored = load_bundle(path)
    assert torch.equal(restored["speed_kph"], bundle["speed_kph"])
    assert restored["corridor_names"] == bundle["corridor_names"]
    assert restored["splits"] == bundle["splits"]


def test_missing_dataset_reports_the_fix(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="generate_traffic_data"):
        load_bundle(tmp_path / "absent.pt")
