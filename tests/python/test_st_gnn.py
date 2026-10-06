"""Unit tests for src.models.st_gnn.

The model ships two evaluation paths for each of its operators: the reference
``nn.Conv1d(dilation=d)`` / ``ChebConv`` implementations, and fused GEMM
rewrites that exist purely for speed. The rewrites are the whole risk surface
of this module -- they are hand-derived from the reference weight layouts, and
a transposed index would train and serve a subtly wrong model without ever
raising. Most of what follows pins them to the reference.

The other properties worth asserting are structural: causality (no output may
depend on a future input step) and the receptive field actually covering the
input window.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.st_gnn import (
    STGNN,
    ChebSpatialConv,
    GatedDilatedCausalConv,
    STGNNConfig,
    STGNNRuntime,
    count_parameters,
    rescaled_laplacian,
)

NUM_NODES = 12
SEQ_LEN = 12


def _ring_edge_index(num_nodes: int = NUM_NODES) -> torch.Tensor:
    """Symmetric ring: every node has exactly two neighbours."""
    forward = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
    pairs = forward + [(v, u) for u, v in forward]
    return torch.tensor(sorted(pairs), dtype=torch.long).t().contiguous()


@pytest.fixture
def edge_index() -> torch.Tensor:
    return _ring_edge_index()


@pytest.fixture
def config() -> STGNNConfig:
    return STGNNConfig(
        num_nodes=NUM_NODES,
        in_channels=5,
        hidden_channels=8,
        skip_channels=16,
        horizons=3,
        seq_len=SEQ_LEN,
    )


# ---------------------------------------------------------------------------
# Fused vs reference equivalence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dilation", [1, 2, 4, 8])
@pytest.mark.parametrize("kernel_size", [2, 3])
def test_temporal_fused_matches_conv1d(dilation: int, kernel_size: int) -> None:
    """The tap-GEMM rewrite must equal nn.Conv1d with the same weights."""
    torch.manual_seed(0)
    fused = GatedDilatedCausalConv(6, 4, kernel_size, dilation, fused=True)
    reference = GatedDilatedCausalConv(6, 4, kernel_size, dilation, fused=False)
    reference.load_state_dict(fused.state_dict())

    x = torch.randn(NUM_NODES, 3, SEQ_LEN, 6)
    with torch.no_grad():
        assert torch.allclose(fused(x), reference(x), atol=1e-5)


def test_temporal_preserves_sequence_length() -> None:
    conv = GatedDilatedCausalConv(4, 7, kernel_size=2, dilation=4)
    out = conv(torch.randn(NUM_NODES, 2, SEQ_LEN, 4))
    assert out.shape == (NUM_NODES, 2, SEQ_LEN, 7)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_spatial_fused_matches_chebconv(edge_index: torch.Tensor, k: int) -> None:
    """The cached-Laplacian recursion must equal PyG's ChebConv."""
    torch.manual_seed(0)
    fused = ChebSpatialConv(5, 3, k, NUM_NODES, edge_index, fused=True)
    reference = ChebSpatialConv(5, 3, k, NUM_NODES, edge_index, fused=False)
    reference.load_state_dict(fused.state_dict())

    x = torch.randn(NUM_NODES, 2, SEQ_LEN, 5)
    with torch.no_grad():
        assert torch.allclose(
            fused(x, edge_index), reference(x, edge_index), atol=1e-5
        )


def test_spatial_falls_back_above_dense_threshold(
    edge_index: torch.Tensor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A graph too large for a dense operator must not allocate one."""
    monkeypatch.setattr(ChebSpatialConv, "dense_max_nodes", 4)
    conv = ChebSpatialConv(2, 2, 3, NUM_NODES, edge_index, fused=True)
    assert conv.fused is False
    assert conv.laplacian is None


def test_rescaled_laplacian_is_symmetric_with_unit_spectrum(
    edge_index: torch.Tensor,
) -> None:
    """Chebyshev polynomials are only valid on eigenvalues in [-1, 1]."""
    laplacian = rescaled_laplacian(edge_index, NUM_NODES)
    assert torch.allclose(laplacian, laplacian.t(), atol=1e-6)
    eigenvalues = torch.linalg.eigvalsh(laplacian)
    assert float(eigenvalues.min()) >= -1.0 - 1e-5
    assert float(eigenvalues.max()) <= 1.0 + 1e-5


def test_model_fused_matches_reference(
    config: STGNNConfig, edge_index: torch.Tensor
) -> None:
    """End to end: the two paths are the same network."""
    torch.manual_seed(0)
    fused = STGNN(config, edge_index, fused=True).eval()
    reference = STGNN(config, edge_index, fused=False).eval()
    reference.load_state_dict(fused.state_dict())

    x = torch.randn(4, NUM_NODES, config.in_channels, SEQ_LEN)
    with torch.no_grad():
        assert torch.allclose(fused(x), reference(x), atol=1e-5)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------
def test_output_shape(config: STGNNConfig, edge_index: torch.Tensor) -> None:
    model = STGNN(config, edge_index).eval()
    with torch.no_grad():
        out = model(torch.randn(7, NUM_NODES, config.in_channels, SEQ_LEN))
    assert out.shape == (7, NUM_NODES, config.horizons)


@pytest.mark.parametrize("dilation", [1, 2, 4])
def test_temporal_conv_is_causal(dilation: int) -> None:
    """No output position may move when a *later* input step changes.

    This is the property left padding buys, and the one thing that would make
    the whole forecaster quietly cheat if it broke: a symmetric pad would let
    step ``t`` read step ``t+1``, and the reported errors would look excellent
    right up until the model met live data with no future to read.
    """
    torch.manual_seed(0)
    conv = GatedDilatedCausalConv(4, 4, kernel_size=2, dilation=dilation).eval()
    x = torch.randn(NUM_NODES, 2, SEQ_LEN, 4)
    step = 5

    with torch.no_grad():
        base = conv(x)
        perturbed = x.clone()
        perturbed[:, :, step] += 100.0
        out = conv(perturbed)

    assert torch.allclose(out[:, :, :step], base[:, :, :step], atol=1e-6)
    # ...and the perturbed step itself must matter, or the test is vacuous.
    assert not torch.allclose(out[:, :, step], base[:, :, step], atol=1e-6)


def test_spatial_conv_does_not_mix_time(edge_index: torch.Tensor) -> None:
    """The graph convolution mixes nodes only; time stays independent."""
    torch.manual_seed(0)
    conv = ChebSpatialConv(4, 4, 3, NUM_NODES, edge_index).eval()
    x = torch.randn(NUM_NODES, 2, SEQ_LEN, 4)
    step = 3

    with torch.no_grad():
        base = conv(x, edge_index)
        perturbed = x.clone()
        perturbed[:, :, step] += 100.0
        out = conv(perturbed, edge_index)

    untouched = [t for t in range(SEQ_LEN) if t != step]
    assert torch.allclose(out[:, :, untouched], base[:, :, untouched], atol=1e-5)


def test_receptive_field_covers_window(edge_index: torch.Tensor) -> None:
    config = STGNNConfig(num_nodes=NUM_NODES, dilations=(1, 2, 4, 8), seq_len=12)
    assert config.receptive_field == 16

    model = STGNN(config, edge_index).eval()
    x = torch.randn(1, NUM_NODES, config.in_channels, config.seq_len)
    with torch.no_grad():
        base = model(x)
        shifted = x.clone()
        shifted[..., 0] += 10.0
        # The earliest step is inside the receptive field, so it must matter.
        assert float((model(shifted) - base).abs().max()) > 1e-6


def test_receptive_field_shorter_than_window_is_rejected(
    edge_index: torch.Tensor,
) -> None:
    config = STGNNConfig(num_nodes=NUM_NODES, dilations=(1, 2), seq_len=12)
    with pytest.raises(ValueError, match="receptive field"):
        STGNN(config, edge_index)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_asymmetric_edge_index_is_rejected(config: STGNNConfig) -> None:
    one_way = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    with pytest.raises(ValueError, match="symmetric"):
        STGNN(config, one_way)


def test_out_of_range_edge_index_is_rejected(config: STGNNConfig) -> None:
    bad = torch.tensor([[0, 99], [99, 0]], dtype=torch.long)
    with pytest.raises(ValueError, match="outside"):
        STGNN(config, bad)


def test_wrong_channel_count_is_rejected(
    config: STGNNConfig, edge_index: torch.Tensor
) -> None:
    model = STGNN(config, edge_index).eval()
    with pytest.raises(ValueError, match="input channels"):
        model(torch.randn(1, NUM_NODES, config.in_channels + 1, SEQ_LEN))


def test_config_round_trips_through_dict(config: STGNNConfig) -> None:
    restored = STGNNConfig.from_dict({**config.to_dict(), "unexpected": 1})
    assert restored == config


# ---------------------------------------------------------------------------
# Serving wrapper
# ---------------------------------------------------------------------------
def _runtime(config: STGNNConfig, edge_index: torch.Tensor) -> STGNNRuntime:
    torch.manual_seed(0)
    return STGNNRuntime(
        model=STGNN(config, edge_index).eval(),
        feature_mean=torch.randn(NUM_NODES, config.in_channels),
        feature_std=torch.rand(NUM_NODES, config.in_channels) + 0.5,
        target_mean=torch.full((NUM_NODES,), 30.0),
        target_std=torch.full((NUM_NODES,), 5.0),
        min_travel_time_s=torch.full((NUM_NODES,), 20.0),
    ).eval()


def test_runtime_applies_scalers(
    config: STGNNConfig, edge_index: torch.Tensor
) -> None:
    runtime = _runtime(config, edge_index)
    x = torch.randn(3, NUM_NODES, config.in_channels, SEQ_LEN)
    normalised = (x - runtime.feature_mean) / runtime.feature_std
    with torch.no_grad():
        expected = runtime.model(normalised) * runtime.target_std + runtime.target_mean
        expected = torch.maximum(expected, runtime.min_travel_time_s)
        assert torch.allclose(runtime(x), expected, atol=1e-6)


def test_runtime_clamps_to_free_flow(
    config: STGNNConfig, edge_index: torch.Tensor
) -> None:
    """Predictions below free-flow travel time are physically impossible."""
    runtime = _runtime(config, edge_index)
    out = runtime(torch.randn(3, NUM_NODES, config.in_channels, SEQ_LEN))
    assert bool((out >= 20.0 - 1e-6).all())


def test_runtime_rejects_mis_shaped_scalers(
    config: STGNNConfig, edge_index: torch.Tensor
) -> None:
    with pytest.raises(ValueError, match="feature_mean"):
        STGNNRuntime(
            model=STGNN(config, edge_index),
            feature_mean=torch.zeros(NUM_NODES),
            feature_std=torch.ones(NUM_NODES, config.in_channels),
            target_mean=torch.zeros(NUM_NODES),
            target_std=torch.ones(NUM_NODES),
        )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def test_onnx_export_matches_torch(
    config: STGNNConfig, edge_index: torch.Tensor, tmp_path
) -> None:
    """The ONNX graph is the artifact that actually gets served."""
    ort = pytest.importorskip("onnxruntime")

    runtime = _runtime(config, edge_index)
    example = torch.randn(1, NUM_NODES, config.in_channels, SEQ_LEN)
    with torch.no_grad():
        reference = runtime(example).numpy()

    path = tmp_path / "model.onnx"
    torch.onnx.export(
        runtime,
        (example,),
        str(path),
        input_names=["features"],
        output_names=["travel_time_s"],
        dynamic_axes={"features": {0: "batch"}, "travel_time_s": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    produced = session.run(None, {"features": example.numpy()})[0]
    assert np.abs(produced - reference).max() < 1e-4

    # The dynamic batch axis has to survive: the router batches queries.
    larger = torch.randn(4, NUM_NODES, config.in_channels, SEQ_LEN)
    with torch.no_grad():
        expected = runtime(larger).numpy()
    produced = session.run(None, {"features": larger.numpy()})[0]
    assert np.abs(produced - expected).max() < 1e-4


def test_count_parameters_is_positive(
    config: STGNNConfig, edge_index: torch.Tensor
) -> None:
    assert count_parameters(STGNN(config, edge_index)) > 0
