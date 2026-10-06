"""Spatio-Temporal Graph Neural Network for dynamic travel-time forecasting.

The network follows the Graph-WaveNet family: interleaved *temporal* and
*spatial* operators stacked into residual blocks.

    input  [B, N, F_in, L]        L past steps of F_in features on N segments
      |  channel projection
      v
    +-- block b (dilation d_b) -------------------------------------+
    |   gated dilated causal convolution   (time)                   |
    |   Chebyshev graph convolution        (space, every step)      |
    |   dropout -> residual add -> channel LayerNorm                |
    +---------------------------------------------------------------+
      |  skip accumulation, read out at the last time step
      v
    output [B, N, H]              travel time at H forecast horizons

Why these two operators:

* **Gated dilated causal convolutions** give an exponentially growing
  receptive field in depth while touching each input step once. Padding is
  applied on the *left only*, so step ``t`` is never a function of step
  ``t+1`` -- there is no future leakage, which an RNN gets for free but a
  plain ``Conv1d`` does not. The ``tanh * sigmoid`` gate lets a block decide
  per channel how much of the temporal update to pass through, which matters
  when a corridor flips between free flow and a queue in one or two steps.

* **Chebyshev graph convolutions** (Defferrard et al., 2016) aggregate over
  the K-hop neighbourhood of the road topology with K localised filters,
  rather than the single hop a GCN layer gives. Congestion travels upstream
  along the physical network, so a K=3 filter sees roughly the distance a
  queue spreads in one five-minute step.

Nodes here are *road segments* (directed edges of the OSM graph), connected
by the line graph built in :mod:`src.data.generate_traffic_data`. Travel time
is a property of a segment, so making segments the nodes keeps the prediction
target and the message-passing unit the same object.

Shapes and layout
-----------------
``forward`` takes ``[B, N, F_in, L]`` and returns ``[B, N, H]``. Internally
everything is held as ``[N, B, L, C]`` -- nodes first, channels last -- and
the two permutes at the boundary are the only ones in the network.

That is deliberate. In this layout every operator reads the axis it needs
without shuffling memory: the graph convolution needs nodes leading (a free
``view`` to ``[N, B*L*C]``), while the temporal convolution, the channel
projections and the LayerNorm all act on the trailing channel axis. Profiled
under ONNX Runtime, an implementation that permuted per operator spent 8.7 ms
of its 17 ms in ``Transpose`` against 1.8 ms in ``MatMul``; layout, not
arithmetic, was the entire cost.

``N`` and the graph are fixed at construction: ``edge_index`` is a buffer, so
a checkpoint carries the topology it was trained on and the ONNX export takes
a single tensor input.

Usage
-----
    from src.models.st_gnn import STGNN, STGNNConfig

    config = STGNNConfig(num_nodes=440, in_channels=5, horizons=3)
    model = STGNN(config, edge_index)
    y = model(torch.randn(8, 440, 5, 12))     # -> [8, 440, 3]
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import ChebConv

__all__ = [
    "STGNNConfig",
    "STGNN",
    "STGNNRuntime",
    "GatedDilatedCausalConv",
    "ChebSpatialConv",
    "rescaled_laplacian",
    "count_parameters",
]


@dataclass(frozen=True)
class STGNNConfig:
    """Hyper-parameters of :class:`STGNN`.

    Stored alongside the weights in every checkpoint so a model can be rebuilt
    without repeating the arguments at the call site.
    """

    num_nodes: int
    in_channels: int = 5
    hidden_channels: int = 32
    skip_channels: int = 64
    horizons: int = 3
    # K in the Chebyshev expansion: filters are localised to K-1 hops.
    cheb_k: int = 3
    # One entry per residual block. Doubling the dilation each block grows the
    # receptive field geometrically; see `receptive_field`.
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    temporal_kernel: int = 2
    dropout: float = 0.1
    # Checked against the receptive field at construction time.
    seq_len: int = 12

    @property
    def receptive_field(self) -> int:
        """Number of past steps the last output position can see."""
        return 1 + sum((self.temporal_kernel - 1) * d for d in self.dilations)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "STGNNConfig":
        """Rebuild from a checkpoint payload, tolerating extra keys."""
        fields = set(cls.__dataclass_fields__)
        kwargs = {k: v for k, v in payload.items() if k in fields}
        if "dilations" in kwargs:
            kwargs["dilations"] = tuple(int(d) for d in kwargs["dilations"])
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Temporal operator
# ---------------------------------------------------------------------------
class GatedDilatedCausalConv(nn.Module):
    """Gated dilated *causal* 1D convolution over the time axis.

    Two convolutions share an input: a ``tanh`` filter branch and a
    ``sigmoid`` gate branch, multiplied elementwise (the WaveNet activation).

    Causality comes from asymmetric padding: ``(kernel - 1) * dilation`` zeros
    are prepended and none appended, so output position ``t`` depends only on
    inputs ``<= t``. The sequence length is preserved, which lets blocks stack
    without bookkeeping.

    Parameters live in two ``nn.Conv1d(..., dilation=d)`` modules, which define
    the operation. ``fused=True`` evaluates them differently but identically:
    each of the ``k`` dilated taps is a strided slice of the padded input, so
    the convolution is a sum of ``k`` channel-mixing GEMMs against the
    reshaped ``Conv1d`` weights, with the filter and gate branches packed into
    one matrix.

    The detour exists because PyTorch's CPU convolution has no fast path for
    ``dilation > 1``: profiled here, a 10 MFLOP dilated ``Conv1d`` took 20 ms
    (~0.5 GFLOPS) against 2 ms for the same shape undilated. Taking the GEMM
    path instead is worth roughly 50x on this workload, and it also keeps the
    tensor in the ``[N, B, L, C]`` layout the rest of the stack uses.

    Shapes: ``[N, B, L, C_in] -> [N, B, L, C_out]``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        fused: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size < 2:
            raise ValueError(f"kernel_size must be >= 2, got {kernel_size}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.fused = fused
        self.left_pad = (kernel_size - 1) * dilation
        self.filter = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )
        self.gate = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )

    def _tap_weights(self) -> tuple[list[Tensor], Tensor]:
        """Per-tap ``[C_in, 2 * C_out]`` matrices and the packed bias.

        ``Conv1d.weight`` is ``[C_out, C_in, k]``; tap ``j`` therefore needs
        ``weight[:, :, j].t()``. Filter and gate are concatenated so each tap
        costs one GEMM rather than two. Rebuilt per call so gradients flow
        back into the ``Conv1d`` parameters, which stay the single source of
        truth for the weights.
        """
        taps = [
            torch.cat(
                [self.filter.weight[:, :, j].t(), self.gate.weight[:, :, j].t()],
                dim=1,
            )
            for j in range(self.kernel_size)
        ]
        return taps, torch.cat([self.filter.bias, self.gate.bias])

    def forward(self, x: Tensor) -> Tensor:
        nodes, batch, length, _ = x.shape
        # Pad the time axis (dim -2) on the left only. F.pad's pair order runs
        # from the last dimension backwards, hence the leading (0, 0).
        padded = F.pad(x, (0, 0, self.left_pad, 0))

        if not self.fused:
            # Reference path: [N, B, L, C] -> [N*B, C, L] for nn.Conv1d.
            flat = padded.permute(0, 1, 3, 2).reshape(nodes * batch, -1, padded.size(2))
            out = torch.tanh(self.filter(flat)) * torch.sigmoid(self.gate(flat))
            return out.reshape(nodes, batch, self.out_channels, length).permute(
                0, 1, 3, 2
            )

        taps, bias = self._tap_weights()
        projected = bias
        for j, weight in enumerate(taps):
            offset = j * self.dilation
            projected = projected + padded[:, :, offset : offset + length] @ weight
        filtered, gated = projected.chunk(2, dim=-1)
        return torch.tanh(filtered) * torch.sigmoid(gated)


# ---------------------------------------------------------------------------
# Spatial operator
# ---------------------------------------------------------------------------
def rescaled_laplacian(
    edge_index: Tensor, num_nodes: int, dtype: torch.dtype = torch.float32
) -> Tensor:
    """Dense ``[N, N]`` rescaled symmetric-normalised Laplacian.

    Reproduces ``ChebConv.__norm__`` for ``normalization="sym"``: build
    ``L = I - D^-1/2 A D^-1/2``, rescale by ``2 / lambda_max`` and subtract the
    identity, giving an operator with spectrum in ``[-1, 1]`` -- the domain the
    Chebyshev polynomials are defined on.

    ``lambda_max`` follows PyG's default of ``2 * max(L)``, which for a
    symmetric-normalised Laplacian on a graph with no isolated nodes is
    exactly 2. Deriving it rather than hard-coding it keeps the operator
    identical to the reference implementation on degenerate graphs too.

    The result is a *constant of the topology*, so it is computed once at
    construction. That is what makes the fused path both fast and exportable:
    the ONNX graph gets a MatMul against an initializer instead of a
    gather/scatter over a per-call Laplacian rebuild.
    """
    from torch_geometric.utils import get_laplacian

    index, weight = get_laplacian(edge_index, None, "sym", dtype, num_nodes)
    lambda_max = 2.0 * weight.max()
    weight = (2.0 * weight) / lambda_max
    weight = weight.masked_fill(weight == float("inf"), 0.0)
    weight = weight - (index[0] == index[1]).to(weight.dtype)

    dense = torch.zeros(num_nodes, num_nodes, dtype=dtype)
    # accumulate=True: get_laplacian emits one entry per input edge, so
    # parallel edges must sum rather than overwrite.
    dense.index_put_((index[0], index[1]), weight, accumulate=True)
    return dense


class ChebSpatialConv(nn.Module):
    """Chebyshev graph convolution applied independently at every time step.

    Wraps :class:`torch_geometric.nn.ChebConv`, which owns the parameters and
    defines the semantics. Evaluation takes one of two paths:

    ``fused=True`` (default)
        Runs the Chebyshev recursion ``T_0 = x``, ``T_1 = L_hat x``,
        ``T_k = 2 L_hat T_{k-1} - T_{k-2}`` as matmuls against a cached
        Laplacian. Mathematically identical -- same ``lins``, same ``bias``,
        same operator -- but far cheaper here: the graph is static and the
        same ``[N, N]`` operator applies to all ``B * L`` slices, so they
        collapse into one GEMM against a ``[N, B*L*C]`` view instead of
        ``B * L`` scatter reductions over a materialised ``[B*L, E, C]``
        message tensor. Measured ~13x faster, and it lowers to a plain ONNX
        ``MatMul``.

    ``fused=False``
        Calls ``ChebConv.forward``. PyG's ``MessagePassing`` uses
        ``node_dim=-2``, so a ``[B*L, N, C]`` input is handled natively. Kept
        as the reference the fused path is tested against, and selected
        automatically when the graph is too large for a dense operator.

    The dense Laplacian costs ``N^2`` floats (~800 KB at corridor scale). Past
    ``dense_max_nodes`` that stops being a good trade and the reference path
    takes over.

    Shapes: ``[N, B, L, C_in] -> [N, B, L, C_out]``.
    """

    dense_max_nodes: int = 4096

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        num_nodes: int,
        edge_index: Tensor,
        fused: bool = True,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        # normalization="sym" fixes lambda_max at 2.0 for a connected graph,
        # so the rescaled Laplacian is a constant of the topology.
        self.conv = ChebConv(in_channels, out_channels, K=k, normalization="sym")
        self.fused = fused and num_nodes <= self.dense_max_nodes
        # persistent=False: the Laplacian is a pure function of edge_index,
        # which is itself saved, so keeping N^2 floats per block out of every
        # checkpoint costs nothing but a rebuild at construction.
        self.register_buffer(
            "laplacian",
            rescaled_laplacian(edge_index, num_nodes) if self.fused else None,
            persistent=False,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        nodes, batch, length, channels = x.shape

        if not self.fused:
            # Reference path: [N, B, L, C] -> [B*L, N, C] for MessagePassing.
            stacked = x.permute(1, 2, 0, 3).reshape(batch * length, nodes, channels)
            out = self.conv(stacked, edge_index)
            return out.reshape(batch, length, nodes, -1).permute(2, 0, 1, 3)

        shape = (nodes, batch, length, channels)
        # Node-major already, so this is a free view: one [N, N] x [N, B*L*C]
        # GEMM propagates every step of every sample at once.
        previous = x.reshape(nodes, -1)
        lins = self.conv.lins
        out = lins[0](x)
        if len(lins) > 1:
            current = self.laplacian @ previous
            out = out + lins[1](current.view(shape))
            for lin in lins[2:]:
                nxt = torch.addmm(
                    previous, self.laplacian, current, beta=-1.0, alpha=2.0
                )
                out = out + lin(nxt.view(shape))
                previous, current = current, nxt

        if self.conv.bias is not None:
            out = out + self.conv.bias
        return out


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------
class STBlock(nn.Module):
    """One temporal-then-spatial residual block on ``[N, B, L, C]``."""

    def __init__(
        self,
        channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
        cheb_k: int,
        dropout: float,
        num_nodes: int,
        edge_index: Tensor,
        fused: bool = True,
    ) -> None:
        super().__init__()
        self.temporal = GatedDilatedCausalConv(
            channels, channels, kernel_size, dilation, fused=fused
        )
        self.spatial = ChebSpatialConv(
            channels,
            channels,
            cheb_k,
            num_nodes=num_nodes,
            edge_index=edge_index,
            fused=fused,
        )
        self.dropout = nn.Dropout(dropout)
        # Channels are the trailing axis, so both of these apply directly.
        self.norm = nn.LayerNorm(channels)
        self.skip_proj = nn.Linear(channels, skip_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(residual_output, skip_contribution)``."""
        h = self.temporal(x)
        h = self.spatial(h, edge_index)
        h = self.dropout(h)
        return self.norm(x + h), self.skip_proj(h)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class STGNN(nn.Module):
    """Spatio-temporal GNN mapping a window of segment features to horizons.

    Parameters
    ----------
    config:
        Hyper-parameters. ``config.num_nodes`` must match ``edge_index``.
    edge_index:
        ``[2, E]`` int64 COO connectivity of the *segment* graph (the line
        graph of the road network). Registered as a buffer, so it travels with
        the checkpoint and is baked into the ONNX graph.
    fused:
        Use the GEMM formulations of the temporal and spatial operators. They
        are numerically identical to the ``nn.Conv1d`` / ``ChebConv``
        reference paths but avoid two slow kernels; see
        :class:`GatedDilatedCausalConv` and :class:`ChebSpatialConv`. Set
        ``False`` to run the reference implementations.

    Notes
    -----
    ``edge_index`` must be symmetric. ``normalization="sym"`` builds a
    symmetric normalised Laplacian regardless, so a one-directional edge list
    silently yields an asymmetric operator with the wrong spectrum; the
    constructor rejects that rather than training a subtly wrong model.
    """

    def __init__(
        self,
        config: STGNNConfig,
        edge_index: Tensor,
        fused: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        _validate_edge_index(edge_index, config.num_nodes)
        self.register_buffer("edge_index", edge_index.to(torch.int64))

        if config.receptive_field < config.seq_len:
            raise ValueError(
                f"receptive field {config.receptive_field} < seq_len "
                f"{config.seq_len}: the earliest input steps can never reach "
                "the output. Add a dilation or widen temporal_kernel."
            )

        self.input_proj = nn.Linear(config.in_channels, config.hidden_channels)
        self.blocks = nn.ModuleList(
            STBlock(
                channels=config.hidden_channels,
                skip_channels=config.skip_channels,
                kernel_size=config.temporal_kernel,
                dilation=dilation,
                cheb_k=config.cheb_k,
                dropout=config.dropout,
                num_nodes=config.num_nodes,
                edge_index=self.edge_index,
                fused=fused,
            )
            for dilation in config.dilations
        )
        # Read-out on the last step only: with causal convolutions that
        # position is the sole one that has seen the whole window.
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(config.skip_channels, config.skip_channels),
            nn.ReLU(),
            nn.Linear(config.skip_channels, config.horizons),
        )

    @property
    def receptive_field(self) -> int:
        return self.config.receptive_field

    def forward(self, x: Tensor) -> Tensor:
        """``[B, N, F_in, L] -> [B, N, H]``.

        ``x`` is expected already normalised (see
        :class:`~src.data.generate_traffic_data.TrafficDataset`); use
        :class:`STGNNRuntime` for a module that takes raw units.
        """
        if x.dim() != 4:
            raise ValueError(f"expected [B, N, F, L], got {tuple(x.shape)}")
        if x.size(2) != self.config.in_channels:
            raise ValueError(
                f"expected {self.config.in_channels} input channels, "
                f"got {x.size(2)}"
            )

        edge_index = self.edge_index
        # The only permute in: [B, N, F, L] -> [N, B, L, F].
        h = self.input_proj(x.permute(1, 0, 3, 2))

        skip_total: Tensor | None = None
        for block in self.blocks:
            h, skip = block(h, edge_index)
            skip_total = skip if skip_total is None else skip_total + skip
        assert skip_total is not None  # STGNNConfig.dilations is never empty

        # Last step, then the only permute out: [N, B, H] -> [B, N, H].
        return self.head(skip_total[:, :, -1]).permute(1, 0, 2)

    def extra_repr(self) -> str:
        return (
            f"num_nodes={self.config.num_nodes}, "
            f"in_channels={self.config.in_channels}, "
            f"hidden={self.config.hidden_channels}, "
            f"horizons={self.config.horizons}, "
            f"dilations={self.config.dilations}, "
            f"receptive_field={self.receptive_field}"
        )


class STGNNRuntime(nn.Module):
    """Serving wrapper: raw features in, travel time in seconds out.

    Folding the scalers into the graph means the ONNX consumer (the C++
    engine) never has to carry a second copy of the normalisation constants,
    and cannot silently drift out of sync with the checkpoint.

    Parameters
    ----------
    model:
        A trained :class:`STGNN`.
    feature_mean, feature_std:
        ``[N, F]`` per-segment input statistics from the training split.
    target_mean, target_std:
        ``[N]`` per-segment target statistics from the training split.
    min_travel_time_s:
        Optional ``[N]`` lower clamp, normally the free-flow travel time. A
        regression head has no notion of physical limits and can undershoot on
        rarely-congested segments; clamping keeps the router's edge weights
        admissible for A*/Dijkstra.
    """

    def __init__(
        self,
        model: STGNN,
        feature_mean: Tensor,
        feature_std: Tensor,
        target_mean: Tensor,
        target_std: Tensor,
        min_travel_time_s: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        num_nodes = model.config.num_nodes
        num_features = model.config.in_channels

        _check_shape("feature_mean", feature_mean, (num_nodes, num_features))
        _check_shape("feature_std", feature_std, (num_nodes, num_features))
        _check_shape("target_mean", target_mean, (num_nodes,))
        _check_shape("target_std", target_std, (num_nodes,))

        # Broadcast-ready against [B, N, F, L] and [B, N, H].
        self.register_buffer("feature_mean", feature_mean.reshape(1, num_nodes, -1, 1))
        self.register_buffer("feature_std", feature_std.reshape(1, num_nodes, -1, 1))
        self.register_buffer("target_mean", target_mean.reshape(1, num_nodes, 1))
        self.register_buffer("target_std", target_std.reshape(1, num_nodes, 1))

        if min_travel_time_s is None:
            self.register_buffer("min_travel_time_s", None)
        else:
            _check_shape("min_travel_time_s", min_travel_time_s, (num_nodes,))
            self.register_buffer(
                "min_travel_time_s", min_travel_time_s.reshape(1, num_nodes, 1)
            )

    def forward(self, x: Tensor) -> Tensor:
        """``[B, N, F, L]`` in raw units -> ``[B, N, H]`` in seconds."""
        normalised = (x - self.feature_mean) / self.feature_std
        out = self.model(normalised) * self.target_std + self.target_mean
        if self.min_travel_time_s is not None:
            out = torch.maximum(out, self.min_travel_time_s)
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_edge_index(edge_index: Tensor, num_nodes: int) -> None:
    """Reject connectivity that would produce a wrong Laplacian, loudly."""
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must be [2, E], got {tuple(edge_index.shape)}")
    if not edge_index.numel():
        raise ValueError("edge_index is empty: the segment graph has no edges")
    if int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes:
        raise ValueError(
            f"edge_index references nodes outside [0, {num_nodes}); "
            f"range is [{int(edge_index.min())}, {int(edge_index.max())}]"
        )

    pairs = {(int(u), int(v)) for u, v in edge_index.t().tolist()}
    missing = [(u, v) for (u, v) in pairs if (v, u) not in pairs]
    if missing:
        raise ValueError(
            f"edge_index must be symmetric for normalization='sym'; "
            f"{len(missing)} edge(s) have no reverse, e.g. {missing[:3]}"
        )


def _check_shape(name: str, tensor: Tensor, expected: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"{name} must have shape {expected}, got {tuple(tensor.shape)}"
        )


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count, for logging."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
