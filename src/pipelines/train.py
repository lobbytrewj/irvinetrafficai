"""Train the STGNN travel-time forecaster and export it to ONNX.

Pipeline
--------
1. Load ``data/processed/traffic_dataset.pt`` and window it into the three
   chronological splits.
2. Fit :class:`~src.models.st_gnn.STGNN` with MSE on z-scored travel time,
   Adam, ``ReduceLROnPlateau`` and early stopping on validation loss.
3. Report MAE / RMSE / MAPE **in seconds** per horizon on the held-out test
   split, next to a persistence baseline.
4. Export the trained network -- scalers folded in -- to
   ``data/processed/traffic_stgnn.onnx`` and verify the ONNX Runtime output
   against PyTorch before declaring success.

Why MSE on normalised targets: the corridors span 200 m arterial blocks and
1 km freeway segments, so raw-second errors differ by an order of magnitude
across nodes and an unnormalised loss would spend all its capacity on the
longest segments. Reporting, by contrast, is in seconds, because that is the
unit the router consumes.

Usage
-----
    python -m src.pipelines.train --epochs 60
    python -m src.pipelines.train --export-only     # re-export a checkpoint
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.generate_traffic_data import (
    DATASET_FILENAME,
    DEFAULT_PROCESSED_DIR,
    TrafficDataset,
    load_bundle,
)
from src.models.st_gnn import STGNN, STGNNConfig, STGNNRuntime, count_parameters

LOGGER = logging.getLogger(__name__)

CHECKPOINT_FILENAME = "traffic_stgnn.pt"
ONNX_FILENAME = "traffic_stgnn.onnx"
METRICS_FILENAME = "traffic_stgnn_metrics.json"


@dataclass(frozen=True)
class TrainConfig:
    """Everything the training run needs that is not in the dataset bundle."""

    dataset_path: Path = DEFAULT_PROCESSED_DIR / DATASET_FILENAME
    output_dir: Path = DEFAULT_PROCESSED_DIR
    epochs: int = 60
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    # Stop after this many epochs without a new best validation loss.
    patience: int = 10
    lr_patience: int = 4
    lr_factor: float = 0.5
    grad_clip: float = 5.0
    hidden_channels: int = 32
    skip_channels: int = 64
    cheb_k: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.1
    # Keep every train_stride-th window. Consecutive windows overlap in all
    # but one input step, so >1 shortens an epoch at almost no cost in
    # information. Validation and test always run at stride 1.
    train_stride: int = 1
    num_workers: int = 0
    device: str = "auto"
    seed: int = 0
    # Cap the batches per epoch; for smoke tests, not for real runs.
    max_batches: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset_path"] = str(self.dataset_path)
        payload["output_dir"] = str(self.output_dir)
        payload["dilations"] = list(self.dilations)
        return payload


def resolve_device(name: str = "auto") -> torch.device:
    """Pick a device. ``auto`` prefers CUDA and otherwise stays on CPU.

    MPS is reachable with ``--device mps`` but is not the automatic choice:
    the gather/scatter kernels message passing relies on have historically
    been the flaky part of that backend, and a corridor-scale graph trains on
    CPU in minutes anyway.
    """
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model / data assembly
# ---------------------------------------------------------------------------
def build_model(bundle: dict[str, Any], config: TrainConfig) -> STGNN:
    """Instantiate an :class:`STGNN` sized from the dataset bundle."""
    model_config = STGNNConfig(
        num_nodes=int(bundle["num_nodes"]),
        in_channels=len(bundle["feature_names"]),
        hidden_channels=config.hidden_channels,
        skip_channels=config.skip_channels,
        horizons=len(bundle["horizons"]),
        cheb_k=config.cheb_k,
        dilations=tuple(config.dilations),
        dropout=config.dropout,
        seq_len=int(bundle["seq_len"]),
    )
    return STGNN(model_config, bundle["edge_index"])


def build_loaders(
    bundle: dict[str, Any], config: TrainConfig
) -> dict[str, DataLoader]:
    """One loader per split. Only the training split is shuffled."""
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = TrafficDataset(
            bundle,
            split=split,
            stride=config.train_stride if split == "train" else 1,
        )
        LOGGER.info("  %s", dataset)
        loaders[split] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            num_workers=config.num_workers,
            drop_last=False,
        )
    return loaders


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def _run_epoch(
    model: STGNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimiser: torch.optim.Optimizer | None = None,
    grad_clip: float = 0.0,
    max_batches: int | None = None,
) -> float:
    """One pass over ``loader``. Trains when ``optimiser`` is given.

    Returns the sample-weighted mean loss, so a short final batch does not
    skew the epoch average.
    """
    training = optimiser is not None
    model.train(training)
    total, seen = 0.0, 0

    with torch.set_grad_enabled(training):
        for index, (x, y) in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            loss = criterion(prediction, y)

            if training:
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimiser.step()

            total += float(loss.detach()) * x.size(0)
            seen += x.size(0)

    if not seen:
        raise RuntimeError("loader yielded no batches")
    return total / seen


@torch.no_grad()
def evaluate(
    model: STGNN,
    loader: DataLoader,
    bundle: dict[str, Any],
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Per-horizon error in seconds on the loader's split.

    Predictions and targets are de-normalised first: a normalised MSE is the
    right training signal but says nothing about whether the router's edge
    weights are off by two seconds or twenty.
    """
    model.eval()
    # Broadcast-ready against [B, N, H]: undoes the per-segment z-score.
    offset = bundle["target_mean"].to(device).view(1, -1, 1)
    scale = bundle["target_std"].to(device).view(1, -1, 1)

    errors, absolute_percentage, count = [], [], 0
    for index, (x, y) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        prediction = model(x) * scale + offset
        truth = y * scale + offset

        residual = prediction - truth
        errors.append(residual.cpu())
        absolute_percentage.append((residual.abs() / truth.clamp_min(1e-3)).cpu())
        count += x.size(0)

    residual = torch.cat(errors)  # [S, N, H]
    mape = torch.cat(absolute_percentage)
    horizons = [int(h) for h in bundle["horizons"]]
    step_minutes = int(bundle["step_minutes"])

    per_horizon = {}
    for i, steps in enumerate(horizons):
        per_horizon[f"{steps * step_minutes}min"] = {
            "mae_s": float(residual[..., i].abs().mean()),
            "rmse_s": float(residual[..., i].pow(2).mean().sqrt()),
            "mape_pct": float(mape[..., i].mean() * 100.0),
        }
    return {
        "samples": count,
        "mae_s": float(residual.abs().mean()),
        "rmse_s": float(residual.pow(2).mean().sqrt()),
        "mape_pct": float(mape.mean() * 100.0),
        "per_horizon": per_horizon,
    }


def persistence_baseline(
    bundle: dict[str, Any], split: str = "test"
) -> dict[str, Any]:
    """"Tomorrow looks like now": predict the last observed travel time.

    Any spatio-temporal model that does not beat this is not earning its
    inference cost, so it is reported next to the trained metrics.
    """
    dataset = TrafficDataset(bundle, split=split, normalise=False)
    travel_time = bundle["travel_time_s"]
    horizons = [int(h) for h in bundle["horizons"]]
    step_minutes = int(bundle["step_minutes"])
    seq_len = dataset.seq_len

    last_index = dataset.starts + seq_len - 1  # [S]
    prediction = travel_time[last_index].unsqueeze(-1)  # [S, N, 1]
    target_index = torch.stack([last_index + h for h in horizons], dim=-1)  # [S, H]
    truth = travel_time[target_index].permute(0, 2, 1)  # [S, N, H]

    residual = prediction - truth
    mape = residual.abs() / truth.clamp_min(1e-3)
    return {
        "samples": int(len(dataset)),
        "mae_s": float(residual.abs().mean()),
        "rmse_s": float(residual.pow(2).mean().sqrt()),
        "mape_pct": float(mape.mean() * 100.0),
        "per_horizon": {
            f"{steps * step_minutes}min": {
                "mae_s": float(residual[..., i].abs().mean()),
                "rmse_s": float(residual[..., i].pow(2).mean().sqrt()),
                "mape_pct": float(mape[..., i].mean() * 100.0),
            }
            for i, steps in enumerate(horizons)
        },
    }


def train(
    bundle: dict[str, Any], config: TrainConfig
) -> tuple[STGNN, dict[str, Any]]:
    """Fit the model. Returns it with the best validation weights restored."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = resolve_device(config.device)
    loaders = build_loaders(bundle, config)
    model = build_model(bundle, config).to(device)
    LOGGER.info(
        "Model: %s params, receptive field %d steps",
        f"{count_parameters(model):,}",
        model.receptive_field,
    )

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode="min",
        factor=config.lr_factor,
        patience=config.lr_patience,
    )

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    history: list[dict[str, float]] = []
    stale = 0

    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        train_loss = _run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimiser=optimiser,
            grad_clip=config.grad_clip,
            max_batches=config.max_batches,
        )
        val_loss = _run_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            max_batches=config.max_batches,
        )
        scheduler.step(val_loss)
        learning_rate = optimiser.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": learning_rate,
            }
        )

        improved = val_loss < best_loss - 1e-6
        if improved:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            # Snapshot on the CPU so a long run does not pin a second copy of
            # the weights in accelerator memory.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1

        LOGGER.info(
            "epoch %3d/%d | train %.5f | val %.5f | lr %.2e | %.1fs%s",
            epoch,
            config.epochs,
            train_loss,
            val_loss,
            learning_rate,
            time.perf_counter() - started,
            "  *" if improved else "",
        )

        if stale >= config.patience:
            LOGGER.info(
                "Early stop: no improvement for %d epochs (best %.5f @ epoch %d)",
                config.patience,
                best_loss,
                best_epoch,
            )
            break

    model.load_state_dict(best_state)
    model.to(device)

    metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "history": history,
        "test": evaluate(
            model, loaders["test"], bundle, device, max_batches=config.max_batches
        ),
        "val": evaluate(
            model, loaders["val"], bundle, device, max_batches=config.max_batches
        ),
        "persistence_baseline_test": persistence_baseline(bundle, "test"),
    }
    return model, metrics


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(
    model: STGNN,
    bundle: dict[str, Any],
    config: TrainConfig,
    metrics: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Persist weights, model config, graph and scalers in one file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / CHECKPOINT_FILENAME

    torch.save(
        {
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "model_config": model.config.to_dict(),
            "train_config": config.to_dict(),
            "edge_index": bundle["edge_index"].cpu(),
            # The scalers belong with the weights: a model served against a
            # differently-scaled feed is wrong in a way nothing will flag.
            "feature_mean": bundle["feature_mean"],
            "feature_std": bundle["feature_std"],
            "target_mean": bundle["target_mean"],
            "target_std": bundle["target_std"],
            "segment_free_flow_time_s": bundle["segment_free_flow_time_s"],
            "feature_names": list(bundle["feature_names"]),
            "horizons": list(bundle["horizons"]),
            "step_minutes": int(bundle["step_minutes"]),
            "corridor_id": bundle["corridor_id"],
            "corridor_names": list(bundle["corridor_names"]),
            "metrics": metrics,
        },
        path,
    )
    LOGGER.info("Wrote %s", path)
    return path


def load_checkpoint(
    path: Path | str, map_location: str | torch.device = "cpu"
) -> tuple[STGNN, dict[str, Any]]:
    """Rebuild a trained :class:`STGNN` and return it with its payload."""
    path = Path(path)
    if path.is_dir():
        path = path / CHECKPOINT_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.pipelines.train` first."
        )
    payload = torch.load(path, map_location=map_location, weights_only=True)
    model = STGNN(
        STGNNConfig.from_dict(payload["model_config"]), payload["edge_index"]
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
def build_runtime(model: STGNN, payload: dict[str, Any]) -> STGNNRuntime:
    """Wrap a trained model with its scalers for serving."""
    return STGNNRuntime(
        model=model,
        feature_mean=payload["feature_mean"],
        feature_std=payload["feature_std"],
        target_mean=payload["target_mean"],
        target_std=payload["target_std"],
        min_travel_time_s=payload.get("segment_free_flow_time_s"),
    ).eval()


def export_onnx(
    model: STGNN,
    payload: dict[str, Any],
    output_dir: Path,
    opset: int = 17,
    benchmark_iters: int = 200,
) -> Path:
    """Export to ONNX, check it against PyTorch, and time it.

    The exported graph takes **raw** features -- the scalers and the
    free-flow clamp are baked in via :class:`STGNNRuntime` -- so the C++
    engine only has to fill a ``[B, N, F, L]`` buffer and read seconds back.
    ``edge_index`` is a buffer, hence a constant in the graph, so the topology
    needs no runtime input either.
    """
    import onnxruntime as ort

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ONNX_FILENAME

    runtime = build_runtime(model.cpu().eval(), payload)
    num_nodes = model.config.num_nodes
    num_features = model.config.in_channels
    seq_len = model.config.seq_len
    example = torch.randn(1, num_nodes, num_features, seq_len)

    with torch.no_grad():
        reference = runtime(example)

    torch.onnx.export(
        runtime,
        (example,),
        str(path),
        input_names=["features"],
        output_names=["travel_time_s"],
        # Only the batch axis is dynamic: N, F and L are fixed by the trained
        # graph, and pinning them lets the runtime pre-plan every allocation.
        dynamic_axes={"features": {0: "batch"}, "travel_time_s": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        # The dynamo exporter needs `onnxscript`, which is not a dependency of
        # this project; the TorchScript path handles the scatter ops that PyG
        # message passing lowers to.
        dynamo=False,
    )

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    produced = session.run(None, {"features": example.numpy()})[0]
    deviation = float(np.abs(produced - reference.numpy()).max())
    # Tolerance is generous relative to fp32 noise but far tighter than any
    # error that would matter in seconds of travel time.
    if not np.isfinite(deviation) or deviation > 1e-3:
        raise RuntimeError(
            f"ONNX output diverges from PyTorch by {deviation:.3g} s; "
            f"refusing to ship {path}"
        )

    batch = {"features": example.numpy()}
    for _ in range(10):  # warm up allocators and thread pools
        session.run(None, batch)
    started = time.perf_counter()
    for _ in range(benchmark_iters):
        session.run(None, batch)
    latency_ms = (time.perf_counter() - started) / benchmark_iters * 1e3

    LOGGER.info(
        "Wrote %s (%.1f KB) | max|onnx-torch| = %.2e s | %.3f ms/inference "
        "for all %d segments",
        path,
        path.stat().st_size / 1e3,
        deviation,
        latency_ms,
        num_nodes,
    )
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _log_metrics(name: str, metrics: dict[str, Any]) -> None:
    LOGGER.info(
        "%s: MAE %.2fs  RMSE %.2fs  MAPE %.2f%%  (%d windows)",
        name,
        metrics["mae_s"],
        metrics["rmse_s"],
        metrics["mape_pct"],
        metrics["samples"],
    )
    for horizon, values in metrics["per_horizon"].items():
        LOGGER.info(
            "    %-6s MAE %6.2fs  RMSE %6.2fs  MAPE %5.2f%%",
            horizon,
            values["mae_s"],
            values["rmse_s"],
            values["mape_pct"],
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_PROCESSED_DIR / DATASET_FILENAME
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr-patience", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--skip-channels", type=int, default=64)
    parser.add_argument("--cheb-k", type=int, default=3)
    parser.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--train-stride",
        type=int,
        default=1,
        help="keep every Nth training window (val/test always use 1)",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="cap batches per epoch (smoke tests only)",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="skip training; re-export the existing checkpoint to ONNX",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    )

    if args.export_only:
        model, payload = load_checkpoint(args.output_dir)
        export_onnx(model, payload, args.output_dir, opset=args.opset)
        return 0

    config = TrainConfig(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        lr_patience=args.lr_patience,
        hidden_channels=args.hidden_channels,
        skip_channels=args.skip_channels,
        cheb_k=args.cheb_k,
        dilations=tuple(args.dilations),
        dropout=args.dropout,
        train_stride=args.train_stride,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        max_batches=args.max_batches,
    )

    bundle = load_bundle(config.dataset_path)
    LOGGER.info(
        "Dataset: %d segments x %d steps, horizons %s (x%d min)",
        bundle["num_nodes"],
        bundle["num_steps"],
        list(bundle["horizons"]),
        bundle["step_minutes"],
    )

    model, metrics = train(bundle, config)
    _log_metrics("test", metrics["test"])
    _log_metrics("persistence baseline (test)", metrics["persistence_baseline_test"])

    lift = (
        1.0
        - metrics["test"]["mae_s"] / metrics["persistence_baseline_test"]["mae_s"]
    )
    LOGGER.info("MAE improvement over persistence: %.1f%%", lift * 100)

    save_checkpoint(model, bundle, config, metrics, config.output_dir)
    metrics_path = Path(config.output_dir) / METRICS_FILENAME
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    LOGGER.info("Wrote %s", metrics_path)

    if not args.no_export:
        _, payload = load_checkpoint(config.output_dir)
        export_onnx(model, payload, config.output_dir, opset=args.opset)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
