"""Original-objective DD-Net70 training and evaluation for TAP-Syn banks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .ddnet import DDNet70, DDNetLoss, curriculum_transform
from ..data import collate_fwi_batch
from .metrics import MetricAccumulator
from .fwi_train import _rng_state, _restore_rng_state, _atomic_torch_save


@dataclass(frozen=True)
class DDNetTrainConfig:
    learning_rate: float = 1e-3
    batch_size: int = 64
    eval_batch_size: int = 5
    weight_decay: float = 0.0
    source_stage_epochs: tuple[int, int, int] = (10, 10, 100)
    target_epochs: int = 50
    velocity_loss_weight: float = 1.0
    boundary_loss_weight: float = 0.1
    device: str = "cpu"
    num_workers: int = 0
    checkpoint_every_epochs: int = 5
    resume_from: str | Path | None = None


@dataclass(frozen=True)
class DDNetRunArtifacts:
    checkpoint_path: Path
    latest_checkpoint: Path
    history: list[dict[str, float]]
    best_metrics: dict[str, float]
    best_epoch: int


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "cuda" and torch.cuda.is_available() else name)


def _loader(dataset: Dataset, config: DDNetTrainConfig, *, training: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size if training else config.eval_batch_size,
        shuffle=training,
        num_workers=config.num_workers,
        drop_last=False,
        collate_fn=collate_fwi_batch,
    )


def load_ddnet70_checkpoint(path: str | Path, *, device: str = "cpu") -> DDNet70:
    payload = torch.load(path, map_location=_device(device), weights_only=False)
    if payload.get("format") != "tap_syn_ddnet70_v1":
        raise ValueError(f"Not a TAP-Syn DD-Net70 checkpoint: {path}")
    model = DDNet70()
    model.load_state_dict(payload["model_state_dict"])
    return model


def _epoch(*, model: DDNet70, loader: DataLoader, optimizer: torch.optim.Optimizer | None, loss_fn: DDNetLoss,
           device: torch.device, curriculum_stage: int | None, full_metrics: bool) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metrics = MetricAccumulator(full=full_metrics)
    sums: dict[str, float] = {"loss": 0.0, "loss_mse": 0.0, "loss_boundary_cross_entropy": 0.0}
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            seismic, target = batch["seismic"].to(device), batch["velocity"].to(device)
            if curriculum_stage is not None:
                seismic = curriculum_transform(seismic, curriculum_stage)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction, boundary_logits = model(seismic)
            loss, parts = loss_fn(prediction, boundary_logits, target)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            batch_size = len(target)
            count += batch_size
            sums["loss"] += float(loss.detach()) * batch_size
            for key, value in parts.items():
                sums[f"loss_{key}"] += float(value.detach()) * batch_size
            metrics.update(prediction, target)
    result = {key: value / max(count, 1) for key, value in sums.items()}
    result.update(metrics.compute())
    return result


def train_ddnet70(*, train_dataset: Dataset, eval_dataset: Dataset | None, config: DDNetTrainConfig,
                  checkpoint_dir: str | Path, initial_checkpoint: str | Path | None = None,
                  source_pretraining: bool = False) -> DDNetRunArtifacts:
    """Original source curriculum or identically initialized direct/proxy transfer.

    Recovery stores model/optimizer, RNG, stage progress and best validation
    selection state only; no sample IDs, data paths, manifests or history files.
    """
    if min(config.source_stage_epochs) < 1 or config.target_epochs < 1:
        raise ValueError("DD training epochs must be positive")
    device = _device(config.device)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if config.resume_from is None and any(checkpoint_dir.glob("*.pt")):
        raise FileExistsError(f"Use resume_from for existing checkpoints: {checkpoint_dir}")
    model = DDNet70().to(device)
    if initial_checkpoint is not None and config.resume_from is None:
        model.load_state_dict(load_ddnet70_checkpoint(initial_checkpoint, device=config.device).state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = DDNetLoss(velocity_weight=config.velocity_loss_weight, boundary_weight=config.boundary_loss_weight).to(device)
    stages = list(enumerate(config.source_stage_epochs, start=1)) if source_pretraining else [(None, config.target_epochs)]
    schedule = [stage for stage, epochs in stages for _ in range(epochs)]
    history: list[dict[str, float]] = []
    best_metrics, best_epoch, best_loss, best_state = {}, 0, float("inf"), None
    completed = 0
    critical = {key: value for key, value in asdict(config).items()
                if key not in {"device", "num_workers", "resume_from", "checkpoint_every_epochs"}}
    if config.resume_from is not None:
        saved = torch.load(config.resume_from, map_location=device, weights_only=False)
        if saved['config'] != critical or saved['schedule'] != schedule:
            raise ValueError("Resume DD architecture/training schedule differs")
        model.load_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        completed = int(saved['epoch'])
        if not 0 <= completed <= len(schedule):
            raise ValueError("Invalid DD resume epoch")
        best_loss, best_epoch = saved['best_loss'], saved['best_epoch']
        best_metrics, best_state = saved['best_metrics'], saved['best_model_state_dict']
        _restore_rng_state(saved['rng_state'])
    train_loader = _loader(train_dataset, config, training=True)
    eval_loader = None if eval_dataset is None else _loader(eval_dataset, config, training=False)

    def payload(epoch, state=None):
        return {"format": "tap_syn_ddnet70_v1", "model_state_dict": model.state_dict() if state is None else state,
                "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "config": critical,
                "schedule": schedule, "best_loss": best_loss, "best_epoch": best_epoch,
                "best_metrics": best_metrics, "best_model_state_dict": best_state, "rng_state": _rng_state()}

    for index in range(completed, len(schedule)):
        epoch = index + 1
        train_metrics = _epoch(model=model, loader=train_loader, optimizer=optimizer, loss_fn=loss_fn,
                              device=device, curriculum_stage=schedule[index], full_metrics=False)
        record = {f"train_{key}": value for key, value in train_metrics.items()}
        if eval_loader is not None:
            metrics = _epoch(model=model, loader=eval_loader, optimizer=None, loss_fn=loss_fn,
                             device=device, curriculum_stage=None, full_metrics=False)
            record.update({f"eval_{key}": value for key, value in metrics.items()})
            if metrics['loss'] < best_loss:
                best_loss, best_epoch, best_metrics = metrics['loss'], epoch, metrics
                best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
                _atomic_torch_save(payload(epoch, best_state), checkpoint_dir / "best.pt")
        else:
            best_epoch, best_metrics = epoch, train_metrics
        history.append(record)
        _atomic_torch_save(payload(epoch), checkpoint_dir / "latest.pt")
        if config.checkpoint_every_epochs > 0 and epoch % config.checkpoint_every_epochs == 0:
            _atomic_torch_save(payload(epoch), checkpoint_dir / f"epoch{epoch:04d}.pt")
        print(f"DD epoch {epoch}/{len(schedule)} stage={schedule[index]} metrics={record}", flush=True)
    if eval_loader is None:
        _atomic_torch_save(payload(len(schedule)), checkpoint_dir / "best.pt")
    elif best_state is not None and not (checkpoint_dir / "best.pt").exists():
        _atomic_torch_save(payload(len(schedule), best_state), checkpoint_dir / "best.pt")
    return DDNetRunArtifacts(checkpoint_dir / "best.pt", checkpoint_dir / "latest.pt", history, best_metrics, best_epoch)


@torch.no_grad()
def evaluate_ddnet70(*, checkpoint_path: str | Path, dataset: Dataset, config: DDNetTrainConfig, full_metrics: bool = True) -> dict[str, float]:
    device = _device(config.device)
    model = load_ddnet70_checkpoint(checkpoint_path, device=config.device).to(device).eval()
    loss_fn = DDNetLoss(velocity_weight=config.velocity_loss_weight, boundary_weight=config.boundary_loss_weight).to(device)
    return _epoch(model=model, loader=_loader(dataset, config, training=False), optimizer=None, loss_fn=loss_fn,
                  device=device, curriculum_stage=None, full_metrics=full_metrics)
