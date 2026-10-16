"""Training loops for the ABA-FWI paper experiments."""

from __future__ import annotations

import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import FWIModelConfig, FWITrainConfig
from ..data import collate_fwi_batch
from .fwi_losses import aba_fwi_loss
from .fwi_models import build_fwi_model as build_configured_fwi_model
from .metrics import MetricAccumulator


@dataclass(frozen=True)
class FWIRunArtifacts:
    checkpoint_path: Path | None
    latest_checkpoint: Path
    history: list[dict[str, float]]
    best_metrics: dict[str, float]
    best_epoch: int | None


def _rng_state() -> dict[str, Any]:
    return {
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _coerce_rng_tensor(state: Any) -> torch.Tensor:
    if isinstance(state, torch.Tensor):
        return state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return torch.as_tensor(state, dtype=torch.uint8, device="cpu").contiguous()


def _restore_rng_state(state: dict[str, Any]) -> None:
    required = {"torch", "cuda", "numpy", "python"}
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError("Resume checkpoint has an incomplete RNG state.")
    torch.random.set_rng_state(_coerce_rng_tensor(state["torch"]))
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("Resume checkpoint contains CUDA RNG state, but CUDA is unavailable.")
        torch.cuda.set_rng_state_all([_coerce_rng_tensor(cuda_state) for cuda_state in state["cuda"]])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _critical_config(model_config: FWIModelConfig, train_config: FWITrainConfig) -> dict[str, Any]:
    return {
        "model": asdict(model_config),
        "train": {
            "batch_size": train_config.batch_size,
            "eval_batch_size": train_config.eval_batch_size,
            # Worker count only changes how samples are fetched. It must not
            # invalidate a checkpoint when reducing I/O pressure on resume.
            "learning_rate": train_config.learning_rate,
            "weight_decay": train_config.weight_decay,
            "eval_every_epochs": train_config.eval_every_epochs,
            "use_amp": train_config.use_amp,
        },
    }


def _autocast_context(device: torch.device, use_amp: bool):
    enabled = use_amp and device.type == "cuda"
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def build_fwi_model(config: FWIModelConfig) -> nn.Module:
    return build_configured_fwi_model(config)


def _make_loader(
    dataset: Dataset,
    train_config: FWITrainConfig,
    *,
    shuffle: bool,
    evaluation: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=train_config.eval_batch_size if evaluation else train_config.batch_size,
        num_workers=train_config.num_workers,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=collate_fwi_batch,
    )


def _run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    use_amp: bool,
    full_metrics: bool,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    metrics = MetricAccumulator(full=full_metrics)
    loss_sum = 0.0
    sample_count = 0
    if is_train and scaler is None:
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp and device.type == "cuda")
    gradient_context = nullcontext() if is_train else torch.no_grad()
    with gradient_context:
        total_batches = len(loader)
        for batch_index, batch in enumerate(loader, start=1):
            seismic = batch["seismic"].to(device)
            velocity = batch["velocity"].to(device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, use_amp):
                prediction = model(seismic)
            if prediction.shape != velocity.shape:
                raise ValueError(
                    f"ABAFWI prediction/target shapes differ: {tuple(prediction.shape)} / "
                    f"{tuple(velocity.shape)}."
                )
            loss, components = aba_fwi_loss(prediction, velocity)
            if is_train and optimizer is not None and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            batch_size = velocity.shape[0]
            loss_sum += float(loss.detach().item()) * batch_size
            sample_count += batch_size
            metrics.update(prediction, velocity)
            for name, value in components.items():
                key = f"loss_{name}"
                metrics._sums[key] = metrics._sums.get(key, 0.0) + float(value.detach().item()) * batch_size
            if batch_index == 1 or batch_index % 100 == 0 or batch_index == total_batches:
                mode = "train" if is_train else "eval"
                print(
                    f"{mode} batch {batch_index}/{total_batches} "
                    f"loss={float(loss.detach().item()):.6f}",
                    flush=True,
                )
    result = metrics.compute()
    if sample_count > 0:
        result["loss"] = loss_sum / sample_count
    return result


def train_fwi_model(
    *,
    model_config: FWIModelConfig,
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    train_config: FWITrainConfig,
    checkpoint_dir: str | Path,
    initial_checkpoint: str | Path | None = None,
    epochs: int | None = None,
) -> FWIRunArtifacts:
    device = _resolve_device(train_config.device)
    total_epochs = train_config.epochs if epochs is None else epochs
    if total_epochs < 1:
        raise ValueError("ABA epochs must be positive")
    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    resume_path = None if train_config.resume_from is None else Path(train_config.resume_from)
    if resume_path is None and any(checkpoint_root.glob("*.pt")):
        raise FileExistsError(
            f"Checkpoint directory already contains training checkpoints: {checkpoint_root}. "
            "Pass resume_from to continue it."
        )
    critical_config = _critical_config(model_config, train_config)


    model = build_fwi_model(model_config).to(device)
    if initial_checkpoint is not None and resume_path is None:
        payload = torch.load(Path(initial_checkpoint), map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
    optimizer_options: dict[str, Any] = {"lr": train_config.learning_rate}
    if train_config.weight_decay != 0.0:
        optimizer_options["weight_decay"] = train_config.weight_decay
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_options)
    scaler = torch.amp.GradScaler(device.type, enabled=train_config.use_amp and device.type == "cuda")
    train_loader = _make_loader(
        train_dataset,
        train_config,
        shuffle=True,
    )
    eval_loader = (
        None
        if eval_dataset is None
        else _make_loader(eval_dataset, train_config, shuffle=False, evaluation=True)
    )
    history: list[dict[str, float]] = []
    best_eval_loss: float | None = None
    best_metrics: dict[str, float] = {}
    best_checkpoint: Path | None = None
    best_model_state_dict: dict[str, torch.Tensor] | None = None
    best_epoch: int | None = None
    start_epoch = 1

    if resume_path is not None:
        resume_payload = torch.load(resume_path, map_location=device, weights_only=False)
        required = {
            "model_state_dict", "optimizer_state_dict", "scaler_state_dict", "epoch",
            "best_eval_loss", "best_metrics", "best_model_state_dict", "best_epoch", "rng_state",
            "total_epochs", "critical_config",
        }
        missing = sorted(required.difference(resume_payload))
        if missing:
            raise ValueError(f"Resume checkpoint is incomplete; missing keys: {missing}.")
        saved_epoch = resume_payload["epoch"]
        if not isinstance(saved_epoch, int) or saved_epoch < 0:
            raise ValueError("Resume checkpoint epoch must be a non-negative integer.")
        if resume_payload["total_epochs"] != total_epochs or saved_epoch > total_epochs:
            raise ValueError(
                f"Resume total epochs mismatch: checkpoint={resume_payload['total_epochs']}, "
                f"requested={total_epochs}, completed={saved_epoch}."
            )
        saved_critical_config = dict(resume_payload["critical_config"])
        saved_train_config = dict(saved_critical_config.get("train", {}))
        saved_train_config.pop("num_workers", None)
        saved_critical_config["train"] = saved_train_config
        if saved_critical_config != critical_config:
            raise ValueError("Resume checkpoint model/training configuration does not match the current run.")
        model.load_state_dict(resume_payload["model_state_dict"])
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scaler.load_state_dict(resume_payload["scaler_state_dict"])
        best_eval_loss = resume_payload["best_eval_loss"]
        best_metrics = dict(resume_payload["best_metrics"])
        best_model_state_dict = resume_payload["best_model_state_dict"]
        best_epoch = resume_payload["best_epoch"]
        best_checkpoint = checkpoint_root / "best.pt" if best_model_state_dict is not None else None
        start_epoch = saved_epoch + 1
        _restore_rng_state(resume_payload["rng_state"])

    def checkpoint_payload(epoch: int, model_state: dict[str, torch.Tensor] | None = None) -> dict[str, Any]:
        return {
            "checkpoint_version": 2,
            "model_state_dict": model.state_dict() if model_state is None else model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "best_eval_loss": best_eval_loss,
            "best_metrics": best_metrics,
            "best_model_state_dict": best_model_state_dict,
            "best_epoch": best_epoch,
            "rng_state": _rng_state(),
            "total_epochs": total_epochs,
            "critical_config": critical_config,
        }

    for epoch in range(start_epoch, total_epochs + 1):
        print(f"epoch {epoch}/{total_epochs} train", flush=True)
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            use_amp=train_config.use_amp,
            full_metrics=False,
            scaler=scaler,
        )
        epoch_metrics = {f"train_{key}": value for key, value in train_metrics.items()}
        should_evaluate = eval_loader is not None and (
            (train_config.eval_every_epochs > 0 and epoch % train_config.eval_every_epochs == 0)
            or epoch == total_epochs
        )
        if should_evaluate:
            print(f"epoch {epoch}/{total_epochs} eval", flush=True)
            eval_metrics = _run_epoch(
                model=model,
                loader=eval_loader,
                optimizer=None,
                device=device,
                use_amp=train_config.use_amp,
                # Checkpoint selection only needs the validation objective.
                # Expensive PIQ metrics are computed once for final reports.
                full_metrics=False,
            )
            epoch_metrics.update({f"eval_{key}": value for key, value in eval_metrics.items()})
            current_eval_loss = eval_metrics["loss"]
            if best_eval_loss is None or current_eval_loss < best_eval_loss:
                best_eval_loss = current_eval_loss
                best_metrics = dict(epoch_metrics)
                best_epoch = epoch
                best_model_state_dict = {
                    name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
                }
        history.append(epoch_metrics)
        if best_epoch == epoch:
            best_checkpoint = checkpoint_root / "best.pt"
            _atomic_torch_save(checkpoint_payload(epoch, best_model_state_dict), best_checkpoint)
        if train_config.checkpoint_every_epochs > 0 and epoch % train_config.checkpoint_every_epochs == 0:
            _atomic_torch_save(checkpoint_payload(epoch), checkpoint_root / f"epoch{epoch:04d}.pt")
        _atomic_torch_save(checkpoint_payload(epoch), checkpoint_root / "latest.pt")
        print(f"epoch {epoch}/{total_epochs} complete metrics={epoch_metrics}", flush=True)

    latest_checkpoint = checkpoint_root / "latest.pt"
    if start_epoch > total_epochs and not latest_checkpoint.exists():
        _atomic_torch_save(checkpoint_payload(total_epochs), latest_checkpoint)
    if best_model_state_dict is not None and not (checkpoint_root / "best.pt").exists():
        best_checkpoint = checkpoint_root / "best.pt"
        _atomic_torch_save(checkpoint_payload(total_epochs, best_model_state_dict), best_checkpoint)
    return FWIRunArtifacts(
        checkpoint_path=best_checkpoint if eval_dataset is not None else latest_checkpoint,
        latest_checkpoint=latest_checkpoint,
        history=history,
        best_metrics=best_metrics,
        best_epoch=best_epoch,
    )


def evaluate_fwi_model(
    *,
    checkpoint_path: str | Path,
    model_config: FWIModelConfig,
    dataset: Dataset,
    train_config: FWITrainConfig,
) -> dict[str, float]:
    device = _resolve_device(train_config.device)
    model = build_fwi_model(model_config).to(device)
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    loader = _make_loader(dataset, train_config, shuffle=False, evaluation=True)
    return _run_epoch(
        model=model,
        loader=loader,
        optimizer=None,
        device=device,
        use_amp=train_config.use_amp,
        full_metrics=train_config.full_eval_metrics,
    )
