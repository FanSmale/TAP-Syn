"""Minimal executable workflow; run `python -m tap_syn --help` for arguments.

Typical commands (use `uv run tap-syn` after installing the project):
  tap-syn rf-train --train SOURCE_TRAIN --validation SOURCE_VAL --output RF_CKPT
  tap-syn lora-train --train TARGET_TRAIN --checkpoint RF_CKPT.pt --output LORA_CKPT
  tap-syn proxy --train TARGET_TRAIN --checkpoint LORA_CKPT.pt --output PROXY_PAIRS
  tap-syn fwi-train --network aba --mode source --train SOURCE_TRAIN --output ABA_SOURCE
  tap-syn fwi-train --network dd --mode direct --train TARGET_TRAIN --validation TARGET_VAL \
      --checkpoint DD_SOURCE.pt --output DD_DIRECT
  tap-syn fwi-train --network aba --mode proxy --train PROXY_PAIRS --validation TARGET_VAL \
      --checkpoint ABA_SOURCE.pt --output ABA_PROXY
  tap-syn evaluate --network aba --train TARGET_TEST --checkpoint ABA_PROXY/best.pt
  tap-syn forward --train VELOCITY_ROOT --output PHYSICAL_PAIRS

Validation/test are separate user-prepared roots; they are not automatically
split. Proxy generation uses only --train. No metadata or result file is saved.
Config modules and torch checkpoints are executable formats; trust their source.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import runpy
import torch
import torch.nn.functional as F

from .settings import Config
from .data import NpyVelocityChunks, PairedNpyDataset
from .training import DatasetConfig, TrainingStrategyConfig, train_training_strategy
from .lora import LoraFinetuneConfig, train_lora_finetune
from .forward import OpenFWIForward
from .generation import PairChunkWriter, load_generation_stack, build_proxy
from .fwi.config import FWIModelConfig
from .fwi.fwi_train import train_fwi_model, evaluate_fwi_model
from .fwi.ddnet_train import train_ddnet70, evaluate_ddnet70


def parser():
    root = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = root.add_subparsers(dest="command", required=True)
    for command in ('rf-train', 'lora-train', 'proxy', 'forward', 'fwi-train', 'evaluate'):
        p = sub.add_parser(command)
        p.add_argument('--config', type=Path, help='Trusted Python file defining CONFIG=Config(...)')
        p.add_argument('--train', type=Path, required=True, help='Input root; evaluate uses test root here')
        p.add_argument('--device', choices=('cpu', 'cuda'))
        p.add_argument('--max-samples', type=int)
        p.add_argument('--batch-size', type=int)
        if command != 'evaluate':
            p.add_argument('--output', type=Path, required=True)
        if command in ('rf-train', 'lora-train', 'fwi-train'):
            p.add_argument('--validation', type=Path)
            p.add_argument('--lr', type=float)
            p.add_argument('--workers', type=int)
            p.add_argument('--resume', type=Path)
        if command in ('lora-train', 'proxy', 'evaluate'):
            p.add_argument('--checkpoint', type=Path, required=True)
        if command == 'rf-train':
            p.add_argument('--stage-epochs', type=int, nargs=4, metavar=('S1', 'S2A', 'S2B', 'S3'))
        if command == 'lora-train':
            p.add_argument('--lora-epochs', type=int, nargs=3, metavar=('BPSI', 'CONDITIONAL', 'BRIDGE'))
            p.add_argument('--rank', type=int)
            p.add_argument('--alpha', type=float)
            p.add_argument('--multiplier', type=float)
        if command == 'proxy':
            p.add_argument('--base-checkpoint', type=Path, help='Required for adapter-only checkpoints')
            p.add_argument('--anchor-count', type=int, help='Per training reference (default 14)')
            p.add_argument('--conditional-count', type=int, help='Per training reference (default 12)')
            p.add_argument('--unconditional-count', type=int, help='Per training reference (default 36)')
            p.add_argument('--steps', type=int)
        if command in ('proxy', 'forward'):
            p.add_argument('--pml-width', type=int)
            p.add_argument('--simulation-scale', type=float)
            p.add_argument('--chunk-size', type=int)
        if command in ('fwi-train', 'evaluate'):
            p.add_argument('--network', choices=('aba', 'dd'), required=True)
        if command == 'fwi-train':
            p.add_argument('--mode', choices=('source', 'direct', 'proxy'), required=True)
            p.add_argument('--checkpoint', type=Path, help='Source-pretrained weights for either transfer mode')
            p.add_argument('--epochs', type=int, help='ABA epochs or DD target epochs')
            p.add_argument('--dd-stage-epochs', type=int, nargs=3)
        if command == 'evaluate':
            p.add_argument('--full-metrics', action='store_true', help='Add PIQ SSIM/PSNR/UQI/LPIPS (requires local weights)')
    return root


def _override(config, args, **mapping):
    return replace(config, **{key: getattr(args, value) for key, value in mapping.items()
                              if getattr(args, value, None) is not None})


def _check_validation(train, validation):
    if validation is None:
        return
    # Detect physical file/range overlap even when symlinks or distinct roots refer
    # to the same data. No split IDs are materialized or persisted.
    for i, path in enumerate(train.files):
        lo = max(train.start, 0 if i == 0 else train.ends[i - 1])
        hi = min(train.stop, train.ends[i])
        if lo >= hi:
            continue
        for j, other in enumerate(validation.files):
            if path.resolve() != other.resolve():
                continue
            other_start = 0 if j == 0 else validation.ends[j - 1]
            vlo, vhi = max(validation.start, other_start), min(validation.stop, validation.ends[j])
            base = 0 if i == 0 else train.ends[i - 1]
            if max(lo - base, vlo - other_start) < min(hi - base, vhi - other_start):
                raise ValueError('Training and validation samples overlap')


def main(argv=None):
    args = parser().parse_args(argv)
    config = Config() if args.config is None else runpy.run_path(str(args.config))['CONFIG']
    if not isinstance(config, Config):
        raise TypeError('CONFIG must be tap_syn.settings.Config')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if getattr(args, 'batch_size', None) is not None and args.batch_size < 1:
        raise ValueError('batch size must be positive')
    if args.command in ('rf-train', 'lora-train'):
        train = DatasetConfig(root=args.train, max_samples=args.max_samples,
                              coarse_method=config.coarse_method, coarse_profile=config.coarse_profile)
        validation = None if args.validation is None else replace(train, root=args.validation, max_samples=None)
        from .training.factory import build_dataset
        _check_validation(build_dataset(train), build_dataset(validation))
        if args.command == 'rf-train':
            runtime = _override(config.rf_runtime, args, device='device', batch_size='batch_size', num_workers='workers')
            runtime = replace(runtime, checkpoint_dir=args.output, resume_from=args.resume)
            optimizer = _override(config.rf_optimizer, args, learning_rate='lr')
            stages = config.stages
            if args.stage_epochs is not None:
                if min(args.stage_epochs) < 1:
                    raise ValueError('RF stage epochs must be positive')
                stages = tuple(replace(s, epochs=n) for s, n in zip(stages, args.stage_epochs))
            result = train_training_strategy(TrainingStrategyConfig(train_dataset=train, eval_dataset=validation,
                model=config.model, loss=config.loss, optimizer=optimizer, runtime=runtime, stages=stages))
            print(f"RF checkpoint: {result['last_checkpoint']}")
        else:
            runtime = _override(config.lora_runtime, args, device='device', batch_size='batch_size', num_workers='workers')
            runtime = replace(runtime, checkpoint_dir=args.output)
            optimizer = _override(config.lora_optimizer, args, learning_rate='lr')
            modules = _override(config.lora, args, rank='rank', alpha='alpha', multiplier='multiplier')
            stages = config.lora_stages
            if args.lora_epochs is not None:
                b, c, bridge = args.lora_epochs
                if b < 0 or c < 1 or bridge < 0:
                    raise ValueError('LoRA requires nonnegative Bpsi/bridge and positive conditional epochs')
                stages = replace(stages, epochs=b+c+bridge, b_psi_epochs=b, unconditional_bridge_epochs=bridge)
            result = train_lora_finetune(LoraFinetuneConfig(train_dataset=train, eval_dataset=validation,
                model=config.model, loss=config.loss, optimizer=optimizer, runtime=runtime, lora=modules,
                lora_runtime=stages, base_checkpoint=args.checkpoint, resume_from=args.resume))
            print(f"LoRA checkpoint: {result['latest_checkpoint']}")
    elif args.command in ('proxy', 'forward'):
        physical = _override(config.forward, args, device='device', pml_width='pml_width', simulation_scale='simulation_scale')
        forward = OpenFWIForward(physical)
        proxy = _override(config.proxy, args, batch_size='batch_size', chunk_size='chunk_size')
        if args.command == 'proxy':
            proxy = _override(proxy, args, anchor_per_reference='anchor_count', conditional_per_reference='conditional_count',
                              unconditional_per_reference='unconditional_count', num_steps='steps')
            flow, b_psi = load_generation_stack(args.checkpoint, model=config.model,
                device=physical.device, base_checkpoint=args.base_checkpoint)
            build_proxy(reference_root=args.train, output_root=args.output, flow=flow, b_psi=b_psi,
                        forward=forward, config=proxy, device=physical.device, max_references=args.max_samples,
                        coarse_method=config.coarse_method, coarse_profile=config.coarse_profile)
        else:
            data = NpyVelocityChunks(args.train, max_samples=args.max_samples)
            writer = PairChunkWriter(args.output, proxy.chunk_size)
            for start in range(0, len(data), proxy.batch_size):
                # Different physical input sizes are allowed across chunks.
                batch = torch.cat([F.interpolate(data[i][None], (70, 70), mode='bilinear', align_corners=False, antialias=True)
                                   for i in range(start, min(start + proxy.batch_size, len(data)))])
                writer.add(batch, forward.simulate(batch))
            writer.flush()
            print(f'forward pairs: {writer.count}')
    else:
        data = PairedNpyDataset(args.train, max_samples=args.max_samples)
        selected = config.aba if args.network == 'aba' else config.dd
        selected = _override(selected, args, device='device')
        if args.command == 'evaluate':
            selected = _override(selected, args, eval_batch_size='batch_size')
            if args.network == 'aba':
                selected = replace(selected, full_eval_metrics=args.full_metrics)
                metrics = evaluate_fwi_model(checkpoint_path=args.checkpoint, model_config=FWIModelConfig(),
                                             dataset=data, train_config=selected)
            else:
                metrics = evaluate_ddnet70(checkpoint_path=args.checkpoint, dataset=data, config=selected,
                                           full_metrics=args.full_metrics)
            print(metrics)
        else:
            if args.mode != 'source' and args.checkpoint is None:
                raise ValueError('Both direct and proxy transfer require --checkpoint from source pretraining')
            validation = None if args.validation is None else PairedNpyDataset(args.validation)
            _check_validation(data, validation)
            selected = _override(selected, args, batch_size='batch_size', learning_rate='lr', num_workers='workers')
            if args.network == 'aba':
                epochs = args.epochs if args.epochs is not None else (selected.epochs if args.mode == 'source' else selected.finetune_epochs)
                selected = replace(selected, resume_from=args.resume)
                result = train_fwi_model(model_config=FWIModelConfig(), train_dataset=data, eval_dataset=validation,
                    train_config=selected, checkpoint_dir=args.output, initial_checkpoint=args.checkpoint, epochs=epochs)
                print(f"ABA checkpoint: {result.checkpoint_path}")
            else:
                if args.resume is not None:
                    selected = replace(selected, resume_from=args.resume)
                if args.epochs is not None:
                    selected = replace(selected, target_epochs=args.epochs)
                if args.dd_stage_epochs is not None:
                    selected = replace(selected, source_stage_epochs=tuple(args.dd_stage_epochs))
                result = train_ddnet70(train_dataset=data, eval_dataset=validation, config=selected,
                    checkpoint_dir=args.output, initial_checkpoint=args.checkpoint, source_pretraining=args.mode == 'source')
                print(f"DD checkpoint: {result.checkpoint_path}")
