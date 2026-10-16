"""Memory-mapped physical-velocity chunks, with no on-disk index/metadata.

Velocity root: arbitrary *.npy files, or root/velocity/**/*.npy. Shapes are
[H,W], [N,H,W], or [N,1,H,W]. Each getitem reads only its in-file sample.
Formal domains use the same interface; prepare independent train/val/test roots
before calling training. The loader neither invents a split nor stores one.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from ..coarse_generator import CoarseVelocityGenerator, coarse_config_for_profile

DEFAULT_DATA_ROOT = Path("data")


def default_velocity_normalizer(velocity: Tensor, vmin: float = 1000.0, vmax: float = 6000.0) -> Tensor:
    if vmax <= vmin:
        raise ValueError("vmax must exceed vmin")
    return ((velocity - vmin) / (vmax - vmin)).clamp(0, 1)


def default_velocity_denormalizer(velocity: Tensor, vmin: float = 1000.0, vmax: float = 6000.0) -> Tensor:
    if vmax <= vmin:
        raise ValueError("vmax must exceed vmin")
    return velocity.clamp(0, 1) * (vmax - vmin) + vmin


class NpyVelocityChunks(Dataset):
    """Small bounded per-worker mmap cache; cumulative lengths replace sample lists."""

    def __init__(self, root: str | Path, *, sample_start: int | None = None,
                 sample_end: int | None = None, max_samples: int | None = None):
        root = Path(root).expanduser().resolve()
        self.root = root / "velocity" if (root / "velocity").is_dir() else root
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self.files = sorted(self.root.rglob("*.npy"))
        if not self.files:
            raise ValueError(f"No npy chunks in {self.root}")
        self.ends, self.shapes = [], []
        count = 0
        for path in self.files:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            shape = tuple(array.shape)
            if len(shape) == 2:
                size = 1
            elif len(shape) == 3:
                size = shape[0]
            elif len(shape) == 4 and shape[1] == 1:
                size = shape[0]
            else:
                raise ValueError(f"Unsupported velocity shape {shape}: {path}")
            if size < 1 or min(shape[-2:]) < 2 or array.dtype.kind not in "fiu":
                raise ValueError(f"Invalid velocity chunk: {path}")
            self.shapes.append(shape)
            count += size
            self.ends.append(count)
        self.start = 0 if sample_start is None else sample_start
        stop = count if sample_end is None else sample_end
        if not 0 <= self.start < stop <= count:
            raise ValueError(f"Invalid sample range [{self.start}, {stop}) of {count}")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self.stop = stop if max_samples is None else min(stop, self.start + max_samples)
        self._cache = OrderedDict()
        self._pid = os.getpid()

    def __len__(self):
        return self.stop - self.start

    def locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        absolute = index + self.start
        chunk = bisect_right(self.ends, absolute)
        return chunk, absolute - (0 if chunk == 0 else self.ends[chunk - 1])

    def _array(self, path: Path):
        if self._pid != os.getpid():
            self._cache = OrderedDict()
            self._pid = os.getpid()
        if path not in self._cache:
            self._cache[path] = np.load(path, mmap_mode="r", allow_pickle=False)
        self._cache.move_to_end(path)
        while len(self._cache) > 4:
            self._cache.popitem(last=False)
        return self._cache[path]

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_cache'] = OrderedDict()
        return state

    def __getitem__(self, index: int) -> Tensor:
        chunk, offset = self.locate(index)
        array = self._array(self.files[chunk])
        sample = array if array.ndim == 2 else array[offset]
        velocity = torch.from_numpy(np.array(sample, dtype=np.float32, copy=True))
        if velocity.ndim == 2:
            velocity = velocity.unsqueeze(0)
        if not torch.isfinite(velocity).all() or float(velocity.min()) <= 0:
            raise ValueError(f"Expected finite positive velocity in m/s: {self.files[chunk]}")
        return velocity


class VelocityNpyDataset(NpyVelocityChunks):
    """Generator dataset: physical D(m), normalized condition and residual at 128²."""

    def __init__(self, root: str | Path = DEFAULT_DATA_ROOT, *, target_size=(128, 128),
                 coarse_method="mixed", coarse_profile="strong_less_smooth", **kwargs):
        super().__init__(root, **kwargs)
        if tuple(target_size) != (128, 128):
            raise ValueError("The generator operates at 128x128")
        self.target_size = tuple(target_size)
        self.coarse_method = coarse_method
        self.coarse_generator = CoarseVelocityGenerator(coarse_config_for_profile(coarse_profile))

    def __getitem__(self, index):
        velocity = super().__getitem__(index)
        velocity = F.interpolate(velocity[None], size=self.target_size, mode="bilinear", align_corners=False)[0]
        velocity_norm = default_velocity_normalizer(velocity)
        coarse = self.coarse_generator(velocity, method=self.coarse_method, return_info=False)
        condition = default_velocity_normalizer(coarse)
        return {"velocity": velocity, "velocity_norm": velocity_norm, "m_c": coarse,
                "m_c_cond": condition, "r0": velocity_norm - condition}
