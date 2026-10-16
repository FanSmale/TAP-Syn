"""Paired chunks: root/{velocity,seismic}/<same relative name>.npy.

Velocity is physical m/s, seismic is raw simulator amplitude. FWI targets are
normalized globally with 1000--6000 m/s, never per-image. Chunk lengths and shapes
must match; missing/extra names fail rather than silently dropping pairs.
"""
from pathlib import Path
import numpy as np
import torch
from .velocity_npy import NpyVelocityChunks, default_velocity_normalizer


class PairedNpyDataset(NpyVelocityChunks):
    def __init__(self, root, **kwargs):
        root = Path(root).expanduser().resolve()
        if not (root / "velocity").is_dir() or not (root / "seismic").is_dir():
            raise ValueError(f"Expected velocity/ and seismic/ in {root}")
        super().__init__(root, **kwargs)
        self.seismic_root = root / "seismic"
        relative = [p.relative_to(self.root) for p in self.files]
        found = {p.relative_to(self.seismic_root) for p in self.seismic_root.rglob('*.npy')}
        if set(relative) != found:
            raise ValueError("Seismic and velocity chunk filenames differ")
        self.seismic_files = [self.seismic_root / p for p in relative]
        for i, (path, vshape) in enumerate(zip(self.seismic_files, self.shapes)):
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            expected_count = self.ends[i] - (0 if i == 0 else self.ends[i - 1])
            if tuple(vshape[-2:]) != (70, 70):
                raise ValueError(f"FWI velocity must be 70x70: {self.files[i]}")
            if array.shape not in [(5, 1000, 70), (expected_count, 5, 1000, 70)]:
                raise ValueError(f"Invalid seismic shape {array.shape}: {path}")
            if array.ndim == 3 and expected_count != 1:
                raise ValueError(f"Pair length mismatch: {path}")

    def __getitem__(self, index):
        velocity_physical = super().__getitem__(index)
        chunk, offset = self.locate(index)
        array = self._array(self.seismic_files[chunk])
        sample = array if array.ndim == 3 else array[offset]
        seismic = torch.from_numpy(np.array(sample, dtype=np.float32, copy=True))
        if not torch.isfinite(seismic).all():
            raise ValueError(f"Nonfinite seismic: {self.seismic_files[chunk]}")
        return {"seismic": seismic, "velocity": default_velocity_normalizer(velocity_physical)}


def collate_fwi_batch(samples):
    return {key: torch.stack([sample[key] for sample in samples]) for key in ("seismic", "velocity")}
