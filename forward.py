"""Batch adapter for the extracted physical acoustic solver, not a pairing stub."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ForwardConfig:
    device: str = "cpu"
    pml_width: int = 120
    simulation_scale: float = 1.0
    max_vel: float = 6000.0


class OpenFWIForward:
    def __init__(self, config: ForwardConfig = ForwardConfig()):
        from .physics import create_openfwi_forward_simulator
        if config.pml_width < 0 or config.simulation_scale <= 0 or config.max_vel <= 0:
            raise ValueError("Invalid forward configuration")
        self.config = config
        self.simulator = create_openfwi_forward_simulator(
            device=config.device, pml_width=config.pml_width, simulation_scale=config.simulation_scale)
        self.simulator.max_vel = config.max_vel

    @torch.no_grad()
    def simulate(self, velocity_physical):
        from .physics import simulate_openfwi
        if velocity_physical.ndim != 4 or tuple(velocity_physical.shape[1:]) != (1, 70, 70):
            raise ValueError("Physical forward requires [B,1,70,70] in m/s")
        if not torch.isfinite(velocity_physical).all() or (velocity_physical <= 0).any():
            raise ValueError("Velocity must be finite and positive")
        if velocity_physical.numel() and float(velocity_physical.max()) > self.config.max_vel:
            raise ValueError(f"Velocity exceeds solver max_vel={self.config.max_vel}")
        outputs = [simulate_openfwi(sample[0].float(), simulator=self.simulator).detach().cpu()
                   for sample in velocity_physical]
        return torch.stack(outputs) if outputs else torch.empty(0, 5, 1000, 70)
