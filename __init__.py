"""TAP-Syn: coarse-conditioned residual rectified flow and DL-FWI transfer."""
from .coarse_generator import CoarseConfig, CoarseVelocityGenerator
from .nets import ResidualVelocityUNet, BPsi
from .methods import RectifiedFlowPath, ResidualVelocityRectifiedFlow
