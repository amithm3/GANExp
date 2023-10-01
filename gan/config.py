from dataclasses import dataclass

from utils.config import Config


@dataclass
class DCGanConfig(Config):
    inp_channels: int = 3
    hidden_channels: int = 64
    out_channels: int = 3
    upsample: int = 4
    n_blocks: int = 9
    n: int = 0
    blocks: tuple = (64, 128, 256, 512)
    betas: tuple = (0.5, 0.999)


__all__ = [
    "DCGanConfig",
]
