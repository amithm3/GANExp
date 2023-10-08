from dataclasses import dataclass

from torch import nn

from . import DCGenerator, PatchDiscriminator, ResGenerator
from utils.config import Config


# === DCGAN ===
@dataclass
class DCGanConfig(Config):
    inp_channels: int = 3
    hidden_channels: int = 64
    out_channels: int = 3
    upsample: int = 4
    n: int = 0
    blocks: tuple = (64, 128, 256, 512)
    betas: tuple = (0.5, 0.999)


def build_DCGan(config: "DCGanConfig"):
    generator = DCGenerator(
        config.inp_channels, config.out_channels, config.hidden_channels,
        n=config.n, p=config.p, norm=config.norm, act=nn.ReLU,
        upsample=config.upsample,
    ).to(config.device)
    discriminator = PatchDiscriminator(
        config.out_channels, config.blocks,
        n=config.n, p=config.p, norm=config.norm, act=nn.LeakyReLU,
    ).to(config.device)
    return generator, discriminator


# === CycleGAN ===
@dataclass
class CycleGanConfig(Config):
    inp_channels: int = 3
    hidden_channels: int = 64
    out_channels: int = 3
    downsample: int = 4
    residuals: int = 9
    n: int = 0
    blocks: tuple = (64, 128, 256, 512)
    betas: tuple = (0.5, 0.999)


def build_CycleGan(config: "CycleGanConfig"):
    generator1 = ResGenerator(
        config.inp_channels, config.out_channels, config.hidden_channels,
        n=config.n, p=config.p, norm=config.norm, act=nn.ReLU,
        downsample=config.downsample, residuals=config.residuals,
    ).to(config.device)
    generator2 = ResGenerator(
        config.inp_channels, config.out_channels, config.hidden_channels,
        n=config.n, p=config.p, norm=config.norm, act=nn.ReLU,
        downsample=config.downsample, residuals=config.residuals,
    ).to(config.device)
    discriminator1 = PatchDiscriminator(
        config.out_channels, config.blocks,
        n=config.n, p=config.p, norm=config.norm, act=nn.LeakyReLU,
    ).to(config.device)
    discriminator2 = PatchDiscriminator(
        config.out_channels, config.blocks,
        n=config.n, p=config.p, norm=config.norm, act=nn.LeakyReLU,
    ).to(config.device)

    return generator1, generator2, discriminator1, discriminator2


__all__ = [
    "DCGanConfig", "build_DCGan",
    "CycleGanConfig", "build_CycleGan",
]
