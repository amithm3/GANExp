from dataclasses import dataclass

from torch import nn

from . import DCGenerator, PatchDiscriminator, ResDiscriminator, ResGenerator, GanConfig


# === DCGAN ===
@dataclass
class DCGanConfig(GanConfig):
    latent_dim: int = 3
    out_channels: int = 3
    hidden_channels: int = 64
    head_kernel: int = 4
    head_stride: int = 1
    head_padding: int = 0


def build_DCGan(config: "DCGanConfig"):
    generator = DCGenerator(
        config.latent_dim, config.out_channels, config.hidden_channels,
        n=config.n, p=config.p, norm=nn.InstanceNorm2d, act=nn.ReLU,
        sampling_layers=config.sampling_layers,
        head_kernel=config.head_kernel, head_stride=config.head_stride, head_padding=config.head_padding,
    ).to(config.device)
    discriminator = PatchDiscriminator(
        config.out_channels, config.blocks,
        n=config.n, p=config.p, norm=nn.InstanceNorm2d, act=nn.LeakyReLU,
    ).to(config.device)
    return generator, discriminator


# === CycleGAN ===
@dataclass
class CycleGanConfig(GanConfig):
    inp_channels: int = 3
    hidden_channels: int = 64
    out_channels: int = 3
    residuals: int = 5
    head_kernel: int = 7
    head_stride: int = 1
    head_padding: int = 3
    sampling_kernel: int = 4
    sampling_stride: int = 2
    sampling_padding: int = 1


def build_CycleGan(config: "CycleGanConfig"):
    generatorA = ResGenerator(
        config.inp_channels, config.out_channels, config.hidden_channels,
        n=config.n, p=config.p, norm=nn.InstanceNorm2d, act=nn.ReLU,
        sampling_layers=config.sampling_layers, residuals=config.residuals,
        head_kernel=config.head_kernel, head_stride=config.head_stride, head_padding=config.head_padding,
        sampling_kernel=config.sampling_kernel, sampling_stride=config.sampling_stride,
        sampling_padding=config.sampling_padding,
    ).to(config.device)
    generatorB = ResGenerator(
        config.inp_channels, config.out_channels, config.hidden_channels,
        n=config.n, p=config.p, norm=nn.InstanceNorm2d, act=nn.ReLU,
        sampling_layers=config.sampling_layers, residuals=config.residuals,
        head_kernel=config.head_kernel, head_stride=config.head_stride, head_padding=config.head_padding,
        sampling_kernel=config.sampling_kernel, sampling_stride=config.sampling_stride,
        sampling_padding=config.sampling_padding,
    ).to(config.device)
    discriminatorA = PatchDiscriminator(
        config.out_channels, config.blocks,
        n=config.n, p=config.p, norm=nn.InstanceNorm2d, act=nn.LeakyReLU,
    ).to(config.device)
    discriminatorB = PatchDiscriminator(
        config.out_channels, config.blocks,
        n=config.n, p=config.p, norm=nn.InstanceNorm2d, act=nn.LeakyReLU,
    ).to(config.device)
    return generatorA, generatorB, discriminatorA, discriminatorB


__all__ = [
    "DCGanConfig", "build_DCGan",
    "CycleGanConfig", "build_CycleGan",
]
