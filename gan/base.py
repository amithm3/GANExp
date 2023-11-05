from dataclasses import dataclass

import torch
from torch import nn

from utils.config import Config


class Generator(nn.Module):
    def __init__(self, **kwargs):
        """
        Base class for all generators.
        :keyword n: Number of layers in each sampling layer.
        :keyword p: Dropout probability.
        :keyword norm: Normalization layer.
        :keyword act: Activation function.
        :keyword sampling_layers: Number of sampling layers.
        """
        n = kwargs.pop("n", 0)
        p = kwargs.pop("p", 0)
        norm = kwargs.pop("norm", nn.InstanceNorm2d)
        act = kwargs.pop("act", nn.ReLU)
        sampling_layers = kwargs.pop("sampling_layers", 2)
        assert isinstance(n, int) and n >= 0, \
            f"Invalid value of n, must be non negative integer, got {n}"
        assert 0 <= p <= 1, \
            f"Invalid value of p, must be between 0 and 1, got {p}"
        assert callable(norm), \
            f"Invalid value of norm, must be a callable, got {norm}"
        assert callable(act), \
            f"Invalid value of act, must be a callable, got {act}"
        assert isinstance(sampling_layers, int) and sampling_layers > 0, \
            f"Invalid value of sampling_layers, must be a positive integer, got {sampling_layers}"
        assert kwargs == {}, \
            f"Unused arguments: {kwargs}"
        super().__init__()
        self.n = n
        self.p = p
        self.norm = norm
        self.act = act
        self.sample_layers = sampling_layers


class LatentGenerator(Generator):
    def __init__(self, latent_dim: int, out_channels: int, hidden_channels: int, **kwargs):
        """
        Base class for all latent generators.
        :param latent_dim: Dimension of latent space.
        :param out_channels: Number of output channels.
        :param hidden_channels: Number of hidden channels.
        :keyword n: Number of layers in each sampling layer.
        :keyword p: Dropout probability.
        :keyword norm: Normalization layer.
        :keyword act: Activation function.
        :keyword sampling_layers: Number of sampling layers.
        """
        assert isinstance(latent_dim, int) and latent_dim > 0, \
            f"Invalid value of latent_dim, must be a positive integer, got {latent_dim}"
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

    def sample_noise(self, batch_size: int, device: str) -> "torch.Tensor":
        """
        Sample noise from latent space.
        :param batch_size: Batch size.
        :param device: Device to sample noise on.
        :return: Noise tensor.
        """
        return torch.randn(batch_size, self.latent_dim, device=device)


class ConvGenerator(Generator):
    def __init__(self, inp_channels: int, out_channels: int, hidden_channels: int, **kwargs):
        """
        Base class for all convolutional generators.
        :param inp_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param hidden_channels: Number of hidden channels.
        :keyword n: Number of layers between each normalization layer.
        :keyword p: Dropout probability.
        :keyword norm: Normalization layer.
        :keyword act: Activation function.
        :keyword sampling_layers: Number of sampling layers.
        """
        assert isinstance(inp_channels, int) and inp_channels > 0, \
            f"Invalid value of inp_channels, must be a positive integer, got {inp_channels}"
        assert isinstance(out_channels, int) and out_channels > 0, \
            f"Invalid value of out_channels, must be a positive integer, got {out_channels}"
        assert isinstance(hidden_channels, int) and hidden_channels > 0, \
            f"Invalid value of hidden_channels, must be a positive integer, got {hidden_channels}"
        super().__init__(**kwargs)
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

    def sample_noise(self, batch_size: int, w: int, h: int, device: str) -> "torch.Tensor":
        """
        Sample noise from latent space.
        :param batch_size: Batch size.
        :param w: Width of noise tensor.
        :param h: Height of noise tensor.
        :param device: Device to sample noise on.
        :return: Noise tensor.
        """
        return torch.randn(batch_size, self.inp_channels, h, w, device=device)


class Discriminator(nn.Module):
    def __init__(self, blocks, **kwargs):
        """
        Base class for all discriminators.
        :param blocks: Number of channels in each block.
        :keyword n: Number of layers in each sampling layer.
        :keyword p: Dropout probability.
        :keyword norm: Normalization layer.
        :keyword act: Activation function.
        """
        n = kwargs.pop("n", 0)
        p = kwargs.pop("p", 0)
        norm = kwargs.pop("norm", nn.InstanceNorm2d)
        act = kwargs.pop("act", nn.ReLU)
        assert isinstance(n, int) and n >= 0, \
            f"Invalid value of n, must be a non negative integer, got {n}"
        assert 0 <= p <= 1, \
            f"Invalid value of p, must be between 0 and 1, got {p}"
        assert callable(norm), \
            f"Invalid value of norm, must be a callable, got {norm}"
        assert callable(act), \
            f"Invalid value of act, must be a callable, got {act}"
        assert kwargs == {}, \
            f"Unused arguments: {kwargs}"
        super().__init__()
        self.blocks = blocks
        self.n = n
        self.p = p
        self.norm = norm
        self.act = act


class Critic(Discriminator):
    def __init__(self, blocks, **kwargs):
        """
        Base class for all critics.
        :param blocks: Number of channels in each block.
        :keyword n: Number of layers in each sampling layer.
        :keyword p: Dropout probability.
        :keyword norm: Normalization layer.
        :keyword act: Activation function.
        """
        super().__init__(blocks, **kwargs)


@dataclass
class GanConfig(Config):
    n: int = 0
    p: float = 0
    sampling_layers: int = 2
    lr: float = 0.0002
    betas: tuple[float, float] = (0.5, 0.999)
    blocks: tuple = (64, 128, 256, 512)


__all__ = [
    "Generator",
    "LatentGenerator",
    "ConvGenerator",
    "Discriminator",
    "Critic",
    "GanConfig",
]
