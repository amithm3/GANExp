import torch
from torch import nn

from utils.blocks import ConvBlock, SkipBlock, ResidualConvBlock
from .base import LatentGenerator, ConvGenerator


class DCGenerator(LatentGenerator):
    def __init__(self, latent_dim: int, out_channels: int, hidden_channels: int, **kwargs):
        """
        Deep Convolutional Generator.
        :param latent_dim: Dimension of latent space.
        :param out_channels: Number of output channels.
        :param hidden_channels: Number of hidden channels.
        :keyword n: Number of layers between each normalization layer.
        :keyword p: Dropout probability.
        :keyword norm: Normalization layer.
        :keyword act: Activation function.
        :keyword sample_layers: Number of sampling layers.
        :keyword head_kernel: Kernel size of the first convolutional layer.
        :keyword head_stride: Stride of the first convolutional layer.
        :keyword head_padding: Padding of the first convolutional layer.
        """
        head_kernel = kwargs.pop("head_kernel", 4)
        head_stride = kwargs.pop("head_stride", 1)
        head_padding = kwargs.pop("head_padding", 0)
        assert isinstance(head_kernel, int) and latent_dim > 0, \
            f"Invalid value of head_kernel, must be a positive integer, got {head_kernel}"
        assert isinstance(head_stride, int) and latent_dim > 0, \
            f"Invalid value of head_stride, must be a positive integer, got {head_stride}"
        assert isinstance(head_padding, int) and latent_dim > 0, \
            f"Invalid value of head_padding, must be a positive integer, got {head_padding}"
        super().__init__(latent_dim, out_channels, hidden_channels, **kwargs)
        n = self.n
        p = self.p
        norm = self.norm
        act = self.act
        sample_layers = self.sample_layers

        self.head = ConvBlock(latent_dim, hidden_channels * 2 ** sample_layers, act,
                              n=n, p=p, act_every_n=False, norm_every_n=True, down=False,
                              kernel_size=head_kernel, stride=head_stride, padding=head_padding)
        self.blocks = nn.Sequential(*reversed([
            ConvBlock(hidden_channels * 2 ** (i + 1), hidden_channels * 2 ** i, act, norm,
                      n=n, p=p, act_every_n=False, norm_every_n=True, down=False,
                      kernel_size=4, stride=2, padding=1)
            for i in range(sample_layers)
        ]))
        self.tail = ConvBlock(hidden_channels, out_channels, nn.Tanh,
                              n=n, p=p, act_every_n=False, norm_every_n=False,
                              kernel_size=4, stride=2, padding=1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        x = self.head(x)
        x = self.blocks(x)
        y = self.tail(x)
        return y


class ResGenerator(ConvGenerator):
    def __init__(self, inp_channels: int, out_channels: int, hidden_channels: int, **kwargs):
        """
        Residual Generator.
        :param inp_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param hidden_channels: Number of hidden channels.
        :keyword act: Activation function.
        :keyword norm: Normalization layer.
        :keyword n: Number of layers between each normalization layer.
        :keyword p: Dropout probability.
        :keyword sample_layers: Number of sampling layers.
        :keyword residuals: Number of residual blocks.
        :keyword head_kernel: Kernel size of the first convolutional layer.
        :keyword head_stride: Stride of the first convolutional layer.
        :keyword head_padding: Padding of the first convolutional layer.
        :keyword sampling_kernel: Kernel size of the sampling convolutional layers.
        :keyword sampling_stride: Stride of the sampling convolutional layers.
        :keyword sampling_padding: Padding of the sampling convolutional layers.
        """
        residuals = kwargs.pop("residuals", 9)
        head_kernel = kwargs.pop("head_kernel", 7)
        head_stride = kwargs.pop("head_stride", 1)
        head_padding = kwargs.pop("head_padding", 3)
        sampling_kernel = kwargs.pop("sampling_kernel", 4)
        sampling_stride = kwargs.pop("sampling_stride", 2)
        sampling_padding = kwargs.pop("sampling_padding", 1)
        assert isinstance(residuals, int) and residuals > 0, \
            f"Invalid value of residuals, must be a positive integer, got {residuals}"
        assert isinstance(head_kernel, int) and head_kernel > 0, \
            f"Invalid value of head_kernel, must be a positive integer, got {head_kernel}"
        assert isinstance(head_stride, int) and head_stride > 0, \
            f"Invalid value of head_stride, must be a positive integer, got {head_stride}"
        assert isinstance(head_padding, int) and head_padding > 0, \
            f"Invalid value of head_padding, must be a positive integer, got {head_padding}"
        assert isinstance(sampling_kernel, int) and sampling_kernel > 0, \
            f"Invalid value of sampling_kernel, must be a positive integer, got {sampling_kernel}"
        assert isinstance(sampling_stride, int) and sampling_stride > 0, \
            f"Invalid value of sampling_stride, must be a positive integer, got {sampling_stride}"
        assert isinstance(sampling_padding, int) and sampling_padding > 0, \
            f"Invalid value of sampling_padding, must be a positive integer, got {sampling_padding}"
        super().__init__(inp_channels, out_channels, hidden_channels, **kwargs)
        n = self.n
        p = self.p
        norm = self.norm
        act = self.act
        sample_layers = self.sample_layers

        self.head = ConvBlock(inp_channels, hidden_channels, act,
                              n=n, p=p, act_every_n=False, norm_every_n=True,
                              kernel_size=head_kernel, stride=head_stride, padding=head_padding)
        downsample_blocks = nn.ModuleList([
            ConvBlock(hidden_channels * 2 ** i, hidden_channels * 2 ** (i + 1), act, norm,
                      n=n, p=p, act_every_n=False, norm_every_n=True, down=True,
                      kernel_size=sampling_kernel, stride=sampling_stride, padding=sampling_padding)
            for i in range(sample_layers)
        ])
        res_channels = hidden_channels * 2 ** sample_layers
        residual_blocks = nn.Sequential(*[
            ResidualConvBlock(res_channels, res_channels, act, norm,
                              n=n, p=p, act_every_n=False, norm_every_n=True,
                              kernel_size=3, stride=1, padding=1)
            for _ in range(residuals)
        ])
        upsample_blocks = nn.ModuleList(reversed([
            ConvBlock(hidden_channels * 2 ** (i + 1) * 2, hidden_channels * 2 ** i, act, norm,
                      n=n, p=p, act_every_n=False, norm_every_n=True, down=False,
                      kernel_size=sampling_kernel, stride=sampling_stride, padding=sampling_padding)
            for i in range(sample_layers)
        ]))
        self.blocks = SkipBlock(downsample_blocks, residual_blocks, upsample_blocks)
        self.tail = ConvBlock(hidden_channels, out_channels, nn.Tanh,
                              n=n, p=p, act_every_n=False, norm_every_n=False,
                              kernel_size=head_kernel, stride=head_stride, padding=head_padding)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = self.head(x)
        x = self.blocks(x)
        y = self.tail(x)
        return y





__all__ = [
    "DCGenerator",
    "ResGenerator",
]
