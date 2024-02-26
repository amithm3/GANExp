import torch
from torch import nn

from utils.blocks import ConvBlock, ResidualConvBlock, SkipBlock

from .base import Vec2ImgGenerator, Img2ImgGenerator


class DCGenerator(Vec2ImgGenerator):
    def __init__(self, latents: int, out_chn: int, hid_chn: int, **kwargs):
        """
        Deep Convolutional Generator.
        :param latents: Dimension of latent space.
        :param out_chn: Number of output channels.
        :param hid_chn: Number of hidden channels.
        :keyword act: Activation function.
        :keyword norm: Normalization layer.
        :keyword n: Number of layers in each sampling layer.
        :keyword p: Dropout probability.
        :keyword sampling: Number of sampling layers.
        :keyword head_kernel: Kernel size of the first convolutional layer.
        :keyword head_stride: Stride of the first convolutional layer.
        :keyword head_padding: Padding of the first convolutional layer.
        :keyword sampling_kernel: Kernel size of the sampling convolutional layers.
        :keyword sampling_stride: Stride of the sampling convolutional layers.
        :keyword sampling_padding: Padding of the sampling convolutional layers.
        """
        head_kernel = kwargs.pop("head_kernel", 4)
        head_stride = kwargs.pop("head_stride", 1)
        head_padding = kwargs.pop("head_padding", 0)
        sampling_kernel = kwargs.pop("sampling_kernel", 4)
        sampling_stride = kwargs.pop("sampling_stride", 2)
        sampling_padding = kwargs.pop("sampling_padding", 1)
        assert isinstance(head_kernel, int) and latents > 0, \
            f"Invalid value of head_kernel, must be a positive integer, got {head_kernel}"
        assert isinstance(head_stride, int) and latents > 0, \
            f"Invalid value of head_stride, must be a positive integer, got {head_stride}"
        assert isinstance(head_padding, int) and latents > 0, \
            f"Invalid value of head_padding, must be a positive integer, got {head_padding}"
        assert isinstance(sampling_kernel, int) and latents > 0, \
            f"Invalid value of sampling_kernel, must be a positive integer, got {sampling_kernel}"
        assert isinstance(sampling_stride, int) and latents > 0, \
            f"Invalid value of sampling_stride, must be a positive integer, got {sampling_stride}"
        super().__init__(latents, out_chn, hid_chn, **kwargs)
        self.head_kernel = head_kernel
        self.head_stride = head_stride
        self.head_padding = head_padding
        self.sampling_kernel = sampling_kernel
        self.sampling_stride = sampling_stride
        self.sampling_padding = sampling_padding

        self.head = ConvBlock(
            latents, hid_chn * 2 ** self.sampling, self.act, self.norm,
            n=self.n, p=self.p, act_every_n=False, norm_every_n=True, down=False,
            kernel_size=self.head_kernel, stride=self.head_stride, padding=self.head_padding
        )
        self.blocks = nn.Sequential(*reversed([
            ConvBlock(
                hid_chn * 2 ** (i + 1), hid_chn * 2 ** i, self.act, self.norm,
                n=self.n, p=self.p, act_every_n=False, norm_every_n=True, down=False,
                kernel_size=self.sampling_kernel, stride=self.sampling_stride, padding=self.sampling_padding
            )
            for i in range(self.sampling)
        ]))
        self.tail = ConvBlock(
            hid_chn, out_chn, nn.Tanh, None,
            n=self.n, p=self.p, act_every_n=False, norm_every_n=False,
            kernel_size=self.sampling_kernel, stride=self.sampling_stride, padding=self.sampling_padding
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        x = self.head(x)
        x = self.blocks(x)
        y = self.tail(x)
        return y


class ResGenerator(Img2ImgGenerator):
    def __init__(self, inp_chn: int, out_chn: int, hid_chn: int, **kwargs):
        """
        Residual Generator
        :param inp_chn: Number of input channels.
        :param out_chn: Number of output channels.
        :param hid_chn: Number of hidden channels.
        :keyword act: Activation function.
        :keyword norm: Normalization layer.
        :keyword n: Number of layers between each normalization layer.
        :keyword p: Dropout probability.
        :keyword sampling: Number of sampling layers.
        :keyword residuals: Number of residual blocks.
        :keyword head_kernel: Kernel size of the first convolutional layer.
        :keyword head_stride: Stride of the first convolutional layer.
        :keyword head_padding: Padding of the first convolutional layer.
        :keyword sampling_kernel: Kernel size of the sampling convolutional layers.
        :keyword sampling_stride: Stride of the sampling convolutional layers.
        :keyword sampling_padding: Padding of the sampling convolutional layers.
        """
        head_kernel = kwargs.pop("head_kernel", 7)
        head_stride = kwargs.pop("head_stride", 1)
        head_padding = kwargs.pop("head_padding", 3)
        sampling_kernel = kwargs.pop("sampling_kernel", 4)
        sampling_stride = kwargs.pop("sampling_stride", 2)
        sampling_padding = kwargs.pop("sampling_padding", 1)
        residuals = kwargs.pop("residuals", 5)
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
        assert isinstance(residuals, int) and residuals > 0, \
            f"Invalid value of residuals, must be a positive integer, got {residuals}"
        super().__init__(inp_chn, out_chn, hid_chn, **kwargs)
        self.head_kernel = head_kernel
        self.head_stride = head_stride
        self.head_padding = head_padding
        self.sampling_kernel = sampling_kernel
        self.sampling_stride = sampling_stride
        self.sampling_padding = sampling_padding
        self.residuals = residuals

        self.head = ConvBlock(
            self.inp_chn, self.hid_chn, self.act, self.norm,
            n=self.n, p=self.p, act_every_n=False, norm_every_n=True,
            kernel_size=self.head_kernel, stride=self.head_stride, padding=self.head_padding
        )
        downsample_blocks = nn.ModuleList([
            ConvBlock(
                self.hid_chn * 2 ** i, self.hid_chn * 2 ** (i + 1), self.act, self.norm,
                n=self.n, p=self.p, act_every_n=False, norm_every_n=True, down=True,
                kernel_size=self.sampling_kernel, stride=self.sampling_stride, padding=self.sampling_padding
            )
            for i in range(self.sampling)
        ])
        bottleneck_blocks = nn.Sequential(*[
            ResidualConvBlock(
                chn := self.hid_chn * 2 ** self.sampling, chn, self.act, self.norm,
                n=self.n, p=self.p, act_every_n=False, norm_every_n=True,
                kernel_size=3, stride=1, padding=1,
            )
            for _ in range(self.residuals)
        ])
        upsample_blocks = nn.ModuleList(reversed([
            ConvBlock(
                self.hid_chn * 2 ** (i + 1) * 2, self.hid_chn * 2 ** i, self.act, self.norm,
                n=self.n, p=self.p, act_every_n=False, norm_every_n=True, down=False,
                kernel_size=self.sampling_kernel, stride=self.sampling_stride, padding=self.sampling_padding
            )
            for i in range(self.sampling)
        ]))
        self.blocks = SkipBlock(downsample_blocks, bottleneck_blocks, upsample_blocks)
        self.tail = ConvBlock(
            self.hid_chn, self.out_chn, nn.Tanh, None,
            n=self.n, p=self.p, act_every_n=False, norm_every_n=False,
            kernel_size=self.head_kernel, stride=self.head_stride, padding=self.head_padding
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = self.head(x)
        x = self.blocks(x)
        y = self.tail(x)
        return y


__all__ = [
    "DCGenerator",
    "ResGenerator",
]
