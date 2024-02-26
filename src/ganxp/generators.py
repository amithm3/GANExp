import torch
from torch import nn

from .base import Vec2ImgGenerator
from utils.blocks import ConvBlock
from utils.assertions import _assert_positive_integer, _assert_integer


class DCGenerator(Vec2ImgGenerator):
    def __init__(self, latents: int, out_chn: int, hid_chn: int, **kwargs):
        """
        Deep Convolutional Generator.
        :param latents: Dimension of latent space.
        :param out_chn: Number of output channels.
        :param hid_chn: Number of hidden channels.
        :keyword act: Activation function.
        :keyword norm: Normalization function.
        :keyword n: Block size of each sampling layer.
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
        _assert_positive_integer(
            head_kernel=head_kernel,
            head_stride=head_stride,
            sampling_kernel=sampling_kernel,
            sampling_stride=sampling_stride,
        )
        _assert_integer(
            head_padding=head_padding,
            head_kernel=head_kernel,
        )

        super().__init__(latents, out_chn, hid_chn, **kwargs)
        self.head_kernel = head_kernel
        self.head_stride = head_stride
        self.head_padding = head_padding
        self.sampling_kernel = sampling_kernel
        self.sampling_stride = sampling_stride
        self.sampling_padding = sampling_padding

        self.head = ConvBlock(
            self.latents, self.hid_chn * 2 ** self.sampling, self.act, self.norm,
            n=self.n, p=self.p, act_every_n=False, norm_every_n=True, down=False,
            kernel_size=self.head_kernel, stride=self.head_stride, padding=self.head_padding
        )
        self.blocks = nn.Sequential(*reversed([
            ConvBlock(
                self.hid_chn * 2 ** (i + 1), self.hid_chn * 2 ** i, self.act, self.norm,
                n=self.n, p=self.p, act_every_n=False, norm_every_n=True, down=False,
                kernel_size=self.sampling_kernel, stride=self.sampling_stride, padding=self.sampling_padding
            )
            for i in range(self.sampling)
        ]))
        self.tail = ConvBlock(
            self.hid_chn, self.out_chn, nn.Tanh, None,
            n=self.n, p=self.p, act_every_n=False, norm_every_n=False,
            kernel_size=self.sampling_kernel, stride=self.sampling_stride, padding=self.sampling_padding
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        x = self.head(x)
        x = self.blocks(x)
        y = self.tail(x)
        return y
