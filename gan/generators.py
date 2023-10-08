from torch import nn

from utils.blocks import ConvBlock, ResidualConvBlock, SkipBlock


class DCGenerator(nn.Module):
    def __init__(self, inp_channels: int, out_channels: int, hidden_channels: int, **kwargs):
        """
        Deep Convolutional Generator.
        :param inp_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param hidden_channels: Number of hidden channels.
        :keyword n: Number of layers between each upsampling layer.
        :keyword p: Dropout probability.
        :keyword norm: Normalization layer.
        :keyword act: Activation function.
        :keyword upsample: Number of upsampling blocks.
        :keyword head_kernel: Kernel size of the first convolutional layer.
        :keyword head_stride: Stride of the first convolutional layer.
        """
        super().__init__()
        n = kwargs.pop("n", 1)
        p = kwargs.pop("p", 0)
        norm = kwargs.pop("norm", nn.InstanceNorm2d)
        act = kwargs.pop("act", nn.ReLU)
        upsample = kwargs.pop("upsample", 4)
        head_kernel = kwargs.pop("kernel_size", 4)
        head_stride = kwargs.pop("stride", 1)
        assert kwargs == {}, \
            f"Unused arguments: {kwargs}"

        self.head = ConvBlock(inp_channels, hidden_channels * 2 ** upsample, act,
                              n=n, p=p, act_every_n=False, norm_every_n=True, down=False,
                              kernel_size=head_kernel, stride=head_stride, padding=0)
        self.blocks = nn.Sequential(*reversed([
            ConvBlock(hidden_channels * 2 ** (i + 1), hidden_channels * 2 ** i, act, norm,
                      n=n, p=p, act_every_n=False, norm_every_n=True, down=False,
                      kernel_size=4, stride=2, padding=1)
            for i in range(upsample)
        ]))
        self.tail = ConvBlock(hidden_channels, out_channels, nn.Tanh,
                              n=n, p=p, act_every_n=False, norm_every_n=False,
                              kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x = self.head(x)
        x = self.blocks(x)
        y = self.tail(x)
        return y


class ResGenerator(nn.Module):
    def __init__(self, inp_channels: int, out_channels: int, hidden_channels: int, **kwargs):
        """
        Residual Generator.
        :param inp_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param hidden_channels: Number of hidden channels.
        :keyword n: Number of layers between each normalization layer.
        :keyword p: Dropout probability.
        :keyword norm: Normalization layer.
        :keyword act: Activation function.
        :keyword downsample: Number of downsampling blocks.
        :keyword residuals: Number of residual blocks.
        """
        super().__init__()
        n = kwargs.pop("n", 1)
        p = kwargs.pop("p", 0)
        norm = kwargs.pop("norm", nn.InstanceNorm2d)
        act = kwargs.pop("act", nn.ReLU)
        downsample = kwargs.pop("downsample", 4)
        residuals = kwargs.pop("residuals", 9)
        assert kwargs == {}, \
            f"Unused arguments: {kwargs}"

        self.head = ConvBlock(inp_channels, hidden_channels, act,
                              n=n, p=p, act_every_n=False, norm_every_n=True,
                              kernel_size=7, stride=1, padding=3)
        downsample_blocks = nn.ModuleList([
            ConvBlock(hidden_channels * 2 ** i, hidden_channels * 2 ** (i + 1), act, norm,
                      n=n, p=p, act_every_n=False, norm_every_n=True, down=True,
                      kernel_size=4, stride=2, padding=1)
            for i in range(downsample)
        ])
        residual_blocks = nn.Sequential(*[
            ResidualConvBlock(hidden_channels * 2 ** downsample, hidden_channels * 2 ** downsample, act, norm,
                              n=n, p=p, act_every_n=False, norm_every_n=True,
                              kernel_size=3, stride=1, padding=1)
            for _ in range(residuals)
        ])
        upsample_blocks = nn.ModuleList(reversed([
            ConvBlock(hidden_channels * 2 ** (i + 1) * 2, hidden_channels * 2 ** i, act, norm,
                      n=n, p=p, act_every_n=False, norm_every_n=True, down=False,
                      kernel_size=4, stride=2, padding=1)
            for i in range(downsample)
        ]))
        self.blocks = SkipBlock(downsample_blocks, residual_blocks, upsample_blocks)
        self.tail = ConvBlock(hidden_channels, out_channels, nn.Tanh,
                              n=n, p=p, act_every_n=False, norm_every_n=False,
                              kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x = self.head(x)
        x = self.blocks(x)
        y = self.tail(x)
        return y


__all__ = [
    "DCGenerator",
    "ResGenerator",
]
