from abc import ABCMeta, abstractmethod

from torch import nn

from utils.blocks import ConvBlock, ResidualConvBlock, SkipBlock, LinearBlock, ResidualLinearBlock


class Generator(nn.Module, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def build_head(inp: int, hidden: int, **kwargs) -> "nn.Module":
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build_blocks(hidden: int, **kwargs) -> "nn.Module":
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build_pred(hidden: int, out: int, **kwargs) -> "nn.Module":
        raise NotImplementedError

    def __init__(self, inp: int, out: int, hidden: int, **kwargs):
        super().__init__()

        self.head = self.build_head(inp, hidden, **kwargs)
        self.blocks = self.build_blocks(hidden, **kwargs)
        self.pred = self.build_pred(hidden, out, **kwargs)

    def forward(self, x):
        x = self.head(x)
        x = self.blocks(x)
        y = self.pred(x)
        return y


class LinGenerator(Generator):
    @staticmethod
    def build_head(inp_features: int, hidden_features: int, **kwargs) -> "nn.Module":
        n = kwargs.pop("n")
        p = kwargs.pop("p")
        act = kwargs.pop("act")

        return LinearBlock(inp_features, hidden_features, act,
                           n=n, p=p, act_every_n=False, norm_every_n=True)

    @staticmethod
    def build_blocks(hidden_features: int, **kwargs) -> "nn.Module":
        n = kwargs.pop("n")
        p = kwargs.pop("p")
        norm = kwargs.pop("norm")
        act = kwargs.pop("act")
        downsample = kwargs.pop("downsample")
        residuals = kwargs.pop("residuals")

        downsample_blocks = nn.ModuleList([
            LinearBlock(hidden_features // 2 ** i, hidden_features // 2 ** (i + 1), act, norm,
                        n=n, p=p, act_every_n=False, norm_every_n=True)
            for i in range(downsample)
        ])
        residual_blocks = nn.Sequential(*[
            ResidualLinearBlock(hidden_features // 2 ** downsample, hidden_features // 2 ** downsample, act, norm,
                                identity=True,
                                n=n, p=p, act_every_n=False, norm_every_n=True)
            for _ in range(residuals)
        ])
        upsample_blocks = nn.ModuleList(reversed([
            LinearBlock(hidden_features // 2 ** (i + 1) * 2, hidden_features // 2 ** i, act, norm,
                        n=n, p=p, act_every_n=False, norm_every_n=True)
            for i in range(downsample)
        ]))

        return SkipBlock(downsample_blocks, residual_blocks, upsample_blocks)

    @staticmethod
    def build_pred(hidden_features: int, out_features: int, **kwargs) -> "nn.Module":
        n = kwargs.pop("n")
        p = kwargs.pop("p")

        return LinearBlock(hidden_features, out_features, nn.Tanh(),
                           n=n, p=p, act_every_n=False, norm_every_n=False)


class FCResGenerator(Generator):
    @staticmethod
    def build_head(inp_channels: int, hidden_channels: int, **kwargs) -> "nn.Module":
        n = kwargs.pop("n")
        p = kwargs.pop("p")
        act = kwargs.pop("act")

        return ConvBlock(inp_channels, hidden_channels, act,
                         n=n, p=p, act_every_n=False, norm_every_n=True,
                         kernel_size=7, stride=1, padding=3)

    @staticmethod
    def build_blocks(hidden_channels: int, **kwargs) -> "nn.Module":
        n = kwargs.pop("n")
        p = kwargs.pop("p")
        norm = kwargs.pop("norm")
        act = kwargs.pop("act")
        downsample = kwargs.pop("downsample")
        residuals = kwargs.pop("residuals")

        downsample_blocks = nn.ModuleList([
            ConvBlock(hidden_channels * 2 ** i, hidden_channels * 2 ** (i + 1), act, norm,
                      n=n, p=p, act_every_n=False, norm_every_n=True, down=True,
                      kernel_size=4, stride=2, padding=1)
            for i in range(downsample)
        ])
        residual_blocks = nn.Sequential(*[
            ResidualConvBlock(hidden_channels * 2 ** downsample, hidden_channels * 2 ** downsample, act, norm,
                              identity=True,
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

        return SkipBlock(downsample_blocks, residual_blocks, upsample_blocks)

    @staticmethod
    def build_pred(hidden_channels: int, out_channels: int, **kwargs) -> "nn.Module":
        n = kwargs.pop("n")
        p = kwargs.pop("p")

        return ConvBlock(hidden_channels, out_channels, nn.Tanh(),
                         n=n, p=p, act_every_n=False, norm_every_n=False,
                         kernel_size=7, stride=1, padding=3)


class DCGenerator(Generator):
    def __init__(self, inp_channels: int, out_channels: int, hidden_channels: int, **kwargs):
        """
        :param inp_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param hidden_channels: Number of hidden channels.
        :keyword n: Number of layers between each normalization layer.
        :keyword p: Dropout probability.
        :keyword norm: Normalization layer.
        :keyword act: Activation function.
        :keyword upsample: Number of upsampling blocks.
        """
        n = kwargs.pop("n", 1)
        p = kwargs.pop("p", 0)
        norm = kwargs.pop("norm", nn.InstanceNorm2d)
        act = kwargs.pop("act", nn.ReLU())
        upsample = kwargs.pop("upsample", 4)
        assert kwargs == {}, \
            f"Unused arguments: {kwargs}"
        super().__init__(inp_channels, out_channels, hidden_channels, n=n, p=p, norm=norm, act=act,
                         upsample=upsample)

    @staticmethod
    def build_head(inp_channels: int, hidden_channels: int, **kwargs) -> "nn.Module":
        n = kwargs.pop("n")
        p = kwargs.pop("p")
        act = kwargs.pop("act")
        upsample = kwargs.pop("upsample")

        return nn.Sequential(
            ConvBlock(inp_channels, hidden_channels * 2 ** upsample, act,
                      n=n, p=p, act_every_n=False, norm_every_n=True, down=False,
                      kernel_size=4, stride=1, padding=0),
        )

    @staticmethod
    def build_blocks(hidden_channels: int, **kwargs) -> "nn.Module":
        n = kwargs.pop("n")
        p = kwargs.pop("p")
        norm = kwargs.pop("norm")
        act = kwargs.pop("act")
        upsample = kwargs.pop("upsample")

        blocks = nn.Sequential(*reversed([
            ConvBlock(hidden_channels * 2 ** (i + 1), hidden_channels * 2 ** i, act, norm,
                      n=n, p=p, act_every_n=False, norm_every_n=True, down=False,
                      kernel_size=4, stride=2, padding=1)
            for i in range(upsample)
        ]))

        return blocks

    @staticmethod
    def build_pred(hidden_channels: int, out_channels: int, **kwargs) -> "nn.Module":
        n = kwargs.pop("n")
        p = kwargs.pop("p")

        return ConvBlock(hidden_channels, out_channels, nn.Tanh(),
                         n=n, p=p, act_every_n=False, norm_every_n=False,
                         kernel_size=7, stride=1, padding=3)


__all__ = [
    "Generator",
    "FCResGenerator",
    "DCGenerator",
    "LinGenerator",
]
