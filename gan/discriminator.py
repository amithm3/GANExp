from abc import ABCMeta, abstractmethod
from typing import Iterable

from torch import nn

from utils.blocks import ConvBlock, LinearBlock


class Discriminator(nn.Module, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def build_head(inp: int, first: int, **kwargs) -> "nn.Module":
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build_blocks(blocks: list[int], **kwargs) -> "nn.Module":
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build_pred(final: int, **kwargs) -> "nn.Module":
        raise NotImplementedError

    def __init__(self, inp: int, blocks: Iterable[int], **kwargs):
        blocks = list(blocks)
        super().__init__()
        assert len(blocks) > 2

        self.head = self.build_head(inp, blocks[0], **kwargs)
        self.blocks = self.build_blocks(blocks, **kwargs)
        self.pred = self.build_pred(blocks[-1], **kwargs)

    def forward(self, x):
        x = self.head(x)
        x = self.blocks(x)
        y = self.pred(x)
        return y


class LinDiscriminator(Discriminator):
    @staticmethod
    def build_head(inp_features: int, first_features: int, **kwargs) -> "nn.Module":
        n = kwargs.pop("n")
        p = kwargs.pop("p")
        act = kwargs.pop("act")

        return LinearBlock(inp_features, first_features, act,
                           n=n, p=p, act_every_n=False, norm_every_n=True)

    @staticmethod
    def build_blocks(blocks: list[int], **kwargs) -> "nn.Module":
        n = kwargs.pop("n")
        p = kwargs.pop("p")
        norm = kwargs.pop("norm")
        act = kwargs.pop("act")

        blocks = nn.Sequential(*[
            LinearBlock(inp_features, out_features, act, norm,
                        n=n, p=p, act_every_n=False, norm_every_n=True)
            for inp_features, out_features in zip(blocks[:-1], blocks[1:])
        ])

        return blocks

    @staticmethod
    def build_pred(final_features: int, **kwargs) -> "nn.Module":
        p = kwargs.pop("p", 0)

        return LinearBlock(final_features, 1, nn.Sigmoid(),
                           n=0, p=p, act_every_n=False, norm_every_n=False)


class PatchDiscriminator(Discriminator):
    def __init__(self, inp_channels: int, blocks: Iterable[int], **kwargs):
        """
        Patch discriminator for 2D input.
        :param inp_channels: number of input channels
        :param blocks: number of channels in each block
        :keyword n: number of layers in each block
        :keyword p: dropout probability
        :keyword norm: normalization layer
        :keyword act: activation function
        """
        n = kwargs.pop("n", 1)
        p = kwargs.pop("p", 0)
        norm = kwargs.pop("norm", nn.InstanceNorm2d)
        act = kwargs.pop("act", nn.LeakyReLU(0.2))
        assert kwargs == {}, \
            f"Unused arguments: {kwargs}"
        blocks = list(blocks)
        super().__init__(inp_channels, blocks, n=n, p=p, norm=norm, act=act)

    @staticmethod
    def build_head(inp_channels: int, first_channels: int, **kwargs) -> "nn.Module":
        n = kwargs.pop("n")
        p = kwargs.pop("p")
        act = kwargs.pop("act")

        return ConvBlock(inp_channels, first_channels, act,
                         n=n, p=p, act_every_n=False, norm_every_n=True,
                         kernel_size=4, stride=2, padding=1)

    @staticmethod
    def build_blocks(blocks: list[int], **kwargs) -> "nn.Module":
        n = kwargs.pop("n")
        p = kwargs.pop("p")
        norm = kwargs.pop("norm")
        act = kwargs.pop("act")

        blocks = nn.Sequential(*[
            ConvBlock(inp_features, out_features, act, norm,
                      n=n, p=p, act_every_n=False, norm_every_n=True,
                      kernel_size=4, stride=2 if i < len(blocks) - 2 else 1, padding=1)
            for i, (inp_features, out_features) in enumerate(zip(blocks[:-1], blocks[1:]))
        ])

        return blocks

    @staticmethod
    def build_pred(final_channels: int, **kwargs) -> "nn.Module":
        n = kwargs.pop("n", 1)
        p = kwargs.pop("p", 0)

        return ConvBlock(final_channels, 1, nn.Sigmoid(),
                         n=n, p=p, act_every_n=False, norm_every_n=False,
                         kernel_size=4, stride=1, padding=1)


__all__ = [
    "Discriminator",
    "PatchDiscriminator",
    "LinDiscriminator",
]
