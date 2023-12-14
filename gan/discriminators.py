from typing import Iterable

from torch import nn

from .base import Discriminator, Critic

from utils.blocks import ConvBlock, ResidualConvBlock


class PatchDiscriminator(Discriminator):
    def __init__(self, inp_channels: int, blocks: Iterable[int], **kwargs):
        """
        Patch discriminator.
        :param inp_channels: number of input channels
        :param blocks: number of channels in each block
        :keyword n: number of layers in each block
        :keyword p: dropout probability
        :keyword norm: normalization layer
        :keyword act: activation function
        :keyword use_sigmoid: whether to use sigmoid activation in the last layer
        """
        use_sigmoid = kwargs.pop("use_sigmoid", True)
        super().__init__(blocks, **kwargs)
        n = self.n
        p = self.p
        norm = self.norm
        act = self.act
        blocks = list(blocks)
        self.head = ConvBlock(inp_channels, blocks[0], None, act,
                              n=n, p=p, act_every_n=False, norm_every_n=True,
                              kernel_size=4, stride=2, padding=1)
        self.blocks = nn.Sequential(*[
            ConvBlock(inp_features, out_features, norm, act,
                      n=n, p=p, act_every_n=False, norm_every_n=True, down=True,
                      kernel_size=4, stride=2 if i < len(blocks) - 2 else 1, padding=1)
            for i, (inp_features, out_features) in enumerate(zip(blocks[:-1], blocks[1:]))
        ])
        self.tail = ConvBlock(blocks[-1], 1, None, nn.Sigmoid if use_sigmoid else None,
                              n=n, p=p, act_every_n=False, norm_every_n=False,
                              kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.blocks(x)
        y = self.tail(x)
        return y


class ResDiscriminator(Discriminator):
    def __init__(self, inp_channels: int, blocks: Iterable[int], **kwargs):
        """
        Residual discriminator.
        :param inp_channels: number of input channels
        :param blocks: number of channels in each block
        :keyword n: number of layers in each block
        :keyword p: dropout probability
        :keyword norm: normalization layer
        :keyword act: activation function
        :keyword use_sigmoid: whether to use sigmoid activation in the last layer
        """
        use_sigmoid = kwargs.pop("use_sigmoid", True)
        super().__init__(blocks, **kwargs)
        n = self.n
        p = self.p
        norm = self.norm
        act = self.act
        blocks = list(blocks)
        self.head = ConvBlock(inp_channels, blocks[0], act,
                              n=n, p=p, act_every_n=False, norm_every_n=True,
                              kernel_size=4, stride=2, padding=1)
        self.blocks = nn.Sequential(*[
            ResidualConvBlock(inp_features, out_features, act, norm,
                              n=n, p=p, act_every_n=False, norm_every_n=True, down=True,
                              kernel_size=4, stride=2 if i < len(blocks) - 2 else 1, padding=1)
            for i, (inp_features, out_features) in enumerate(zip(blocks[:-1], blocks[1:]))
        ])
        self.tail = ConvBlock(blocks[-1], 1, nn.Sigmoid if use_sigmoid else None,
                              n=n, p=p, act_every_n=False, norm_every_n=False,
                              kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.blocks(x)
        y = self.tail(x)
        return y


class PatchCritic(Critic, PatchDiscriminator):
    def __init__(self, inp_channels: int, blocks: Iterable[int], **kwargs):
        """
        Patch critic.
        :param inp_channels: number of input channels
        :param blocks: number of channels in each block
        :keyword n: number of layers in each block
        :keyword p: dropout probability
        :keyword norm: normalization layer
        :keyword act: activation function
        """
        super().__init__(inp_channels, blocks, use_sigmoid=False, **kwargs)


class ResCritic(Critic, ResDiscriminator):
    def __init__(self, inp_channels: int, blocks: Iterable[int], **kwargs):
        """
        Residual critic.
        :param inp_channels: number of input channels
        :param blocks: number of channels in each block
        :keyword n: number of layers in each block
        :keyword p: dropout probability
        :keyword norm: normalization layer
        :keyword act: activation function
        """
        super().__init__(inp_channels, blocks, use_sigmoid=False, **kwargs)


__all__ = [
    "PatchDiscriminator",
    "PatchCritic",
    "ResDiscriminator",
    "ResCritic",
]
