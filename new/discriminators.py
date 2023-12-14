from typing import Iterable

import torch
from torch import nn

from utils.blocks import ConvBlock
from .base import Discriminator, Critic


class PatchDiscriminator(Discriminator):
    def __init__(self, inp_chn: int, blocks: int | Iterable[int], **kwargs):
        """
        Patch discriminator.
        :param inp_chn: number of input channels
        :param blocks: number of channels in each block
        :keyword act: activation function
        :keyword norm: normalization layer
        :keyword n: number of layers in each block
        :keyword p: dropout probability
        :keyword m_block: multiplier for number of channels in each block
        """
        super().__init__(blocks, **kwargs)
        self.head = ConvBlock(
            inp_chn, self.blocks[0], self.act, None,
            n=self.n, p=self.p, act_every_n=False, norm_every_n=True,
            kernel_size=4, stride=2, padding=1
        )
        self.main = nn.Sequential(*[
            ConvBlock(
                inp_chn, out_chn, self.act, self.norm,
                n=self.n, p=self.p, act_every_n=False, norm_every_n=True, down=True,
                kernel_size=4, stride=2 if i < len(self.blocks) - 2 else 1, padding=1
            )
            for i, (inp_chn, out_chn) in enumerate(zip(self.blocks[:-1], self.blocks[1:]))
        ])
        self.tail = ConvBlock(
            self.blocks[-1], 1, nn.Sigmoid if self.USE_SIGMOID else None, None,
            n=self.n, p=self.p, act_every_n=False, norm_every_n=False,
            kernel_size=4, stride=1, padding=1
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = self.head(x)
        x = self.main(x)
        y = self.tail(x)
        return y


class PatchCritic(PatchDiscriminator, Critic):
    pass


__all__ = [
    "PatchDiscriminator",
    "PatchCritic"
]
