from typing import Iterable

from torch import nn

from utils.blocks import ConvBlock


class PatchDiscriminator(nn.Module):
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
        n = kwargs.pop("n", 1)
        p = kwargs.pop("p", 0)
        norm = kwargs.pop("norm", nn.InstanceNorm2d)
        act = kwargs.pop("act", lambda: nn.LeakyReLU)
        use_sigmoid = kwargs.pop("use_sigmoid", True)
        assert kwargs == {}, \
            f"Unused arguments: {kwargs}"
        blocks = list(blocks)
        super().__init__()

        self.head = ConvBlock(inp_channels, blocks[0], act,
                              n=n, p=p, act_every_n=False, norm_every_n=True,
                              kernel_size=4, stride=2, padding=1)
        self.blocks = nn.Sequential(*[
            ConvBlock(inp_features, out_features, act, norm,
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


class PatchCritic(PatchDiscriminator):
    def __init__(self, inp_channels: int, blocks: Iterable[int], **kwargs):
        """
        Patch discriminator.
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
]
