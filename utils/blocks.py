import torch
from torch import nn


class LinearBlock(nn.Sequential):
    def __init__(
            self,
            inp_features: int,
            out_features: int,
            act: "nn.Module" = None,
            norm: type["nn.Module"] = None,
            **kwargs,
    ):
        """
        Compact linear block with optional activation and normalization layers.
        :param inp_features: number of input features
        :param out_features: number of output features
        :param act: activation function
        :param norm: normalization layer
        :param kwargs: arguments to pass to the linear layers
        :keyword n: number of intermediate linear layers
        :keyword p: dropout probability
        :keyword act_every_n: whether to apply activation after every n linear layers
        :keyword norm_every_n: whether to apply normalization after every n linear layers
        """
        n = kwargs.pop("n", 0)
        p = kwargs.pop("p", 0)
        act_every_n = kwargs.pop("act_every_n", False)
        norm_every_n = kwargs.pop("norm_every_n", True)

        self.inp_features = inp_features
        self.out_features = out_features
        self.n = n
        self.p = p

        layers = [
            nn.Linear(inp_features, out_features, bias=not norm, **kwargs),
            act if act_every_n and act else None,
            norm(out_features) if norm_every_n and norm else None,
            *[module
              for module in (
                  nn.Linear(out_features, out_features, bias=not norm, **kwargs),
                  act if act_every_n and act else None,
                  norm(out_features) if norm_every_n and norm else None,
              )
              for _ in range(n)],
            act if not act_every_n and act else None,
            norm(out_features) if not norm_every_n and norm else None,
            nn.Dropout(p) if p else None,
        ]

        super().__init__(*[layer for layer in layers if layer is not None])


class ConvBlock(nn.Sequential):
    def __init__(
            self,
            inp_channels: int,
            out_channels: int,
            act: "nn.Module" = None,
            norm: type["nn.Module"] = None,
            **kwargs,
    ):
        """
        Compact convolutional block with optional activation and normalization layers.
        :param inp_channels: number of input channels
        :param out_channels: number of output channels
        :param act: activation function
        :param norm: normalization layer
        :param kwargs: arguments to pass to the convolutional layers
        :keyword n: number of intermediate convolutions
        :keyword p: dropout probability
        :keyword act_every_n: whether to apply activation after every n convolutions
        :keyword norm_every_n: whether to apply normalization after every n convolutions
        :keyword down: whether to downsample, if False, then upsample
        """

        n = kwargs.pop("n", 0)
        p = kwargs.pop("p", 0)
        act_every_n = kwargs.pop("act_every_n", False)
        norm_every_n = kwargs.pop("norm_every_n", True)
        down = kwargs.pop("down", True)

        CONV = nn.Conv2d if down else nn.ConvTranspose2d
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.n = n
        self.p = p

        layers = [
            CONV(inp_channels, out_channels, bias=not norm, **kwargs),
            act if act_every_n and act else None,
            norm(out_channels) if norm_every_n and norm else None,
            *[module
              for module in (
                  CONV(out_channels, out_channels, bias=not norm, kernel_size=3, stride=1, padding=1),
                  act if act_every_n and act else None,
                  norm(out_channels) if norm_every_n and norm else None,
              )
              for _ in range(n)],
            act if not act_every_n and act else None,
            norm(out_channels) if not norm_every_n and norm else None,
            nn.Dropout(p) if p else None,
        ]

        super().__init__(*[layer for layer in layers if layer is not None])


class ResidualLinearBlock(LinearBlock):
    def __init__(self, *args, **kwargs):
        identity = kwargs.pop("identity", False)
        super().__init__(*args, **kwargs)
        self.identity = identity
        self.shortcut = LinearBlock(*args, **kwargs) if not identity else nn.Identity()


class ResidualConvBlock(ConvBlock):
    def __init__(self, *args, **kwargs):
        identity = kwargs.pop("identity", False)
        super().__init__(*args, **kwargs)
        self.identity = identity
        kwargs.pop("n", None)
        kwargs.pop("p", None)
        kwargs.pop("norm", None)
        kwargs.pop("act", None)
        args = args[:min(len(args), 2)]
        self.shortcut = ConvBlock(*args, **kwargs) if not identity else nn.Identity()

    def forward(self, x):
        y = x
        for layer in self:
            if layer != self.shortcut: y = layer(y)
        return y + self.shortcut(x)


class SkipBlock(nn.Module):
    def __init__(self, encoder: "nn.ModuleList", blocks: "nn.Module", decoder: "nn.ModuleList"):
        super().__init__()
        self.encoder = encoder
        self.blocks = blocks
        self.decoder = decoder

    def forward(self, x):
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
        x = self.blocks(x)
        for block in self.decoder:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block(x)
        return x


__all__ = [
    "LinearBlock",
    "ConvBlock",
    "ResidualLinearBlock",
    "ResidualConvBlock",
    "SkipBlock",
]
