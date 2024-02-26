from typing import Callable

import torch
from torch import nn


class LinBlock(nn.Module):
    @property
    def params(self):
        return {**self._params}

    @property
    def kwargs(self):
        return {**self._kwargs}

    def __init__(
            self,
            inp_features: int,
            out_features: int,
            act: Callable[[], "nn.Module"] = None,
            norm: Callable[[int], "nn.Module"] = None,
            **kwargs,
    ):
        """
        Compact linear block
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
        n = kwargs.pop("n", 1)
        p = kwargs.pop("p", 0)
        act_every_n = kwargs.pop("act_every_n", False)
        norm_every_n = kwargs.pop("norm_every_n", True)
        assert act is None or callable(act), \
            f"Invalid value of act, must be callable, got {act}"
        assert norm is None or callable(norm), \
            f"Invalid value of norm, must be callable, got {norm}"
        assert isinstance(n, int) and n > 0, \
            f"Invalid value of n, must be a positive integer, got {n}"
        assert isinstance(p, (int, float)) and 0 <= p < 1, \
            f"Invalid value of p, must be a float in [0, 1), got {p}"
        assert isinstance(act_every_n, bool), \
            f"Invalid value of act_every_n, must be a bool, got {act_every_n}"
        assert isinstance(norm_every_n, bool), \
            f"Invalid value of norm_every_n, must be a bool, got {norm_every_n}"
        super().__init__()

        LINEAR = nn.Linear
        layers = [
            LINEAR(inp_features, out_features, bias=not norm, **kwargs),
            act() if act_every_n and act else None,
            norm(out_features) if norm_every_n and norm else None,
        ]
        for _ in range(1, n):
            layers.extend([
                LINEAR(out_features, out_features, bias=not norm, **kwargs),
                act() if act_every_n and act else None,
                norm(out_features) if norm_every_n and norm else None,
            ])
        layers.extend([
            act() if not act_every_n and act else None,
            norm(out_features) if not norm_every_n and norm else None,
            nn.Dropout(p) if p else None,
        ])

        self._params = {
            "n": n,
            "p": p,
            "norm": norm,
            "act": act,
            "act_every_n": act_every_n,
            "norm_every_n": norm_every_n,
        }
        self._kwargs = kwargs
        self.layers = nn.ModuleList([layer for layer in layers if layer is not None])

    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x


class ConvBlock(nn.Module):
    @property
    def params(self):
        return {**self._params}

    @property
    def kwargs(self):
        return {**self._kwargs}

    def __init__(
            self,
            inp_channels: int,
            out_channels: int,
            act: Callable[[], "nn.Module"] = None,
            norm: Callable[[int], "nn.Module"] = None,
            **kwargs,
    ):
        """
        Compact convolutional block
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
        n = kwargs.pop("n", 1)
        p = kwargs.pop("p", 0)
        act_every_n = kwargs.pop("act_every_n", False)
        norm_every_n = kwargs.pop("norm_every_n", True)
        down = kwargs.pop("down", True)
        assert act is None or callable(act), \
            f"Invalid value of act, must be callable, got {act}"
        assert norm is None or callable(norm), \
            f"Invalid value of norm, must be callable, got {norm}"
        assert isinstance(n, int) and n > 0, \
            f"Invalid value of n, must be a positive integer, got {n}"
        assert isinstance(p, (int, float)) and 0 <= p < 1, \
            f"Invalid value of p, must be a float in [0, 1), got {p}"
        assert isinstance(act_every_n, bool), \
            f"Invalid value of act_every_n, must be a bool, got {act_every_n}"
        assert isinstance(norm_every_n, bool), \
            f"Invalid value of norm_every_n, must be a bool, got {norm_every_n}"
        assert isinstance(down, bool), \
            f"Invalid value of down, must be a bool, got {down}"
        super().__init__()

        CONV = nn.Conv2d if down else nn.ConvTranspose2d
        layers = [
            CONV(inp_channels, out_channels, bias=not norm, **kwargs),
            act() if act_every_n and act else None,
            norm(out_channels) if norm_every_n and norm else None,
        ]
        for _ in range(1, n):
            layers.extend([
                CONV(out_channels, out_channels, bias=not norm, **kwargs),
                act() if act_every_n and act else None,
                norm(out_channels) if norm_every_n and norm else None,
            ])
        layers.extend([
            act() if not act_every_n and act else None,
            norm(out_channels) if not norm_every_n and norm else None,
            nn.Dropout(p) if p else None,
        ])

        self._params = {
            "n": n,
            "p": p,
            "norm": norm,
            "act": act,
            "act_every_n": act_every_n,
            "norm_every_n": norm_every_n,
            "down": down,
        }
        self._kwargs = kwargs
        self.layers = nn.ModuleList([layer for layer in layers if layer is not None])

    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x


class ResidualLinBlock(LinBlock):
    def __init__(
            self,
            inp_features: int,
            out_features: int,
            act: Callable[[], "nn.Module"] = None,
            norm: Callable[[int], "nn.Module"] = None,
            **kwargs,
    ):
        """
        Compact Residual linear block
        :param inp_features: number of input features
        :param out_features: number of output features
        :param norm: normalization layer
        :param act: activation function
        :param kwargs: arguments to pass to the linear layers
        :keyword n: number of intermediate linear layers
        :keyword p: dropout probability
        :keyword act_every_n: whether to apply activation after every n linear layers
        :keyword norm_every_n: whether to apply normalization after every n linear layers
        """
        super().__init__(inp_features, out_features, act, norm, **kwargs)
        identity = inp_features == out_features
        self._params.update({
            "identity": identity,
        })
        self.shortcut = LinBlock(inp_features, out_features, **self.kwargs) if not identity else nn.Identity()

    def forward(self, x):
        return super().forward(x) + self.shortcut(x)


class ResidualConvBlock(ConvBlock):
    def __init__(
            self,
            inp_channels: int,
            out_channels: int,
            act: Callable[[], "nn.Module"] = None,
            norm: Callable[[int], "nn.Module"] = None,
            **kwargs,
    ):
        """
        Compact Residual convolutional block
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
        super().__init__(inp_channels, out_channels, act, norm, **kwargs)
        with torch.no_grad():
            inp_shape = (1, inp_channels, 64, 64)
            out_shape = self.layers[0](torch.randn(inp_shape)).shape
            identity = inp_shape == out_shape
        self._params.update({
            "identity": identity,
        })
        self.shortcut = ConvBlock(inp_channels, out_channels, **self.kwargs) if not identity else nn.Identity()

    def forward(self, x):
        return super().forward(x) + self.shortcut(x)


class SkipBlock(nn.Module):
    def __init__(self, encoder: "nn.ModuleList", bottleneck: "nn.Module", decoder: "nn.ModuleList"):
        """
        Skip Connection block
        :param encoder: forward encoder
        :param bottleneck: bottleneck module
        :param decoder: backward decoder
        """
        assert isinstance(encoder, nn.ModuleList), \
            f"Invalid type of encoder, must be nn.ModuleList, got {type(encoder)}"
        assert isinstance(bottleneck, nn.Module), \
            f"Invalid type of bottleneck, must be nn.Module, got {type(bottleneck)}"
        assert isinstance(decoder, nn.ModuleList), \
            f"Invalid type of decoder, must be nn.ModuleList, got {type(decoder)}"
        super().__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder

    def forward(self, x):
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
        x = self.bottleneck(x)
        for block in self.decoder:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block(x)
        y = x
        return y


__all__ = [
    "LinBlock",
    "ConvBlock",
    "ResidualLinBlock",
    "ResidualConvBlock",
    "SkipBlock",
]
