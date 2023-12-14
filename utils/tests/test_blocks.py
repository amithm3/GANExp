import unittest

import torch
from torch import nn

from ..blocks import LinBlock, ConvBlock, ResidualLinBlock, ResidualConvBlock, SkipBlock


class TestBlocks(unittest.TestCase):
    def test_LinBlock(self):
        block = LinBlock(
            ife := 10,
            ofe := 20,
            nn.ReLU,
            nn.InstanceNorm1d,
            n=5,
            p=0.5,
            norm_every_n=True,
            act_every_n=True,
        )
        print(block)
        x = torch.randn(bs := 32, ife)
        y = block(x)
        self.assertEqual(y.shape, (bs, ofe))

    def test_ConvBlock(self):
        block = ConvBlock(
            ich := 10,
            och := 20,
            nn.ReLU,
            nn.InstanceNorm2d,
            n=5,
            p=0.5,
            norm_every_n=True,
            act_every_n=True,
            kernel_size=3,
            stride=1,
            padding=1,
            down=True,
        )
        print(block)
        x = torch.randn(bs := 32, ich, h := 64, w := 64)
        y = block(x)
        self.assertEqual(y.shape, (bs, och, h, w))

    def test_ResidualLinBlock(self):
        block = ResidualLinBlock(
            ife := 10,
            ofe := 20,
            nn.ReLU,
            nn.InstanceNorm1d,
            n=5,
            p=0.5,
            norm_every_n=True,
            act_every_n=True,
        )
        print(block)
        x = torch.randn(bs := 32, ife)
        y = block(x)
        self.assertEqual(y.shape, (bs, ofe))

    def test_ResidualConvBlock(self):
        block = ResidualConvBlock(
            ich := 10,
            och := 20,
            nn.ReLU,
            nn.InstanceNorm2d,
            n=5,
            p=0.5,
            norm_every_n=True,
            act_every_n=True,
            kernel_size=3,
            stride=1,
            padding=1,
            down=True,
        )
        print(block)
        x = torch.randn(bs := 32, ich, h := 64, w := 64)
        y = block(x)
        self.assertEqual(y.shape, (bs, och, h, w))

    def test_SkipBlock(self):
        encoder = nn.ModuleList([
            ConvBlock(3, 64, nn.LeakyReLU, nn.InstanceNorm2d,
                      n=2,
                      p=0.5,
                      norm_every_n=True,
                      act_every_n=True,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      down=True),
            ConvBlock(64, 128, nn.LeakyReLU, nn.InstanceNorm2d,
                      n=2,
                      p=0.5,
                      norm_every_n=True,
                      act_every_n=True,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      down=True),
        ])
        bottleneck = ConvBlock(128, 128, nn.LeakyReLU, nn.InstanceNorm2d,
                                 n=2,
                                 p=0.5,
                                 norm_every_n=True,
                                 act_every_n=True,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 down=False)
        decoder = nn.ModuleList([
            ConvBlock(256, 64, nn.ReLU, nn.InstanceNorm2d,
                      n=2,
                      p=0.5,
                      norm_every_n=True,
                      act_every_n=True,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      down=False),
            ConvBlock(128, 3, nn.ReLU, nn.InstanceNorm2d,
                      n=2,
                      p=0.5,
                      norm_every_n=True,
                      act_every_n=True,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      down=False),
        ])
        block = SkipBlock(encoder, bottleneck, decoder)
        print(block)
        x = torch.randn(bs := 32, ch := 3, h := 64, w := 64)
        y = block(x)
        self.assertEqual(y.shape, (bs, ch, h, w))
