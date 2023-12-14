import unittest

import torch

from ..discriminators import PatchDiscriminator, PatchCritic


class TestDiscriminators(unittest.TestCase):
    def test_PatchDiscriminator(self):
        disc = PatchDiscriminator(chn := 3, blocks := [64, 128, 256, 512, 1])
        print(disc)
        x = torch.randn(1, chn, h := 256, w := 256)
        y = disc(x)
        self.assertEqual(y.shape, (1, 1, h // 2 ** (len(blocks) - 1) - 2, w // 2 ** (len(blocks) - 1) - 2))

    def test_PatchCritic(self):
        disc = PatchCritic(chn := 3, blocks := [64, 128, 256, 512, 1])
        print(disc)
        x = torch.randn(1, chn, h := 256, w := 256)
        y = disc(x)
        self.assertEqual(y.shape, (1, 1, h // 2 ** (len(blocks) - 1) - 2, w // 2 ** (len(blocks) - 1) - 2))
