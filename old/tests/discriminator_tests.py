import unittest

import torch

from gan import PatchDiscriminator, ResDiscriminator


class DiscriminatorTests(unittest.TestCase):
    BATCH_SIZE = 32

    def test_PatchDiscriminator(self):
        discriminator = PatchDiscriminator(3, [64, 128, 256, 512], n=1)
        art = discriminator(torch.randn(self.BATCH_SIZE, 3, 128, 128))
        print(discriminator)
        print(art.shape)

    def test_ResDiscriminator(self):
        discriminator = ResDiscriminator(3, [64, 128, 256, 512])
        art = discriminator(torch.randn(self.BATCH_SIZE, 3, 128, 128))
        print(discriminator)
        print(art.shape)
