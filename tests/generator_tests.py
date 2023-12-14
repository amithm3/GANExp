import unittest

from new import DCGenerator, ResGenerator


class GeneratorTests(unittest.TestCase):
    BATCH_SIZE = 32

    def test_DCGenerator(self):
        generator = DCGenerator(128, 3, 64, sampling_layers=4, n=1)
        noise = generator.sample_noise(self.BATCH_SIZE, "cpu")
        art = generator(noise)
        print(generator)
        self.assertEqual(art.shape, (self.BATCH_SIZE, 3, 2 ** generator.sample_layers * 2, 2 ** generator.sample_layers * 2))

    def test_ResGenerator(self):
        generator = ResGenerator(3, 5, 64)
        noise = generator.sample_noise(self.BATCH_SIZE, 128, 128, "cpu")
        art = generator(noise)
        print(generator)
        self.assertEqual(art.shape, (self.BATCH_SIZE, generator.out_chn, noise.shape[2], noise.shape[3]))
