import unittest

from ..generators import DCGenerator, ResGenerator


class TestGenerators(unittest.TestCase):
    def test_DCGenerator(self):
        gen = DCGenerator(16, chn := 3, 64, sampling=(smp := 3))
        print(gen)
        x = gen.sample_noise(1, "cpu")
        y = gen(x)
        self.assertEqual(y.shape, (1, chn, 2 ** smp * 2, 2 ** smp * 2))

    def test_ResGenerator(self):
        gen = ResGenerator(16, chn := 3, 64, sampling=(smp := 3))
        print(gen)
        x = gen.sample_noise(bs := 8, w := 64, h := 64, "cpu")
        y = gen(x)
        self.assertEqual(y.shape, (bs, chn, h, w))
