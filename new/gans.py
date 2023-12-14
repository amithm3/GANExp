from .base import KLGan
from .generators import DCGenerator
from .discriminators import PatchDiscriminator


class DCGan(KLGan):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generator = DCGenerator(**kwargs)
        self.discriminator = PatchDiscriminator(**kwargs)
