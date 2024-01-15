from typing import Callable

import torch
from torch import nn, optim

from .base import KLGan
from .generators import DCGenerator
from .base import Discriminator


class DCGan(KLGan):
    def __init__(self, generator: "DCGenerator", discriminator: "Discriminator", **kwargs):
        lr = kwargs.pop("lr", 0.0002)
        betas = kwargs.pop("betas", (0.5, 0.999))
        super().__init__(**kwargs)
        self.modelG = generator
        self.modelD = discriminator
        self.optimizerG = self.optimizer(self.generator.parameters(), lr=lr, betas=betas)
        self.optimizerD = self.optimizer(self.discriminator.parameters(), lr=lr, betas=betas)

    def forward(self, DATA) -> dict[str, float]:
        # ===Train===#
        real, noise = self.data_extractor(DATA)
        metrics = self.train_step(self.modelG, self.modelD, self.optimizerG, self.optimizerD, real, noise)
        # ---End Train---#

        # ===Logging===
        # ---End Logging---#

        # === Checkpoint ===
        # ---End Checkpoint---#

        return metrics
