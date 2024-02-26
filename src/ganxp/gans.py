import torch

from .base import KLGan, Discriminator
from .generators import DCGenerator


class DCGan(KLGan):
    def __init__(self, generator: "DCGenerator", discriminator: "Discriminator", **kwargs):
        """
        Deep Convolutional GAN.
        :param generator:
        :param discriminator:
        :keyword lr: Learning rate.
        :keyword betas: Betas for the optimizer.
        """
        lr = kwargs.pop("lr", 0.0002)
        betas = kwargs.pop("betas", (0.5, 0.999))
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.optim_gen = self.optimizer(self.generator.parameters(), lr=lr, betas=betas)
        self.optim_dis = self.optimizer(self.discriminator.parameters(), lr=lr, betas=betas)

    def forward(self, real: "torch.Tensor", noise: "torch.Tensor"):
        metrics = self.loss_step(
            self.generator, self.discriminator, self.optim_gen, self.optim_dis,
            noise, real,
            self.perceptual, self.perceptual_lambda
        )
        return metrics
