from typing import Callable, Iterable

import torch
from torch import nn, optim
from torchvision import models
from torchvision.utils import make_grid

from utils.benchmark import PerceptualLoss
from utils.checkpoints import save_checkpoint


def build_gan_trainer(
        generator: "nn.Module", discriminator: "nn.Module",
        optimizerG: "optim.Optimizer", optimizerD: "optim.Optimizer",
        loss_step_fn: Callable[..., dict[str, float]],
        data_extractor: Callable[..., Iterable["torch.Tensor"]],
        **kwargs,
):
    """
    :param generator:
    :param discriminator:
    :param optimizerG:
    :param optimizerD:
    :param loss_step_fn:
    :param data_extractor:
    :keyword device:
    :keyword writer:
    :keyword writer_period:
    :keyword fixed_inp:
    :keyword perceptual_loss:
    :keyword lambda_perceptual:
    :keyword save_path:
    :keyword save_period:
    :return:
    """
    device = kwargs.pop("device", None)
    writer = kwargs.pop("writer", None)
    writer_period = kwargs.pop("writer_period", 100)
    fixed_inp = kwargs.pop("fixed_inp", None)
    perceptual_loss = kwargs.pop("perceptual_loss", None)
    lambda_perceptual = kwargs.pop("lambda_perceptual", 1)
    save_path = kwargs.pop("save_path", None)
    save_period = kwargs.pop("save_period", writer_period * 2)

    if writer:
        assert fixed_inp is not None, \
            "parameters `writer` and `fixed_inp` are mutually inclusive"
    if perceptual_loss is True:
        perceptual_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:29].eval().to(device)
        perceptual_loss = PerceptualLoss(perceptual_model)

    def trainer(DATA, step: int) -> dict[str, float]:
        real, noise = data_extractor(DATA)
        metrics = loss_step_fn(
            generator, discriminator,
            optimizerG, optimizerD,
            real, noise,
            perceptual_loss, lambda_perceptual,
        )

        # ===Logging===
        if writer and step % writer_period == 0:
            for k, v in metrics.items(): writer.add_scalar(k, v, step)
            grid = make_grid(generator(fixed_inp), nrow=int(len(fixed_inp) ** 0.5), normalize=True)
            writer.add_image("image", grid, step)
        # ---End Logging---

        # ===Checkpoint===
        if save_path and step % save_period == 0:
            save_checkpoint(
                save_path,
                models={
                    "generator": generator,
                    "discriminator": discriminator,
                },
                optimizers={
                    "generator": optimizerG,
                    "discriminator": optimizerD,
                },
                step=step,
            )
        # ---End Checkpoint---

        metrics["step"] = step
        return metrics

    return trainer

