from typing import Callable, Union, TYPE_CHECKING, Iterable

import torch
from torch.nn import functional as F
from torch import nn, optim
from torchvision import models
from torchvision.utils import make_grid

from utils.benchmark import PerceptualLoss
from utils.checkpoints import save_checkpoint


def kl_gan_loss_step(
        generator: "nn.Module", discriminator: "nn.Module",
        optimizerG: "optim.Optimizer", optimizerD: "optim.Optimizer",
        real: "torch.Tensor", noise: "torch.Tensor",
        perceptual_loss: Union["nn.Module", bool] = False, lambda_perceptual: float = 1,
):
    fake = generator(noise)

    # ===Discriminator Loss===
    # adversarial loss
    pred_real = discriminator(real)
    pred_fake_true = discriminator(fake.detach())
    loss_adversarialD = (F.binary_cross_entropy(pred_real, torch.ones_like(pred_real)) +
                         F.binary_cross_entropy(pred_fake_true, torch.zeros_like(pred_fake_true)))
    # total loss
    loss_D = loss_adversarialD
    # backprop
    optimizerD.zero_grad()
    loss_D.backward()
    optimizerD.step()
    # ---End Discriminator Loss---

    # ===Generator Loss===
    # adversarial loss
    pred_fake_false = discriminator(fake)
    loss_adversarialG = F.binary_cross_entropy(pred_fake_false, torch.ones_like(pred_fake_false))
    # perceptual loss
    loss_perceptual = perceptual_loss(fake, real) if perceptual_loss else torch.tensor(0.0)
    # total loss
    loss_G = loss_adversarialG + lambda_perceptual * loss_perceptual
    # backprop
    optimizerG.zero_grad()
    loss_G.backward()
    optimizerG.step()
    # ---End Generator Loss---

    # ===Misc===
    total_loss = loss_D + loss_G
    metrics = {
        "loss/total": total_loss.item(),
        "loss/adversarialD": loss_adversarialD.item(),
        "loss/adversarialG": loss_adversarialG.item(),
        "loss/perceptual": loss_perceptual.item(),
        "loss/D": loss_D.item(),
        "loss/G": loss_G.item(),
        "pred/real": pred_real.mean().item(),
        "pred/fake_true": pred_fake_true.mean().item(),
        "pred/fake_false": pred_fake_false.mean().item(),
    }
    # ---End Misc---

    return metrics


def build_gan_trainer(
        generator: "nn.Module", discriminator: "nn.Module",
        optimizerG: "optim.Optimizer", optimizerD: "optim.Optimizer",
        loss_step_fn: Callable[..., dict[str, float]],
        data_extractor: Callable[..., Iterable["torch.Tensor"]],
        **kwargs,
):
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


def kl_cycle_gan_loss_step(
        generatorA: "nn.Module", generatorB: "nn.Module",
        discriminatorA: "nn.Module", discriminatorB: "nn.Module",
        optimizerG: "optim.Optimizer", optimizerD: "optim.Optimizer",
        realA: "torch.Tensor", realB: "torch.Tensor",
        perceptual_loss: Union["nn.Module", bool] = False,
        lambda_perceptual: float = 1, lambda_cycle: float = 10, lambda_identity: float = 0.5,
):
    fakeA, fakeB = generatorA(realB), generatorB(realA)
    backA, backB = generatorA(fakeB), generatorB(fakeA)
    sameA, sameB = generatorA(realA), generatorB(realB)

    # ===Discriminator Loss===
    # adversarial Loss
    pred_realA, pred_realB = discriminatorA(realA), discriminatorB(realB)
    pred_fakeA_true, pred_fakeB_true = discriminatorA(fakeA.detach()), discriminatorB(fakeB.detach())
    loss_adversarialDA = (F.binary_cross_entropy(pred_realA, torch.ones_like(pred_realA)) +
                          F.binary_cross_entropy(pred_fakeA_true, torch.zeros_like(pred_fakeA_true)))
    loss_adversarialDB = (F.binary_cross_entropy(pred_realB, torch.ones_like(pred_realB)) +
                          F.binary_cross_entropy(pred_fakeB_true, torch.zeros_like(pred_fakeB_true)))
    loss_adversarialD = (loss_adversarialDA + loss_adversarialDB) / 2
    # total loss
    loss_D = loss_adversarialD
    # backprop
    optimizerD.zero_grad()
    loss_D.backward()
    optimizerD.step()
    # ---End Discriminator Loss---

    # ===Generator Loss===
    # adversarial loss
    pred_fakeA_false, pred_fakeB_false = discriminatorA(fakeA), discriminatorB(fakeB)
    loss_adversarialGA = F.binary_cross_entropy(pred_fakeA_false, torch.ones_like(pred_fakeA_false))
    loss_adversarialGB = F.binary_cross_entropy(pred_fakeB_false, torch.ones_like(pred_fakeB_false))
    loss_adversarialG = (loss_adversarialGA + loss_adversarialGB) / 2
    # cycle loss
    loss_cycleA = F.l1_loss(backA, realA)
    loss_cycleB = F.l1_loss(backB, realB)
    loss_cycle = (loss_cycleA + loss_cycleB) / 2
    # identity loss
    loss_identityA = F.l1_loss(sameA, realA)
    loss_identityB = F.l1_loss(sameB, realB)
    loss_identity = (loss_identityA + loss_identityB) / 2
    # perceptual loss
    if perceptual_loss:
        loss_perceptualA = perceptual_loss(sameA, realA)
        loss_perceptualB = perceptual_loss(sameA, realB)
        loss_perceptual = (loss_perceptualA + loss_perceptualB) / 2
    else:
        loss_perceptual = torch.tensor(0)
    # total loss
    loss_G = loss_adversarialG + lambda_cycle * loss_cycle + lambda_identity * loss_identity + loss_perceptual * lambda_perceptual
    # backprop
    optimizerG.zero_grad()
    loss_G.backward()
    optimizerG.step()
    # ---End Generator Loss---

    # ===Misc===
    loss_total = loss_D + loss_G
    metrics = {
        "loss/total": loss_total.item(),
        "loss/adversarialD": loss_adversarialD.item(),
        "loss/adversarialG": loss_adversarialG.item(),
        "loss/cycle": loss_cycle.item(),
        "loss/identity": loss_identity.item(),
        "loss/perceptual": loss_perceptual.item(),
        "loss/D": loss_D.item(),
        "loss/G": loss_G.item(),
        "pred/realA": pred_realA.mean().item(),
        "pred/realB": pred_realB.mean().item(),
        "pred/fakeA_true": pred_fakeA_true.mean().item(),
        "pred/fakeB_true": pred_fakeB_true.mean().item(),
        "pred/fakeA_false": pred_fakeA_false.mean().item(),
        "pred/fakeB_false": pred_fakeB_false.mean().item(),
    }

    return metrics


def build_cycle_gan_trainer(
        generatorA: "nn.Module", generatorB: "nn.Module", discriminatorA: "nn.Module", discriminatorB: "nn.Module",
        optimizerG: "optim.Optimizer", optimizerD: "optim.Optimizer",
        loss_step_fn: Callable[..., dict[str, float]],
        data_extractor: Callable[..., Iterable["torch.Tensor"]],
        **kwargs
):
    """
    A helper function to build a trainer for CycleGAN.
    :param generatorA: generator for domain A
    :param generatorB: generator for domain B
    :param discriminatorA: discriminator for domain A
    :param discriminatorB: discriminator for domain B
    :param optimizerG: optimizer for generator
    :param optimizerD: optimizer for discriminator
    :param loss_step_fn: loss function
    :param data_extractor: callable to extract domain A and domain B data from batch
    :keyword writer: tensorboard writer
    :keyword writer_period: logging period
    :keyword fixed_inp: fixed input for logging
    :keyword save_path: checkpoint save path
    :keyword save_period: checkpoint save period
    :keyword perceptual_loss: use perceptual loss
    :keyword lambda_perceptual: perceptual loss weight
    :keyword lambda_cycle: cycle loss weight
    :keyword lambda_identity: identity loss weight
    :keyword device: device to run on
    :return: trainer function
    """
    writer = kwargs.pop("writer", None)
    writer_period = kwargs.pop("writer_period", 100)
    fixed_inp = kwargs.pop("fixed_inp", None)
    save_path = kwargs.pop("save_path", None)
    save_period = kwargs.pop("save_period", 200)
    perceptual_loss = kwargs.pop("perceptual_loss", False)
    lambda_perceptual = kwargs.pop("lambda_perceptual", 1)
    lambda_cycle = kwargs.pop("lambda_cycle", 10)
    lambda_identity = kwargs.pop("lambda_identity", 1)
    device = kwargs.pop("device", None)

    if writer:
        assert fixed_inp is not None, \
            "parameters `writer` and `fixed_inp` are mutually inclusive"
        assert len(fixed_inp) == 2, \
            "parameter `fixed_inp` must be a list of length 2"
    if perceptual_loss is True:
        perceptual_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:29].eval().to(device)
        perceptual_loss = PerceptualLoss(perceptual_model)

    first = True

    def trainer(DATA, step: int) -> dict[str, float]:
        nonlocal first

        realA, realB = data_extractor(DATA)
        metrics = loss_step_fn(
            generatorA, generatorB,
            discriminatorA, discriminatorB,
            optimizerG, optimizerD,
            realA, realB,
            perceptual_loss,
            lambda_perceptual, lambda_cycle, lambda_identity,
        )

        # ===Logging===
        if writer and step % writer_period == 0:
            for k, v in metrics.items(): writer.add_scalar(k, v, step)
            realA, realB = fixed_inp
            grid_realA = make_grid(realA, nrow=len(realA), normalize=True)
            grid_realB = make_grid(realB, nrow=len(realB), normalize=True)
            grid_fakeA = make_grid(generatorA(realB), nrow=len(realA), normalize=True)
            grid_fakeB = make_grid(generatorB(realA), nrow=len(realB), normalize=True)
            gridA = make_grid([grid_realA, grid_fakeB], nrow=2, normalize=True)
            gridB = make_grid([grid_realB, grid_fakeA], nrow=2, normalize=True)
            writer.add_image("image/gradA", gridA, step)
            writer.add_image("image/gridB", gridB, step)
        # ---End Logging---

        # ===Checkpoint===
        if save_path and step % save_period == 0 and not first:
            save_checkpoint(
                save_path,
                models={
                    "generatorA": generatorA,
                    "generatorB": generatorB,
                    "discriminatorA": discriminatorA,
                    "discriminatorB": discriminatorB,
                },
                optimizers={
                    "generator": optimizerG,
                    "discriminator": optimizerD,
                },
                step=step,
            )
        # ---End Checkpoint---

        if first: first = False
        metrics["_step"] = step
        return metrics

    return trainer


__all__ = [
    "build_gan_trainer",
    "build_cycle_gan_trainer",
    "kl_gan_loss_step",
    "kl_cycle_gan_loss_step",
]
