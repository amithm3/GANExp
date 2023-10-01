from typing import TYPE_CHECKING, Union, Callable

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import models
from torchvision.utils import make_grid

from . import DCGenerator, PatchDiscriminator, DCGanConfig
from utils.benchmark import PerceptualLoss
from utils.checkpoints import save_checkpoint

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


def build_DCGan(config: "DCGanConfig"):
    generator = DCGenerator(
        config.inp_channels, config.out_channels, config.hidden_channels,
        n=config.n, p=config.p, norm=config.norm, act=nn.ReLU(),
        upsample=config.upsample,
    ).to(config.device)
    discriminator = PatchDiscriminator(
        config.out_channels, config.blocks,
        n=config.n, p=config.p, norm=config.norm, act=nn.LeakyReLU(),
    ).to(config.device)
    return generator, discriminator


def build_DCGan_trainer(
        generator: "DCGenerator", discriminator: "PatchDiscriminator",
        optimizerG: "optim.Optimizer", optimizerD: "optim.Optimizer",
        inp_channels: int,
        data_extractor: Callable[..., "torch.Tensor"],
        save_path: str = None, save_period: int = 200,
        writer: "SummaryWriter" = False, writer_period: int = 100, fixed_inp: "torch.Tensor" = None,
        perceptual_loss: Union["nn.Module", bool] = False, lambda_perceptual: float = 1,
        device=None,
):
    if writer:
        assert fixed_inp is not None, \
            "parameters `writer` and `fixed_inp` are mutually inclusive"
    if perceptual_loss is True:
        perceptual_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:29].eval().to(device)
        perceptual_loss = PerceptualLoss(perceptual_model)

    def trainer(DATA, step: int) -> dict[str, float]:
        real = data_extractor(DATA)
        noise = torch.randn(real.shape[0], inp_channels, 1, 1, device=device)
        fake = generator(noise)

        # ===Discriminator Loss===
        # Adversarial Loss
        pred_real = discriminator(real)
        pred_fake_true = discriminator(fake.detach())
        loss_adversarialD = (F.binary_cross_entropy(pred_real, torch.ones_like(pred_real)) +
                             F.binary_cross_entropy(pred_fake_true, torch.zeros_like(pred_fake_true)))
        # Total Loss
        loss_D = loss_adversarialD
        # backprop
        optimizerD.zero_grad()
        loss_D.backward()
        optimizerD.step()
        # ---End Discriminator Loss---

        # ===Generator Loss===
        # Adversarial Loss
        pred_fake_false = discriminator(fake)
        loss_adversarialG = F.binary_cross_entropy(pred_fake_false, torch.ones_like(pred_fake_false))
        # Perceptual Loss
        loss_perceptual = perceptual_loss(fake, real) if perceptual_loss else torch.tensor(0.0)
        # Total Loss
        loss_G = loss_adversarialG + lambda_perceptual * loss_perceptual
        # backprop
        optimizerG.zero_grad()
        loss_G.backward()
        optimizerG.step()
        # ---End Generator Loss---

        # ===Logging===
        if writer and step % writer_period == 0:
            writer.add_scalar("loss/adversarialD", loss_adversarialD.item(), step)
            writer.add_scalar("loss/adversarialG", loss_adversarialG.item(), step)
            writer.add_scalar("loss/perceptual", loss_perceptual.item(), step)
            writer.add_scalar("loss/D", loss_D.item(), step)
            writer.add_scalar("loss/G", loss_G.item(), step)
            writer.add_scalar("pred/real", pred_real.mean().item(), step)
            writer.add_scalar("pred/fake_true", pred_fake_true.mean().item(), step)
            writer.add_scalar("pred/fake_false", pred_fake_false.mean().item(), step)

            grid = make_grid(generator(fixed_inp), nrow=len(fixed_inp), normalize=True)
            writer.add_image("image", grid, step)
        # ---End Logging---

        # ===Checkpoint===
        if save_path and step % save_period == 0 and step != 0:
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

        return {
            "loss/adversarialD": loss_adversarialD.item(),
            "loss/adversarialG": loss_adversarialG.item(),
            "loss/perceptual": loss_perceptual.item(),
            "loss/D": loss_D.item(),
            "loss/G": loss_G.item(),
            "pred/real": pred_real.mean().item(),
            "pred/fake_true": pred_fake_true.mean().item(),
            "pred/fake_false": pred_fake_false.mean().item(),
        }

    return trainer
