from typing import Union

import torch
from torch import nn, optim
from torch.nn import functional as F


def kl_loss_step(
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
