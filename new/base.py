from abc import ABCMeta, abstractmethod
from typing import Iterable, Union

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import models

from utils.benchmark import PerceptualLoss


class Generator(nn.Module, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        """
        Base class for all generators
        :keyword act: Activation function
        :keyword norm: Normalization layer
        :keyword n: Number of layers in each sampling layer
        :keyword p: Dropout probability
        :keyword sampling: Number of sampling layers
        """
        act = kwargs.pop("act", nn.ReLU)
        norm = kwargs.pop("norm", nn.InstanceNorm2d)
        n = kwargs.pop("n", 1)
        p = kwargs.pop("p", 0)
        sampling = kwargs.pop("sampling", 2)
        assert isinstance(sampling, int) and sampling > 0, \
            f"Invalid value for sampling, must be a positive integer, got {sampling}"
        super().__init__()
        self.act = act
        self.norm = norm
        self.n = n
        self.p = p
        self.sampling = sampling

    @abstractmethod
    def sample_noise(self, *args, **kwargs) -> "torch.Tensor":
        """
        Sample noise from latent space.
        :return: Noise tensor
        """
        raise NotImplementedError


class Vec2ImgGenerator(Generator):
    def __init__(self, latents: int, out_chn: int, hid_chn: int, **kwargs):
        """
        Base class for all latent generators
        :param latents: Dimension of latent space.
        :param out_chn: Number of output channels.
        :param hid_chn: Number of hidden channels.
        :keyword act: Activation function.
        :keyword norm: Normalization layer.
        :keyword n: Number of layers in each sampling layer.
        :keyword p: Dropout probability.
        :keyword sampling: Number of sampling layers.
        """
        assert isinstance(latents, int) and latents > 0, \
            f"Invalid value of latent_dim, must be a positive integer, got {latents}"
        assert isinstance(out_chn, int) and out_chn > 0, \
            f"Invalid value of out_chn, must be a positive integer, got {out_chn}"
        assert isinstance(hid_chn, int) and hid_chn > 0, \
            f"Invalid value of hid_chn, must be a positive integer, got {hid_chn}"
        super().__init__(**kwargs)
        self.latents = latents
        self.out_chn = out_chn
        self.hid_chn = hid_chn

    def sample_noise(self, batch_size: int, device: str) -> "torch.Tensor":
        """
        Sample noise from latent space.
        :param batch_size: Batch size.
        :param device: Device to sample noise on.
        :return: Noise tensor
        """
        return torch.randn(batch_size, self.latents, device=device)


class Img2ImgGenerator(Generator):
    def __init__(self, inp_chn: int, out_chn: int, hid_chn: int, **kwargs):
        """
        Base class for all image generators
        :param inp_chn: Number of input channels.
        :param out_chn: Number of output channels.
        :param hid_chn: Number of hidden channels.
        :keyword act: Activation function.
        :keyword norm: Normalization layer.
        :keyword n: Number of layers in each sampling layer.
        :keyword p: Dropout probability.
        :keyword sampling: Number of sampling layers.
        """
        assert isinstance(inp_chn, int) and inp_chn > 0, \
            f"Invalid value of in_chn, must be a positive integer, got {inp_chn}"
        assert isinstance(out_chn, int) and out_chn > 0, \
            f"Invalid value of out_chn, must be a positive integer, got {out_chn}"
        assert isinstance(hid_chn, int) and hid_chn > 0, \
            f"Invalid value of hid_chn, must be a positive integer, got {hid_chn}"
        super().__init__(**kwargs)
        self.inp_chn = inp_chn
        self.out_chn = out_chn
        self.hid_chn = hid_chn

    def sample_noise(self, batch_size: int, height: int, width: int, device: str) -> "torch.Tensor":
        """
        Sample noise from latent space.
        :param batch_size: Batch size.
        :param height: Height of image.
        :param width: Width of image.
        :param device: Device to sample noise on.
        :return: Noise tensor.
        """
        return torch.randn(batch_size, self.inp_chn, height, width, device=device)


class Discriminator(nn.Module):
    USE_SIGMOID = True

    def __init__(self, blocks: int | Iterable[int], **kwargs):
        """
        Base class for all discriminators.
        :param blocks: Number of sampling layers | Number of channels in each sampling layer.
        :keyword act: Activation function.
        :keyword norm: Normalization layer.
        :keyword n: Number of layers in each sampling layer.
        :keyword p: Dropout probability.
        :keyword m_block: multiplier for number of channels in each block.
        """
        n = kwargs.pop("n", 1)
        p = kwargs.pop("p", 0)
        norm = kwargs.pop("norm", nn.InstanceNorm2d)
        act = kwargs.pop("act", nn.LeakyReLU)
        if isinstance(blocks, int):
            m_block = kwargs.pop("m_block", 32)
            blocks = (m_block * 2 ** i for i in range(blocks))
        else:
            assert isinstance(blocks, Iterable) and "block_core" not in kwargs, \
                f"parameter block_core must be only used with integer valued parameter blocks"
        blocks = list(blocks)
        assert all(isinstance(b, int) and b > 0 for b in blocks), \
            f"Invalid value of blocks, must be a positive integer or a list of positive integers, got {blocks}"
        super().__init__()
        self.blocks: list[int] = blocks
        self.act = act
        self.norm = norm
        self.n = n
        self.p = p


class Critic(Discriminator):
    USE_SIGMOID = False

    def __init__(self, blocks: int | list[int], **kwargs):
        """
        Base class for all critics.
        :param blocks: Number of sampling layers | Number of channels in each sampling layer.
        :keyword act: Activation function.
        :keyword norm: Normalization layer.
        :keyword n: Number of layers in each sampling layer.
        :keyword p: Dropout probability.
        :keyword m_block: multiplier for number of channels in each block.
        """
        super().__init__(blocks, **kwargs)


class GAN(nn.Module, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        """
        Base class for all GANs.
        :param generators: List of generators.
        :param discriminators: List of discriminators.
        :keyword perceptual: Use perceptual loss.
        :keyword perceptual_lambda: Weight of perceptual loss.
        """
        perceptual = kwargs.pop("perceptual", False)
        perceptual_lambda = kwargs.pop("perceptual_lambda", 1)
        if perceptual is True:
            perceptual_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:29].eval()
            perceptual = PerceptualLoss(perceptual_model)
        assert perceptual is False or isinstance(perceptual, PerceptualLoss), \
            f"Invalid value of perceptual, must be a boolean or an instance of PerceptualLoss, got {perceptual}"
        assert isinstance(perceptual_lambda, float) and perceptual_lambda > 0, \
            f"Invalid value of perceptual_lambda, must be a positive float, got {perceptual_lambda}"
        super().__init__()
        self.perceptual = perceptual
        self.perceptual_lambda = perceptual_lambda

    @abstractmethod
    def loss_step(
            self,
            generator: "nn.Module",
            discriminator: "nn.Module",
            optimizerG: "optim.Optimizer",
            optimizerD: "optim.Optimizer",
            noise: "torch.Tensor",
            real: "torch.Tensor",
    ) -> dict[str, float]:
        raise NotImplementedError


class KLGan(GAN):
    def loss_step(
            self,
            generator: "nn.Module",
            discriminator: "nn.Module",
            optimizerG: "optim.Optimizer",
            optimizerD: "optim.Optimizer",
            noise: "torch.Tensor",
            real: "torch.Tensor",
    ) -> dict[str, float]:
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
        loss_perceptual = self.perceptual(fake, real) if self.perceptual else torch.tensor(0.0)
        # total loss
        loss_G = loss_adversarialG + self.perceptual_lambda * loss_perceptual
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


__all__ = [
    "Generator",
    "Vec2ImgGenerator",
    "Img2ImgGenerator",
    "Discriminator",
    "Critic",
    "GAN",
    "KLGan",
]
