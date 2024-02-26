import os
from dataclasses import dataclass
from typing import Union, TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


@dataclass
class Config:
    model_name: str
    model_version: str = 'v1'
    model_dir: str = "./models/"
    log_dir: str = "./logs/"
    device: str = "cuda" if torch.backends.cuda.is_built() else "mps" if torch.backends.mps.is_built() else "cpu"
    writer: Union["SummaryWriter", bool] = False

    @property
    def checkpoint_path(self) -> str:
        return f"{self.model_dir}/{self.model_name}/{self.model_version}/"

    @property
    def log_path(self) -> str:
        return f"{self.log_dir}/{self.model_name}/{self.model_version}/"

    def __post_init__(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        if self.writer is True:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_path)

    def copy(self, **kwargs):
        return type(self)(**{**self.__dict__, **kwargs})  # type: ignore


def conv_weights_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0.0)


__all__ = [
    "Config",
    "conv_weights_init",
]
