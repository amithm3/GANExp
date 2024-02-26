from torch import nn


class PerceptualLoss:
    """
    Perceptual loss.
    :param model: model to use for perceptual loss
    :param criterion: cost function to use for perceptual loss
    """
    def __init__(self, model: "nn.Module", criterion=None):
        if criterion is None: criterion = nn.L1Loss()
        self.model = model.eval().requires_grad_(False)
        self.criterion = criterion

    def __call__(self, x, y):
        return self.criterion(self.model(x), self.model(y))
