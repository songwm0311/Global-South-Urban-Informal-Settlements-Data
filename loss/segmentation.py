"""Loss functions used by the informal-settlement segmentation models."""

import torch
import torch.nn.functional as F

from loss.structural_similarity import MSSSIMLoss


def _masked_mean(values, mask=None):
    if mask is None:
        return values.mean()
    denominator = mask.sum().clamp_min(torch.finfo(values.dtype).eps)
    return (values * mask).sum() / denominator


class L1Loss(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()

    def forward(self, prediction, target, mask=None):
        return _masked_mean(torch.abs(prediction - target), mask)


class WeightMSELoss(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()

    def forward(self, prediction, target, mask=None):
        error = torch.abs(prediction - target)
        weights = torch.exp(8 * target)
        return _masked_mean(weights * error, mask)


class MSELoss(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()

    def forward(self, prediction, target, mask=None):
        return _masked_mean((prediction - target) ** 2, mask)


class LogCoshLoss(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()

    def forward(self, prediction, target, mask=None):
        error = prediction - target
        values = error + F.softplus(-2 * error) - torch.log(
            torch.tensor(2.0, dtype=error.dtype, device=error.device)
        )
        return _masked_mean(values, mask)


class MSSSIMMaskedLoss(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.criterion = MSSSIMLoss(data_range=1)

    def forward(self, prediction, target, mask=None):
        if mask is not None:
            prediction = prediction * mask
            target = target * mask
        return self.criterion(prediction, target)


# Backward-compatible name used by existing command-line options.
msssimLoss = MSSSIMMaskedLoss
