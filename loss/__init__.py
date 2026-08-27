"""Loss selection for model training."""

from .segmentation import L1Loss
from .segmentation import LogCoshLoss
from .segmentation import MSELoss
from .segmentation import WeightMSELoss
from .segmentation import msssimLoss


LossSelector = {
    "mseloss": MSELoss,
    "l1loss": L1Loss,
    "weightmseloss": WeightMSELoss,
    "logcoshloss": LogCoshLoss,
    "msssimloss": msssimLoss,
}
