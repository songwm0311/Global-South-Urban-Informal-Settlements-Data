# -*- coding: utf-8 -*-


from .Regression import MSELoss
from .Regression import L1Loss
from .Regression import WeightMSELoss
from .Regression import LogCoshLoss
from .Regression import msssimLoss



LossSelector = {'mseloss': MSELoss,
                'l1loss': L1Loss,
                'weightmseloss': WeightMSELoss,
                'logcoshloss':LogCoshLoss,
                'msssimloss':msssimLoss}
