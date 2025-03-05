import torch
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as F
from loss.structural_similarity import MSSSIMLoss

# ------------------------------------------------------------------------

class L1Loss(torch.nn.Module):  # input,target
    def __init__(self,weight=None):
        super().__init__()
    def forward(self, y_t, y_prime_t):
        loss = torch.nn.L1Loss()
        return loss(y_t, y_prime_t)


class WeightMSELoss(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()
    def forward(self, y_pre, y_label,mask):
        # B, C, H, W = y_label.size()
        e = torch.abs(y_pre-y_label) * mask
        weight_value1 = torch.exp(8*y_label)
        loss1 = torch.sum(weight_value1 * e) / (mask.sum())
        return loss1

class MSELoss(torch.nn.Module):
    def __init__(self,weight=None):
        super().__init__()
    def forward(self, y_pre, y_label,mask):
        e = torch.abs(y_pre - y_label) * mask
        loss1 = torch.sum(e*e) / (mask.sum())
        return   loss1

#class MSELoss(torch.nn.Module):
    #def __init__(self,weight=None):
        #super().__init__()
    #def forward(self, y_t, y_prime_t):
        #loss = torch.nn.MSELoss()
        #return   loss(y_t, y_prime_t)

class LogCoshLoss(torch.nn.Module):
    def __init__(self,weight=None):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
#        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))
        return torch.sum(torch.log(torch.cosh(ey_t + 1e-12)))


class XTanhLoss(torch.nn.Module):
    def __init__(self,weight=None):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(ey_t * torch.tanh(ey_t))


class XSigmoidLoss(torch.nn.Module):
    def __init__(self,weight=None):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        # return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)
        return torch.mean(2 * ey_t * torch.sigmoid(ey_t) - ey_t)


class AlgebraicLoss(torch.nn.Module):
    def __init__(self,weight=None):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(ey_t * ey_t / torch.sqrt(1 + ey_t * ey_t))


# class msssimLoss(torch.nn.Module):
#     def __init__(self, weight=None):
#         super().__init__()
#         self.alpha = 0.15  # 0.15
#         self.w_fact = torch.Tensor([0.0005, 0.001]).cuda()
#         self.w_exponent = torch.Tensor([0.001, 0.045]).cuda()
#         self.data_range = 1
#
#     def forward(self, output, target, mask):
#         weights = torch.exp(8 * target)
#         criterion = MSSSIMLoss(data_range=self.data_range)
#         e = torch.abs(output - target)*mask
#         loss1 = torch.sum(weights * e)/(mask.sum())
#         loss = self.alpha * loss1 + (1. - self.alpha) * criterion(output*mask, target*mask)
#         return loss

class msssimLoss(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.alpha = 0.15  # 0.15
        self.w_fact = torch.Tensor([0.0005, 0.001]).cuda()
        self.w_exponent = torch.Tensor([0.001, 0.045]).cuda()
        self.data_range = 1

    def forward(self, output, target, mask):
        criterion = MSSSIMLoss(data_range=self.data_range)
        loss = criterion(output*mask, target*mask)
        return loss