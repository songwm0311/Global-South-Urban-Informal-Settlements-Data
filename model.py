"""U-Net and false-positive-aware BCE plus Tversky loss."""
import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, input_channels, out_channels=1):
        super().__init__()
        self.enc1, self.enc2 = self.block(input_channels, 64), self.block(64, 128)
        self.enc3, self.enc4 = self.block(128, 256), self.block(256, 512)
        self.center = self.block(512, 1024)
        self.dec4, self.dec3 = self.block(1536, 512), self.block(768, 256)
        self.dec2, self.dec1 = self.block(384, 128), self.block(192, 64)
        self.final = nn.Conv2d(64, out_channels, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    @staticmethod
    def block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.ReLU(inplace=True), nn.GroupNorm(8, out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.ReLU(inplace=True), nn.GroupNorm(8, out_channels),
        )

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        center = self.center(self.pool(e4))
        d4 = self.dec4(torch.cat([e4, self.up(center)], 1)); d3 = self.dec3(torch.cat([e3, self.up(d4)], 1))
        d2 = self.dec2(torch.cat([e2, self.up(d3)], 1)); d1 = self.dec1(torch.cat([e1, self.up(d2)], 1))
        return self.final(d1)


class CombinedSegmentationLoss(nn.Module):
    def __init__(self, pos_weight=1, alpha=.8, beta=.2, tversky_weight=.25, negative_weight=1.25, smooth=1e-6):
        super().__init__(); self.pos_weight, self.alpha, self.beta = float(pos_weight), alpha, beta
        self.tversky_weight, self.negative_weight, self.smooth = tversky_weight, negative_weight, smooth

    def forward(self, logits, targets, valid):
        bce_map = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weight = targets * self.pos_weight + (1 - targets) * self.negative_weight
        bce = (bce_map * weight * valid).sum() / valid.sum().clamp_min(1)
        probabilities, target = torch.sigmoid(logits) * valid, targets * valid
        tp = (probabilities * target).sum(); fp = (probabilities * (1 - targets) * valid).sum(); fn = ((1 - probabilities) * target).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return bce + self.tversky_weight * (1 - tversky)

