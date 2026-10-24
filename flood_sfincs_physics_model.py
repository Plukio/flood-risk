"""Spatial and local controls for 10 m SFINCS water-state pretraining."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def water_projection(logits: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """Enforce nonnegative depth and exact closed-domain water storage."""
    # Image channel 3 is current depth / .05 m; channel 4 is rain rate / 60
    # mm/h. Each transition spans exactly 300 s.
    current = image[:, 3:4]*.05
    rain_increment = image[:, 4:5]*.005
    target_mean = (current+rain_increment).mean(dim=(-2, -1), keepdim=True)
    weights = F.softplus(logits)+1e-6
    return weights/weights.mean(dim=(-2, -1), keepdim=True)*target_mean


class Block(nn.Module):
    def __init__(self, source: int, target: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(source, target, 3, padding=1),
                                    nn.SiLU(),
                                    nn.Conv2d(target, target, 3, padding=1),
                                    nn.SiLU())

    def forward(self, x):
        return self.layers(x)


class SpatialFlowUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = Block(5, 16)
        self.down2 = Block(16, 32)
        self.down3 = Block(32, 64)
        self.middle = Block(64, 96)
        self.up3 = Block(96+64, 64)
        self.up2 = Block(64+32, 32)
        self.up1 = Block(32+16, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, image):
        a = self.down1(image)
        b = self.down2(F.avg_pool2d(a, 2))
        c = self.down3(F.avg_pool2d(b, 2))
        middle = self.middle(F.avg_pool2d(c, 2))
        u = self.up3(torch.cat((F.interpolate(middle, size=c.shape[-2:],
                                            mode="bilinear", align_corners=False), c), 1))
        u = self.up2(torch.cat((F.interpolate(u, size=b.shape[-2:],
                                            mode="bilinear", align_corners=False), b), 1))
        u = self.up1(torch.cat((F.interpolate(u, size=a.shape[-2:],
                                            mode="bilinear", align_corners=False), a), 1))
        return water_projection(self.head(u), image)


class PlainSpatialCNN(nn.Module):
    """Full-resolution dilated CNN with a broad receptive field, no U-Net."""
    def __init__(self):
        super().__init__()
        self.entry = nn.Conv2d(5, 48, 3, padding=1)
        dilation = (1, 2, 4, 8, 16, 16, 8, 4, 2, 1)
        self.layers = nn.ModuleList(nn.Conv2d(48, 48, 3, padding=d,
                                               dilation=d) for d in dilation)
        self.head = nn.Conv2d(48, 1, 1)

    def forward(self, image):
        features = F.silu(self.entry(image))
        for layer in self.layers:
            features = F.silu(features+.2*layer(features))
        return water_projection(self.head(features), image)


class LocalFlowMLP(nn.Module):
    """Same local input and mass constraint, without learned spatial context."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(5, 32, 1), nn.SiLU(),
                                    nn.Conv2d(32, 32, 1), nn.SiLU(),
                                    nn.Conv2d(32, 1, 1))

    def forward(self, image):
        return water_projection(self.layers(image), image)
