"""
Deep learning architectures for chest X-ray (multi-label) and brain MRI (multi-class).
Uses ImageNet-pretrained backbones + custom heads (transfer learning).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


CHEST_LABELS = [
    "pneumonia",
    "covid19",
    "effusion",
    "fibrosis",
    "cardiomegaly",
    "atelectasis",
]

BRAIN_LABELS = ["glioma", "meningioma", "pituitary", "normal"]


class MultiLabelChestNet(nn.Module):
    """
    ResNet50 backbone, multi-label logits (BCEWithLogitsLoss in training).
    Output shape: (N, num_labels).
    """

    def __init__(self, num_labels: int = len(CHEST_LABELS), backbone: str = "resnet50"):
        super().__init__()
        self._name = backbone
        if backbone == "vgg16":
            net = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
            self.features = net.features
            self.avgpool = net.avgpool
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(1024, num_labels),
            )
            self.resnet = None
        else:
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, num_labels)
            self.resnet = net
            self.features = None
            self.avgpool = None
            self.head = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._name == "vgg16":
            assert self.features is not None
            x = self.features(x)
            x = self.avgpool(x)
            return self.head(x)
        assert self.resnet is not None
        return self.resnet(x)

    def gradcam_submodel(self) -> tuple[nn.Module, nn.Module]:
        """Return (forward_module, target_layer) for Grad-CAM."""
        if self._name == "resnet50" and self.resnet is not None:
            return self.resnet, self.resnet.layer4
        if self.features is not None:
            # VGG features: hook last conv (index -1 is ReLU after last conv in typical block — use -3 for last conv)
            return self, self.features[-2]
        raise RuntimeError("Model not initialized")


class BrainMRINet(nn.Module):
    """ResNet18 for lighter MRI inference (4-class softmax)."""

    def __init__(self, num_classes: int = len(BRAIN_LABELS)):
        super().__init__()
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_f = net.fc.in_features
        net.fc = nn.Linear(in_f, num_classes)
        self.inner = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)

    def gradcam_submodel(self) -> tuple[nn.Module, nn.Module]:
        return self.inner, self.inner.layer4


class DoubleConv(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """
    Compact U-Net for binary tumor/abnormality segmentation (256x256).
    """

    def __init__(self, in_ch: int = 3, base: int = 32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.out_conv = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)
