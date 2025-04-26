# -----------------------------------------------------------------------------
# model.py – simple ResNet18 regression head ⇒ single score
# -----------------------------------------------------------------------------
import torch.nn as nn
import torchvision.models as models


def get_scoring_model():
    base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in base.parameters():  # freeze feature extractor for speed
        param.requires_grad = False
    num_feats = base.fc.in_features
    base.fc = nn.Sequential(nn.Linear(num_feats, 1), nn.Sigmoid())  # 0‑1
    return base

