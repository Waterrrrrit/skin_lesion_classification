from __future__ import annotations
import torch
from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_B3_Weights

def build_efficientnet_b3(
    num_classes: int,
    device: torch.device,
    pretrained: bool = True,
    dropout_p: float = 0.3
) -> nn.Module:
    """
    이미지 계획 반영: EfficientNet-B3 사용 및 Dropout(0.3) 적용
    """
    weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b3(weights=weights)

    # Classifier 구조 변경: Dropout 포함
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_p, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    return model.to(device)
