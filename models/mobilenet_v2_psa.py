import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from psa import PSA

class MobileNetV2_PSA(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetV2_PSA, self).__init__()
        base_model = mobilenet_v2(weights=None)
        self.features = base_model.features

        # Insert PSA after last bottleneck block
        self.psa = PSA(in_channels=1280)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.psa(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
