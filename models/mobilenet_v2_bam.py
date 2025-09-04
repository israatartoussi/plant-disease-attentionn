import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision.models.mobilenetv2 import InvertedResidual
from bam import BAM

class MobileNetV2_BAM(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetV2_BAM, self).__init__()
        original_model = mobilenet_v2(pretrained=True)

        self.features = nn.Sequential()
        for idx, layer in enumerate(original_model.features):
            self.features.add_module(f"{idx}", layer)
            # Insert BAM after layer 14 and 17 (based on channels)
            if idx == 14:
                self.features.add_module("bam_14", BAM(in_channels=160))
            elif idx == 17:
                self.features.add_module("bam_17", BAM(in_channels=320))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
