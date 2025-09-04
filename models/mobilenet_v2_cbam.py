from torchvision.models import mobilenet_v2
import torch.nn as nn
from cbam import CBAM


class MobileNetV2_CBAM(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2_CBAM, self).__init__()
        base = mobilenet_v2(pretrained=True)
        self.features = base.features
        self.cbam = CBAM(in_planes=1280)  # آخر إخراج MobileNetV2
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(base.last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.classifier(x)
