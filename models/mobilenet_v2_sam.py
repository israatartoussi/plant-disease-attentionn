import torch
import torch.nn as nn
from torchvision import models

# SAM Attention Module
class SAM(nn.Module):
    def __init__(self, in_planes):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv1(x))
        return x * attention

# MobileNetV2 + SAM
class MobileNetV2_SAM(nn.Module):
    def __init__(self, base_model, num_classes):
        super(MobileNetV2_SAM, self).__init__()
        self.features = nn.Sequential()
        self.sam_layers = nn.ModuleDict()

        for name, module in base_model.features._modules.items():
            self.features.add_module(name, module)

            # Add SAM after each inverted residual block that has Conv2d
            if hasattr(module, "conv"):
                conv_layers = [layer for layer in module.conv if isinstance(layer, nn.Conv2d)]
                out_channels = conv_layers[-1].out_channels if conv_layers else base_model.last_channel

                # Add SAM
                sam = SAM(out_channels)  # ✅ Fixed: pass in_planes
                self.features.add_module(f"sam_{name}", sam)
                self.sam_layers[name] = sam

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(base_model.last_channel, num_classes)

    def forward(self, x):
        for name, module in self.features._modules.items():
            x = module(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
