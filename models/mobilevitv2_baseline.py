import torch
import torch.nn as nn
import timm  # pip install timm

class MobileViTv2_Baseline(nn.Module):
    def __init__(self, num_classes=4, backbone="mobilevitv2_100", pretrained=True):
        super().__init__()
        # timm provides a classifier head already
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
        self.has_forward_head = hasattr(self.backbone, "forward_head")

        # Fallback only if the backbone has no forward_head
        if not self.has_forward_head:
            with torch.no_grad():
                self.backbone.eval()
                feat = self.backbone.forward_features(torch.zeros(1, 3, 224, 224))
                ch = feat.shape[1]
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(ch, num_classes)

    def forward(self, x):
        feat = self.backbone.forward_features(x)
        if self.has_forward_head:
            return self.backbone.forward_head(feat, pre_logits=False)
        feat = self.pool(feat)
        feat = torch.flatten(feat, 1)
        return self.classifier(feat)

def get_mobilevitv2_base(num_classes):
    return MobileViTv2_Baseline(num_classes=num_classes, backbone="mobilevitv2_100", pretrained=True)
