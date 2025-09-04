# bootstrap so absolute-path execution finds the package root
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # /home/itartoussi/classification/project
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mobilevitv2_baseline import get_mobilevitv2_base
from models.cbam import CBAM

class MobileViTv2_CBAM(nn.Module):
    """
    Wraps a MobileViT-v2 (timm) backbone, applies CBAM on the final feature map,
    then uses the original classification head.
    """
    def __init__(self, num_classes=4, dummy_size=224):
        super().__init__()
        # Build baseline (timm MobileViT-v2 under the hood)
        self.base = get_mobilevitv2_base(num_classes=num_classes)

        # Infer channels from forward_features
        with torch.no_grad():
            self.base.backbone.eval()
            feat = self.base.backbone.forward_features(torch.zeros(1, 3, dummy_size, dummy_size))
            if feat.dim() != 4:
                raise RuntimeError(f"Expected (B,C,H,W) features, got {feat.shape}")
            ch = int(feat.shape[1])

        self.cbam = CBAM(channels=ch)
        self.has_forward_head = hasattr(self.base.backbone, "forward_head")

    def forward(self, x):
        feat = self.base.backbone.forward_features(x)  # (B, C, H, W)
        feat = self.cbam(feat)
        if self.has_forward_head:
            return self.base.backbone.forward_head(feat, pre_logits=False)
        # Rare fallback
        feat = F.adaptive_avg_pool2d(feat, 1)
        feat = torch.flatten(feat, 1)
        return self.base.classifier(feat)

def get_mobilevitv2_cbam(num_classes: int):
    return MobileViTv2_CBAM(num_classes=num_classes)
