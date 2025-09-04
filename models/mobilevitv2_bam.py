# dual-import so it works as package (-m) or by absolute path
try:
    from models.mobilevitv2_baseline import get_mobilevitv2_base
    from models.bam import BAM
except ImportError:
    from mobilevitv2_baseline import get_mobilevitv2_base
    from bam import BAM

import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileViTv2_BAM(nn.Module):
    """
    Wraps a MobileViT-v2 (timm) backbone, applies BAM on the final feature map,
    then uses the original classification head.
    """
    def __init__(self, num_classes: int = 4, dummy_size: int = 224):
        super().__init__()
        self.base = get_mobilevitv2_base(num_classes=num_classes)

        # infer channels from forward_features
        with torch.no_grad():
            self.base.backbone.eval()
            feat = self.base.backbone.forward_features(torch.zeros(1, 3, dummy_size, dummy_size))
            if feat.dim() != 4:
                raise RuntimeError(f"Expected (B,C,H,W) features, got {feat.shape}")
            ch = int(feat.shape[1])

        self.bam = BAM(in_channels=ch)
        self.has_forward_head = hasattr(self.base.backbone, "forward_head")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.base.backbone.forward_features(x)  # (B, C, H, W)
        feat = self.bam(feat)
        if self.has_forward_head:
            return self.base.backbone.forward_head(feat, pre_logits=False)
        # rare fallback
        feat = F.adaptive_avg_pool2d(feat, 1)
        feat = torch.flatten(feat, 1)
        return self.base.classifier(feat)

def get_mobilevitv2_bam(num_classes: int):
    return MobileViTv2_BAM(num_classes=num_classes)
