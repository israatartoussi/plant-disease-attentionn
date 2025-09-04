import torch
import torch.nn as nn

class BAM(nn.Module):
    """
    Bottleneck Attention Module operating on (B, C, H, W).
    Uses sigmoid(channel + spatial) as in the common BAM variant.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16, dilation: int = 4):
        super().__init__()
        hidden = max(4, in_channels // reduction_ratio)

        # Channel attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False),
        )

        # Spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ch = self.channel_att(x)
        sp = self.spatial_att(x)
        att = self.sigmoid(ch + sp)
        return x * att
