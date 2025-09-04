import torch
import torch.nn as nn

class C2PSA(nn.Module):
    """
    Channel-Compressed Position Self-Attention.
    Operates on (B, C, H, W) and returns (B, C, H, W).
    """
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        c_red = max(4, in_channels // reduction)
        self.q = nn.Conv2d(in_channels, c_red, kernel_size=1, bias=False)
        self.k = nn.Conv2d(in_channels, c_red, kernel_size=1, bias=False)
        self.v = nn.Conv2d(in_channels, c_red, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(c_red, in_channels, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        q = self.q(x).view(B, -1, N).transpose(1, 2)  # (B, N, C')
        k = self.k(x).view(B, -1, N)                  # (B, C', N)
        attn = torch.bmm(q, k) / (k.size(1) ** 0.5)   # (B, N, N)
        attn = self.softmax(attn)

        v = self.v(x).view(B, -1, N).transpose(1, 2)  # (B, N, C')
        out = torch.bmm(attn, v).transpose(1, 2)      # (B, C', N)
        out = out.contiguous().view(B, -1, H, W)      # (B, C', H, W)
        out = self.proj(out)                          # (B, C, H, W)
        return out + x                                 # residual
