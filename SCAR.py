
"""SCAR and C2f_SCAR modules for YOLO-style detectors.

Implementation follows the Spatial-Contextual Adaptive Reasoning (SCAR)
description in the uploaded CDR-YOLO paper and keeps the coding style of the
provided C2f_iRMB_EMA example.

The module is intentionally dependency-light: only PyTorch is required.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SCAR", "SCARBlock", "C2f_SCAR"]


# -----------------------------
# YOLO-style basic layers
# -----------------------------
def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard YOLO convolution: Conv2d + BatchNorm2d + SiLU."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


# -----------------------------
# Helper functions
# -----------------------------
def _valid_group_count(channels: int, groups: int) -> int:
    """Return the largest group count <= groups that divides channels."""
    groups = max(1, min(groups, channels))
    while channels % groups != 0:
        groups -= 1
    return groups


def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    """Channel shuffle used after concatenating local and global streams."""
    b, c, h, w = x.shape
    if groups <= 1 or c % groups != 0:
        return x
    x = x.reshape(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.reshape(b, c, h, w)


def _adaptive_pool_by_tokens(x: torch.Tensor, max_tokens: Optional[int]) -> torch.Tensor:
    """Pool a feature map so that H*W <= max_tokens while preserving aspect ratio.

    If max_tokens is None or <= 0, no pooling is applied and full spatial attention
    is used. Full attention can be memory-heavy on large feature maps.
    """
    if max_tokens is None or max_tokens <= 0:
        return x

    b, c, h, w = x.shape
    n = h * w
    if n <= max_tokens:
        return x

    scale = math.sqrt(float(max_tokens) / float(n))
    ph = max(1, int(round(h * scale)))
    pw = max(1, int(round(w * scale)))

    # Ensure that ph * pw does not exceed max_tokens.
    while ph * pw > max_tokens:
        if ph >= pw and ph > 1:
            ph -= 1
        elif pw > 1:
            pw -= 1
        else:
            break
    return F.adaptive_avg_pool2d(x, output_size=(ph, pw))


# -----------------------------
# SCAR module
# -----------------------------
class SCAR(nn.Module):
    """Spatial-Contextual Adaptive Reasoning module.

    Main structure:
      1) Local spatial excitation stream:
         group split -> GAP/1x1 branch + bounded 3x3 local branch -> mask -> reweight.
      2) Global semantic inference stream:
         Q/K/V point-wise projections -> scaled attention -> residual semantic feature.
      3) Fusion:
         concat(local, global) -> channel shuffle -> 1x1 projection.

    Args:
        channels: Number of input/output channels.
        scar_groups: Number of channel groups G in the local spatial stream.
        reduction: Channel reduction ratio for Q/K/V hidden dimension dk.
        max_tokens: Maximum key/value spatial tokens in global inference.
            Default 64 makes the module runnable even for large smoke-test inputs.
            Set to 0 or None to use exact full H*W by H*W attention on small maps.
        qkv_bias: Whether to use bias in Q/K/V projections.
        attn_drop: Dropout ratio for the attention map.
        proj_drop: Dropout ratio after the global output projection.
    """

    def __init__(
        self,
        channels: int,
        scar_groups: int = 4,
        reduction: int = 4,
        max_tokens: Optional[int] = 64,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")

        self.channels = channels
        self.scar_groups = _valid_group_count(channels, scar_groups)
        self.group_channels = channels // self.scar_groups
        self.max_tokens = max_tokens

        # Local spatial excitation stream, shared by all channel groups after reshape.
        cg = self.group_channels
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.channel_response = nn.Conv2d(cg, cg, kernel_size=1, stride=1, padding=0, bias=True)
        # Depth-wise 3x3 convolution keeps the local branch lightweight while preserving spatial boundaries.
        self.local_spatial = nn.Conv2d(cg, cg, kernel_size=3, stride=1, padding=1, groups=cg, bias=True)
        self.local_aggregation = nn.Conv2d(2 * cg, cg, kernel_size=1, stride=1, padding=0, bias=True)

        # Global semantic inference stream.
        dk = max(1, channels // max(1, reduction))
        self.dk = dk
        self.scale = dk ** -0.5
        self.q = nn.Conv2d(channels, dk, kernel_size=1, bias=qkv_bias)
        self.k = nn.Conv2d(channels, dk, kernel_size=1, bias=qkv_bias)
        self.v = nn.Conv2d(channels, dk, kernel_size=1, bias=qkv_bias)
        self.global_proj = nn.Conv2d(dk, channels, kernel_size=1, bias=True)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

        # Stream fusion: Concat(X_spatial, X_global) -> shuffle -> 1x1 Conv.
        self.out_proj = Conv(2 * channels, channels, k=1, s=1)

    def local_spatial_excitation(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        g, cg = self.scar_groups, self.group_channels

        x_group = x.reshape(b * g, cg, h, w)
        z = self.gap(x_group)
        u = self.channel_response(z).expand(-1, -1, h, w)
        v = self.local_spatial(x_group)
        mask = torch.sigmoid(self.local_aggregation(torch.cat((u, v), dim=1)))
        x_spatial = x_group * mask
        return x_spatial.reshape(b, c, h, w)

    def global_semantic_inference(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        q = self.q(x).flatten(2).transpose(1, 2)              # B, N, dk
        kv_source = _adaptive_pool_by_tokens(x, self.max_tokens)
        k = self.k(kv_source).flatten(2).transpose(1, 2)      # B, M, dk
        v = self.v(kv_source).flatten(2).transpose(1, 2)      # B, M, dk

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # B, N, M
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)                           # B, N, dk
        out = out.transpose(1, 2).reshape(b, self.dk, h, w)
        out = self.proj_drop(self.global_proj(out))
        return out + x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_spatial = self.local_spatial_excitation(x)
        x_global = self.global_semantic_inference(x)
        y = torch.cat((x_spatial, x_global), dim=1)
        y = channel_shuffle(y, groups=2)
        return self.out_proj(y)


class SCARBlock(nn.Module):
    """A residual wrapper around SCAR, convenient for cascaded use in C2f_SCAR."""

    def __init__(
        self,
        c1: int,
        c2: Optional[int] = None,
        shortcut: bool = True,
        scar_groups: int = 4,
        reduction: int = 4,
        max_tokens: Optional[int] = 64,
    ):
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.proj = Conv(c1, c2, k=1, s=1) if c1 != c2 else nn.Identity()
        self.scar = SCAR(c2, scar_groups=scar_groups, reduction=reduction, max_tokens=max_tokens)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.scar(self.proj(x))
        return x + y if self.add else y


class C2f_SCAR(nn.Module):
    """C2f block integrated with cascaded SCAR units.

    This keeps the YOLOv8 C2f split-concat topology and replaces the transformation
    branch bottlenecks with SCAR reasoning units, as described for C2f_SCAR.

    Args are kept compatible with common Ultralytics C2f signatures:
        c1, c2, n=1, shortcut=False, g=1, e=0.5
    Additional SCAR controls can be passed when using the module directly.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,       # kept for YAML/API compatibility; SCAR uses scar_groups instead
        e: float = 0.5,
        scar_groups: int = 4,
        reduction: int = 4,
        max_tokens: Optional[int] = 64,
    ):
        super().__init__()
        self.c = int(c2 * e)
        if self.c <= 0:
            raise ValueError(f"hidden channels must be positive, got c2={c2}, e={e}")
        self.cv1 = Conv(c1, 2 * self.c, k=1, s=1)
        self.cv2 = Conv((2 + n) * self.c, c2, k=1, s=1)
        self.m = nn.ModuleList(
            SCARBlock(
                self.c,
                self.c,
                shortcut=shortcut,
                scar_groups=scar_groups,
                reduction=reduction,
                max_tokens=max_tokens,
            )
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))


if __name__ == "__main__":
    # Smoke tests: same style as the provided iEMA code.
    torch.manual_seed(0)

    for shape in [(1, 64, 80, 80), (1, 64, 640, 640)]:
        image = torch.rand(*shape)
        model = C2f_SCAR(64, 64, n=1, shortcut=False, scar_groups=4, reduction=4, max_tokens=64)
        model.eval()
        with torch.no_grad():
            out = model(image)
        print(f"input: {tuple(image.shape)} -> output: {tuple(out.shape)}")
