# Ultralytics YOLO-compatible SGDR implementation
# Implements the paper's Saliency-Guided Dynamic Routing (SGDR) mechanism
# and its C2f integration (C2f_SGDR).
#
# Design:
#   z = GAP(X)
#   s = sigmoid(W2(ReLU(W1(z))))
#   M_c = 1[s_c > tau]
#   active channels: 3x3 convolution
#   dormant channels: identity mapping
#   output: feature reintegration in the original channel order
#
# Training uses a straight-through hard gate so the saliency estimator and
# learnable threshold receive gradients. Evaluation can use true dynamic
# channel slicing to skip dormant-channel convolution.

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SGDR", "C2f_SGDR"]


def autopad(
    k: Union[int, Tuple[int, int]],
    p: Optional[Union[int, Tuple[int, int]]] = None,
    d: int = 1,
) -> Union[int, Tuple[int, int]]:
    """Return padding that preserves the spatial resolution."""
    if d > 1:
        if isinstance(k, int):
            k = d * (k - 1) + 1
        else:
            k = tuple(d * (x - 1) + 1 for x in k)
    if p is None:
        p = k // 2 if isinstance(k, int) else tuple(x // 2 for x in k)
    return p


class Conv(nn.Module):
    """Ultralytics-style Conv-BN-activation block."""

    default_act = nn.SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: Union[int, Tuple[int, int]] = 1,
        s: int = 1,
        p: Optional[Union[int, Tuple[int, int]]] = None,
        g: int = 1,
        d: int = 1,
        act: Union[bool, nn.Module] = True,
    ) -> None:
        super().__init__()
        if c1 <= 0 or c2 <= 0:
            raise ValueError(f"c1 and c2 must be positive, got c1={c1}, c2={c2}.")
        if c1 % g != 0 or c2 % g != 0:
            raise ValueError(f"groups={g} must divide c1={c1} and c2={c2}.")

        self.conv = nn.Conv2d(
            c1,
            c2,
            k,
            s,
            autopad(k, p, d),
            groups=g,
            dilation=d,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class SGDR(nn.Module):
    """Saliency-Guided Dynamic Routing.

    Args:
        dim: Number of input/output channels.
        reduction: Bottleneck reduction ratio in the saliency estimator.
        tau: Initial routing threshold in (0, 1).
        learnable_tau: Whether tau is optimized end-to-end.
        temperature: Temperature used by the straight-through soft surrogate.
        dynamic_inference: If True, evaluation with batch size 1 or eager PyTorch
            performs actual channel slicing and convolves only active channels.
        min_active_channels: Optional lower bound on the active-channel count.
            The paper's strict threshold rule corresponds to 0.
        bias: Whether the active 3x3 convolution uses bias.

    Notes:
        1. The hard threshold is non-differentiable. During training, a
           straight-through estimator is used:
               M_st = M_hard + M_soft - stop_gradient(M_soft)
           so the forward pass remains a hard routing decision.
        2. Dynamic channel slicing is not ONNX/TensorRT friendly. During tracing
           or ONNX export, the module automatically uses the mathematically
           equivalent fixed-shape masked formulation.
        3. Feature reintegration preserves the original channel order. This is
           necessary for a well-defined batched tensor and is equivalent to
           concatenating the two routed branches followed by restoring their
           original channel positions.
    """

    def __init__(
        self,
        dim: int,
        reduction: int = 8,
        tau: float = 0.5,
        learnable_tau: bool = True,
        temperature: float = 0.1,
        dynamic_inference: bool = True,
        min_active_channels: int = 0,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}.")
        if reduction <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}.")
        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau must lie in (0, 1), got {tau}.")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        if not 0 <= min_active_channels <= dim:
            raise ValueError(
                f"min_active_channels must be in [0, {dim}], got {min_active_channels}."
            )

        self.dim = int(dim)
        self.reduction = int(reduction)
        self.temperature = float(temperature)
        self.dynamic_inference = bool(dynamic_inference)
        self.min_active_channels = int(min_active_channels)

        hidden = max(1, self.dim // self.reduction)

        # Global saliency estimator: GAP -> FC -> ReLU -> FC -> Sigmoid
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(self.dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, self.dim, bias=True)

        # Parameterize tau with a logit so it always remains inside (0, 1).
        tau_logit = math.log(tau / (1.0 - tau))
        tau_tensor = torch.tensor(float(tau_logit), dtype=torch.float32)
        if learnable_tau:
            self.tau_logit = nn.Parameter(tau_tensor)
        else:
            self.register_buffer("tau_logit", tau_tensor)

        # A single maximum-size 3x3 kernel bank. For an active set A, dynamic
        # inference uses W[A, A, :, :], which realizes a standard convolution
        # over the selected active channels only.
        self.weight = nn.Parameter(torch.empty(self.dim, self.dim, 3, 3))
        self.bias = nn.Parameter(torch.zeros(self.dim)) if bias else None

        self.register_buffer(
            "_last_active_ratio", torch.tensor(0.0), persistent=False
        )
        self.register_buffer("_last_tau", torch.tensor(float(tau)), persistent=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    @property
    def tau(self) -> torch.Tensor:
        """Current scalar threshold constrained to (0, 1)."""
        return torch.sigmoid(self.tau_logit)

    @torch.no_grad()
    def set_tau(self, value: float) -> None:
        """Set tau explicitly, useful for sensitivity/ablation experiments."""
        if not 0.0 < value < 1.0:
            raise ValueError(f"tau must lie in (0, 1), got {value}.")
        value = min(max(float(value), 1e-6), 1.0 - 1e-6)
        self.tau_logit.copy_(
            torch.tensor(
                math.log(value / (1.0 - value)),
                device=self.tau_logit.device,
                dtype=self.tau_logit.dtype,
            )
        )

    def saliency_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute input-dependent channel saliency coefficients s in (0, 1)."""
        z = self.avg_pool(x).flatten(1)
        return torch.sigmoid(self.fc2(F.relu(self.fc1(z), inplace=False)))

    def _routing_masks(
        self, scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return hard mask and straight-through mask."""
        tau = self.tau.to(dtype=scores.dtype, device=scores.device)
        hard = (scores > tau).to(scores.dtype)

        if self.min_active_channels > 0:
            k = min(self.min_active_channels, self.dim)
            topk_idx = scores.topk(k=k, dim=1, largest=True, sorted=False).indices
            hard = hard.scatter(1, topk_idx, 1.0)

        soft = torch.sigmoid((scores - tau) / self.temperature)
        straight_through = hard + soft - soft.detach()
        return hard, straight_through

    def _forward_masked(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Fixed-shape formulation used for training and model export."""
        mask4 = mask.unsqueeze(-1).unsqueeze(-1)
        active_input = x * mask4
        active_output = F.conv2d(
            active_input,
            self.weight,
            self.bias,
            stride=1,
            padding=1,
        )
        return active_output * mask4 + x * (1.0 - mask4)

    def _forward_dynamic(
        self, x: torch.Tensor, hard_mask: torch.Tensor
    ) -> torch.Tensor:
        """Actual per-sample dynamic channel slicing for eager inference."""
        outputs = []
        for batch_index in range(x.shape[0]):
            xb = x[batch_index : batch_index + 1]
            active_idx = torch.nonzero(
                hard_mask[batch_index] > 0.5, as_tuple=False
            ).flatten()

            if active_idx.numel() == 0:
                outputs.append(xb)
                continue

            x_active = xb.index_select(1, active_idx)
            w_active = (
                self.weight.index_select(0, active_idx)
                .index_select(1, active_idx)
                .contiguous()
            )
            b_active = (
                self.bias.index_select(0, active_idx).contiguous()
                if self.bias is not None
                else None
            )
            y_active = F.conv2d(
                x_active,
                w_active,
                b_active,
                stride=1,
                padding=1,
            )

            # Feature reintegration: processed active channels are written back
            # into their original positions; dormant channels remain unchanged.
            yb = xb.clone()
            yb.index_copy_(1, active_idx, y_active)
            outputs.append(yb)

        return torch.cat(outputs, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"SGDR expects a 4-D BCHW tensor, received shape {tuple(x.shape)}."
            )
        if x.shape[1] != self.dim:
            raise ValueError(
                f"SGDR was initialized with dim={self.dim}, "
                f"but received {x.shape[1]} channels."
            )

        scores = self.saliency_scores(x)
        hard_mask, straight_through_mask = self._routing_masks(scores)

        with torch.no_grad():
            self._last_active_ratio.copy_(hard_mask.float().mean())
            self._last_tau.copy_(self.tau.detach().float())

        export_or_trace = (
            torch.jit.is_tracing()
            or torch.jit.is_scripting()
            or torch.onnx.is_in_onnx_export()
        )

        if self.training:
            # Hard routing in the forward pass, differentiable surrogate in
            # the backward pass.
            return self._forward_masked(x, straight_through_mask)

        if self.dynamic_inference and not export_or_trace:
            return self._forward_dynamic(x, hard_mask)

        return self._forward_masked(x, hard_mask)

    def routing_statistics(self) -> Dict[str, float]:
        """Return the most recently observed routing statistics."""
        return {
            "tau": float(self._last_tau.item()),
            "active_ratio": float(self._last_active_ratio.item()),
            "dormant_ratio": float(1.0 - self._last_active_ratio.item()),
        }

    def extra_repr(self) -> str:
        learnable = isinstance(self.tau_logit, nn.Parameter)
        return (
            f"dim={self.dim}, reduction={self.reduction}, "
            f"tau={float(self.tau.detach()):.4f}, "
            f"learnable_tau={learnable}, temperature={self.temperature}, "
            f"dynamic_inference={self.dynamic_inference}, "
            f"min_active_channels={self.min_active_channels}"
        )


class C2f_SGDR(nn.Module):
    """C2f block whose transformation branch contains cascaded SGDR units.

    The constructor keeps the usual Ultralytics C2f signature so it can be used
    directly by the YOLO model parser.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        reduction: int = 8,
        tau: float = 0.5,
        learnable_tau: bool = True,
        temperature: float = 0.1,
        dynamic_inference: bool = True,
        min_active_channels: int = 0,
    ) -> None:
        super().__init__()
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}.")
        if not 0.0 < e <= 1.0:
            raise ValueError(f"e must lie in (0, 1], got {e}.")

        # shortcut and g are retained for parser/API compatibility. SGDR itself
        # follows the paper's active-convolution/dormant-identity formulation.
        _ = shortcut, g

        self.c = max(1, int(c2 * e))
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            SGDR(
                dim=self.c,
                reduction=reduction,
                tau=tau,
                learnable_tau=learnable_tau,
                temperature=temperature,
                dynamic_inference=dynamic_inference,
                min_active_channels=min_active_channels,
                bias=False,
            )
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(module(y[-1]) for module in self.m)
        return self.cv2(torch.cat(y, dim=1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), dim=1))
        y.extend(module(y[-1]) for module in self.m)
        return self.cv2(torch.cat(y, dim=1))

    def set_dynamic_inference(self, enabled: bool = True) -> None:
        for module in self.m:
            module.dynamic_inference = bool(enabled)

    @torch.no_grad()
    def set_tau(self, value: float) -> None:
        for module in self.m:
            module.set_tau(value)

    def routing_statistics(self) -> Dict[str, float]:
        stats = [module.routing_statistics() for module in self.m]
        if not stats:
            return {"tau": 0.0, "active_ratio": 0.0, "dormant_ratio": 0.0}
        return {
            "tau": sum(item["tau"] for item in stats) / len(stats),
            "active_ratio": sum(item["active_ratio"] for item in stats)
            / len(stats),
            "dormant_ratio": sum(item["dormant_ratio"] for item in stats)
            / len(stats),
        }


class CSPPC_SE(C2f_SGDR):
    """Backward-compatible alias.

    Existing YAML files that reference ``CSPPC_SE`` will now instantiate the
    paper-consistent C2f_SGDR implementation without requiring a YAML rename.
    """


def _self_test() -> None:
    torch.manual_seed(7)

    model = C2f_SGDR(
        c1=64,
        c2=128,
        n=2,
        reduction=8,
        tau=0.5,
        learnable_tau=True,
        temperature=0.1,
        dynamic_inference=True,
    )

    # Forward and backward test.
    model.train()
    x = torch.randn(2, 64, 32, 32, requires_grad=True)
    y = model(x)
    assert y.shape == (2, 128, 32, 32)
    y.mean().backward()
    assert x.grad is not None
    assert all(module.fc1.weight.grad is not None for module in model.m)
    assert all(module.tau_logit.grad is not None for module in model.m)

    # Dynamic and masked evaluation paths must be numerically equivalent.
    model.eval()
    x_eval = torch.randn(1, 64, 24, 24)
    model.set_dynamic_inference(False)
    with torch.no_grad():
        y_masked = model(x_eval)
    model.set_dynamic_inference(True)
    with torch.no_grad():
        y_dynamic = model(x_eval)

    torch.testing.assert_close(
        y_dynamic, y_masked, rtol=1e-4, atol=1e-5
    )

    print("SGDR self-test passed.")
    print("Input shape :", tuple(x_eval.shape))
    print("Output shape:", tuple(y_dynamic.shape))
    print("Routing stats:", model.routing_statistics())


if __name__ == "__main__":
    _self_test()
