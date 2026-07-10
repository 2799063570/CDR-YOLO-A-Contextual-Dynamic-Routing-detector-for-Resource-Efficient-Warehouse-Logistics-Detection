import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================= Inner WIoU: begin =========================
class InnerWIoU(nn.Module):
    """Inner WIoU loss for bounding-box regression.

    This implementation follows the paper formulation:
        1) scale predicted/GT boxes in an auxiliary overlap space;
        2) compute auxiliary IoU I_aux on the scaled boxes;
        3) keep CIoU-style geometric regularization on the original boxes;
        4) apply the dynamic gradient gain phi(beta).

    Boxes are expected to be aligned pairs with shape (..., 4).
    Default hyper-parameters follow the final paper setting:
        ratio=0.9, alpha=1.0, mu=0.5, gamma=2.0
    """

    def __init__(
        self,
        ratio=0.9,
        alpha=1.0,
        mu=0.5,
        gamma=2.0,
        momentum=0.01,
        eps=1e-7,
        gain_clip=10.0,
        detach_gain=True,
    ):
        super().__init__()
        self.ratio = ratio
        self.alpha = alpha
        self.mu = mu
        self.gamma = gamma
        self.momentum = momentum
        self.eps = eps
        self.gain_clip = gain_clip
        self.detach_gain = detach_gain
        self.register_buffer("liou_ema", torch.tensor(1.0))

    @staticmethod
    def _xyxy_to_xywh(box):
        x1, y1, x2, y2 = box.unbind(-1)
        w = (x2 - x1).clamp(min=0)
        h = (y2 - y1).clamp(min=0)
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        return x, y, w, h

    @staticmethod
    def _xywh_to_xyxy(x, y, w, h):
        half_w, half_h = w * 0.5, h * 0.5
        return x - half_w, y - half_h, x + half_w, y + half_h

    def _aligned_iou_xyxy(self, box1, box2):
        """Aligned IoU for box1 and box2, both in xyxy format with shape (..., 4)."""
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)

        inter_w = (torch.minimum(b1_x2, b2_x2) - torch.maximum(b1_x1, b2_x1)).clamp(min=0)
        inter_h = (torch.minimum(b1_y2, b2_y2) - torch.maximum(b1_y1, b2_y1)).clamp(min=0)
        inter = inter_w * inter_h

        area1 = (b1_x2 - b1_x1).clamp(min=0) * (b1_y2 - b1_y1).clamp(min=0)
        area2 = (b2_x2 - b2_x1).clamp(min=0) * (b2_y2 - b2_y1).clamp(min=0)
        union = area1 + area2 - inter
        return inter / (union + self.eps)

    def forward(self, pred, target, xywh=False, return_details=False):
        """Compute aligned Inner WIoU.

        Args:
            pred: predicted boxes, shape (..., 4).
            target: ground-truth boxes, shape (..., 4).
            xywh: True if boxes are in center xywh format, otherwise xyxy.
            return_details: if True, also return original IoU, auxiliary IoU and gain.

        Returns:
            loss with shape (...,). If return_details=True:
            (loss, I_ori, I_aux, phi).
        """
        if pred.numel() == 0:
            empty = pred.new_zeros(pred.shape[:-1])
            return (empty, empty, empty, empty) if return_details else empty

        if xywh:
            x_pr, y_pr, w_pr, h_pr = pred.unbind(-1)
            x_gt, y_gt, w_gt, h_gt = target.unbind(-1)
            pred_xyxy = torch.stack(self._xywh_to_xyxy(x_pr, y_pr, w_pr, h_pr), dim=-1)
            target_xyxy = torch.stack(self._xywh_to_xyxy(x_gt, y_gt, w_gt, h_gt), dim=-1)
        else:
            pred_xyxy, target_xyxy = pred, target
            x_pr, y_pr, w_pr, h_pr = self._xyxy_to_xywh(pred_xyxy)
            x_gt, y_gt, w_gt, h_gt = self._xyxy_to_xywh(target_xyxy)

        # Step 1-3: auxiliary boxes and auxiliary IoU I_aux.
        w_pr_aux, h_pr_aux = w_pr * self.ratio, h_pr * self.ratio
        w_gt_aux, h_gt_aux = w_gt * self.ratio, h_gt * self.ratio

        pred_aux = torch.stack(self._xywh_to_xyxy(x_pr, y_pr, w_pr_aux, h_pr_aux), dim=-1)
        target_aux = torch.stack(self._xywh_to_xyxy(x_gt, y_gt, w_gt_aux, h_gt_aux), dim=-1)
        i_aux = self._aligned_iou_xyxy(pred_aux, target_aux)

        # Original IoU and original-space geometric regularization.
        i_ori = self._aligned_iou_xyxy(pred_xyxy, target_xyxy)

        p_x1, p_y1, p_x2, p_y2 = pred_xyxy.unbind(-1)
        t_x1, t_y1, t_x2, t_y2 = target_xyxy.unbind(-1)

        rho2 = (x_pr - x_gt).pow(2) + (y_pr - y_gt).pow(2)
        cw = torch.maximum(p_x2, t_x2) - torch.minimum(p_x1, t_x1)
        ch = torch.maximum(p_y2, t_y2) - torch.minimum(p_y1, t_y1)
        c2 = cw.pow(2) + ch.pow(2) + self.eps

        v = (4.0 / math.pi ** 2) * (
            torch.atan(w_gt / (h_gt + self.eps)) - torch.atan(w_pr / (h_pr + self.eps))
        ).pow(2)
        lambda_v = v / (1.0 - i_ori + v + self.eps)
        r_geo = rho2 / c2 + lambda_v * v

        l_inner = 1.0 - i_aux + r_geo

        # Step 4-5: outlier degree beta and dynamic gradient gain phi(beta).
        beta = rho2 / c2
        l_iou_star = (1.0 - i_ori).detach()

        if self.training:
            with torch.no_grad():
                cur_liou = l_iou_star.mean().clamp(min=self.eps)
                self.liou_ema.mul_(1.0 - self.momentum).add_(cur_liou * self.momentum)

        liou_norm = self.liou_ema.detach().clamp(min=self.eps)
        phi = (l_iou_star / (liou_norm + self.eps)) * self.alpha * (
            (self.mu ** self.gamma) / ((beta + self.mu).pow(self.gamma) + self.eps)
        )

        if self.gain_clip is not None:
            phi = phi.clamp(min=0.0, max=self.gain_clip)

        # For stable YOLO training, phi is usually used as a gradient gain/weight.
        # Set detach_gain=False if you want beta inside phi to participate in backprop.
        if self.detach_gain:
            phi = phi.detach()

        loss = phi * l_inner
        if return_details:
            return loss, i_ori, i_aux, phi
        return loss


def bbox_inner_wiou(
    pred,
    target,
    xywh=False,
    ratio=0.9,
    alpha=1.0,
    mu=0.5,
    gamma=2.0,
    eps=1e-7,
    liou_ema=None,
    gain_clip=10.0,
    detach_gain=True,
):
    """Functional Inner WIoU.

    This is useful for quick testing. For training, prefer the InnerWIoU module
    because it maintains the moving-average IoU loss L_IoU.
    """
    loss_fn = InnerWIoU(
        ratio=ratio,
        alpha=alpha,
        mu=mu,
        gamma=gamma,
        eps=eps,
        gain_clip=gain_clip,
        detach_gain=detach_gain,
    ).to(pred.device)
    if liou_ema is not None:
        with torch.no_grad():
            loss_fn.liou_ema.fill_(float(liou_ema))
    return loss_fn(pred, target, xywh=xywh, return_details=True)
# ========================== Inner WIoU: end ==========================



class BboxLoss(nn.Module):
    """Criterion class for YOLOv8 box loss, using Inner WIoU by default."""

    def __init__(
        self,
        reg_max,
        use_dfl=False,
        use_inner_wiou=True,
        ratio=0.9,
        alpha=1.0,
        mu=0.5,
        gamma=2.0,
        gain_clip=10.0,
    ):
        """Initialize BboxLoss.

        Args:
            reg_max: DFL regularization maximum.
            use_dfl: whether to use DFL.
            use_inner_wiou: True uses the proposed Inner WIoU; False falls back to CIoU.
            ratio, alpha, mu, gamma: Inner WIoU hyper-parameters from the paper.
        """
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl
        self.use_inner_wiou = use_inner_wiou
        self.inner_wiou = InnerWIoU(
            ratio=ratio,
            alpha=alpha,
            mu=mu,
            gamma=gamma,
            gain_clip=gain_clip,
        )

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, hw=None):
        """Box regression loss.

        pred_bboxes and target_bboxes are YOLO decoded boxes in xyxy format.
        """
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        if self.use_inner_wiou:
            inner_loss, i_ori, i_aux, gain = self.inner_wiou(
                pred_bboxes[fg_mask],
                target_bboxes[fg_mask],
                xywh=False,
                return_details=True,
            )
            loss_iou = (inner_loss.unsqueeze(-1) * weight).sum() / target_scores_sum
        else:
            # Optional fallback: original CIoU branch from the reference loss.py.
            iou = bbox_iou(
                pred_bboxes[fg_mask], target_bboxes[fg_mask],
                xywh=False, GIoU=False, DIoU=False, CIoU=True, EIoU=False, SIoU=False, WIoU=False,
                ShapeIoU=False, hw=hw[fg_mask] if hw is not None else None, mpdiou=False, Inner=False,
                Focaleriou=False, d=0.00, u=0.95, ratio=0.75, eps=1e-7, scale=0.0
            )
            loss_iou = ((1.0 - iou).unsqueeze(-1) * weight).sum() / target_scores_sum

        # DFL loss: keep exactly the YOLOv8 calculation style.
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss.
        """
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


if __name__ == "__main__":
    torch.manual_seed(0)

    pred = torch.tensor([
        [10.0, 10.0, 30.0, 30.0],
        [12.0, 13.0, 31.0, 33.0],
        [50.0, 50.0, 80.0, 90.0],
    ], requires_grad=True)
    target = torch.tensor([
        [10.0, 10.0, 30.0, 30.0],
        [10.0, 10.0, 30.0, 30.0],
        [55.0, 60.0, 88.0, 95.0],
    ])

    criterion = InnerWIoU(ratio=0.9, alpha=1.0, mu=0.5, gamma=2.0)
    loss, i_ori, i_aux, gain = criterion(pred, target, xywh=False, return_details=True)
    print("Inner WIoU loss:", loss)
    print("Original IoU:", i_ori)
    print("Auxiliary IoU:", i_aux)
    print("Dynamic gain:", gain)
    loss.mean().backward()
    print("Backward OK, grad shape:", pred.grad.shape)
