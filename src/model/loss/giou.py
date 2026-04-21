import torch
import torch.nn as nn

def bbox_overlaps(bboxes1: torch.Tensor, bboxes2: torch.Tensor, is_aligned: bool = False) -> torch.Tensor:
    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]

        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
        union = area1 + area2 - inter
        return inter / union.clamp(min=1e-6)

    N, M = bboxes1.size(0), bboxes2.size(0)
    lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])
    rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def giou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred_area = (pred[:, 2] - pred[:, 0]).clamp(0) * (pred[:, 3] - pred[:, 1]).clamp(0)
    target_area = (target[:, 2] - target[:, 0]).clamp(0) * (target[:, 3] - target[:, 1]).clamp(0)

    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    union = pred_area + target_area - inter
    iou = inter / union.clamp(min=eps)

    enc_lt = torch.min(pred[:, :2], target[:, :2])
    enc_rb = torch.max(pred[:, 2:], target[:, 2:])
    enc_wh = (enc_rb - enc_lt).clamp(min=0)
    enc_area = enc_wh[:, 0] * enc_wh[:, 1]

    giou = iou - (enc_area - union) / enc_area.clamp(min=eps)
    return 1.0 - giou


class GIoULoss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        loss = giou_loss(pred, target)
        if weight is not None:
            loss = loss * weight

        if self.reduction == "mean":
            loss = loss.mean() if avg_factor is None else loss.sum() / max(avg_factor, 1.0)
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss * self.loss_weight
