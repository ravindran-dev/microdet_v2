import torch
import torch.nn as nn
import torch.nn.functional as F

class QualityFocalLoss(nn.Module):
    def __init__(self, beta=2.0, reduction="mean", eps=1e-6):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, target, weight=None):
        p = torch.sigmoid(logits)

        pos_mask = (target > 0).float()
        neg_mask = 1.0 - pos_mask

        pos_bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pos_w = torch.pow(torch.abs(target - p).clamp(min=self.eps), self.beta)
        pos_loss = pos_bce * pos_w * pos_mask

        neg_bce = F.binary_cross_entropy_with_logits(
            logits, torch.zeros_like(logits), reduction="none"
        )
        neg_w = torch.pow(p.clamp(min=self.eps), self.beta)
        neg_loss = neg_bce * neg_w * neg_mask

        loss = pos_loss + neg_loss

        if weight is not None:
            loss = loss * weight

        if self.reduction == "mean":
            denom = (
                (weight if weight is not None else torch.ones_like(loss))
                .sum()
                .clamp(min=1.0)
            )
            return loss.sum() / denom

        if self.reduction == "sum":
            return loss.sum()

        return loss
