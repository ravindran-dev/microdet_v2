import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DistributionFocalLoss(nn.Module):
    def __init__(self, reg_max=7, reduction='mean', label_smooth_eps=0.0):
        super().__init__()
        self.reg_max = reg_max
        self.K = reg_max + 1
        self.reduction = reduction
        self.eps = label_smooth_eps

    def forward(self, pred, target):
        N = pred.shape[0]
        pred = pred.view(N, 4, self.K)

        t = target.clamp(min=0.0, max=float(self.reg_max))
        tl = t.floor().long()
        tr = (tl + 1).clamp(max=self.reg_max)

        wl = (tr.float() - t).clamp(min=0.0, max=1.0)
        wr = 1.0 - wl

        if self.eps > 0:
            wl = wl * (1 - self.eps) + self.eps / self.K
            wr = wr * (1 - self.eps) + self.eps / self.K

        logp = F.log_softmax(pred, dim=2)

        tl_logp = logp.gather(2, tl.unsqueeze(-1)).squeeze(-1)
        tr_logp = logp.gather(2, tr.unsqueeze(-1)).squeeze(-1)

        loss = -(wl * tl_logp + wr * tr_logp).sum(dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


@torch.no_grad()
def dfl_decode(pred, reg_max=7):
    K = reg_max + 1
    N = pred.shape[0]
    pred = pred.view(N, 4, K)
    prob = F.softmax(pred, dim=2)
    proj = torch.arange(0, K, device=pred.device, dtype=prob.dtype).view(1, 1, K)
    return (prob * proj).sum(dim=2)
