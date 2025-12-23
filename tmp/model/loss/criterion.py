from tmp.src.common_imports import *
from tmp.model.loss.qfl import QualityFocalLoss
from tmp.model.loss.dfl import DistributionFocalLoss
from tmp.model.loss.giou import GIoULoss
from tmp.model.loss.dfl import dfl_decode


class DetectionCriterion(nn.Module):
    def __init__(self, loss_cfg, *, strides, assigner, reg_max=None):
        super().__init__()
        self.strides = list(strides)
        self.assigner = assigner
        self.reg_max = int(loss_cfg.get("reg_max", reg_max if reg_max is not None else 7))

        self.w_qfl = float(loss_cfg.get("lambda_qfl", 1.0))
        self.w_dfl = float(loss_cfg.get("lambda_dfl", 0.25))
        self.w_box = float(loss_cfg.get("lambda_box", 2.0))

        self.qfl = QualityFocalLoss(
            beta=loss_cfg.get("qfl_beta", 2.0),
            reduction="mean",
        )
        self.dfl = DistributionFocalLoss(
            reg_max=self.reg_max,
            reduction="mean",
            label_smooth_eps=loss_cfg.get("dfl_label_smooth_eps", 0.0),
        )
        self.box = GIoULoss(reduction="mean")

    def forward(self, preds, targets):
        if isinstance(preds, dict):
            cls_per_level = preds["cls_logits"]
            reg_per_level = preds["reg_dfl"]
        else:
            cls_per_level = [p["cls"] for p in preds]
            reg_per_level = [p["reg"] for p in preds]

        B = cls_per_level[0].shape[0]
        device = cls_per_level[0].device

        points_all, cls_all, reg_all = [], [], []
        for lvl, (cl, rg) in enumerate(zip(cls_per_level, reg_per_level)):
            _, _, H, W = cl.shape
            ys, xs = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing="ij",
            )
            stride = self.strides[lvl]
            px = (xs + 0.5) * stride
            py = (ys + 0.5) * stride
            pts = torch.stack([px.reshape(-1), py.reshape(-1)], 1)
            points_all.append(pts)
            cls_all.append(cl.permute(0, 2, 3, 1).reshape(B, -1, 1))
            reg_all.append(rg.permute(0, 2, 3, 1).reshape(B, -1, 4 * (self.reg_max + 1)))

        points_all = torch.cat(points_all, 0)
        cls_cat = torch.cat(cls_all, 1)
        reg_cat = torch.cat(reg_all, 1)

        total_qfl, total_dfl, total_box = 0.0, 0.0, 0.0
        num_pos_total = 0

        for b in range(B):
            cls_b = cls_cat[b]
            reg_b = reg_cat[b]

            (
                q_targets,
                pos_mask,
                ltrb_all,
                gt_xyxy,
                *_
            ) = self.assigner.build_targets(points_all, self.strides, targets[b])

            q_targets = q_targets.to(device).float()
            pos_mask = pos_mask.to(device)
            gt_xyxy = gt_xyxy.to(device)

            total_qfl += self.qfl(cls_b, q_targets.unsqueeze(1))

            if pos_mask.any():
                idx = pos_mask.nonzero(as_tuple=False).squeeze(1)

                reg_pos = reg_b.index_select(0, idx)
                ltrb_pos = ltrb_all.index_select(0, idx).to(device)

                total_dfl += self.dfl(reg_pos, ltrb_pos)

                with torch.no_grad():
                    dist = dfl_decode(reg_pos, reg_max=self.reg_max)
                    pts_pos = points_all.index_select(0, idx).to(device)
                    pred_xyxy = torch.stack(
                        [
                            pts_pos[:, 0] - dist[:, 0],
                            pts_pos[:, 1] - dist[:, 1],
                            pts_pos[:, 0] + dist[:, 2],
                            pts_pos[:, 1] + dist[:, 3],
                        ],
                        1,
                    )

                gt_pos = gt_xyxy.index_select(0, idx)
                total_box += self.box(pred_xyxy, gt_pos)

                num_pos_total += idx.numel()
            else:
                total_dfl += reg_b.sum() * 0.0
                total_box += reg_b.sum() * 0.0

        if num_pos_total > 0:
            total_dfl /= num_pos_total
            total_box /= num_pos_total

        total = (
            self.w_qfl * total_qfl
            + self.w_dfl * total_dfl
            + self.w_box * total_box
        )

        return total, self.w_qfl * total_qfl, self.w_dfl * total_dfl, self.w_box * total_box
