from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from src.model.loss.dfl import dfl_decode
from src.model.module.nms import batched_nms_class_agnostic


def canonicalize_preds(preds):
    if isinstance(preds, dict):
        if "cls_logits" in preds and "reg_dfl" in preds:
            return {
                "cls_logits": list(preds["cls_logits"]),
                "reg_dfl": list(preds["reg_dfl"]),
            }
        if "cls" in preds and "reg" in preds:
            return {"cls_logits": [preds["cls"]], "reg_dfl": [preds["reg"]]}

    if isinstance(preds, (list, tuple)):
        if (
            len(preds) == 2
            and isinstance(preds[0], (list, tuple))
            and isinstance(preds[1], (list, tuple))
        ):
            return {"cls_logits": list(preds[0]), "reg_dfl": list(preds[1])}

        if all(isinstance(p, dict) and "cls" in p and "reg" in p for p in preds):
            return {
                "cls_logits": [p["cls"] for p in preds],
                "reg_dfl": [p["reg"] for p in preds],
            }

        if len(preds) == 2 and all(torch.is_tensor(p) for p in preds):
            return {"cls_logits": [preds[0]], "reg_dfl": [preds[1]]}

    if torch.is_tensor(preds):
        t = preds
        dummy_reg = torch.zeros(
            (t.size(0), 4, t.size(2), t.size(3)),
            device=t.device,
            dtype=t.dtype,
        )
        return {"cls_logits": [t], "reg_dfl": [dummy_reg]}

    raise TypeError(f"Unsupported prediction format: {type(preds)}")


def infer_model_strides(model, num_levels: int) -> List[int]:
    if hasattr(model, "strides") and model.strides is not None:
        s = [int(v) for v in model.strides]
        if len(s) >= num_levels:
            return s[:num_levels]

    if hasattr(model, "head") and hasattr(model.head, "strides"):
        s = [int(v) for v in model.head.strides]
        if len(s) >= num_levels:
            return s[:num_levels]

    default = [8, 16, 32]
    if num_levels <= len(default):
        return default[:num_levels]
    out = list(default)
    while len(out) < num_levels:
        out.append(out[-1] * 2)
    return out


def decode_predictions(
    preds,
    *,
    model,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    min_box: float = 0.0,
    score_temperature: float = 1.0,
):
    p = canonicalize_preds(preds)
    cls_per_level = p["cls_logits"]
    reg_per_level = p["reg_dfl"]

    B = cls_per_level[0].shape[0]
    device = cls_per_level[0].device
    strides = infer_model_strides(model, len(cls_per_level))

    all_boxes: List[torch.Tensor] = []
    all_scores: List[torch.Tensor] = []

    for lvl, (cl, rg) in enumerate(zip(cls_per_level, reg_per_level)):
        _, C, H, W = cl.shape
        K = rg.shape[1] // 4

        logits = cl.permute(0, 2, 3, 1).reshape(B, -1, C)
        logits = logits / max(float(score_temperature), 1e-6)
        scores = torch.sigmoid(logits)
        scores, _ = scores.max(dim=-1)

        reg_flat = rg.permute(0, 2, 3, 1).reshape(B, -1, 4 * K)

        ys, xs = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        stride = strides[lvl]
        px = (xs + 0.5) * stride
        py = (ys + 0.5) * stride
        pts = torch.stack([px.reshape(-1), py.reshape(-1)], dim=1)

        dist = dfl_decode(reg_flat.reshape(-1, 4 * K), reg_max=K - 1).view(B, -1, 4)

        x1 = pts[:, 0][None, :] - dist[:, :, 0]
        y1 = pts[:, 1][None, :] - dist[:, :, 1]
        x2 = pts[:, 0][None, :] + dist[:, :, 2]
        y2 = pts[:, 1][None, :] + dist[:, :, 3]
        boxes = torch.stack([x1, y1, x2, y2], dim=2)

        if min_box > 0:
            wh = boxes[:, :, 2:4] - boxes[:, :, 0:2]
            valid = (wh[:, :, 0] >= min_box) & (wh[:, :, 1] >= min_box)
            scores = scores * valid.float()

        all_boxes.append(boxes)
        all_scores.append(scores)

    boxes_cat = torch.cat(all_boxes, dim=1)
    scores_cat = torch.cat(all_scores, dim=1)

    return batched_nms_class_agnostic(
        boxes_cat,
        scores_cat,
        iou_thresh=float(iou_thres),
        conf_thresh=float(conf_thres),
        max_det=int(max_det),
    )


def undo_letterbox(
    boxes_xyxy: torch.Tensor,
    letterbox_meta: Dict,
):
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy

    r = float(letterbox_meta.get("scale", 1.0))
    top = float(letterbox_meta.get("pad_top", 0.0))
    left = float(letterbox_meta.get("pad_left", 0.0))
    ow = float(letterbox_meta.get("orig_w", 0.0))
    oh = float(letterbox_meta.get("orig_h", 0.0))

    out = boxes_xyxy.clone()
    out[:, [0, 2]] = (out[:, [0, 2]] - left) / max(r, 1e-6)
    out[:, [1, 3]] = (out[:, [1, 3]] - top) / max(r, 1e-6)

    if ow > 0 and oh > 0:
        out[:, [0, 2]] = out[:, [0, 2]].clamp(0.0, ow)
        out[:, [1, 3]] = out[:, [1, 3]].clamp(0.0, oh)

    return out