import torch
from tmp.src.common_imports  import *
from tmp.model.loss.dfl import dfl_decode
from tmp.model.module.nms import nms_infer_wrapper
from tmp.model.loss.criterion import DetectionCriterion
from tmp.train.pred_utils import normalize_preds

@torch.no_grad()
def eval_model(model, val_loader, post_cfg):
    device = next(model.parameters()).device

    conf_thres = float(post_cfg.get("conf_thres", 0.90))
    iou_thres = float(post_cfg.get("iou_thres", 0.50))
    max_det = int(post_cfg.get("max_det", 50))
    min_box = int(post_cfg.get("min_box", 12))

    for batch in val_loader:
        images, targets, metas = batch
        images = images.to(device)

       
        out = model(images)
        out = normalize_preds(out)


        if isinstance(out, dict):
            cls_per_level = out["cls_logits"]
            reg_per_level = out["reg_dfl"]
        else:
            cls_per_level = [o["cls"] for o in out]
            reg_per_level = [o["reg"] for o in out]

        B = images.size(0)
        all_boxes = []
        all_scores = []

        for lvl, (cl, rg) in enumerate(zip(cls_per_level, reg_per_level)):
           
            _, _, H, W = cl.shape
            K = rg.shape[1] // 4

            
            cls_flat = cl.permute(0, 2, 3, 1).reshape(B, -1)         
            rg_flat = rg.permute(0, 2, 3, 1).reshape(B, -1, 4 * K)   

           
            ys, xs = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing="ij"
            )
            # Option 1 (recommended): use criterion strides
            # Get strides from model (head or detect)
            if hasattr(model, "strides"):
                strides = model.strides
            elif hasattr(model, "head") and hasattr(model.head, "strides"):
                strides = model.head.strides
            else:
                raise AttributeError("Could not find strides in model or model.head")

            stride = strides[lvl]


            px = (xs + 0.5) * stride
            py = (ys + 0.5) * stride
            pts = torch.stack([px.reshape(-1), py.reshape(-1)], dim=1) 

            
            
            dist = dfl_decode(rg_flat.reshape(-1, 4 * K), reg_max=K - 1)

            dist = dist.view(B, -1, 4)

            x1 = pts[:, 0][None, :] - dist[:, :, 0]
            y1 = pts[:, 1][None, :] - dist[:, :, 1]
            x2 = pts[:, 0][None, :] + dist[:, :, 2]
            y2 = pts[:, 1][None, :] + dist[:, :, 3]

            boxes = torch.stack([x1, y1, x2, y2], dim=2)

           
            w = boxes[:, :, 2] - boxes[:, :, 0]
            h = boxes[:, :, 3] - boxes[:, :, 1]
            valid = (w >= min_box) & (h >= min_box)

            all_boxes.append(boxes)
            all_scores.append(cls_flat)

        boxes_cat = torch.cat(all_boxes, dim=1)       
        scores_cat = torch.cat(all_scores, dim=1)    

        det_boxes, det_scores, det_batch = nms_infer_wrapper(
            scores_cat.unsqueeze(-1),
            boxes_cat,
            conf_thresh=conf_thres,
            iou_thresh=iou_thres,
            max_det=max_det,
        )

        

   
    return {
        "mAP": 0.0,
        "AP50": 0.0,
        "AP_small": 0.0,
        "prec@0.90": 0.0,
        "rec@0.90": 0.0,
        "FPPF@0.90": 0.0,
    }
