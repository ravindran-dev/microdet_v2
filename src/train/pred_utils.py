import torch

def normalize_preds(preds):
    if isinstance(preds, dict):
        return preds

    if isinstance(preds, (list, tuple)):
        # ([cls_list], [reg_list])
        if (
            len(preds) == 2
            and isinstance(preds[0], (list, tuple))
            and isinstance(preds[1], (list, tuple))
        ):
            return {
                "cls_logits": list(preds[0]),
                "reg_dfl": list(preds[1]),
            }

        # [{'cls': t, 'reg': t}, ...]
        if all(isinstance(p, dict) for p in preds):
            return preds

    if torch.is_tensor(preds):
        t = preds
        dummy_reg = torch.zeros(
            (t.size(0), 4, t.size(2), t.size(3)),
            device=t.device,
            dtype=t.dtype,
        )
        return {"cls_logits": [t], "reg_dfl": [dummy_reg]}

    raise TypeError(f"Unsupported prediction format: {type(preds)}")
