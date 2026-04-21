import re
from typing import List, Tuple, Dict, Any

import numpy as np
import torch


np_str_obj_array_pattern = re.compile(r"[SaUO]")


def _stack_images(batch_imgs: List[torch.Tensor]) -> torch.Tensor:
    if isinstance(batch_imgs, torch.Tensor):
        return batch_imgs

    out = None
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None and len(batch_imgs) > 0:
        numel = sum(x.numel() for x in batch_imgs)
        storage = batch_imgs[0].storage()._new_shared(numel)
        out = batch_imgs[0].new(storage)

    return torch.stack(batch_imgs, 0, out=out)


def _to_tensor_from_numpy(arr: np.ndarray) -> torch.Tensor:
    if arr.dtype.kind in ("i", "u"):
        return torch.from_numpy(arr).long()
    if arr.dtype.kind in ("f",):
        return torch.from_numpy(arr).float()
    return torch.from_numpy(arr)


def coco_collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]]
):
    imgs, targets, metas = zip(*batch)

    images = _stack_images(list(imgs)).float()

    targets_out: List[Dict[str, Any]] = []
    for t in targets:
        td: Dict[str, Any] = {}
        for k, v in t.items():
            if isinstance(v, np.ndarray):
                if v.dtype.kind in ("i", "u", "f"):
                    td[k] = _to_tensor_from_numpy(v)
                else:
                    td[k] = v
            elif isinstance(v, torch.Tensor):
                td[k] = v
            else:
                td[k] = v

        td.setdefault("boxes", torch.zeros((0, 4), dtype=torch.float32))
        td.setdefault("labels", torch.zeros((0,), dtype=torch.long))
        targets_out.append(td)

    metas_out: List[Dict[str, Any]] = []
    for m in metas:
        md: Dict[str, Any] = {}
        for k, v in m.items():
            if isinstance(v, np.ndarray) and v.shape == ():
                md[k] = v.item()
            else:
                md[k] = v
        metas_out.append(md)

    return images, targets_out, metas_out
