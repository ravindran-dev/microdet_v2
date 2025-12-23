from tmp.src.common_imports import torch
from typing import Optional, Dict, Any


class AssignResult:
    def __init__(
        self,
        num_gts: int,
        gt_inds: torch.Tensor,
        max_overlaps: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        self.num_gts = int(num_gts)
        self.gt_inds = gt_inds.long()
        self.max_overlaps = max_overlaps.float()
        self.labels = labels.long() if labels is not None else None
        self._extra_properties: Dict[str, Any] = {}

    @property
    def num_preds(self) -> int:
        return int(self.gt_inds.numel())

    @property
    def info(self) -> Dict[str, Any]:
        out = {
            "num_gts": self.num_gts,
            "num_preds": self.num_preds,
            "gt_inds": self.gt_inds,
            "max_overlaps": self.max_overlaps,
            "labels": self.labels,
        }
        out.update(self._extra_properties)
        return out

    def set_extra_property(self, key: str, value: Any):
        if key in self._extra_properties:
            raise KeyError(f"Property '{key}' already set.")
        self._extra_properties[key] = value

    def get_extra_property(self, key: str):
        return self._extra_properties.get(key, None)

    def add_gt_(self, gt_labels: torch.Tensor):
        device = gt_labels.device
        G = gt_labels.numel()
        new_inds = torch.arange(1, G + 1, dtype=torch.long, device=device)
        self.gt_inds = torch.cat([new_inds, self.gt_inds], dim=0)
        self.max_overlaps = torch.cat([torch.ones(G, device=device), self.max_overlaps], dim=0)
        if self.labels is not None:
            self.labels = torch.cat([gt_labels.long(), self.labels], dim=0)

    def __repr__(self):
        return (
            f"<AssignResult(num_gts={self.num_gts}, "
            f"preds={self.num_preds}, "
            f"gt_inds={tuple(self.gt_inds.shape)}, "
            f"max_overlaps={tuple(self.max_overlaps.shape)}, "
            f"labels={'None' if self.labels is None else tuple(self.labels.shape)})>"
        )
