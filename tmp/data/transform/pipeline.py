# src/data/transforms/pipeline.py
from tmp.src.common_imports  import functools, Dict, Tuple, Dataset
from src import color_aug_and_norm, ShapeTransform

class Pipeline:
    def __init__(self, cfg: Dict, keep_ratio: bool = True):
        self.shape_transform = ShapeTransform(keep_ratio=keep_ratio, **cfg)
        self.color = functools.partial(color_aug_and_norm, kwargs=cfg)

    def __call__(self, dataset: Dataset, meta: Dict, dst_shape: Tuple[int, int]):
        meta = self.shape_transform(meta, dst_shape=dst_shape)
        meta = self.color(meta=meta)
        return meta
