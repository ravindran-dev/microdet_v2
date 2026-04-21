import torch

from src.model.postprocess import canonicalize_preds

def normalize_preds(preds):
    return canonicalize_preds(preds)
