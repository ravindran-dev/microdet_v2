# src/data/transforms/color.py
from tmp.src.common_imports import *

def random_brightness(img, delta: float):
    return img + random.uniform(-delta, delta)

def random_contrast(img, alpha_low: float, alpha_up: float):
    return img * random.uniform(alpha_low, alpha_up)

def random_saturation(img, alpha_low: float, alpha_up: float):
    hsv = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
    hsv[..., 1] *= random.uniform(alpha_low, alpha_up)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def _normalize(img, mean, std):
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3) / 255.0
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3) / 255.0
    return (img - mean) / std

def color_aug_and_norm(meta, kwargs):
    img = meta["img"].astype(np.float32) / 255.0

    if "brightness" in kwargs and random.random() < 0.5:
        img = random_brightness(img, kwargs["brightness"])

    if "contrast" in kwargs and random.random() < 0.5:
        img = random_contrast(img, *kwargs["contrast"])

    if "saturation" in kwargs and random.random() < 0.5:
        img = random_saturation(img, *kwargs["saturation"])

    if "normalize" in kwargs:
        img = _normalize(img, *kwargs["normalize"])

    meta["img"] = img
    return meta
