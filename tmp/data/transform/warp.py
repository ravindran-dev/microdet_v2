# src/data/transforms/warp.py
from tmp.src.common_imports import *

def get_flip_matrix(prob=0.5):
    M = np.eye(3)
    if random.random() < prob:
        M[0, 0] = -1
    return M

def get_perspective_matrix(v=0.0):
    M = np.eye(3)
    M[2, 0] = random.uniform(-v, v)
    M[2, 1] = random.uniform(-v, v)
    return M

def get_rotation_matrix(deg=0.0):
    M = np.eye(3)
    a = random.uniform(-deg, deg)
    M[:2] = cv2.getRotationMatrix2D((0, 0), a, 1.0)
    return M

def get_scale_matrix(ratio=(1, 1)):
    M = np.eye(3)
    s = random.uniform(*ratio)
    M[0, 0] = s
    M[1, 1] = s
    return M

def get_stretch_matrix(wx=(1, 1), hy=(1, 1)):
    M = np.eye(3)
    M[0, 0] *= random.uniform(*wx)
    M[1, 1] *= random.uniform(*hy)
    return M

def get_shear_matrix(deg=0.0):
    M = np.eye(3)
    v = math.tan(random.uniform(-deg, deg) * math.pi / 180)
    M[0, 1] = v
    M[1, 0] = v
    return M

def get_translate_matrix(t, w, h):
    M = np.eye(3)
    M[0, 2] = random.uniform(0.5 - t, 0.5 + t) * w
    M[1, 2] = random.uniform(0.5 - t, 0.5 + t) * h
    return M

def get_resize_matrix(raw, dst, keep_ratio):
    rw, rh = raw
    dw, dh = dst
    M = np.eye(3)
    if keep_ratio:
        C = np.eye(3)
        C[0, 2] = -rw / 2
        C[1, 2] = -rh / 2
        ratio = (dh / rh) if (rw / rh < dw / dh) else (dw / rw)
        R = np.eye(3)
        R[0, 0] = R[1, 1] = ratio
        T = np.eye(3)
        T[0, 2] = 0.5 * dw
        T[1, 2] = 0.5 * dh
        return T @ R @ C
    M[0, 0] = dw / rw
    M[1, 1] = dh / rh
    return M

def warp_boxes(boxes, M, w, h):
    n = len(boxes)
    if n == 0:
        return boxes
    pts = np.ones((n * 4, 3))
    pts[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
    pts = (pts @ M.T)
    pts = (pts[:, :2] / pts[:, 2:3]).reshape(n, 8)
    xs = pts[:, 0::2]
    ys = pts[:, 1::2]
    out = np.stack([xs.min(1), ys.min(1), xs.max(1), ys.max(1)], 1)
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
    return out.astype(np.float32)

class ShapeTransform:
    def __init__(
        self,
        keep_ratio=True,
        perspective=0.0,
        scale=(1.0, 1.0),
        stretch=((1.0, 1.0), (1.0, 1.0)),
        rotation=0.0,
        shear=0.0,
        translate=0.0,
        flip=0.0,
    ):
        self.keep_ratio = keep_ratio
        self.perspective = perspective
        self.scale = scale
        self.stretch = stretch
        self.rotation = rotation
        self.shear = shear
        self.translate = translate
        self.flip = flip

    def __call__(self, meta, dst_shape):
        img = meta["img"]
        h, w = img.shape[:2]

        C = np.eye(3)
        C[0, 2] = -w / 2
        C[1, 2] = -h / 2

        M = (
            get_translate_matrix(self.translate, w, h)
            @ get_flip_matrix(self.flip)
            @ get_shear_matrix(self.shear)
            @ get_rotation_matrix(self.rotation)
            @ get_stretch_matrix(*self.stretch)
            @ get_scale_matrix(self.scale)
            @ get_perspective_matrix(self.perspective)
            @ C
        )
        M = get_resize_matrix((w, h), dst_shape, self.keep_ratio) @ M

        img2 = cv2.warpPerspective(img, M, dst_shape, borderValue=(114, 114, 114))
        meta["img"] = img2
        meta["warp_matrix"] = M

        if "boxes" in meta:
            meta["boxes"] = warp_boxes(meta["boxes"], M, dst_shape[0], dst_shape[1])
        return meta
