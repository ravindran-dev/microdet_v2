from tmp.src.common_imports  import cv2, np

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw = (new_shape[1] - new_w) / 2
    dh = (new_shape[0] - new_h) / 2

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(dh), int(dh)
    left, right = int(dw), int(dw)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def inverse_letterbox(boxes, ratio, pad, original_shape):
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes /= ratio
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, original_shape[1])
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, original_shape[0])
    return boxes
